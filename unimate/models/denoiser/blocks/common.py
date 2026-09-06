"""Layers and blocks shared by every denoiser variant.

- Primitives: ``modulate`` (adaLN), ``LlamaRMSNorm``, ``SwiGLUFFN``
- Embedders: ``TimestepEmbedder``, ``InputLayer``, ``TposPool`` /
  ``TposCrossAttentionPool``
- Full attention: ``RoPEAttention``, ``DiTBlock`` (adaLN only) and
  ``DiTCrossBlock`` (adaLN + a text cross-attention stage) — the two blocks
  the ``attention='full'`` variants stack
- Output: ``FinalLayer``

The factored (``attention='graph'``) attention lives in
:mod:`unimate.models.denoiser.blocks.graph`, text cross-attention in
:mod:`unimate.models.denoiser.blocks.cross`, and the variant classes that
assemble all of this in :mod:`unimate.models.denoiser`.
"""

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from unimate.models.denoiser.rope import GraphRoPE

__all__ = [
    "modulate",
    "LlamaRMSNorm",
    "SwiGLUFFN",
    "TimestepEmbedder",
    "TposCrossAttentionPool",
    "TposPool",
    "InputLayer",
    "RoPEAttention",
    "DiTBlock",
    "DiTCrossBlock",
    "FinalLayer",
]



# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features, bias: bool = True) -> None:
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, dtype=torch.float32):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
        ).to(device=t.device, dtype=dtype)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype=torch.bfloat16):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, dtype=dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class TposCrossAttentionPool(nn.Module):
    """PMA-style attention pool (Set Transformer / Perceiver Resampler).

    K learnable queries cross-attend to per-joint tpos embeddings, with
    residual + FFN (both pre-LN), then reduce K queries to one (B, D)
    vector via mean + RMSNorm + linear projection.

    Design notes:
      * Residual on the cross-attention output keeps a gradient shortcut so
        the queries cannot drift wholesale away from initialization.
      * A post-attention SwiGLU FFN with its own residual gives the standard
        "attention + FFN" representational power.
      * The K → 1 reduction is mean(K) followed by Linear(D → D) rather than
        Linear(K·D → D), which keeps the parameter count independent of K
        and bounds the output magnitude.
      * A final RMSNorm before the projection caps the magnitude flowing
        into the adaLN driver ``y``.
    """

    def __init__(self, latent_dim, num_queries=4, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, latent_dim) * 0.02)

        # Cross-attention block (pre-LN + residual).
        self.norm_q = LlamaRMSNorm(latent_dim, eps=1e-6)
        self.norm_kv = LlamaRMSNorm(latent_dim, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, batch_first=True,
        )

        # FFN block (pre-LN + residual).
        self.norm_ff = LlamaRMSNorm(latent_dim, eps=1e-6)
        self.ffn = SwiGLUFFN(latent_dim, int(latent_dim * mlp_ratio))

        # K → 1 reduction (mean across queries, then norm + project).
        self.norm_out = LlamaRMSNorm(latent_dim, eps=1e-6)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, tpos_emb, joints_valid):
        """
        Args:
            tpos_emb: (B, J, latent_dim) per-joint tpos embeddings.
            joints_valid: (B, J) boolean mask (True = valid joint).
        Returns:
            (B, latent_dim) pooled conditioning vector.
        """
        B = tpos_emb.shape[0]
        q = self.queries.expand(B, -1, -1)  # (B, K, D)

        # Cross-attention with residual. K/V padding handled via PyTorch's
        # key_padding_mask (True = ignore). Queries are not padded.
        key_padding_mask = ~joints_valid  # (B, J)
        kv = self.norm_kv(tpos_emb)
        attn_out, _ = self.cross_attn(
            self.norm_q(q), kv, kv, key_padding_mask=key_padding_mask,
        )
        q = q + attn_out  # (B, K, D)

        # FFN with residual.
        q = q + self.ffn(self.norm_ff(q))  # (B, K, D)

        # Reduce K → 1: mean is magnitude-bounded; out_proj is zero-inited
        # by ``UniMateDenoiserBase.initialize_weights`` so its contribution to ``y`` starts
        # at zero (adaLN-Zero pattern).
        pooled = q.mean(dim=1)  # (B, D)
        return self.out_proj(self.norm_out(pooled))  # (B, D)


class TposPool(nn.Module):
    """Pools per-joint tpos embeddings into a single (B, latent_dim) vector.

    num_queries == 0 -> masked mean-pool + linear projection.
    num_queries  > 0 -> cross-attention with K learnable queries.
    """

    def __init__(self, latent_dim, num_queries=0):
        super().__init__()
        self.num_queries = num_queries
        if num_queries > 0:
            self.pool = TposCrossAttentionPool(latent_dim, num_queries=num_queries)
        else:
            self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, tpos_emb, joints_valid):
        """
        Args:
            tpos_emb: (B, J, latent_dim) per-joint tpos embeddings.
            joints_valid: (B, J) boolean mask (True = valid joint).
        Returns:
            (B, latent_dim)
        """
        if self.num_queries > 0:
            return self.pool(tpos_emb, joints_valid)
        mask = joints_valid.unsqueeze(-1).to(tpos_emb.dtype)  # (B, J, 1)
        token_count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        pooled = (tpos_emb * mask).sum(dim=1) / token_count  # (B, latent_dim)
        return self.proj(pooled)


class InputLayer(nn.Module):
    """Encodes raw motion and tpos features into latent tokens.

    Handles root/joint separation (root = joint index 0 with global motion,
    others = local rotations), 2-layer MLP encoding, and tpos+motion
    concatenation. Tpos pooling (for adaLN conditioning) is handled
    externally by the optional ``TposPool`` module.

    Args:
        feature_len: Per-joint feature dimension (12 for the UniMate representation).
        latent_dim: Transformer hidden dimension.
        max_joints: Maximum number of joints (for padding / masking).
        concat_parent_features: If True, non-root joint tpos input is concatenated
            with parent joint features (2*D input dim). Root joint is unaffected.
    """

    def __init__(self, feature_len, latent_dim, max_joints,
                 concat_parent_features=False):
        super().__init__()
        self.max_joints = max_joints
        self.concat_parent_features = concat_parent_features

        # Root embedders always take D features (root has no meaningful parent)
        self.root_tpos_embedder = nn.Sequential(
            nn.Linear(feature_len, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.root_x_embedder = nn.Sequential(
            nn.Linear(feature_len, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
        )
        # Non-root joint embedders
        self.joint_tpos_embedder = nn.Sequential(
            nn.Linear(feature_len, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
        )
        if concat_parent_features:
            self.parent_tpos_embedder = nn.Sequential(
                nn.Linear(feature_len, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
            )
            self.tpos_fuse = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
            )
        self.joint_x_embedder = nn.Sequential(
            nn.Linear(feature_len, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x, cond):
        """
        Args:
            x: (B, J, D, F) raw motion tensor.
            cond: dict with 'tpos_first_frame', 'n_joints',
                  optionally 'tpos_first_frame_parents'.

        Returns:
            x: (B, F+1, J, latent_dim) embedded tokens (tpos frame prepended).
            tpos_emb: (B, J, latent_dim) per-joint tpos embeddings (pre-prepend).
            joints_valid: (B, J) boolean validity mask.
        """
        device = x.device

        # --- Tpos encoding ---
        tpos = cond['tpos_first_frame'].to(device)  # (B, J, D)
        tpos = einops.rearrange(tpos, 'b j d -> b 1 j d')  # (B, 1, J, D)

        tpos_root = self.root_tpos_embedder(tpos[:, :, :1, :])  # (B, 1, 1, latent_dim)
        tpos_joint_proj = self.joint_tpos_embedder(tpos[:, :, 1:, :])  # (B, 1, J-1, latent_dim)
        if self.concat_parent_features:
            tpos_parents = cond['tpos_first_frame_parents'].to(device)  # (B, J, D)
            tpos_parents = einops.rearrange(tpos_parents, 'b j d -> b 1 j d')
            tpos_parent_proj = self.parent_tpos_embedder(tpos_parents[:, :, 1:, :])  # (B, 1, J-1, latent_dim)
            tpos_joints = self.tpos_fuse(torch.cat([tpos_joint_proj, tpos_parent_proj], dim=-1))  # (B, 1, J-1, latent_dim)
        else:
            tpos_joints = tpos_joint_proj  # (B, 1, J-1, latent_dim)
        tpos_emb = torch.cat([tpos_root, tpos_joints], dim=2)  # (B, 1, J, latent_dim)

        joints_valid = self._lengths_to_mask(cond['n_joints'], self.max_joints)  # (B, J)

        # --- Motion encoding ---
        x = einops.rearrange(x, 'b j d f -> b f j d')  # (B, F, J, D)
        x_root = self.root_x_embedder(x[:, :, :1, :])    # (B, F, 1, latent_dim)
        x_joints = self.joint_x_embedder(x[:, :, 1:, :])  # (B, F, J-1, latent_dim)
        x = torch.cat([x_root, x_joints], dim=2)  # (B, F, J, latent_dim)

        # --- Prepend conditioning frame ---
        x = torch.cat([tpos_emb, x], dim=1)  # (B, F+1, J, latent_dim)
        return x, tpos_emb[:, 0], joints_valid

    @staticmethod
    def _lengths_to_mask(lengths, max_len):
        return torch.arange(max_len, device=lengths.device).expand(
            len(lengths), max_len
        ) < lengths.unsqueeze(1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, rope=None, qk_norm=True,
                 attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope = rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if qk_norm:
            self.q_norm = LlamaRMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = LlamaRMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, position_ids=None, attention_mask=None, cond=None,
                nframes=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            # GraphRoPE: spectral coords from cond, plus nframes for native (T*J)
            if isinstance(self.rope, GraphRoPE):
                spectral_coords = cond['spectral_feats']
                q, k = self.rope(q, k, spectral_coords, nframes=nframes)
            else:
                q, k = self.rope(q, k, position_ids)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Full-attention transformer block with adaLN-Zero modulation.
    Operates on flattened tokens: x (B, L, D), c (B, D).
    """
    def __init__(self, hidden_size, num_heads, mlp_size=1024, rope=None,
                 qk_norm=True, dropout=0.0):
        super().__init__()
        self.norm1 = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                  qk_norm=qk_norm, rope=rope,
                                  attn_drop=dropout, proj_drop=dropout)
        self.norm2 = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attention_mask=None, position_ids=None, cond=None,
                nframes=None):
        dtype = x.dtype
        # c: (B, D) → unsqueeze for broadcasting with (B, L, D)
        c = c.unsqueeze(1) if c.dim() == 2 else c
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        norm_x1 = self.norm1(x.to(torch.float32)).to(dtype)
        attn_input_x = modulate(norm_x1, shift_msa, scale_msa)
        attn_output_x = self.attn(attn_input_x, attention_mask=attention_mask,
                                  position_ids=position_ids, cond=cond,
                                  nframes=nframes)
        x = x + gate_msa * attn_output_x

        norm_x2 = self.norm2(x.to(torch.float32)).to(dtype)
        gate_input_x = modulate(norm_x2, shift_mlp, scale_mlp)
        gate_output_x = self.mlp(gate_input_x)
        x = x + gate_mlp * gate_output_x
        return x


# ---------------------------------------------------------------------------
# Output layers
# ---------------------------------------------------------------------------

class DiTCrossBlock(nn.Module):
    """Full-attention block with a text cross-attention stage.

    The ``full`` + ``cross_attn`` pairing: one self-attention over the
    flattened spatio-temporal tokens, then a stage where every token attends
    the caption's token sequence as K/V, then the MLP.

    Self-attention and the MLP are adaLN-Zero gated; the cross stage is not.
    It gets its own norm and is added plainly, which is what Wan 2.1 does
    (its 6 modulation chunks all go to self-attn and the FFN) and what
    PixArt-alpha does (no norm either). adaLN here is driven by the timestep
    alone, so a cross gate could only say "how much text at step t" — a
    freedom neither reference needs, and one that costs 3*d^2 per layer and
    invites the failure where a zero gate and a zero output projection
    deadlock each other. The cross branch instead starts silent through a
    zero-init ``c_attn.proj``, which stays learnable because nothing
    multiplies its output by zero.

    Operates on flattened tokens: ``x (B, L, D)``, ``c (B, D)``,
    ``text_mem (B, T, D)``.
    """

    def __init__(self, hidden_size, num_heads, mlp_size=1024, rope=None,
                 qk_norm=True, dropout=0.0):
        super().__init__()
        from unimate.models.denoiser.blocks.cross import TextCrossAttention
        self.norm1 = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads=num_heads,
                                  qkv_bias=True, qk_norm=qk_norm, rope=rope,
                                  attn_drop=dropout, proj_drop=dropout)
        self.norm_c = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.c_attn = TextCrossAttention(hidden_size, num_heads=num_heads,
                                         qkv_bias=True, qk_norm=qk_norm,
                                         attn_drop=dropout, proj_drop=dropout)
        self.norm2 = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, text_mem, attention_mask=None, text_mask=None,
                position_ids=None, cond=None, nframes=None):
        dtype = x.dtype
        c = c.unsqueeze(1) if c.dim() == 2 else c
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=-1)

        h = modulate(self.norm1(x.to(torch.float32)).to(dtype), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(h, attention_mask=attention_mask,
                                     position_ids=position_ids, cond=cond,
                                     nframes=nframes)

        # TextCrossAttention works on (B, Fr, J, D); the flattened token axis
        # is fed as a single "frame" so the same module serves both variants.
        h = self.norm_c(x.to(torch.float32)).to(dtype)
        x = x + self.c_attn(h.unsqueeze(2), text_mem, text_mask=text_mask).squeeze(2)

        h = modulate(self.norm2(x.to(torch.float32)).to(dtype), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    """Output projection shared by all UniMate variants.
    Accepts x as (B, F, J, D) or (B, F*J, D) and outputs (B, D_out, F, J).

    Mirrors InputLayer's root/joint split: separate 2-layer MLPs for root
    (global velocity + orientation) and non-root joints (local rotations).
    Before projection, root tokens aggregate whole-body information from
    joint tokens via per-frame cross-attention.
    """
    def __init__(self, hidden_size, output_size, joint=24, num_heads=4):
        super().__init__()
        self.norm_final = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.joint = joint

        # Root aggregation: cross-attention from root (query) to joints (key/value)
        self.root_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True,
        )
        self.root_cross_norm_q = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.root_cross_norm_kv = LlamaRMSNorm(hidden_size, eps=1e-6)

        # Separate output heads for root vs non-root joints
        self.root_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, output_size),
        )
        self.joint_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, output_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c, joints_valid=None):
        # c: (B, D)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)

        if x.dim() == 4:
            B, Fm, J, D = x.shape
            x = x.reshape(B, Fm * J, D)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError("x must be (B,F,J,D) or (B,F*J,D)")

        norm_x = self.norm_final(x.to(torch.float32)).to(x.dtype)
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        x = modulate(norm_x, shift, scale)

        # Unflatten to (B, F, J, D) for root/joint split
        x = einops.rearrange(x, 'b (f j) d -> b f j d', j=self.joint)
        B, F, J, D = x.shape

        root_tokens = x[:, :, :1, :]    # (B, F, 1, D)
        joint_tokens = x[:, :, 1:, :]   # (B, F, J-1, D)

        # Per-frame cross-attention: root queries joints for whole-body aggregation
        root_flat = root_tokens.reshape(B * F, 1, D)
        joints_flat = joint_tokens.reshape(B * F, J - 1, D)
        # key_padding_mask: True = ignore (PyTorch convention), exclude padding joints
        if joints_valid is not None:
            # joints_valid: (B, J) -> exclude root col -> (B, J-1) -> expand per frame -> (B*F, J-1)
            kv_mask = ~joints_valid[:, 1:]  # (B, J-1), True = padding
            kv_mask = kv_mask.unsqueeze(1).expand(-1, F, -1).reshape(B * F, J - 1)
        else:
            kv_mask = None
        root_agg, _ = self.root_cross_attn(
            self.root_cross_norm_q(root_flat),
            self.root_cross_norm_kv(joints_flat),
            joints_flat,
            key_padding_mask=kv_mask,
        )
        root_tokens = root_tokens + root_agg.reshape(B, F, 1, D)

        root = self.root_out(root_tokens)          # (B, F, 1, D_out)
        joints = self.joint_out(joint_tokens)       # (B, F, J-1, D_out)
        x = torch.cat([root, joints], dim=2)        # (B, F, J, D_out)

        x = einops.rearrange(x, 'b f j d -> b d f j')
        return x


