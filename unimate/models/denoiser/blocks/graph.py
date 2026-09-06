"""Graph-aware spatio-temporal transformer blocks for UniMate.

Contains attention modules that incorporate skeleton graph structure
(distance + edge-type biases) into the spatial attention, plus a
factored spatial-temporal transformer block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from unimate.models.denoiser.blocks.common import (
    LlamaRMSNorm,
    SwiGLUFFN,
    modulate,
)
from unimate.models.denoiser.rope import SpectralJointRoPE


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def build_graph_bias_params(module, dim, num_heads, max_path_len=5, edge_types=6):
    """Register the Graphormer bias tables on *module*, in place.

    A free function rather than a base class so the parameters can live either
    directly on an attention (their historical home, and the names checkpoints
    carry) or on a standalone :class:`GraphAttnBias`, with one definition and
    one registration order either way.
    """
    module.graph_dist_embedding = nn.Embedding(max_path_len + 1, dim // 4)
    module.graph_dist_proj = nn.Linear(dim // 4, num_heads)
    module.graph_dist_scale = nn.Parameter(torch.ones(1) * 0.02)

    module.graph_rel_embedding = nn.Embedding(edge_types, dim // 4)
    module.graph_rel_proj = nn.Linear(dim // 4, num_heads)
    module.graph_rel_scale = nn.Parameter(torch.ones(1) * 0.02)


def compute_graph_bias(module, cond, attention_mask):
    """Masked graph bias at (B, H, J, J) from *module*'s tables.

    ``A_ij = QK^T/sqrt(d) + b_dist(i,j) + b_edge(i,j)`` — Graphormer's spatial
    and edge encodings, here over the skeleton graph. Shared across frames.
    """
    dist_emb = module.graph_dist_proj(
        module.graph_dist_embedding(cond["graph_dist"])
    ) * module.graph_dist_scale
    rel_emb = module.graph_rel_proj(
        module.graph_rel_embedding(cond["joint_relations"])
    ) * module.graph_rel_scale

    # (B, J, J, H) -> (B, H, J, J)
    graph_bias = (dist_emb + rel_emb).permute(0, 3, 1, 2)
    # Joint validity folded in here rather than applied as a second mask:
    # attention_mask is (B, 1, 1, J) and broadcasts over the query axis.
    return graph_bias.masked_fill(~attention_mask.to(torch.bool), float('-inf'))


class GraphAttnBias(nn.Module):
    """The graph bias as a standalone module, for sharing across layers.

    Graphormer itself learns one bias for the whole stack. Held by the model
    (see ``share_graph_attn_bias``) it is evaluated once per forward and handed
    to every layer, instead of ``num_layers`` private copies each rebuilding
    their own. Per-layer copies are the default: more expressive, at
    ``num_layers`` times these (small) tables.
    """

    def __init__(self, dim, num_heads, max_path_len=5, edge_types=6):
        super().__init__()
        build_graph_bias_params(self, dim, num_heads, max_path_len, edge_types)

    def forward(self, cond, attention_mask):
        return compute_graph_bias(self, cond, attention_mask)


class SpatialGraphSelfAttention(nn.Module):
    """Spatial self-attention over joints, optionally biased by skeleton graph.
    Operates per-frame: reshapes (B, F, J, D) → (B*F, J, D).

    With ``use_graph_attn_bias=True`` (default): adds learned graph-distance
    and edge-type biases. Bias (B, H, J, J) is shared across frames and
    broadcast over the frame dim during manual attention, avoiding a full
    (B*F, H, J, J) copy and keeping the path SDPA-free.

    With ``use_graph_attn_bias=False`` (ablation): plain self-attention via
    SDPA; the joint validity mask (folded into the bias when on) is supplied
    as SDPA's attn_mask instead.

    ``own_graph_bias=False`` leaves the bias parameters out of this module: the
    model then holds a single :class:`GraphAttnBias` for the whole stack and
    hands the computed ``graph_bias`` to every layer's ``forward``.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, rope=None, qk_norm=True,
                 max_path_len=5, edge_types=6, attn_drop=0., proj_drop=0.,
                 use_graph_attn_bias=True, own_graph_bias=True, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope = rope
        self.use_graph_attn_bias = use_graph_attn_bias

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

        # Graph distance/edge-type biases — skipped when ablated (the joint mask
        # then goes through SDPA's attn_mask in forward instead), and skipped
        # when the model owns one shared bias and passes it into forward. When
        # we do own them they hang directly off this module, the names and the
        # registration order existing checkpoints were written with.
        self.own_graph_bias = self.use_graph_attn_bias and own_graph_bias
        if self.own_graph_bias:
            build_graph_bias_params(self, dim, num_heads, max_path_len, edge_types)

    def forward(self, x, attention_mask=None, position_ids=None, cond=None,
                graph_bias=None):
        B, Fr, J, _ = x.shape
        C = self.num_heads * self.head_dim

        # QKV projection for all frames at once
        x_flat = einops.rearrange(x, 'b f j d -> (b f) j d')
        qkv = self.qkv(x_flat).reshape(B * Fr, J, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*F, H, J, D)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            if isinstance(self.rope, SpectralJointRoPE):
                # Spectral rope: angles depend only on (B, J), so the encoder
                # runs once and the rotation broadcasts across the frame dim.
                spectral_coords = cond['spectral_feats']  # (B, J, K)
                q, k = self.rope.forward_per_frame(
                    q, k, spectral_coords, n_frames=Fr,
                )
            else:
                q, k = self.rope(q, k, position_ids)

        if self.use_graph_attn_bias:
            # Graph bias: (B, H, J, J) — shared across all frames. Supplied by
            # the model when it owns one bias for every layer, else our own.
            if graph_bias is None:
                assert self.own_graph_bias, (
                    'This attention was built with own_graph_bias=False, so '
                    'the model must pass the shared graph_bias into forward.')
                graph_bias = compute_graph_bias(self, cond, attention_mask)
            # Manual attention with broadcasting: avoid expanding (B,H,J,J) bias
            # to (B*F,H,J,J) — broadcast over the frame dim instead.
            q = q.reshape(B, Fr, self.num_heads, J, self.head_dim)
            k = k.reshape(B, Fr, self.num_heads, J, self.head_dim)
            v = v.reshape(B, Fr, self.num_heads, J, self.head_dim)

            attn_weights = (q @ k.transpose(-1, -2)) * self.scale  # (B, F, H, J, J)
            attn_weights = attn_weights + graph_bias.unsqueeze(1)   # broadcast (B, 1, H, J, J)
            # Softmax in fp32. This path is hand-rolled rather than SDPA (the
            # bias cannot broadcast over the frame axis once frames are folded
            # into the batch), so it does not inherit SDPA's internal fp32
            # accumulation: under bf16 autocast a bf16 softmax here costs about
            # an order of magnitude of accuracy against the full-attention
            # variant, which would skew any comparison between them. Free in
            # fp32, where .float() is a no-op.
            attn_weights = attn_weights.float().softmax(dim=-1).to(v.dtype)
            drop_p = self.attn_drop.p if self.training else 0.
            if drop_p > 0.:
                attn_weights = F.dropout(attn_weights, p=drop_p, training=self.training)
            out = (attn_weights @ v).reshape(B * Fr, self.num_heads, J, self.head_dim)
        else:
            # Ablation: no graph bias. Apply joint validity via SDPA's attn_mask
            # (broadcast across frames) so padded joints are still excluded.
            attn_mask = einops.repeat(
                attention_mask, 'b 1 1 j -> (b f) 1 1 j', f=Fr,
            ).to(torch.bool)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

        out = out.transpose(1, 2).reshape(B * Fr, J, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        out = einops.rearrange(out, '(b f) j d -> b f j d', b=B, f=Fr)
        return out


class TemporalSelfAttention(nn.Module):
    """Temporal self-attention applied per-joint.
    Reshapes (B, F, J, D) → (B*J, F, D).
    """
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

    def forward(self, x, attention_mask=None, position_ids=None):
        B_orig, Fr, J, _ = x.shape
        x = einops.rearrange(x, 'b f j d -> (b j) f d')
        attention_mask = einops.repeat(attention_mask, 'b 1 1 f -> (b j) 1 1 f', j=J).to(torch.bool)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q, k, position_ids)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = einops.rearrange(x, '(b j) f d -> b f j d', b=B_orig, j=J)
        return x


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class SpatioTemporalBlock(nn.Module):
    """Spatial self-attn (graph bias optional) → Temporal self-attn → MLP.
    All operate on x: (B, F, J, D), c: (B, D).
    """
    def __init__(self, hidden_size, num_heads, mlp_size=1024,
                 rope_spatial=None, rope_temporal=None,
                 qk_norm=True, max_path_len=5, edge_types=6, dropout=0.0,
                 use_graph_attn_bias=True, own_graph_bias=True,
                 gradient_checkpointing=False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        self.norm_s = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.s_attn = SpatialGraphSelfAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            qk_norm=qk_norm, rope=rope_spatial,
            max_path_len=max_path_len, edge_types=edge_types,
            attn_drop=dropout, proj_drop=dropout,
            use_graph_attn_bias=use_graph_attn_bias,
            own_graph_bias=own_graph_bias,
        )

        self.norm_t = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.t_attn = TemporalSelfAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            qk_norm=qk_norm, rope=rope_temporal,
            attn_drop=dropout, proj_drop=dropout,
        )

        self.norm_mlp = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_size))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def _forward_spatial(self, xs, spatial_mask, position_ids_spatial, cond,
                         graph_bias):
        return self.s_attn(xs, attention_mask=spatial_mask,
                           position_ids=position_ids_spatial, cond=cond,
                           graph_bias=graph_bias)

    def _forward_temporal(self, xt, temporal_mask, position_ids_temporal):
        return self.t_attn(xt, attention_mask=temporal_mask,
                           position_ids=position_ids_temporal)

    def forward(self, x, c, spatial_mask=None, temporal_mask=None,
                position_ids_spatial=None, position_ids_temporal=None, cond=None,
                graph_bias=None):
        dtype = x.dtype
        (shift_s, scale_s, gate_s,
         shift_t, scale_t, gate_t,
         shift_m, scale_m, gate_m) = self.adaLN_modulation(c).chunk(9, dim=-1)

        def expand_to_x(z):
            return z[:, None, None, :]

        use_ckpt = self.gradient_checkpointing and self.training

        # spatial attn
        xs = self.norm_s(x.to(torch.float32)).to(dtype)
        xs = modulate(xs, expand_to_x(shift_s), expand_to_x(scale_s))
        if use_ckpt:
            xs = torch.utils.checkpoint.checkpoint(
                self._forward_spatial, xs, spatial_mask,
                position_ids_spatial, cond, graph_bias, use_reentrant=False,
            )
        else:
            xs = self._forward_spatial(xs, spatial_mask, position_ids_spatial,
                                       cond, graph_bias)
        x = x + expand_to_x(gate_s) * xs

        # temporal attn
        xt = self.norm_t(x.to(torch.float32)).to(dtype)
        xt = modulate(xt, expand_to_x(shift_t), expand_to_x(scale_t))
        if use_ckpt:
            xt = torch.utils.checkpoint.checkpoint(
                self._forward_temporal, xt, temporal_mask,
                position_ids_temporal, use_reentrant=False,
            )
        else:
            xt = self._forward_temporal(xt, temporal_mask, position_ids_temporal)
        x = x + expand_to_x(gate_t) * xt

        # mlp
        xm = self.norm_mlp(x.to(torch.float32)).to(dtype)
        xm = modulate(xm, expand_to_x(shift_m), expand_to_x(scale_m))
        xm = self.mlp(xm)
        x = x + expand_to_x(gate_m) * xm

        return x
