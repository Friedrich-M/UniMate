"""Cross-attention variant of the graph spatio-temporal transformer.

Same factored spatial (graph-biased) + temporal attention as
``blocks_graph``, with an added text cross-attention stage per
block. Time and tpos still drive AdaLN; text is routed through cross-attn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from unimate.models.denoiser.blocks.common import (
    LlamaRMSNorm,
    SwiGLUFFN,
    modulate,
)
from unimate.models.denoiser.blocks.graph import (
    SpatialGraphSelfAttention,
    TemporalSelfAttention,
)


# ---------------------------------------------------------------------------
# Cross-attention (text as K/V)
# ---------------------------------------------------------------------------

class TextCrossAttention(nn.Module):
    """Cross-attention with x as query and text memory as key/value.

    x: (B, F, J, D) -> flattened internally to (B, F*J, D).
    text_mem: (B, L_text, D). No rope; no positional structure on text.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True,
                 attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if qk_norm:
            self.q_norm = LlamaRMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = LlamaRMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, text_mem, text_mask=None):
        B, Fr, J, D = x.shape
        N = Fr * J
        L = text_mem.shape[1]

        q = self.q_proj(x.reshape(B, N, D))
        kv = self.kv_proj(text_mem)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = kv.reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # text_mask: (B, L) bool with True = valid. Build SDPA attn_mask where
        # True = keep, False = mask (PyTorch SDPA convention for bool masks).
        attn_mask = None
        if text_mask is not None:
            attn_mask = text_mask.to(torch.bool)[:, None, None, :]  # (B,1,1,L)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out.reshape(B, Fr, J, D)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class SpatioTemporalCrossBlock(nn.Module):
    """Spatial (graph) → Temporal → Text cross-attn → MLP.
    x: (B, F, J, D), c: (B, D), text_mem: (B, L_text, D).
    """
    def __init__(self, hidden_size, num_heads, mlp_size=1024,
                 rope_spatial=None, rope_temporal=None,
                 qk_norm=True, max_path_len=5, edge_types=6, dropout=0.0,
                 own_graph_bias=True,
                 gradient_checkpointing=False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        self.norm_s = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.s_attn = SpatialGraphSelfAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            qk_norm=qk_norm, rope=rope_spatial,
            max_path_len=max_path_len, edge_types=edge_types,
            attn_drop=dropout, proj_drop=dropout,
            own_graph_bias=own_graph_bias,
        )

        self.norm_t = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.t_attn = TemporalSelfAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            qk_norm=qk_norm, rope=rope_temporal,
            attn_drop=dropout, proj_drop=dropout,
        )

        self.norm_c = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.c_attn = TextCrossAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            qk_norm=qk_norm, attn_drop=dropout, proj_drop=dropout,
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

    def _forward_cross(self, xc, text_mem, text_mask):
        return self.c_attn(xc, text_mem, text_mask=text_mask)

    def forward(self, x, c, text_mem, spatial_mask=None, temporal_mask=None,
                text_mask=None,
                position_ids_spatial=None, position_ids_temporal=None, cond=None,
                graph_bias=None):
        dtype = x.dtype
        # The cross stage takes no modulation chunk: it gets its own norm and
        # is added plainly, as in Wan 2.1 / PixArt-alpha. See DiTCrossBlock.
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

        # text cross attn
        xc = self.norm_c(x.to(torch.float32)).to(dtype)
        if use_ckpt:
            xc = torch.utils.checkpoint.checkpoint(
                self._forward_cross, xc, text_mem, text_mask,
                use_reentrant=False,
            )
        else:
            xc = self._forward_cross(xc, text_mem, text_mask)
        x = x + xc

        # mlp
        xm = self.norm_mlp(x.to(torch.float32)).to(dtype)
        xm = modulate(xm, expand_to_x(shift_m), expand_to_x(scale_m))
        xm = self.mlp(xm)
        x = x + expand_to_x(gate_m) * xm

        return x
