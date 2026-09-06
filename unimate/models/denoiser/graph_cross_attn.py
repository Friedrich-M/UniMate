"""``attention='graph'`` + ``text_cond='cross_attn'`` — factored + text stage."""

import torch
import torch.nn as nn
import einops

from unimate.models.denoiser.base import UniMateDenoiserBase
from unimate.models.denoiser.blocks.cross import SpatioTemporalCrossBlock
from unimate.models.denoiser.blocks.graph import GraphAttnBias


class UniMateGraphCrossAttn(UniMateDenoiserBase):
    """Factored graph attention, with the caption entering as cross-attention.

    Time and (optionally) pooled tpos drive AdaLN. Text enters each block as
    K/V in a dedicated cross-attention stage — it never contributes to the
    AdaLN conditioning vector.
    """

    uses_text_cross = True

    def __init__(self, share_graph_attn_bias=False,
                 gradient_checkpointing=False, **kwargs):
        super().__init__(**kwargs)
        self.gradient_checkpointing = gradient_checkpointing

        # One bias for the whole stack (Graphormer's arrangement) instead of a
        # private one per block: built here, evaluated once per forward, and
        # handed to every layer.
        self.graph_bias = (GraphAttnBias(self.latent_dim, self.num_heads)
                           if share_graph_attn_bias else None)

        # Positional encoding: separate rope module per axis.
        self._build_axis_ropes()

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioTemporalCrossBlock(
                    hidden_size=self.latent_dim,
                    num_heads=self.num_heads,
                    mlp_size=self.ff_size,
                    rope_spatial=self.rope_j,
                    rope_temporal=self.rope_t,
                    qk_norm=True,
                    dropout=self.dropout,
                    own_graph_bias=self.graph_bias is None,
                    gradient_checkpointing=self.gradient_checkpointing,
                )
                for _ in range(self.num_layers)
            ]
        )

        self._finish_build()

    def initialize_weights(self):
        super().initialize_weights()
        # Zero-init each block's cross-attn output projection so the text
        # branch starts silent, as PixArt-alpha does. Safe now that the branch
        # carries no adaLN gate: d(loss)/d(proj) = upstream * attn_out is
        # nonzero, so it can leave zero. Zeroing a gate AND the projection
        # would deadlock both — neither could ever become nonzero and the
        # whole text path would train at exactly zero gradient.
        for block in self.transformer_blocks:
            nn.init.constant_(block.c_attn.proj.weight, 0)
            nn.init.constant_(block.c_attn.proj.bias, 0)

    def forward(self, x, timesteps, cond=None, force_mask=False):
        """
        Args:
            x: [batch_size, max_joints, nfeats, max_frames], denoted x_t in the paper
            timesteps: [batch_size] (int)
            cond: Optional dict with conditioning information
        Returns:
            output: [batch_size, max_joints, nfeats, max_frames]
        """
        bs, njoints, nfeats, nframes = x.shape

        # Timestep embedding → adaLN vector y (text is NOT added to y).
        y = self.time_embedder(timesteps, dtype=x.dtype)  # (B, latent_dim)

        # Text memory for cross-attention: the caption's TOKEN sequence, so
        # each joint/frame can attend the words it needs. A pooled vector here
        # would make this a one-token memory — an adaLN with extra cost.
        if self.cond_mode == 'text':
            raw_text, text_mask = self._caption_tokens(
                bs, cond, x.dtype, x.device, force_mask)      # (B, T, text_dim)
            text_mem = self.cond_embedder(raw_text)           # (B, T, latent_dim)
        else:  # 'no_cond'
            text_mem = torch.zeros(bs, 1, self.latent_dim, dtype=x.dtype, device=x.device)
            text_mask = torch.ones(bs, 1, dtype=torch.bool, device=x.device)

        # Input layer: encode tpos + motion, prepend the conditioning frame
        x, tpos_emb, joints_valid = self.input_layer(x, cond)
        if self.inject_tpos_to_adaln:
            y = y + self.tpos_pool(tpos_emb, joints_valid)  # (B, latent_dim)

        # Optional graph / depth / joint-name token embeddings
        x = self._apply_token_embeddings(x, cond)

        # Prepare attention masks
        temporal_valid = self.lengths_to_mask(cond['motion_length'] + self.n_prefix, nframes + self.n_prefix)  # (B, F+n_prefix)
        temporal_mask = temporal_valid.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, F+1)
        joint_mask = joints_valid.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, J)

        # Spatial rope: spectral pulls from cond; standard uses integer joint ids.
        spatial_pos_ids = self._spatial_position_ids(njoints)
        # Shared graph bias (when enabled): one (B, H, J, J) build for the stack.
        graph_bias = (self.graph_bias(cond, joint_mask)
                      if self.graph_bias is not None else None)
        for block in self.transformer_blocks:
            x = block(x, y, text_mem=text_mem, text_mask=text_mask,
                      spatial_mask=joint_mask, temporal_mask=temporal_mask,
                      position_ids_spatial=spatial_pos_ids,
                      position_ids_temporal=self.temporal_position_ids[:, :nframes + self.n_prefix],
                      cond=cond, graph_bias=graph_bias)

        # Final layer
        x = self.final_layer(x, y, joints_valid=joints_valid)  # (B, feature_len, F+n_prefix, J)
        x = einops.rearrange(x, 'b d f j -> b j d f')  # (B, J, D, F+n_prefix)
        x = x[:, :, :, self.n_prefix:]  # remove prefix frames (B, J, D, F)

        return x
