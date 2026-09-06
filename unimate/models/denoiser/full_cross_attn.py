"""``attention='full'`` + ``text_cond='cross_attn'`` — DiT with a text stage."""

import torch
import torch.nn as nn
import einops

from unimate.models.denoiser.rope import RopeND, GraphRoPE
from unimate.models.denoiser.base import UniMateDenoiserBase
from unimate.models.denoiser.blocks import DiTCrossBlock


class UniMateFullCrossAttn(UniMateDenoiserBase):
    """Full attention over the flattened (T*J) tokens, plus a text stage.

    Same attention stack as :class:`UniMateFullAdaLN`; the caption enters
    each block as cross-attention K/V instead of through the adaLN vector, so
    every spatio-temporal token can attend the words it needs. Time and
    (optionally) pooled tpos still drive adaLN.

    Position encoding is controlled by ``use_spectral_rope``:
      - True: GraphRoPE (sign-invariant spectral joint + sinusoidal temporal).
      - False: 2D sinusoidal RopeND with integer (frame, joint) indices.
    """

    uses_text_cross = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Positional Encoding. Either spectral (GraphRoPE) or standard 2D RoPE.
        head_dim = self.latent_dim // self.num_heads
        max_frames = self.max_motion_length + self.n_prefix  # F + n_prefix
        if self.use_spectral_rope:
            self.rope = GraphRoPE(
                head_dim=head_dim,
                num_eigvecs=self.max_freqs,
                use_signnet=self.use_signnet,
                max_time=max_frames,
            )
        else:
            self.rope = RopeND(
                head_dim=head_dim, nd=2, nd_split=[1, 1],
                max_lens=[max_frames, self.max_joints],
            )
            pos_ids_f = torch.arange(max_frames).unsqueeze(1).repeat(1, self.max_joints)
            pos_ids_j = torch.arange(self.max_joints).unsqueeze(0).repeat(max_frames, 1)
            self.register_buffer(
                'position_ids_precompute',
                torch.stack([pos_ids_f, pos_ids_j], dim=0),
                persistent=False,
            )

        # Transformer blocks (the rope module is shared across blocks)
        self.transformer_blocks = torch.nn.ModuleList(
            [
                DiTCrossBlock(
                    hidden_size=self.latent_dim,
                    num_heads=self.num_heads,
                    mlp_size=self.ff_size,
                    rope=self.rope,
                    qk_norm=True,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self._finish_build()

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

        # Text memory for cross-attention: the caption's TOKEN sequence.
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

        # Flatten spatial-temporal dims for full attention
        x = einops.rearrange(x, 'b f j d -> b d f j')  # (B, latent_dim, F+1, J)
        x = x.flatten(2).transpose(1, 2)  # (B, (F+1)*J, latent_dim)

        # Prepare attention mask
        # Length must be the batch's own frame count, not the config maximum:
        # the token axis below is flattened over (nframes + n_prefix), so a
        # shorter clip would give a mask that no longer lines up with it.
        temporal_mask = self.lengths_to_mask(cond['motion_length'] + self.n_prefix, nframes + self.n_prefix)  # (B, F+n_prefix)
        attention_mask = temporal_mask.unsqueeze(-1) & joints_valid.unsqueeze(1)  # (B, F+n_prefix, J)
        attention_mask = attention_mask.flatten(1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, (F+n_prefix)*J)

        # Positional encodings: spectral needs nframes; standard needs position_ids
        if self.use_spectral_rope:
            rope_nframes = nframes + self.n_prefix
            position_ids = None
        else:
            rope_nframes = None
            position_ids = self.position_ids_precompute[:, :(nframes + self.n_prefix), :njoints]
            position_ids = position_ids.flatten(1)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, y, text_mem, attention_mask=attention_mask,
                      text_mask=text_mask, position_ids=position_ids,
                      cond=cond, nframes=rope_nframes)

        # Final layer
        x = self.final_layer(x, y, joints_valid=joints_valid)  # (B, feature_len, F+n_prefix, J)
        x = einops.rearrange(x, 'b d f j -> b j d f')  # (B, J, D, F+n_prefix)
        x = x[:, :, :, self.n_prefix:]  # remove prefix frames (B, J, D, F)

        return x
