"""Shared base class for the denoiser variants.

The four variants differ only along the two config axes — ``attention``
(full flattened vs. factored spatio-temporal) and ``text_cond`` (adaLN vs. a
cross-attention stage). Everything else — configuration, conditioning
embedders, optional token embeddings, adaLN-Zero weight init, CFG masking —
is identical and lives here.

Subclass contract (order matters so a fixed seed reproduces identical
initial weights):

1. ``super().__init__(...)`` — stashes config and builds the shared
   embedders (cond / timestep / input layer / tpos pool / joint-name /
   graph / depth).
2. Build positional-encoding modules (``_build_axis_ropes`` for the
   factored variants) and ``self.transformer_blocks``.
3. ``self._finish_build()`` — builds ``self.final_layer`` and runs
   ``initialize_weights``.
"""

import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv

from unimate.models.denoiser.blocks import (
    TimestepEmbedder,
    InputLayer,
    FinalLayer,
    TposPool,
)
from unimate.models.denoiser.rope import RopeND, SpectralJointRoPE


class UniMateDenoiserBase(nn.Module):
    """Configuration, conditioning, and initialization shared by all UniMate variants."""

    #: Subclasses that feed the caption in as cross-attention K/V set this,
    #: so the base only builds the parts that variant actually uses.
    uses_text_cross = False

    def __init__(
        self,
        feature_len,
        max_motion_length=60,
        max_joints=24,
        max_depth=0,
        latent_dim=256,
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.0,
        use_spectral_rope=False,
        max_freqs=8,
        use_signnet=True,
        cond_mode='no_cond',
        cond_mask_prob=0.0,
        text_dim=512,
        use_joint_name_emb=False,
        use_graph_emb=False,
        use_depth_emb=False,
        concat_parent_features=False,
        num_tpos_queries=0,
        inject_tpos_to_adaln=False,
    ):
        super().__init__()

        self.max_motion_length = max_motion_length
        self.max_joints = max_joints
        self.feature_len = feature_len
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.text_dim = text_dim
        self.cond_mode = cond_mode
        self.cond_mask_prob = cond_mask_prob

        self.use_spectral_rope = use_spectral_rope
        self.max_freqs = max_freqs
        self.use_signnet = use_signnet

        self.use_joint_name_emb = use_joint_name_emb
        self.use_graph_emb = use_graph_emb
        self.use_depth_emb = use_depth_emb
        self.concat_parent_features = concat_parent_features
        self.num_tpos_queries = num_tpos_queries
        self.inject_tpos_to_adaln = inject_tpos_to_adaln
        # One conditioning frame (tpos / first-frame tokens) prepended to the
        # temporal axis; stripped again after the final layer.
        self.n_prefix = 1

        # Caption embedding → latent (text mode) or identity (no_cond).
        if self.cond_mode == 'text':
            self.cond_embedder = nn.Linear(self.text_dim, self.latent_dim)
            # Learnable unconditional caption — only for variants that route
            # text through cross-attention. Creating it unconditionally would
            # leave the adaLN variants with a parameter that never receives a
            # gradient, which DDP rejects at its default
            # find_unused_parameters=False.
            # Zeroing the text memory instead would leave attention with
            # softmax(q·0) over an all-zero V — the value bias, one constant
            # with no capacity — so the unconditional branch that CFG
            # extrapolates from could not be shaped at all. PixArt-alpha's
            # CaptionEmbedder carries the same idea (its ``y_embedding`` null
            # sequence), including this initialisation scale. The adaLN path
            # keeps zeroing its vector: there a zero really is "no signal"
            # added to the modulation, which is well defined.
            if self.uses_text_cross:
                self.null_caption = nn.Parameter(
                    torch.randn(1, 1, self.text_dim) / self.text_dim ** 0.5)
        elif self.cond_mode == 'no_cond':
            self.cond_embedder = nn.Identity()
        else:
            raise ValueError(f"Unknown cond_mode: {self.cond_mode}")

        # Timestep embedder
        self.time_embedder = TimestepEmbedder(hidden_size=self.latent_dim)

        # Input layer (root/joint split, MLP encoding)
        self.input_layer = InputLayer(
            feature_len=self.feature_len,
            latent_dim=self.latent_dim,
            max_joints=self.max_joints,
            concat_parent_features=self.concat_parent_features,
        )

        # Tpos pooling → adaLN conditioning (only instantiated if enabled)
        if self.inject_tpos_to_adaln:
            self.tpos_pool = TposPool(self.latent_dim, num_queries=self.num_tpos_queries)

        # Joint name embedding
        if self.use_joint_name_emb:
            self.joint_name_embedder = nn.Linear(self.text_dim, self.latent_dim)
            self.joint_name_dropout = nn.Dropout(self.dropout)

        # Graph embedding
        if self.use_graph_emb:
            self.graph_conv = GCNConv(self.latent_dim, self.latent_dim)

        # Kinematic depth embedding
        if self.use_depth_emb:
            # max_depth is auto-computed from data; fallback to max_joints if not provided
            self.max_depth = max_depth if max_depth > 0 else max_joints
            self.depth_embedding = nn.Embedding(self.max_depth + 1, self.latent_dim)

    # ------------------------------------------------------------------
    # Construction helpers (called from subclass __init__)
    # ------------------------------------------------------------------

    def _build_axis_ropes(self):
        """Per-axis RoPE for the factored (Graph / Cross) variants.

        Spatial (joint) axis: spectral RoPE when ``use_spectral_rope``,
        otherwise 1D sinusoidal with integer joint indices. Temporal axis:
        always sinusoidal. Position-id buffers are non-persistent (they are
        pure index tables).
        """
        head_dim = self.latent_dim // self.num_heads
        max_frames = self.max_motion_length + self.n_prefix

        if self.use_spectral_rope:
            self.rope_j = SpectralJointRoPE(
                head_dim=head_dim,
                num_eigvecs=self.max_freqs,
                use_signnet=self.use_signnet,
            )
        else:
            self.rope_j = RopeND(
                head_dim=head_dim, nd=1, nd_split=[1], max_lens=[self.max_joints],
            )
            self.register_buffer(
                'spatial_position_ids',
                torch.arange(self.max_joints).unsqueeze(0),
                persistent=False,
            )

        self.rope_t = RopeND(
            head_dim=head_dim, nd=1, nd_split=[1], max_lens=[max_frames],
        )
        self.register_buffer(
            'temporal_position_ids',
            torch.arange(max_frames).unsqueeze(0),
            persistent=False,
        )

    def _finish_build(self):
        """Build the output layer and initialize weights. Call last in __init__."""
        self.final_layer = FinalLayer(
            hidden_size=self.latent_dim, output_size=self.feature_len,
            joint=self.max_joints,
        )
        self.initialize_weights()

    # ------------------------------------------------------------------
    # Weight initialization (adaLN-Zero)
    # ------------------------------------------------------------------

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Caption projection: the same std=0.02 normal DiT and PixArt-alpha
        # give their conditioning embedders, rather than the xavier default
        # _basic_init would leave (0.0395 for 768->512, twice the variance).
        # An embedding projection sets how loud the conditioning signal is
        # relative to the backbone's activations at step 0, which matters most
        # for text_cond='cross_attn' where it feeds the attention K/V directly.
        if self.cond_mode == 'text':
            nn.init.normal_(self.cond_embedder.weight, std=0.02)
            nn.init.constant_(self.cond_embedder.bias, 0)

        # Zero-out adaLN modulation layers in UniMate blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        for out_mlp in [self.final_layer.root_out, self.final_layer.joint_out]:
            nn.init.constant_(out_mlp[-1].weight, 0)
            nn.init.constant_(out_mlp[-1].bias, 0)
        # Zero-init cross-attention output so residual starts as identity
        nn.init.constant_(self.final_layer.root_cross_attn.out_proj.weight, 0)
        nn.init.constant_(self.final_layer.root_cross_attn.out_proj.bias, 0)

        # Zero-init the tpos cross-attention pool's output projection so its
        # contribution to the adaLN driver `y` starts at zero — same adaLN-Zero
        # pattern used for adaLN_modulation / final-layer outputs above. Without
        # this, randomly-initialized tpos_pool feeds noise into y from step 0,
        # destabilizing the adaLN modulation magnitudes.
        if self.inject_tpos_to_adaln and self.num_tpos_queries > 0:
            nn.init.constant_(self.tpos_pool.pool.out_proj.weight, 0)
            nn.init.constant_(self.tpos_pool.pool.out_proj.bias, 0)

    # ------------------------------------------------------------------
    # Shared forward pieces
    # ------------------------------------------------------------------

    def _caption_vector(self, bs, cond, dtype, device, force_mask):
        """CFG-masked caption embedding vector (pre ``cond_embedder``).

        Returns ``(B, text_dim)`` in text mode (zeros when the batch has no
        captions) or ``(B, latent_dim)`` zeros in no_cond mode — matching
        what ``cond_embedder`` (Linear vs. Identity) expects.
        """
        if self.cond_mode == 'text':
            # Missing key → whole batch had no captions; treat as unconditional.
            cond_vector = cond.get('caption_emb')
            if cond_vector is None:
                cond_vector = torch.zeros(bs, self.text_dim, dtype=dtype, device=device)
        else:  # 'no_cond' (validated in __init__)
            cond_vector = torch.zeros(bs, self.latent_dim, dtype=dtype, device=device)
        return self.mask_cond(cond_vector, force_mask=force_mask)

    def _caption_tokens(self, bs, cond, dtype, device, force_mask):
        """CFG-masked caption token sequence + validity mask for cross-attn.

        Returns ``((B, T, text_dim), (B, T) bool)``. A CFG-dropped sample gets
        the learnable null caption as a one-token memory; its mask keeps only
        that position, so no row is ever fully masked (which would make
        scaled_dot_product_attention return NaN).
        """
        tokens = cond.get('caption_tokens') if self.cond_mode == 'text' else None
        if tokens is None and self.cond_mode == 'text':
            # A caller that only produced the pooled vector (the inference
            # paths encode one prompt at a time) still conditions the model,
            # as a one-token memory. Falling through to zeros here would make
            # cross-attention silently unconditional.
            pooled = cond.get('caption_emb')
            if pooled is not None:
                tokens = pooled.unsqueeze(1)                       # (B, 1, D)
        if tokens is None:
            # no_cond, or a text batch with no captions at all. In text mode
            # the null caption is the unconditional representation the model
            # was trained on, so use it rather than zeros — and referencing it
            # here keeps it in the graph on this path too.
            if self.cond_mode == 'text':
                tokens = self.null_caption.to(dtype).expand(bs, 1, -1)
            else:
                tokens = torch.zeros(bs, 1, self.text_dim, dtype=dtype, device=device)
            return tokens, torch.ones(bs, 1, dtype=torch.bool, device=device)

        tokens = tokens.to(dtype)
        mask = cond.get('caption_mask')
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)

        # Same dropout decision as ``mask_cond``, but a dropped sample is
        # replaced by the learnable null caption rather than zeroed: its memory
        # becomes that single token, and its mask attends only that position.
        # Every row keeps one attendable position, so SDPA never sees a fully
        # masked row (which returns NaN).
        if force_mask:
            drop = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        elif self.training and self.cond_mask_prob > 0.:
            drop = torch.bernoulli(
                torch.ones(tokens.shape[0], device=tokens.device) * self.cond_mask_prob
            ).to(torch.bool)
        else:
            drop = None

        if drop is not None:
            # torch.where rather than indexing the dropped rows: it keeps
            # null_caption in the graph for every sample, so the parameter
            # still receives a (zero) gradient on steps where nothing was
            # dropped. Indexing would leave it unused on those steps, and DDP
            # at its default find_unused_parameters=False errors out on that.
            # PixArt-alpha's CaptionEmbedder.token_drop does the same.
            null = self.null_caption.to(dtype).expand(tokens.shape[0], 1, -1)
            head = torch.where(drop[:, None, None], null, tokens[:, :1])
            tokens = torch.cat([head, tokens[:, 1:]], dim=1)
            only_first = torch.zeros_like(mask)
            only_first[:, 0] = True
            mask = torch.where(drop[:, None], only_first, mask)
        return tokens, mask

    def _build_adaln_y(self, x, timesteps, cond, force_mask):
        """adaLN driver ``y = timestep_emb + caption_emb`` (``text_cond='adaln'``)."""
        timesteps_emb = self.time_embedder(timesteps, dtype=x.dtype)  # (B, latent_dim)
        cond_vector = self._caption_vector(
            x.shape[0], cond, x.dtype, x.device, force_mask,
        )
        return timesteps_emb + self.cond_embedder(cond_vector)  # (B, latent_dim)

    def _apply_token_embeddings(self, x, cond):
        """Add the optional graph / depth / joint-name embeddings to the tokens.

        Args:
            x: (B, F+1, J, latent_dim) embedded tokens (tpos frame prepended).
            cond: conditioning dict.
        """
        device = x.device

        # Graph embedding (per-sample GCN over the skeleton edges)
        if self.use_graph_emb:
            x_graph_emb = []
            for i in range(x.shape[0]):
                edge_index = cond['edge_indexs'][i].to(device)  # [2, E]
                x_graph_emb.append(self.graph_conv(x[i], edge_index))  # (F+1, J, D)
            x = x + torch.stack(x_graph_emb, dim=0)  # (B, F+1, J, latent_dim)

        # Kinematic depth embedding
        if self.use_depth_emb:
            joint_depths = cond['joint_depths'].long().clamp(max=self.max_depth)  # (B, J)
            depth_emb = self.depth_embedding(joint_depths)  # (B, J, latent_dim)
            x = x + depth_emb.unsqueeze(1)  # (B, F+1, J, latent_dim)

        # Joint name embedding
        if self.use_joint_name_emb:
            joint_names_emb = cond['joint_names_emb']  # (B, J, text_dim)
            joint_name_embedded = self.joint_name_dropout(
                self.joint_name_embedder(joint_names_emb)
            )  # (B, J, latent_dim)
            x = x + joint_name_embedded.unsqueeze(1)  # (B, F+1, J, latent_dim)

        return x

    def _spatial_position_ids(self, njoints):
        """Spatial position ids for the factored variants (None when spectral
        RoPE pulls its coordinates from ``cond['spectral_feats']`` instead)."""
        if self.use_spectral_rope:
            return None
        return self.spatial_position_ids[:, :njoints]

    def mask_cond(self, cond, force_mask=False):
        """Classifier-free-guidance dropout of the conditioning vector."""
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    @staticmethod
    def lengths_to_mask(lengths, max_len):
        return torch.arange(max_len, device=lengths.device).expand(
            len(lengths), max_len) < lengths.unsqueeze(1)
