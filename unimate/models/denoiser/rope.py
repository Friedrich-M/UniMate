"""Rotary Position Embedding modules.

``RopeND`` — N-dimensional sinusoidal RoPE with per-axis frequency tables.
Used for integer position indices on regular sequences.

``SpectralJointRoPE`` — joint-axis spectral RoPE. Projects Laplacian
eigenvectors to rotation angles filling the full head dimension. Used for
per-frame spatial attention where temporal encoding is handled separately.

``GraphRoPE`` — combined spectral-joint + sinusoidal-temporal RoPE. Splits
the head dimension between a spectral graph half and a sinusoidal temporal
half. Used for full attention over flattened ``(T*J)`` tokens.

The spectral modules offer two projection backends, selected by
``use_signnet``:

  - ``True`` (default): ``SignNetSpectralEncoder`` implements the
    ``rho([phi(v_k) + phi(-v_k)])`` construction — sign-invariant by
    design, robust to Laplacian eigenvector sign flips.
  - ``False``: ``WIRESpectralEncoder`` (learnable frequencies, from *Rotary
    Position Encodings for Graphs*) — a single unbiased linear projection
    ``angles = W @ v`` where each row of ``W`` is a learned frequency vector
    ``ω_i ∈ ℝ^m``. Parameter count is ``(d/2) * m`` — the minimal WIRE
    parameterization. Not sign-invariant on its own.
"""

import math

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap and negate the two halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    The last dimension of ``q``/``k`` is rotated as standard RoPE:
    ``out = x * cos + rotate_half(x) * sin``. Callers must pass ``cos``/``sin``
    already shaped to broadcast against ``q``/``k`` (no implicit unsqueezing).

    Args:
        q, k: Same shape. The last dim is rotated, earlier dims are batched.
        cos, sin: Broadcastable to ``q`` and ``k``.
    """
    dtype = q.dtype
    q_f, k_f = q.float(), k.float()
    q_out = (q_f * cos) + (_rotate_half(q_f) * sin)
    k_out = (k_f * cos) + (_rotate_half(k_f) * sin)
    return q_out.to(dtype), k_out.to(dtype)


# ---------------------------------------------------------------------------
# RopeND
# ---------------------------------------------------------------------------

class RopeND:
    """N-dimensional Rotary Position Embedding.

    Args:
        head_dim: Total dimension per attention head.
        nd: Number of positional axes.
        max_lens: Maximum sequence length along each axis.
        nd_split: Relative dimension allocation per axis (summed then
            normalised to ``head_dim``).
        bases: Frequency bases per axis (used when ``auto_base=False``).
        auto_base: If True, derive bases empirically so that ``cos(theta)``
            reaches -1 at ~8x the max length (similar to the standard 10k
            base for length 4096).
        cache_longer: Multiplier on max_len for the cached frequency table.
    """

    def __init__(
        self,
        head_dim: int = 64,
        nd: int = 3,
        max_lens: List[int] = [1024, 64, 64],
        nd_split: List[int] = [2, 1, 1],
        bases: List[float] = [1000, 1000, 1000],
        auto_base: bool = True,
        cache_longer: int = 1,
    ):
        self.nd = nd
        self.head_dim = head_dim
        self.max_lens = max_lens
        self.nd_split = nd_split
        self.cache_longer = cache_longer

        # Dimension allocation per axis: 2 * ratio * (head_dim // 2 // sum)
        self.split_dims = [2 * s * (head_dim // 2 // sum(nd_split)) for s in nd_split]
        assert sum(self.split_dims) == head_dim, (
            f"split_dims {self.split_dims} do not sum to head_dim={head_dim}"
        )

        if auto_base:
            # Empirical rule: base ≈ 8L / π  (rounded to nearest 100)
            self.bases = [(int(8 * L / math.pi) // 100 + 1) * 100 for L in max_lens]
        else:
            self.bases = bases

    # ------------------------------------------------------------------
    # Frequency table generation & caching
    # ------------------------------------------------------------------

    def _generate_cos_sin(
        self, max_len: int, dim: int, device: torch.device, base: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build cosine / sine frequency tables of shape ``(max_len, dim)``."""
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device).float() / dim)
        )
        assert inv_freq.size(0) * 2 == dim

        t = torch.arange(max_len * self.cache_longer, device=device).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        freqs = torch.cat([freqs, freqs], dim=1)
        return freqs.cos().float(), freqs.sin().float()

    def _get_pos_embs(
        self, position_ids: torch.Tensor, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Look up (or lazily create) per-axis embeddings and concatenate.

        Args:
            position_ids: ``(nd, L)`` integer position indices per axis.
            device: Target device for the frequency tables.

        Returns:
            ``(cos, sin)`` each of shape ``(L, head_dim)``.
        """
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        cos_parts, sin_parts = [], []
        for i in range(self.nd):
            # Lazy cache per axis; regenerate if device changed
            cache_key_cos, cache_key_sin = f"cos_{i}", f"sin_{i}"
            cached_cos = getattr(self, cache_key_cos, None)
            if cached_cos is None or cached_cos.device != device:
                c, s = self._generate_cos_sin(
                    self.max_lens[i], self.split_dims[i], device, self.bases[i],
                )
                setattr(self, cache_key_cos, c)
                setattr(self, cache_key_sin, s)

            cos_parts.append(getattr(self, cache_key_cos)[position_ids[i, :], :])
            sin_parts.append(getattr(self, cache_key_sin)[position_ids[i, :], :])

        return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply N-D RoPE to query and key tensors.

        Args:
            q: ``(B, H, L, C)`` query tensor.
            k: ``(B, H, L, C)`` key tensor.
            position_ids: ``(nd, L)`` per-axis position indices.

        Returns:
            Rotated ``(q, k)`` with the same shapes.
        """
        cos_emb, sin_emb = self._get_pos_embs(position_ids, device=q.device)
        # Shape (L, C) → (1, 1, L, C) to broadcast over (B, H, L, C)
        cos_emb = cos_emb.unsqueeze(0).unsqueeze(0)
        sin_emb = sin_emb.unsqueeze(0).unsqueeze(0)
        return _apply_rotary_pos_emb(q, k, cos_emb, sin_emb)


# ---------------------------------------------------------------------------
# SignNet sign-invariant spectral encoder
# ---------------------------------------------------------------------------

class SignNetSpectralEncoder(nn.Module):
    """Sign-invariant projection from spectral coordinates to angles.

    Implements ``rho([phi(v_k) + phi(-v_k)]_{k=1..K})`` where ``phi`` is a
    per-frequency scalar MLP and ``rho`` is an aggregation MLP. Because
    ``phi(v) + phi(-v)`` is unchanged when ``v`` is negated, the output is
    invariant to the arbitrary sign flips inherent to Laplacian eigenvectors.

    Args:
        num_eigvecs: Number of spectral frequencies K.
        hidden_dim: Hidden width of phi and rho.
        out_dim: Output dimension (typically ``graph_dim // 2`` rotation angles).
    """

    def __init__(self, num_eigvecs: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.num_eigvecs = num_eigvecs

        # phi: per-frequency scalar -> hidden
        self.phi = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # rho: aggregate K frequencies -> out_dim
        self.rho = nn.Sequential(
            nn.Linear(num_eigvecs * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        nn.init.normal_(self.rho[-1].weight, std=0.02)
        nn.init.zeros_(self.rho[-1].bias)

    def forward(self, spectral_coords: torch.Tensor) -> torch.Tensor:
        """Project spectral coordinates to sign-invariant angles.

        Args:
            spectral_coords: ``(B, J, K)`` Laplacian eigenvector values.

        Returns:
            ``(B, J, out_dim)`` rotation angles.
        """
        v = spectral_coords.unsqueeze(-1)              # (B, J, K, 1)
        h = self.phi(v) + self.phi(-v)                  # (B, J, K, hidden)
        h = h.flatten(start_dim=-2)                     # (B, J, K * hidden)
        return self.rho(h)                              # (B, J, out_dim)


# ---------------------------------------------------------------------------
# WIRE learnable-frequency spectral encoder
# ---------------------------------------------------------------------------

class WIRESpectralEncoder(nn.Module):
    """Learnable-frequency spectral encoder from *Rotary Position Encodings
    for Graphs* (WIRE).

    The only learnable parameters are the frequencies
    ``(ω_i)_{i=1..out_dim} ⊂ ℝ^{num_eigvecs}``, stored as a single
    ``nn.Parameter`` of shape ``(out_dim, num_eigvecs)``. For spectral
    coords ``v ∈ ℝ^{num_eigvecs}`` the output is ``angles[i] = <ω_i, v>``,
    giving ``out_dim * num_eigvecs`` parameters total — much smaller than a
    SignNet head. Not sign-invariant: if Laplacian eigenvectors are
    sign-flipped, the angles are negated.

    Args:
        num_eigvecs: Number of spectral frequencies K (= dim of ω_i).
        out_dim: Number of rotation angles produced (= head_dim / 2 or
            graph_dim / 2 depending on the caller).
    """

    def __init__(self, num_eigvecs: int, out_dim: int):
        super().__init__()
        self.num_eigvecs = num_eigvecs
        self.out_dim = out_dim
        # Rows are frequency vectors ω_i ∈ ℝ^{num_eigvecs}
        self.frequencies = nn.Parameter(torch.empty(out_dim, num_eigvecs))
        nn.init.normal_(self.frequencies, std=0.02)

    def forward(self, spectral_coords: torch.Tensor) -> torch.Tensor:
        """Project spectral coordinates to rotation angles via learned frequencies.

        Args:
            spectral_coords: ``(B, J, K)`` Laplacian eigenvector values.

        Returns:
            ``(B, J, out_dim)`` rotation angles ``[<ω_i, v>]``.
        """
        # (B, J, K) @ (K, out_dim) -> (B, J, out_dim)
        return spectral_coords @ self.frequencies.t()


def _build_spectral_encoder(
    num_eigvecs: int, out_dim: int, use_signnet: bool, signnet_hidden: int,
) -> nn.Module:
    """Construct the spectral encoder selected by ``use_signnet``."""
    if use_signnet:
        return SignNetSpectralEncoder(
            num_eigvecs=num_eigvecs,
            hidden_dim=signnet_hidden,
            out_dim=out_dim,
        )
    return WIRESpectralEncoder(num_eigvecs=num_eigvecs, out_dim=out_dim)


# ---------------------------------------------------------------------------
# SpectralJointRoPE — joint-axis (spectral only) rotary embedding
# ---------------------------------------------------------------------------

class SpectralJointRoPE(nn.Module):
    """Spectral rotary embedding for the joint axis only.

    Projects Laplacian eigenvectors to rotation angles covering the entire
    head dimension. Used for per-frame spatial attention where temporal
    differentiation is handled separately. The projection backend is
    selected by ``use_signnet`` (see module docstring).

    Args:
        head_dim: Total dimension per attention head (must be even).
        num_eigvecs: Number of Laplacian eigenvectors used as spectral coords.
        use_signnet: If True, use sign-invariant SignNet encoder. If False,
            use the minimal WIRE learnable-frequency projection.
        signnet_hidden: Hidden width of the SignNet phi/rho MLPs. Ignored
            when ``use_signnet=False``.
    """

    def __init__(
        self,
        head_dim: int = 64,
        num_eigvecs: int = 8,
        use_signnet: bool = True,
        signnet_hidden: int = 64,
    ):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        self.head_dim = head_dim
        self.num_eigvecs = num_eigvecs
        self.use_signnet = use_signnet

        self.spectral_encoder = _build_spectral_encoder(
            num_eigvecs=num_eigvecs,
            out_dim=head_dim // 2,
            use_signnet=use_signnet,
            signnet_hidden=signnet_hidden,
        )

    def _compute_cos_sin(
        self, spectral_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project spectral coords to cos/sin of rotation angles."""
        angles = self.spectral_encoder(spectral_coords)   # (..., J, C/2)
        angles = torch.cat([angles, angles], dim=-1)      # (..., J, C)
        return angles.cos(), angles.sin()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        spectral_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply joint-axis spectral RoPE to query and key.

        Args:
            q, k: ``(B, H, J, C)`` query/key tensors.
            spectral_coords: ``(B, J, num_eigvecs)`` Laplacian eigenvectors.

        Returns:
            Rotated ``(q, k)``.
        """
        cos, sin = self._compute_cos_sin(spectral_coords)  # (B, J, C)
        # Shape (B, J, C) → (B, 1, J, C) to broadcast over H
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return _apply_rotary_pos_emb(q, k, cos, sin)

    def forward_per_frame(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        spectral_coords: torch.Tensor,
        n_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Efficient per-frame variant for use in factored spatial attention.

        The caller has folded the frame dim into the batch: ``q``, ``k``
        arrive as ``(B*F, H, J, C)``. Spectral coordinates do not depend on
        the frame, so the encoder runs once on ``(B, J, K)`` and the
        rotation is broadcast across F via a reshape to ``(B, F, H, J, C)``.

        This avoids:
          - materializing a ``(B*F, J, K)`` copy of spectral_coords, and
          - running the SignNet/WIRE encoder F times on identical inputs.

        Args:
            q, k: ``(B*F, H, J, C)`` query/key tensors.
            spectral_coords: ``(B, J, num_eigvecs)`` Laplacian eigenvectors.
            n_frames: F — number of frames folded into the batch.

        Returns:
            Rotated ``(q, k)`` with the original ``(B*F, H, J, C)`` shape.
        """
        B, J = spectral_coords.shape[0], spectral_coords.shape[1]
        H, C = q.shape[1], q.shape[3]

        cos, sin = self._compute_cos_sin(spectral_coords)  # (B, J, C)
        # Broadcast shape: (B, 1, 1, J, C) → over F (dim 1) and H (dim 2)
        cos = cos.view(B, 1, 1, J, C)
        sin = sin.view(B, 1, 1, J, C)

        # ``reshape`` because q/k are non-contiguous after ``.permute().unbind()``.
        q = q.reshape(B, n_frames, H, J, C)
        k = k.reshape(B, n_frames, H, J, C)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        return (
            q.reshape(B * n_frames, H, J, C),
            k.reshape(B * n_frames, H, J, C),
        )


# ---------------------------------------------------------------------------
# GraphRoPE — combined spectral-joint + sinusoidal-temporal rotary embedding
# ---------------------------------------------------------------------------

class GraphRoPE(nn.Module):
    """Combined spectral-joint + sinusoidal-temporal Rotary Position Embedding.

    Follows *Rotary Position Encodings for Graphs*: the head dimension is
    split into a graph half and a temporal half. The graph half projects
    Laplacian eigenvectors to rotation angles (backend selected by
    ``use_signnet``); the temporal half uses standard sinusoidal RoPE
    frequencies (rotations depend on frame index). Used for full attention
    over flattened ``(T*J)`` tokens.

    Args:
        head_dim: Total dimension per attention head.
        num_eigvecs: Number of Laplacian eigenvectors K used as spectral
            coordinates per joint.
        nd_split: Relative dim allocation ``[graph_ratio, time_ratio]``
            between the joint and temporal halves. Mirrors ``RopeND``'s
            ``nd_split`` — ``[1, 1]`` (default) is equal split, ``[2, 1]``
            gives the joint half twice as many channels as the temporal
            half, etc. ``sum(nd_split)`` must divide ``head_dim // 2`` so
            each half gets an even number of channels.
        use_signnet: If True, use sign-invariant SignNet encoder. If False,
            use the minimal WIRE learnable-frequency projection.
        signnet_hidden: Hidden width of the SignNet phi/rho MLPs. Ignored
            when ``use_signnet=False``.
        max_time: Maximum temporal sequence length. Used both to size the
            precomputed cos/sin table and to derive ``time_base`` when
            ``time_base`` is None.
        time_base: Frequency base for the temporal RoPE axis. ``None``
            selects the RopeND auto rule ``8·max_time/π`` rounded up to the
            nearest 100, which matches ``RopeND``'s ``auto_base=True``.
    """

    def __init__(
        self,
        head_dim: int = 64,
        num_eigvecs: int = 8,
        nd_split: List[int] = [1, 1],
        use_signnet: bool = True,
        signnet_hidden: int = 64,
        max_time: int = 1024,
        time_base: Optional[float] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_eigvecs = num_eigvecs
        self.use_signnet = use_signnet
        self.max_time = max_time

        # Split head_dim between the two axes, following RopeND's rule.
        # Each axis gets `2 * ratio * (head_dim // 2 // sum(nd_split))`
        # channels, which guarantees each axis has an even count (required
        # by rotate_half) as long as sum(nd_split) | head_dim // 2.
        assert len(nd_split) == 2, f"nd_split must be length 2, got {nd_split}"
        self.nd_split = list(nd_split)
        split_dims = [
            2 * s * (head_dim // 2 // sum(nd_split)) for s in nd_split
        ]
        assert sum(split_dims) == head_dim, (
            f"split_dims {split_dims} do not sum to head_dim={head_dim}. "
            f"Ensure sum(nd_split)={sum(nd_split)} divides head_dim//2={head_dim//2}."
        )
        self.graph_dim, self.time_dim = split_dims

        self.spectral_encoder = _build_spectral_encoder(
            num_eigvecs=num_eigvecs,
            out_dim=self.graph_dim // 2,
            use_signnet=use_signnet,
            signnet_hidden=signnet_hidden,
        )

        # Temporal sinusoidal RoPE — mirrors RopeND for consistency.
        #
        # Auto-derive the base using the same rule as RopeND(auto_base=True):
        #   base ≈ 8·L/π, rounded up to the nearest 100
        # This guarantees the lowest-frequency channel rotates a meaningful
        # fraction of a turn across max_time without wrapping.
        if time_base is None:
            time_base = float((int(8 * max_time / math.pi) // 100 + 1) * 100)
        self.time_base = time_base

        # Pre-compute the full cos/sin table up to max_time. Registered as
        # non-persistent buffers so they follow the module across devices /
        # dtypes automatically (analogous to RopeND's lazy per-axis cache,
        # but always-on since max_time is known at construction).
        inv_freq = 1.0 / (
            time_base ** (torch.arange(0, self.time_dim, 2).float() / self.time_dim)
        )
        t = torch.arange(max_time).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)       # (max_time, time_dim/2)
        freqs = torch.cat([freqs, freqs], dim=-1)          # (max_time, time_dim)
        self.register_buffer("t_cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("t_sin_cache", freqs.sin(), persistent=False)

    def _compute_graph_cos_sin(
        self, spectral_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project spectral coordinates to rotation angles via spectral encoder."""
        angles = self.spectral_encoder(spectral_coords)  # (B, J, graph_dim // 2)
        angles = torch.cat([angles, angles], dim=-1)     # (B, J, graph_dim)
        return angles.cos(), angles.sin()

    def _compute_time_cos_sin(
        self, nframes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Look up cached temporal frequencies (matches RopeND cache style)."""
        assert nframes <= self.max_time, (
            f"nframes={nframes} exceeds max_time={self.max_time}"
        )
        return self.t_cos_cache[:nframes], self.t_sin_cache[:nframes]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        spectral_coords: torch.Tensor,
        nframes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply combined graph + temporal RoPE to query and key.

        Axis-wise rotation: split head_dim into graph_dim + time_dim, then
        apply a proper 2D RoPE independently within each half. The cos/sin
        tensors are never materialized to the full ``(B, T*J, C)`` shape —
        broadcasting over the reshaped ``(B, H, T, J, C)`` view handles it.

        Args:
            q, k: ``(B, H, T*J, C)`` query/key over flattened spatio-temporal tokens.
            spectral_coords: ``(B, J, num_eigvecs)`` Laplacian eigenvectors.
            nframes: T (number of frames).

        Returns:
            Rotated ``(q, k)`` with the same shapes.
        """
        B, H, L, C = q.shape
        njoints = spectral_coords.shape[1]
        assert L == nframes * njoints, (
            f"L={L} != nframes*njoints={nframes}*{njoints}"
        )
        assert C == self.head_dim, f"C={C} != head_dim={self.head_dim}"

        # Per-axis angles (no expansion — broadcasting only).
        g_cos, g_sin = self._compute_graph_cos_sin(spectral_coords)  # (B, J, graph_dim)
        t_cos, t_sin = self._compute_time_cos_sin(nframes)            # (T, time_dim)

        # Unflatten tokens so T and J are separate axes for broadcasting.
        # ``reshape`` (not ``view``) because q/k come from ``.permute().unbind()``
        # in RoPEAttention and may be non-contiguous.
        q = q.reshape(B, H, nframes, njoints, C)
        k = k.reshape(B, H, nframes, njoints, C)

        # Split channels — each half gets its own proper RoPE rotation.
        q_g, q_t = q[..., :self.graph_dim], q[..., self.graph_dim:]
        k_g, k_t = k[..., :self.graph_dim], k[..., self.graph_dim:]

        # Graph rotation: (B, J, graph_dim) → (B, 1, 1, J, graph_dim)
        g_cos = g_cos.unsqueeze(1).unsqueeze(2)
        g_sin = g_sin.unsqueeze(1).unsqueeze(2)
        q_g, k_g = _apply_rotary_pos_emb(q_g, k_g, g_cos, g_sin)

        # Temporal rotation: (T, time_dim) → (1, 1, T, 1, time_dim)
        t_cos = t_cos.view(1, 1, nframes, 1, self.time_dim)
        t_sin = t_sin.view(1, 1, nframes, 1, self.time_dim)
        q_t, k_t = _apply_rotary_pos_emb(q_t, k_t, t_cos, t_sin)

        # Concatenate channels back and restore flat token layout.
        q = torch.cat([q_g, q_t], dim=-1).reshape(B, H, L, C)
        k = torch.cat([k_g, k_t], dim=-1).reshape(B, H, L, C)
        return q, k