"""Collate function for the Mixture dataset.

Assembles per-sample dicts (numpy arrays from ``MotionDataset.__getitem__``)
into a padded ``(motion_tensor, cond_dict)`` batch. All joint-axis fields are
zero-padded to ``max_joints`` and all time-axis fields to ``max_motion_length``.
Optional fields (captions, split tags, ...) are only emitted when present on
every sample (or, for ``caption_emb``, on any sample).
"""

import torch


# ---------------------------------------------------------------------------
# Temporal / spatial mask helpers
# ---------------------------------------------------------------------------

def lengths_to_mask(lengths, max_len):
    """(B,) int lengths -> (B, max_len) bool mask."""
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# ---------------------------------------------------------------------------
# Collate helpers
# ---------------------------------------------------------------------------

def create_padded_matrix(relation_np, max_joints, n_joints, dtype=torch.float32):
    """Pad a (n_joints, n_joints) numpy matrix to (max_joints, max_joints) torch tensor."""
    relation = torch.as_tensor(relation_np, dtype=dtype)
    padded_relation = torch.zeros((max_joints, max_joints), dtype=dtype)
    padded_relation[:n_joints, :n_joints] = relation
    return padded_relation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_joint_axis(batch, key, max_joints, tail_shape, dtype=torch.float32):
    """Stack variable-joint numpy arrays into ``(B, max_joints, *tail_shape)``.

    Each ``batch[i][key]`` has shape ``(n_joints_i, *tail_shape)`` and is
    copied into the leading ``n_joints_i`` slots; trailing slots stay zero.
    """
    B = len(batch)
    out = torch.zeros((B, max_joints, *tail_shape), dtype=dtype)
    for i, data in enumerate(batch):
        arr = data[key]
        out[i, :arr.shape[0]] = torch.from_numpy(arr)
    return out


def _collect_scalar(batch, key, dtype=torch.long):
    """Stack ``batch[i][key]`` (scalar) into a ``(B,)`` tensor."""
    return torch.tensor([b[key] for b in batch], dtype=dtype)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def mixture_batch_collate(batch):
    """Collate per-sample dicts into ``(motion, cond)`` batched tensors."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None

    B = len(batch)
    max_joints = batch[0]['max_joints']
    n_feats = batch[0]['motion'].shape[-1]
    max_len = batch[0]['motion'].shape[0]  # already padded to max_motion_length

    # --- Required fields: sizes / indices ---
    n_joints_list = torch.tensor(
        [len(b['parents']) for b in batch], dtype=torch.long
    )
    motion_lengths = _collect_scalar(batch, 'motion_length')
    crop_start_inds = _collect_scalar(batch, 'start_idx')
    parents_list = [b['parents'] for b in batch]
    edge_indexs_list = [torch.as_tensor(b['edge_indexs'], dtype=torch.long) for b in batch]

    # --- Motion tensor: (B, J, D, T); per-sample (F, nj, D) -> (nj, D, F) ---
    motion = torch.zeros((B, max_joints, n_feats, max_len))
    for i, data in enumerate(batch):
        nj = n_joints_list[i].item()
        m = torch.from_numpy(data['motion']).float()         # (F, nj, D)
        motion[i, :nj, :, :] = m.permute(1, 2, 0)            # (nj, D, F)

    # --- Joint-axis padded fields (always present) ---
    tpos_first_frame = _pad_joint_axis(batch, 'tpos_first_frame', max_joints, (n_feats,))
    mean = _pad_joint_axis(batch, 'mean', max_joints, (n_feats,))
    std = _pad_joint_axis(batch, 'std', max_joints, (n_feats,))
    joint_depths = _pad_joint_axis(batch, 'joint_depths', max_joints, (), dtype=torch.long)

    # std must default to 1 on padding slots so division in the model is safe
    std_ones = torch.ones_like(std)
    std_mask = (torch.arange(max_joints)[None, :] < n_joints_list[:, None]).unsqueeze(-1)
    std = torch.where(std_mask, std, std_ones)

    # --- Pairwise joint-joint matrices (NxN) ---
    # Stored as torch.long since both are consumed as nn.Embedding indices.
    joint_relations = torch.zeros((B, max_joints, max_joints), dtype=torch.long)
    graph_dist = torch.zeros((B, max_joints, max_joints), dtype=torch.long)
    for i, data in enumerate(batch):
        nj = n_joints_list[i].item()
        joint_relations[i] = create_padded_matrix(
            data['joint_relations'], max_joints, nj, dtype=torch.long,
        )
        graph_dist[i] = create_padded_matrix(
            data['joint_graph_dist'], max_joints, nj, dtype=torch.long,
        )

    # --- Optional fields (present on every / any sample) ---
    has_spectral_feats = all('spectral_feats' in b for b in batch)
    has_joint_names_emb = all('joint_names_emb' in b for b in batch)
    has_offsets = all('offsets' in b for b in batch)
    has_tpos_ff_parents = all('tpos_first_frame_parents' in b for b in batch)
    has_caption_emb = any('caption_emb' in b for b in batch)

    spectral_feats = None
    if has_spectral_feats:
        # spectral width (k) can vary across samples — pad both joints and k
        max_freqs = max(b['spectral_feats'].shape[1] for b in batch)
        spectral_feats = torch.zeros((B, max_joints, max_freqs))
        for i, data in enumerate(batch):
            ev = data['spectral_feats']                      # (nj, k_i)
            spectral_feats[i, :ev.shape[0], :ev.shape[1]] = torch.from_numpy(ev)

    joint_names_emb = (
        _pad_joint_axis(
            batch, 'joint_names_emb', max_joints,
            (batch[0]['joint_names_emb'].shape[-1],),
        )
        if has_joint_names_emb else None
    )
    offsets = (
        _pad_joint_axis(batch, 'offsets', max_joints, (3,))
        if has_offsets else None
    )
    tpos_ff_parents = (
        _pad_joint_axis(batch, 'tpos_first_frame_parents', max_joints, (n_feats,))
        if has_tpos_ff_parents else None
    )
    caption_emb = None
    caption_tokens = None
    caption_mask = None
    if has_caption_emb:
        text_dim = next(b['caption_emb'].shape[-1] for b in batch if 'caption_emb' in b)
        caption_emb = torch.zeros((B, text_dim))
        # Token sequences for text_cond='cross_attn', right-padded to the
        # batch's longest caption with a validity mask. Every row keeps at
        # least one valid position: an all-masked row makes SDPA return NaN,
        # and a caption-less sample must contribute zeros, not NaNs.
        max_tok = max((b['caption_tokens'].shape[0]
                       for b in batch if 'caption_tokens' in b), default=1)
        caption_tokens = torch.zeros((B, max_tok, text_dim))
        caption_mask = torch.zeros((B, max_tok), dtype=torch.bool)
        caption_mask[:, 0] = True
        for i, data in enumerate(batch):
            if 'caption_emb' in data:
                caption_emb[i] = torch.from_numpy(data['caption_emb'])
            if 'caption_tokens' in data:
                toks = torch.from_numpy(data['caption_tokens'])   # (T_i, D)
                caption_tokens[i, :toks.shape[0]] = toks
                caption_mask[i, :toks.shape[0]] = True

    # --- Masks ---
    joint_mask = lengths_to_mask(
        n_joints_list, max_len=max_joints,
    ).unsqueeze(1).unsqueeze(1)                              # (B, 1, 1, J)

    lengths_mask = lengths_to_mask(
        motion_lengths, max_len=max_len,
    ).unsqueeze(1).unsqueeze(1)                              # (B, 1, 1, F)

    # --- Assemble cond dict ---
    cond = {
        'crop_start_ind': crop_start_inds,
        'motion_length': motion_lengths,
        'lengths_mask': lengths_mask,
        'n_joints': n_joints_list,
        'joint_mask': joint_mask,
        'tpos_first_frame': tpos_first_frame,
        'parents': parents_list,
        'edge_indexs': edge_indexs_list,
        'joint_relations': joint_relations,
        'graph_dist': graph_dist,
        'joint_depths': joint_depths,
        'mean': mean,
        'std': std,
    }

    # Optional tensor fields
    for key, value in (
        ('spectral_feats', spectral_feats),
        ('joint_names_emb', joint_names_emb),
        ('offsets', offsets),
        ('tpos_first_frame_parents', tpos_ff_parents),
        ('caption_emb', caption_emb),
        ('caption_tokens', caption_tokens),
        ('caption_mask', caption_mask),
    ):
        if value is not None:
            cond[key] = value

    # Optional string/list fields (per-sample scalars, no padding)
    for key in ('object_type', 'split_tag', 'caption'):
        if all(key in b for b in batch):
            cond[key] = [b[key] for b in batch]

    return motion, cond
