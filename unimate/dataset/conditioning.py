"""Build conditioning batches for sampling / visualization / inference.

``create_sample_condition`` selects reference clips from the loaded
``MotionDataset`` (see the mode priority in its docstring), reruns the same
crop → condition-extraction → normalization → padding pipeline used for
training samples, and collates the result with ``mixture_batch_collate`` so
the generative model sees exactly the training-time cond format.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from Quaternions import Quaternions

from unimate.configs.schema import MainConfig
from unimate.dataset.transforms import (
    apply_cropping, extract_conditions, apply_normalization,
    apply_padding, build_parent_features,
)
from unimate.dataset.mixture.collate import mixture_batch_collate
from unimate.utils.motion_utils import realign_unimate_clip
from unimate.utils.topology_utils import compute_edge_indexs


def _resolve_dataset(data):
    """Accept either a Dataset or DataLoader, return the underlying Dataset."""
    if isinstance(data, torch.utils.data.DataLoader):
        return data.dataset
    if isinstance(data, torch.utils.data.Dataset):
        return data
    raise ValueError("data must be either a Dataset or DataLoader instance.")


def create_sample_condition(
    config: MainConfig,
    data,
    num_samples: int = 3,
    object_types: Optional[List[str]] = None,
    test_case_captions: Optional[List[Tuple]] = None,
    gt_start_frame: Optional[int] = None,
):
    """Build a conditioning batch for sampling / visualization / inference.

    Modes (in priority order):
      1. ``test_case_captions`` — list of
         ``(object_type, caption, caption_enc[, clip_name])``, where
         ``caption_enc`` is ``{'caption_emb': (D,)[, 'caption_tokens':
         (T, D)]}``. One reference
         clip per entry provides the target skeleton; the clip's own caption
         is replaced by the user-supplied text + embedding. If a 4th element
         ``clip_name`` is provided, that exact clip is used (resolved against
         train then eval motion dicts); otherwise a random clip is picked.
         The explicit-clip variant is required for motion in-betweening so
         the generated batch lines up with the GT being clamped at sample
         time. Used by ``unimate/inference/sample.py``.
      2. ``object_types`` — pick one train clip and/or one eval clip per
         listed object_type (each emitted only when its map has clips).
      3. default — auto-sample ``num_samples`` types per dataset, routed by
         split mode so visualization always covers both train and eval:
           * ``'object_type'`` (truebones, objaverse): pick ``num_samples``
             train types + ``num_samples`` eval types, one random clip each.
           * ``'clip'`` (mixamo): pick ``num_samples`` types and emit one
             train + one eval clip per picked type (eval skipped when
             ``eval_map`` is empty for that type).

    Args:
        gt_start_frame: when set, the GT clip is cropped starting at this
            absolute frame index instead of the default random ('tpos') /
            zero ('first_frame') start. If fewer than
            ``max_motion_length`` frames remain, the crop is trimmed and the
            tail is zero-padded. ``cond["motion_length"]`` does NOT report
            that: it is deliberately pinned to ``max_motion_length`` so the
            model generates (and attends over) the whole padded window. The
            caller tracks the real length separately — see
            ``sample._gt_valid_lengths`` — and trims outputs at save time.
    """
    dataset = _resolve_dataset(data)
    md = dataset.motion_dataset

    topology_cond_type = config.dataset.topology_condition_type

    train_map = md.train_object_motions_map
    eval_map = md.eval_object_motions_map
    split_mode = md.split_mode_by_dataset

    def _pick(pool, n):
        if not pool:
            return []
        return random.sample(pool, n) if len(pool) >= n else random.choices(pool, k=n)

    # selection: (object_type, clip_name, split_tag) tuples consumed by the
    # batch-building loop below. caption overrides keyed by *selection index*
    # so multiple test cases can share the same reference clip without
    # overwriting each other's caption.
    selection = []
    caption_override_by_idx: Dict[int, Tuple[str, np.ndarray]] = {}

    if test_case_captions is not None:
        # Reference clip provides topology only; the user override supplies
        # the caption. Prefer a train clip; fall back to eval for held-out
        # types. When a 4th tuple element is given, use that exact clip
        # (in-betweening / motion editing pin a specific GT motion).
        for entry in test_case_captions:
            if len(entry) == 4:
                obj_type, caption, caption_enc, explicit_clip = entry
            else:
                obj_type, caption, caption_enc = entry
                explicit_clip = None

            train_clips = train_map.get(obj_type, [])
            eval_clips = eval_map.get(obj_type, [])

            if explicit_clip is not None:
                if explicit_clip in train_clips:
                    clip_name, tag = explicit_clip, 'train'
                elif explicit_clip in eval_clips:
                    clip_name, tag = explicit_clip, 'eval'
                else:
                    raise ValueError(
                        f"clip_name={explicit_clip!r} not found for "
                        f"object_type={obj_type!r} in train or eval maps."
                    )
            else:
                primary = train_clips or eval_clips
                tag = 'train' if train_clips else 'eval'
                if not primary:
                    continue
                clip_name = random.choice(primary)

            caption_override_by_idx[len(selection)] = (caption, caption_enc)
            selection.append((obj_type, clip_name, tag))
    elif object_types is not None:
        # One train + one eval clip per requested type when both pools have
        # entries. With ``test_split_ratio == 0`` or a fully held-out type,
        # emit just one — no synthetic duplicate.
        for object_type in object_types:
            train_clips = train_map.get(object_type, [])
            eval_clips = eval_map.get(object_type, [])
            if not (train_clips or eval_clips):
                continue
            if train_clips:
                selection.append((object_type, random.choice(train_clips), 'train'))
            if eval_clips:
                selection.append((object_type, random.choice(eval_clips), 'eval'))
    else:
        for dataset_type, obj_counts in md.dataset_object_count.items():
            mode = split_mode.get(dataset_type, 'clip')

            if mode == 'object_type':
                train_types = [t for t in obj_counts if train_map.get(t)]
                eval_types = [t for t in obj_counts if eval_map.get(t)]
                for t in _pick(train_types, num_samples):
                    selection.append((t, random.choice(train_map[t]), 'train'))
                for t in _pick(eval_types, num_samples):
                    selection.append((t, random.choice(eval_map[t]), 'eval'))
            else:
                # 'clip' mode: emit train (+eval when present) per picked type.
                # With test_split_ratio==0 (eval_map empty) the train pool is
                # not sampled twice just to hit a sample count.
                all_types = list(obj_counts.keys())
                for t in _pick(all_types, num_samples):
                    train_clips = train_map.get(t, [])
                    eval_clips = eval_map.get(t, [])
                    if not (train_clips or eval_clips):
                        continue
                    if train_clips:
                        selection.append((t, random.choice(train_clips), 'train'))
                    if eval_clips:
                        selection.append((t, random.choice(eval_clips), 'eval'))

    batches = []
    for sel_idx, (object_type, selected_motion_name, _split_tag) in enumerate(selection):
        condition = md.cond_dict[object_type].copy()
        source = md.train_motion_dict if _split_tag == 'train' else md.eval_motion_dict
        motion_data = source[selected_motion_name].copy()
        assert motion_data['object_type'] == object_type

        raw_motion = motion_data['motion']  # (F, J, D)
        parents = motion_data['parents']    # (J,)
        tpos = np.array(condition['tpos_first_frame'])

        max_motion_length = config.dataset.max_motion_length
        raw_motion, start_idx = apply_cropping(
            raw_motion, topology_cond_type, max_motion_length,
            start_idx=gt_start_frame,
        )

        # Re-align so frame-0 facing is identity (matches training-time
        # invariant assumed by the feature pipeline).
        if start_idx > 0:
            raw_motion = realign_unimate_clip(raw_motion, parents)

        # Pad tpos (J, 3) to motion feature dim (J, 12): identity 6D
        # rotation + zero velocity, matching the layout of motion features.
        feature_len = config.dataset.feature_len
        if tpos.shape[-1] < feature_len:
            J = tpos.shape[0]
            identity_6d = Quaternions.id(1).rotation_matrix(cont6d=True)[0]
            pad = np.zeros((J, feature_len - 3))
            pad[:, :6] = identity_6d
            tpos = np.concatenate([tpos, pad], axis=-1)

        conds = extract_conditions(raw_motion, tpos, topology_cond_type)
        motion = conds['motion']
        tpos_first_frame = conds['tpos_first_frame']
        njoints, nfeats = tpos_first_frame.shape

        # Dataset-level normalization stats (root vs. local split).
        dataset_type = motion_data['dataset_type']
        stats = md.dataset_stats[dataset_type]
        mean = np.zeros((njoints, nfeats))
        std = np.zeros((njoints, nfeats))
        mean[0, :], std[0, :] = stats['mean_root'], stats['std_root']
        mean[1:, :], std[1:, :] = stats['mean_local'], stats['std_local']

        motion = apply_normalization(motion, mean, std)
        tpos_first_frame = apply_normalization(tpos_first_frame, mean, std)

        # Pad to fixed length so the generator always produces same-shape outputs.
        motion, _ = apply_padding(motion, max_motion_length)

        parent_feats = build_parent_features(tpos_first_frame, parents)
        # Reuse the clip's stored eigenvectors (loaded via ``create_condition``)
        # rather than recomputing: eigendecomposition signs are arbitrary, and
        # the stored copy is the sign-pinned version the model saw at training
        # time (matters for the non-sign-invariant WIRE spectral backend).
        spectral_feats = motion_data['spectral_feats']

        batch = {
            'motion': motion,
            'max_motion_length': config.dataset.max_motion_length,
            # Always the full window, even for a short crop: the model is
            # asked to generate every frame. See the gt_start_frame note above.
            'motion_length': max_motion_length,
            'max_joints': config.dataset.max_joints,
            'parents': parents,
            'edge_indexs': compute_edge_indexs(parents),
            'tpos_first_frame': tpos_first_frame,
            'tpos_first_frame_parents': parent_feats['tpos_first_frame_parents'],
            'offsets': motion_data['offsets'],
            'joint_graph_dist': motion_data['joint_graph_dist'],
            'joint_relations': motion_data['joint_relations'],
            'joint_depths': motion_data['joint_depths'],
            'spectral_feats': spectral_feats,
            'joint_names_emb': motion_data['joint_names_emb'],
            'object_type': object_type,
            'start_idx': start_idx,
            'mean': mean,
            'std': std,
            'split_tag': _split_tag,
        }

        if sel_idx in caption_override_by_idx:
            caption, caption_enc = caption_override_by_idx[sel_idx]
            batch['caption'] = caption
            # caption_enc is {'caption_emb': (D,)[, 'caption_tokens': (T, D)]};
            # a caller that only produced the pooled vector still works.
            batch.update(caption_enc)
        else:
            for key in ('caption', 'caption_emb', 'caption_tokens'):
                if key in motion_data:
                    batch[key] = motion_data[key]

        batches.append(batch)

    motion, cond = mixture_batch_collate(batches)
    return motion, cond
