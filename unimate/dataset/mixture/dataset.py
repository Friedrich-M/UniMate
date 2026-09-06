"""The unified Mixture dataset: heterogeneous-skeleton motion clips in memory.

Three layers:
  * :class:`Mixture` — top-level ``torch.utils.data.Dataset`` used by the
    dataloader factory. Loads per-dataset ``cond.npy`` / captions /
    ``test_objects.txt``, applies subset + joint-count filters, then builds
    the inner :class:`MotionDataset`.
  * :class:`MotionDataset` — holds all clips in memory, pre-encodes text,
    applies the train/eval split, augmentations, normalization stats, and
    serves per-sample dicts consumed by ``collate.mixture_batch_collate``.
  * :class:`MixtureSampler` — optional power-law balanced sampler across
    object types.
"""

import json
import os
import random
from collections import defaultdict
from os.path import join as pjoin
from typing import Optional, Set

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from Quaternions import Quaternions

from unimate.dataset.transforms import (
    apply_cropping, extract_conditions, apply_normalization,
    apply_padding, build_parent_features,
)
from unimate.utils.motion_utils import (
    realign_unimate_clip,
    compute_unimate_motion_feats,
    compute_rots_from_tpos,
    recover_unimate_joint_pos_from_ric,
    recover_unimate_joint_pos_from_rot,
)
from unimate.utils.topology_utils import compute_laplacian_eigenvectors
from unimate.models.text_encoder.factory import create_text_encoder
from unimate.utils.text_emb_cache import load_cache, load_caches, pool as pool_tokens
from unimate.utils.logger import get_logger
from unimate.dataset.mixture import augmentations as aug_ops
from Animation import offsets_from_positions


logger = get_logger(file_name=__file__, debug="mixture_dataset")


# Per-dataset train/eval split mode:
#   - 'object_type' — hold out whole object types as eval. Default for
#                     multi-topology datasets (truebones, objaverse): training
#                     never sees the held-out skeletons, so eval measures
#                     generalization to unseen topologies.
#   - 'clip'        — stratified per-object-type at the clip level (train/eval
#                     drawn from each type). Used for single-topology datasets
#                     (mixamo) where there is only one object_type and an
#                     object_type-level holdout would empty the train set.
#
# An explicit ``test_objects.txt`` next to ``cond.npy`` overrides the mode
# above for that dataset: the listed object types become the eval set
# (object_type-style split), regardless of ``test_split_ratio``. Useful when
# the test set is fixed by hand (e.g. truebones evaluation list).
_DATASET_SPLIT_MODE = {
    'mixamo':    'clip',
    'truebones': 'object_type',
    'objaverse': 'object_type',
}

# The UniMate representation stores per-frame deltas (root trajectory and
# per-joint local velocity) without normalizing by the frame rate, so the
# frame rate is part of the feature scale: the same physical motion sampled
# at 60 fps yields half the velocity of a 30 fps one. Stage 4 emits 30 fps
# clips (``data_process/utils/motion_features.py::_load_npz_anim`` downsamples
# anything faster), so a clip declaring anything else was built under a
# different assumption and is rejected rather than silently mixed in.
EXPECTED_CLIP_FPS = 30

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_condition(cond_dict, max_freqs=8):
    """Extract skeleton topology fields from a per-object condition dict.

    All fields are pre-computed by
    ``data_process.utils.motion_features.build_topology_cond`` — reading them here
    avoids dataloader-init recomputation and pins eigenvector signs that
    ``compute_laplacian_eigenvectors`` would otherwise pick non-deterministically.
    """
    parents = np.asarray(cond_dict['parents'])
    spectral_feats = np.asarray(cond_dict['spectral_feats'])
    if spectral_feats.shape[-1] != max_freqs:
        logger.warning(
            f"Saved spectral_feats has shape[-1]={spectral_feats.shape[-1]} "
            f"but max_freqs={max_freqs}; recomputing from parents."
        )
        spectral_feats, _ = compute_laplacian_eigenvectors(parents, max_freqs=max_freqs)

    return {
        'parents':          parents,                                        # (J,)
        'edge_indexs':      np.asarray(cond_dict['edge_indexs']),           # (2, 2*(J-1))
        'joint_graph_dist': np.asarray(cond_dict['joint_graph_dists']),     # (J, J)
        'joint_relations':  np.asarray(cond_dict['joint_relations']),       # (J, J)
        'joint_depths':     np.asarray(cond_dict['joint_depths']),          # (J,)
        'spectral_feats':   spectral_feats,                                 # (J, K)
        'tpos':             np.asarray(cond_dict['tpos_first_frame']),      # (J, 3)
    }


def _validate_motion(motion_data, parents=None, offsets=None,
                     gt_global_positions=None, recovery_tol=1e-4,
                     cross_check=False):
    """Check motion array for NaN/Inf/zero-length. Returns (is_valid, reason).

    For 12-dim UniMate features:
      - ``gt_global_positions`` provided → round-trip via RIC and FK; both
        must match GT within ``recovery_tol``.
      - ``cross_check=True`` (no GT, e.g. post-aug) → RIC and FK must agree;
        divergence signals the aug broke the position↔rotation invariant.
    """
    if motion_data.size == 0 or motion_data.shape[0] == 0:
        return False, "empty motion (0 frames)"
    if np.isnan(motion_data).any():
        nan_pct = 100 * np.isnan(motion_data).sum() / motion_data.size
        return False, f"contains NaN ({nan_pct:.1f}% of values)"
    if np.isinf(motion_data).any():
        inf_count = np.isinf(motion_data).sum()
        return False, f"contains {inf_count} Inf values"

    if gt_global_positions is not None:
        gt = gt_global_positions[:-1]
        err_ric = np.abs(recover_unimate_joint_pos_from_ric(motion_data) - gt).max()
        err_fk = np.abs(
            recover_unimate_joint_pos_from_rot(motion_data, parents, offsets) - gt
        ).max()
        if err_ric > recovery_tol or err_fk > recovery_tol:
            return False, (
                f"recovery mismatch: ric_max={err_ric:.4g}, fk_max={err_fk:.4g} "
                f"(tol={recovery_tol:g})"
            )
    elif cross_check:
        ric_pos = recover_unimate_joint_pos_from_ric(motion_data)
        fk_pos = recover_unimate_joint_pos_from_rot(motion_data, parents, offsets)
        err = np.abs(ric_pos - fk_pos).max()
        if err > recovery_tol:
            return False, (
                f"RIC↔FK divergence: max={err:.4g} (tol={recovery_tol:g})"
            )

    return True, ""


def model_joint_names(cond_entry, object_type):
    """Joint names the model is conditioned on — the CLEANED vocabulary.

    ``clean_joint_names`` is stage 3's output (`data_process/joint_annotation`):
    the rig's arbitrary bone strings (``Bip01_L_Thigh``, ``BN__Neck_L_01``)
    mapped onto one shared anatomical vocabulary (``Left Thigh``, ``Neck``).
    That shared vocabulary is the whole point of embedding joint names across
    heterogeneous skeletons — raw rig strings would make the same anatomical
    joint embed differently per asset, which is exactly what this conditioning
    exists to avoid.

    Falling back to the raw names keeps an old ``cond.npy`` loadable, but it
    silently degrades conditioning, so say so rather than doing it quietly.
    The same rule is applied by ``unimate/tools/precompute_text_emb.py``; the
    two must agree or the cached embeddings will not cover what is looked up.
    """
    names = cond_entry.get('clean_joint_names')
    n_expected = len(cond_entry['joint_names'])
    if names is not None and len(names) == n_expected and all(
            str(n).strip() for n in names):
        return [str(n) for n in names]

    reason = ('absent' if names is None
              else f'{len(names)} entries for {n_expected} joints'
              if len(names) != n_expected else 'contains blank names')
    logger.warning(
        f"[{object_type}] clean_joint_names {reason}; conditioning on the raw "
        f"rig names instead. Re-run data_process stage 3 (joint_annotation) "
        f"for this asset — raw names do not share a vocabulary across rigs."
    )
    return [str(n) for n in cond_entry['joint_names']]


# ---------------------------------------------------------------------------
# MotionDataset — stores all motion clips in memory
# ---------------------------------------------------------------------------

class MotionDataset(data.Dataset):
    """In-memory dataset of motion clips across heterogeneous skeletons.

    ``__getitem__`` only yields train clips (from ``train_motion_dict`` /
    ``train_name_list``). Eval clips are kept in memory and exposed via
    ``eval_motion_dict`` / ``eval_name_list`` / ``eval_object_motions_map``
    for inference and visualization. Normalization stats are computed from
    train clips only to prevent leakage.
    """

    def __init__(
        self,
        data_dict: dict,
        topology_condition_type: str,
        max_motion_length: int,
        config_max_depth: int = 0,
        feature_len: int = 12,
        text_encoder_type: str = "t5",
        text_encoder_version: str = "t5-base",
        use_dataset_stats: bool = True,
        balanced_stats: bool = False,
        tie_std: bool = False,
        use_addition_aug: bool = False,
        use_removal_aug: bool = False,
        use_pooling_aug: bool = False,
        use_perturbation_aug: bool = False,
        test_split_ratio: float = 0.0,
        split_seed: int = 42,
        max_freqs: int = 8,
        debug_validate_aug: bool = False,
        ground_motion_height: bool = True,
        realign_feature: bool = True,
        prebuilt_max_joints: int = 0,
        prebuilt_max_depth: int = 0,
        target_object_types: Optional[Set[str]] = None,
        target_clip_stems: Optional[Set[str]] = None,
        stats_path: Optional[str] = None,
    ):
        self._init_config(
            topology_condition_type, max_motion_length, config_max_depth,
            feature_len, use_dataset_stats, balanced_stats,
            tie_std, use_addition_aug, use_removal_aug, use_pooling_aug,
            use_perturbation_aug, test_split_ratio, split_seed, max_freqs,
            debug_validate_aug, ground_motion_height, realign_feature,
        )
        self._target_object_types = target_object_types
        self._target_clip_stems = target_clip_stems
        self._stats_path = stats_path
        self._init_data_state(data_dict)
        if self._target_object_types is not None:
            # Sampling uses both train and eval maps; honoring test_objects.txt
            # would route a target type entirely to eval and trip the
            # _eval_clips_explicit "don't drain the dataset" guard.
            self.explicit_eval_objects = {dt: [] for dt in self.explicit_eval_objects}
        self._build_text_embeddings(
            data_dict, text_encoder_type, text_encoder_version,
        )
        motion_dict, name_list, selected_object_types = self._load_all_clips(data_dict)
        self._finalize_splits(motion_dict, name_list, selected_object_types)
        if prebuilt_max_joints > 0 and prebuilt_max_depth > 0:
            # On-disk data can grow between training and inference; trusting
            # the saved values keeps the tensor dim matched to the checkpoint.
            self.max_joints, self.max_depth = prebuilt_max_joints, prebuilt_max_depth
            logger.info(
                f'Using prebuilt max_joints={self.max_joints} | '
                f'max_depth={self.max_depth} (skipping data scan).'
            )
        else:
            self.max_joints, self.max_depth = self._compute_max_joints_and_depth()

        logger.info(
            f'MotionDataset loaded {len(self.train_motion_dict)} train + '
            f'{len(self.eval_motion_dict)} eval clips '
            f'from {len(self.selected_object_types)} object types '
            f'| max_joints={self.max_joints} | max_depth={self.max_depth}'
        )

        self._calculate_dataset_stats()

    # ------------------------------------------------------------------
    # Init phases
    # ------------------------------------------------------------------

    def _init_config(self, topology_condition_type, max_motion_length,
                     config_max_depth, feature_len,
                     use_dataset_stats, balanced_stats, tie_std,
                     use_addition_aug, use_removal_aug, use_pooling_aug,
                     use_perturbation_aug, test_split_ratio, split_seed,
                     max_freqs, debug_validate_aug,
                     ground_motion_height, realign_feature):
        """Stash hyperparameters as instance attributes."""
        self.topology_condition_type = topology_condition_type
        self.max_motion_length = max_motion_length
        self.config_max_depth = config_max_depth
        self.feature_len = feature_len
        self.use_dataset_stats = use_dataset_stats
        self.balanced_stats = balanced_stats
        self.tie_std = tie_std
        self.use_addition_aug = use_addition_aug
        self.use_removal_aug = use_removal_aug
        self.use_pooling_aug = use_pooling_aug
        self.use_perturbation_aug = use_perturbation_aug
        self.test_split_ratio = test_split_ratio
        self.split_seed = split_seed
        self.max_freqs = max_freqs
        self.debug_validate_aug = debug_validate_aug
        self.ground_motion_height = ground_motion_height
        self.realign_feature = realign_feature
        # Motion dirs already warned about for a missing clip fps field.
        self._fps_unset_dirs = set()

    def _init_data_state(self, data_dict):
        """Initialize per-dataset state derived from ``data_dict``."""
        self.dataset_stats = {}
        self.cond_dict = {}
        self.dataset_object_count = {}
        self.split_mode_by_dataset = dict(_DATASET_SPLIT_MODE)
        # ``explicit_eval_objects[dt]`` (from ``test_objects.txt``) overrides
        # the per-dataset split mode and bypasses ``test_split_ratio``.
        self.explicit_eval_objects = {
            dt: list(info.get('test_objects') or [])
            for dt, info in data_dict.items()
        }

    def _build_text_embeddings(self, data_dict, text_encoder_type,
                               text_encoder_version):
        """Pre-encode joint names + captions, then drop the encoder.

        Anything already in a feature directory's text-embedding cache (see
        ``unimate/tools/precompute_text_emb.py``) is read from disk; the
        encoder is built only if something is still missing, so a complete
        cache keeps it off the GPU entirely.
        """
        self._encoder_spec = (text_encoder_type, text_encoder_version)
        self.text_encoder = None
        roots = [info.get('root_dir') for info in data_dict.values()]
        # Joint names are shared vocabulary — the same string encodes
        # identically everywhere, so merging them by name across datasets is
        # safe. Clip keys are NOT: the datasets name clips by different
        # conventions and nothing guarantees they never collide, so a caption
        # is keyed by (dataset_type, clip) and a collision is impossible
        # rather than merely unlikely.
        joint_cache = load_caches(roots, 'joint', text_encoder_type,
                                  text_encoder_version)
        text_cache = {
            (ds_type, clip): tokens
            for ds_type, info in data_dict.items()
            for clip, tokens in load_cache(info.get('root_dir'), 'caption',
                                           text_encoder_type,
                                           text_encoder_version).items()
        }
        self._preencode_joint_names(data_dict, joint_cache)
        self._preencode_captions(data_dict, text_cache)
        if self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _ensure_text_encoder(self):
        """Build the text encoder on first use (i.e. on the first cache miss)."""
        if self.text_encoder is None:
            encoder_type, encoder_version = self._encoder_spec
            logger.info(f'Text cache incomplete; loading {encoder_type} encoder')
            self.text_encoder = create_text_encoder(
                encoder_type=encoder_type,
                encoder_version=encoder_version,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                pool=False,   # token sequences; pooling happens below
            )
        return self.text_encoder

    def _load_all_clips(self, data_dict):
        """Run each dataset's loader and return (motion_dict, names, types)."""
        motion_dict = {}
        name_list = []
        selected_object_types = set()

        loaders = {
            'truebones': self._load_multi_topology,
            'mixamo':    self._load_single_topology,
            'objaverse': self._load_multi_topology,
        }
        for dataset_type, dataset_info in data_dict.items():
            loader = loaders.get(dataset_type)
            if loader is None:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            loader(dataset_info, dataset_type, motion_dict,
                   selected_object_types, name_list)

        return motion_dict, name_list, selected_object_types

    def _finalize_splits(self, motion_dict, name_list, selected_object_types):
        """Apply the train/eval split and build the per-object_type maps."""
        train_names, eval_names = self._split_names(
            name_list, motion_dict, self.test_split_ratio, self.split_seed,
            self.split_mode_by_dataset, self.explicit_eval_objects,
        )
        if not train_names:
            if self._target_object_types is None:
                raise ValueError(
                    f"Empty train split (test_split_ratio={self.test_split_ratio}).")
            logger.info("Empty train split at inference; sampling will use eval map.")

        self.train_motion_dict = {n: motion_dict[n] for n in train_names}
        self.eval_motion_dict = {n: motion_dict[n] for n in eval_names}
        self.train_name_list = train_names
        self.eval_name_list = eval_names

        self.train_object_motions_map = self._group_by_object(self.train_motion_dict)
        self.eval_object_motions_map = self._group_by_object(self.eval_motion_dict)
        self.selected_object_types = sorted(selected_object_types)

        total = len(train_names) + len(eval_names)
        effective_ratio = len(eval_names) / total if total else 0.0
        logger.info(
            f"Clip split: {len(train_names)} train / {len(eval_names)} eval "
            f"(ratio={effective_ratio:.3f}, seed={self.split_seed})"
        )

    @staticmethod
    def _group_by_object(clip_dict):
        grouped = {}
        for nm, entry in clip_dict.items():
            grouped.setdefault(entry['object_type'], []).append(nm)
        return grouped

    # ------------------------------------------------------------------
    # Per-dataset loaders
    # ------------------------------------------------------------------

    def _batch_encode_unique(self, texts, label: str, chunk_size: int = 256,
                             cache=None):
        """Dedupe ``texts`` and encode the unique set in fixed-size chunks.

        Returns ``{text: (T_i, emb_dim) ndarray}`` — the token sequence with
        padding removed; falsy strings dropped. Chunking bounds peak GPU
        memory for long sequences.
        """
        unique, seen = [], set()
        total = 0
        for t in texts:
            if not t:
                continue
            total += 1
            if t not in seen:
                seen.add(t)
                unique.append(t)

        if not unique:
            return {}

        cache = cache or {}
        out = {t: cache[t] for t in unique if t in cache}
        todo = [t for t in unique if t not in out]
        if not todo:
            logger.info(
                f'Pre-encoding {label}: {len(unique)} unique / {total} total '
                f'— all from the text-embedding cache'
            )
            return out

        logger.info(
            f'Pre-encoding {label}: {len(unique)} unique / {total} total '
            f'({len(out)} cached, {len(todo)} to encode, chunk_size={chunk_size})'
        )
        encoder = self._ensure_text_encoder()
        n_chunks = (len(todo) + chunk_size - 1) // chunk_size
        for i in tqdm(range(0, len(todo), chunk_size),
                      desc=f'Encoding {label}', total=n_chunks):
            chunk = todo[i:i + chunk_size]
            inputs = encoder.tokenize(chunk)
            hidden = encoder(inputs).detach().cpu().numpy()      # (B, T, D)
            lengths = inputs['attention_mask'].sum(dim=-1).cpu().numpy()
            for j, text in enumerate(chunk):
                # Drop padding so the stored rows match the cache's ragged
                # form; one row is kept even for an empty string.
                out[text] = hidden[j, :max(int(lengths[j]), 1)]
        return out

    def _preencode_joint_names(self, data_dict, cache=None):
        """Encode every unique joint name → ``self._joint_name_emb``.

        Scoped to ``target_object_types`` when set.
        """
        targets = self._target_object_types
        all_names = [
            name
            for dataset_info in data_dict.values()
            for ot, cond_entry in dataset_info['cond_dict'].items()
            if targets is None or ot in targets
            for name in model_joint_names(cond_entry, ot)
        ]
        # One vector per joint. The cache already stores joint names pooled
        # (it is all the model reads of them), so only freshly encoded
        # sequences still need pooling here.
        self._joint_name_emb = {
            name: emb if emb.ndim == 1 else pool_tokens(emb)
            for name, emb in self._batch_encode_unique(
                all_names, 'joint names', cache=cache).items()
        }

    def _preencode_captions(self, data_dict, cache=None):
        """Encode each clip's caption → ``self._caption_tokens`` (+ pooled).

        Skipped when ``target_object_types`` is set: captions come from the
        per-test-case spec and are encoded inline at sample time.
        """
        if self._target_object_types is not None:
            self._caption_emb = {}
            self._caption_tokens = {}
            logger.info('Skipping caption pre-encoding (inference mode).')
            return
        caption_by_clip = {
            (ds_type, clip): cap
            for ds_type, dataset_info in data_dict.items()
            for clip, cap in dataset_info.get('captions', {}).items()
            if cap
        }
        # Keyed by (dataset_type, clip): a clip's
        # features are one lookup, with no caption-string round trip. Repeated
        # captions are still encoded once — _batch_encode_unique dedupes the
        # strings before the forward pass.
        cache = cache or {}
        todo = {k: c for k, c in caption_by_clip.items() if k not in cache}
        by_text = self._batch_encode_unique(list(todo.values()), 'captions')

        # A cache hit carries both views: the token sequence text_cond=
        # 'cross_attn' attends, and the pooled vector text_cond='adaln' reads
        # — the latter exactly as the encoder produced it, so adaLN gets the
        # same number the pooled encoder always returned. Only a cache MISS
        # falls back to pooling the sequence here.
        self._caption_tokens, self._caption_emb = {}, {}
        for k in caption_by_clip:
            if k in cache:
                tokens, pooled = cache[k]
            else:
                tokens = by_text.get(caption_by_clip[k])
                if tokens is None:
                    continue
                pooled = pool_tokens(tokens)
            self._caption_tokens[k] = tokens
            self._caption_emb[k] = pooled

    def _precompute_object_type_meta(self, object_type, cond_entry):
        """Per-object-type meta reused across all clips of that type."""
        condition = create_condition(cond_entry, max_freqs=self.max_freqs)

        # Ground the T-pose to match how ``_load_motion_file`` grounds motion;
        # otherwise the network sees two different Y references when
        # ``topology_condition_type='tpos'``.
        if self.ground_motion_height:
            condition['tpos'] = condition['tpos'].copy()
            condition['tpos'][..., 1] -= condition['tpos'][..., 1].min()

        joint_names = model_joint_names(cond_entry, object_type)
        joint_names_emb = np.stack(
            [self._joint_name_emb[n] for n in joint_names]
        )  # (J, emb_dim)
        assert joint_names_emb.shape[0] == condition['parents'].shape[0], (
            f"Joints names embedding / joint count mismatch for {object_type}."
        )

        offsets = offsets_from_positions(condition['tpos'], condition['parents'])  # (J, 3)

        return {
            'object_type': object_type,
            'joint_names_emb': joint_names_emb,
            'condition': condition,
            'offsets': offsets,
            'tpos_local_rotations': np.asarray(cond_entry['tpos_local_rotations']),
        }

    def _load_motion_file(self, motion_dir, motion_name, dataset_type, meta, captions):
        """Load one clip, validate it, and build its entry dict. Returns
        ``None`` when the file is invalid or has no caption.
        """
        # Check caption first — depends only on the filename, so failing here
        # avoids the NPZ load + feature compute for any clip we'd discard.
        clip_key = os.path.splitext(motion_name)[0]
        caption = captions.get(clip_key, '')
        if not caption and self._target_object_types is None:
            logger.warning(f'Skipping {motion_name}: missing caption for {clip_key}')
            return None

        parents = meta['condition']['parents']
        offsets = meta['offsets']
        npz = np.load(pjoin(motion_dir, motion_name), allow_pickle=False)
        global_positions = npz['global_positions']              # (F, J, 3)
        local_rotations = npz['local_rotations']                # (F, J, 4)
        root_facing_quat = npz['root_facing_quat']              # (F, 4)
        self._check_clip_fps(motion_dir, motion_name, npz)

        if self.ground_motion_height:
            global_positions = global_positions.copy()
            global_positions[..., 1] -= global_positions[..., 1].min()

        # Rebase against the T-pose reference so features live in a
        # topology-consistent rotation frame.
        tpos_q = np.broadcast_to(
            meta['tpos_local_rotations'][None],
            local_rotations.shape,
        ).copy()

        rebased = compute_rots_from_tpos(
            Quaternions(tpos_q), Quaternions(local_rotations), parents,
        )
        motion_feats = compute_unimate_motion_feats(
            global_positions, rebased, parents, Quaternions(root_facing_quat),
        )  # (F-1, J, 12)

        valid, reason = _validate_motion(
            motion_feats,
            parents=parents,
            offsets=offsets,
            gt_global_positions=global_positions,
        )
        if not valid:
            logger.warning(f'Skipping {motion_name}: {reason}')
            return None

        entry = {
            'motion': motion_feats,                              # (F-1, J, 12)
            'offsets': offsets,                                  # (J, 3)
            'object_type': meta['object_type'],
            'joint_names_emb': meta['joint_names_emb'],          # (J, emb_dim)
            'dataset_type': dataset_type,
        }
        entry.update(meta['condition'])

        entry['caption'] = caption
        cap_key = (dataset_type, clip_key)
        if cap_key in self._caption_emb:
            entry['caption_emb'] = self._caption_emb[cap_key]        # (D,)
            entry['caption_tokens'] = self._caption_tokens[cap_key]  # (T, D)
        return entry

    def _check_clip_fps(self, motion_dir, motion_name, npz):
        """Reject clips whose frame rate is not the one the features assume.

        Pre-``fps`` feature sets predate the field; they were all built at 30
        fps, so a missing field is accepted with one warning per directory
        rather than failing the run.
        """
        if 'fps' not in npz:
            if motion_dir not in self._fps_unset_dirs:
                self._fps_unset_dirs.add(motion_dir)
                logger.warning(
                    f'{motion_dir}: clips carry no fps field (pre-fps feature '
                    f'set); assuming {EXPECTED_CLIP_FPS} fps.'
                )
            return
        fps = int(npz['fps'])
        if fps != EXPECTED_CLIP_FPS:
            raise ValueError(
                f'{pjoin(motion_dir, motion_name)}: clip is {fps} fps, but the '
                f'motion features assume {EXPECTED_CLIP_FPS} fps — velocities '
                f'would be off by {fps / EXPECTED_CLIP_FPS:.3g}x. Re-run stage 4 '
                f'(data_process/scripts/run_extract_features.sh) for this dataset.'
            )

    @staticmethod
    def _log_dataset_loaded(dataset_type, loaded, total, n_types):
        logger.info(
            f'{dataset_type.capitalize()} loaded {loaded}/{total} valid motions '
            f'across {n_types} object type{"s" if n_types != 1 else ""}'
        )

    def _load_multi_topology(self, dataset_info, dataset_type, motion_dict,
                             selected_object_types, name_list):
        """Loader for multi-topology datasets (truebones, objaverse)."""
        cond_dict = dataset_info['cond_dict']
        motion_dir = dataset_info['motion_dir']
        captions = dataset_info.get('captions', {})

        all_object_types = sorted(cond_dict.keys())
        logger.info(f'{dataset_type.capitalize()} dataset has {len(all_object_types)} object types')

        # Bucket files by object_type via one directory scan. Filenames are
        # ``{object_type}-{clip_id}.npz`` and object_type names never contain
        # ``-``, so the type is everything before the first ``-``.
        type_set = set(all_object_types)
        files_by_type = defaultdict(list)
        for f in sorted(os.listdir(motion_dir)):
            if not f.endswith('.npz'):
                continue
            sep_idx = f.find('-')
            if sep_idx <= 0:
                continue
            object_type = f[:sep_idx]
            if object_type in type_set:
                files_by_type[object_type].append(f)

        self.dataset_object_count[dataset_type] = {}
        total_seen = 0
        total_loaded = 0
        skipped_no_files = 0
        skipped_off_target = 0
        for object_type in tqdm(
            all_object_types, total=len(all_object_types),
            desc=f'Loading {dataset_type.capitalize()} motions',
        ):
            object_motions = files_by_type[object_type]
            if not object_motions:
                skipped_no_files += 1
                continue

            if (self._target_object_types is not None
                    and object_type not in self._target_object_types):
                skipped_off_target += 1
                continue

            if self._target_clip_stems is not None:
                object_motions = [
                    f for f in object_motions
                    if os.path.splitext(f)[0] in self._target_clip_stems
                ]
            elif self._target_object_types is not None:
                # Headroom > 1 so one bad NPZ doesn't drop the whole type;
                # create_sample_condition picks one at random anyway.
                object_motions = object_motions[:3]

            self.cond_dict[object_type] = cond_dict[object_type]
            meta = self._precompute_object_type_meta(
                object_type, cond_dict[object_type],
            )

            total_seen += len(object_motions)
            loaded_count = 0
            for motion_name in object_motions:
                entry = self._load_motion_file(
                    motion_dir, motion_name, dataset_type, meta, captions,
                )
                if entry is None:
                    continue
                motion_dict[motion_name] = entry
                selected_object_types.add(object_type)
                name_list.append(motion_name)
                loaded_count += 1

            self.dataset_object_count[dataset_type][object_type] = loaded_count
            total_loaded += loaded_count

        if skipped_no_files:
            logger.info(
                f'{dataset_type.capitalize()}: skipped {skipped_no_files}/'
                f'{len(all_object_types)} object types with no motion files in {motion_dir}'
            )
        if skipped_off_target:
            logger.info(
                f'{dataset_type.capitalize()}: skipped {skipped_off_target}/'
                f'{len(all_object_types)} object types outside inference target set'
            )

        self._log_dataset_loaded(
            dataset_type, total_loaded, total_seen,
            len(self.dataset_object_count[dataset_type]),
        )

    def _load_single_topology(self, dataset_info, dataset_type, motion_dict,
                              selected_object_types, name_list):
        """Loader for single-topology datasets (e.g. mixamo)."""
        cond_dict = dataset_info['cond_dict']
        motion_dir = dataset_info['motion_dir']
        captions = dataset_info.get('captions', {})

        assert len(cond_dict) == 1, (
            f'Single-topology loader expects exactly 1 object type, got {len(cond_dict)}'
        )
        object_type = next(iter(cond_dict))
        cond_data = cond_dict[object_type]

        logger.info(f'{dataset_type.capitalize()} dataset has 1 object type: {object_type}')

        self.dataset_object_count[dataset_type] = {}
        if (self._target_object_types is not None
                and object_type not in self._target_object_types):
            logger.info(
                f'{dataset_type.capitalize()}: object_type {object_type!r} not in '
                f'inference target set; skipping entire dataset.'
            )
            return

        object_motions = sorted(
            f for f in os.listdir(motion_dir) if f.endswith('.npz')
        )

        # Mixamo clip names are prefixed at load (``{object_type}_{motion_name}``);
        # the target stem comparison must mirror that prefix.
        if self._target_clip_stems is not None:
            object_motions = [
                f for f in object_motions
                if f'{object_type}_{os.path.splitext(f)[0]}' in self._target_clip_stems
            ]
        elif self._target_object_types is not None:
            object_motions = object_motions[:3]

        if not object_motions:
            logger.info(
                f'{dataset_type.capitalize()}: no motion files in {motion_dir}; '
                f'skipping object type {object_type}'
            )
            return

        # Register only after confirming clips exist, so empty types don't
        # leak into ``self.cond_dict`` / ``_compute_max_joints_and_depth``.
        self.cond_dict[object_type] = cond_data
        meta = self._precompute_object_type_meta(object_type, cond_data)

        loaded_count = 0
        for motion_name in tqdm(
            object_motions, total=len(object_motions),
            desc=f'Loading {dataset_type.capitalize()} motions',
        ):
            entry = self._load_motion_file(
                motion_dir, motion_name, dataset_type, meta, captions,
            )
            if entry is None:
                continue
            prefixed_name = f'{object_type}_{motion_name}'
            motion_dict[prefixed_name] = entry
            selected_object_types.add(object_type)
            name_list.append(prefixed_name)
            loaded_count += 1

        self.dataset_object_count[dataset_type][object_type] = loaded_count
        self._log_dataset_loaded(dataset_type, loaded_count, len(object_motions), 1)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.train_name_list)

    def __getitem__(self, idx):
        data = self.train_motion_dict[self.train_name_list[idx]]
        dataset_type = data['dataset_type']
        object_type = data['object_type']

        mean, std = self._get_normalization_stats(data, dataset_type)  # (J, D)
        aug = self._apply_augmentations(data, mean, std)

        augmented_motion = aug['motion']
        mean, std = aug['mean'], aug['std']

        assert augmented_motion.ndim == 3
        assert mean.shape == std.shape == augmented_motion.shape[1:]

        # Augmentations rewrite both position and rotation channels; RIC and
        # FK independently decode the same pose from each view, so divergence
        # signals a broken aug op (addition/removal/pooling/perturbation).
        if self.debug_validate_aug:
            valid, reason = _validate_motion(
                augmented_motion,
                parents=aug['parents'],
                offsets=aug['offsets'],
                cross_check=True,
            )
            if not valid:
                logger.warning(
                    f"[debug_validate_aug] {object_type} idx={idx}: {reason}"
                )

        augmented_motion, start_idx = apply_cropping(
            augmented_motion, self.topology_condition_type, self.max_motion_length,
        )

        # Re-align clips when cropping moved frame-0 forward, so the
        # new frame-0 facing direction is identity (matches training-time
        # invariant assumed by the feature pipeline).
        if self.realign_feature and start_idx > 0:
            augmented_motion = realign_unimate_clip(augmented_motion, aug['parents'])

        # Pad tpos (J, 3) to motion feature dim (J, 12): identity 6D
        # rotation + zero velocity, matching the layout of motion features.
        tpos = aug['tpos']
        if tpos.shape[-1] < self.feature_len:
            J = tpos.shape[0]
            identity_6d = Quaternions.id(1).rotation_matrix(cont6d=True)[0]  # (6,)
            pad = np.zeros((J, self.feature_len - 3))
            pad[:, :6] = identity_6d
            tpos = np.concatenate([tpos, pad], axis=-1)

        conds = extract_conditions(
            augmented_motion, tpos, self.topology_condition_type,
        )
        motion = conds['motion']
        tpos_first_frame = conds['tpos_first_frame']

        motion = apply_normalization(motion, mean, std)
        tpos_first_frame = apply_normalization(tpos_first_frame, mean, std)
        motion, motion_length = apply_padding(motion, self.max_motion_length)

        parents = aug['parents']
        parent_feats = build_parent_features(tpos_first_frame, parents)

        results = {
            'motion': motion,
            'max_motion_length': self.max_motion_length,
            'motion_length': motion_length,
            'max_joints': self.max_joints,
            'parents': parents,
            'edge_indexs': aug['edge_indexs'],
            'tpos_first_frame': tpos_first_frame,
            'tpos_first_frame_parents': parent_feats['tpos_first_frame_parents'],
            'offsets': aug['offsets'],
            'joint_graph_dist': aug['joint_graph_dist'],
            'joint_relations': aug['joint_relations'],
            'joint_depths': aug['joint_depths'],
            'spectral_feats': aug['spectral_feats'],
            'joint_names_emb': aug['joint_names_emb'],
            'object_type': object_type,
            'start_idx': start_idx,
            'mean': mean,
            'std': std,
        }
        results['caption'] = data['caption']
        results['caption_emb'] = data['caption_emb']
        results['caption_tokens'] = data['caption_tokens']

        return results

    # ------------------------------------------------------------------
    # Augmentation pipeline
    # ------------------------------------------------------------------
    # Aug ops live in ``augmentations.py`` as plain functions; this wrapper
    # builds the aug dict and threads ``max_freqs`` through.

    @staticmethod
    def _extract_aug_dict(data, mean, std):
        """Copy of a motion_dict entry + stats so augs can mutate freely."""
        return {
            'motion':            data['motion'].copy(),
            'parents':           data['parents'].copy(),
            'edge_indexs':       data['edge_indexs'].copy(),
            'joint_graph_dist':  data['joint_graph_dist'].copy(),
            'joint_relations':   data['joint_relations'].copy(),
            'joint_depths':      data['joint_depths'].copy(),
            'spectral_feats':    data['spectral_feats'].copy(),
            'tpos':              data['tpos'].copy(),
            'offsets':           data['offsets'].copy(),
            'joint_names_emb':   data['joint_names_emb'].copy(),
            'mean':              mean.copy(),
            'std':               std.copy(),
        }

    def _apply_augmentations(self, data, mean, std):
        """Apply at most one enabled augmentation to a copy of ``data``.

        A no-op is always a candidate, so augmentation probability scales
        inversely with the number of enabled flags.
        """
        aug = self._extract_aug_dict(data, mean, std)

        candidates = [
            ('addition',     self.use_addition_aug,     self._aug_addition),
            ('removal',      self.use_removal_aug,      self._aug_removal),
            ('pooling',      self.use_pooling_aug,      self._aug_pooling),
            ('perturbation', self.use_perturbation_aug, self._aug_perturbation),
        ]
        ops = [None] + [fn for _, enabled, fn in candidates if enabled]
        if len(ops) == 1:
            return aug

        op = random.choice(ops)
        return aug if op is None else op(aug)

    def _aug_addition(self, aug):
        return aug_ops.apply_joint_addition_ellipsoid(aug, self.max_freqs)

    def _aug_removal(self, aug):
        rate = random.uniform(0.05, 0.15)
        return aug_ops.apply_joint_removal(aug, rate, self.max_freqs)

    def _aug_pooling(self, aug):
        rate = random.uniform(0.1, 0.3)
        return aug_ops.apply_skeleton_pooling(aug, rate, self.max_freqs)

    def _aug_perturbation(self, aug):
        return aug_ops.apply_joint_perturbation(aug, max_scale=0.1)

    # ------------------------------------------------------------------
    # Max joints computation
    # ------------------------------------------------------------------

    def _compute_max_joints_and_depth(self):
        """Scan loaded skeletons for max_joints / max_depth, apply
        augmentation headroom, and clamp depth to the config cap.
        """
        max_joints_by_dataset = {}
        max_depths_by_dataset = {}
        for dataset_type, object_types in self.dataset_object_count.items():
            for object_type in object_types:
                cond_data = self.cond_dict[object_type]
                n_joints = len(cond_data['parents'])
                depth = int(np.max(cond_data['joint_depths']))
                max_joints_by_dataset[dataset_type] = max(
                    max_joints_by_dataset.get(dataset_type, 0), n_joints)
                max_depths_by_dataset[dataset_type] = max(
                    max_depths_by_dataset.get(dataset_type, 0), depth)

        data_max_joints = max(max_joints_by_dataset.values())
        data_max_depths = max(max_depths_by_dataset.values())

        for dt in max_joints_by_dataset:
            logger.info(f'{dt.capitalize()} max_joints={max_joints_by_dataset[dt]} '
                        f'max_depths={max_depths_by_dataset[dt]}')
        logger.info(f'Data max_joints={data_max_joints} | max_depths={data_max_depths}')

        # Headroom: joint-addition aug can add 1 joint.
        if self.use_addition_aug:
            data_max_joints += 1

        if self.config_max_depth > 0 and data_max_depths > self.config_max_depth:
            logger.info(f'Clamping max_depths from {data_max_depths} to {self.config_max_depth}')
            data_max_depths = self.config_max_depth

        return data_max_joints, data_max_depths

    # ------------------------------------------------------------------
    # Normalization & statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _split_names(name_list, motion_dict, test_ratio, seed,
                     split_mode_by_dataset=None, explicit_eval_objects=None):
        """Per-dataset train/eval split. Deterministic for (seed, names).

        Per-dataset selection rule:
          1. If ``explicit_eval_objects[dataset_type]`` is non-empty → use
             that fixed object_type list as eval (``test_ratio`` ignored).
          2. Else if ``test_ratio > 0`` → use ``split_mode_by_dataset`` mode:
               - ``'clip'``        : hold out ``round(n * test_ratio)`` clips
                                     per object_type (≥1 train clip kept).
               - ``'object_type'`` : hold out ``round(n_types * test_ratio)``
                                     whole object_types (≥1 train type kept).
          3. Else → that dataset contributes 0 eval clips.

        Returns (train_names, eval_names) preserving original ordering within
        each split.
        """
        explicit_eval_objects = {
            dt: set(v) for dt, v in (explicit_eval_objects or {}).items() if v
        }
        if test_ratio <= 0.0 and not explicit_eval_objects:
            return list(name_list), []
        if test_ratio > 0.0 and not 0.0 < test_ratio < 1.0:
            raise ValueError(f"test_split_ratio must be in [0, 1), got {test_ratio}")

        split_mode_by_dataset = split_mode_by_dataset or {}

        by_dataset_obj = defaultdict(lambda: defaultdict(list))
        for nm in name_list:
            entry = motion_dict[nm]
            by_dataset_obj[entry['dataset_type']][entry['object_type']].append(nm)

        rng = random.Random(seed)
        eval_set = set()

        for dataset_type in sorted(by_dataset_obj.keys()):
            obj_to_clips = by_dataset_obj[dataset_type]
            explicit = explicit_eval_objects.get(dataset_type)

            if explicit:
                eval_set.update(MotionDataset._eval_clips_explicit(
                    dataset_type, explicit, obj_to_clips,
                ))
                continue
            if test_ratio <= 0.0:
                continue

            mode = split_mode_by_dataset.get(dataset_type, 'clip')
            if mode == 'object_type':
                eval_set.update(MotionDataset._eval_clips_by_object_type(
                    dataset_type, obj_to_clips, test_ratio, rng,
                ))
            elif mode == 'clip':
                eval_set.update(MotionDataset._eval_clips_by_clip(
                    obj_to_clips, test_ratio, rng,
                ))
            else:
                raise ValueError(
                    f"Unknown split mode {mode!r} for dataset_type={dataset_type!r}"
                )

        train_names = [nm for nm in name_list if nm not in eval_set]
        eval_names = [nm for nm in name_list if nm in eval_set]
        return train_names, eval_names

    @staticmethod
    def _eval_clips_explicit(dataset_type, explicit, obj_to_clips):
        """Eval = all clips of the listed object_types (test_ratio ignored)."""
        available = set(obj_to_clips.keys())
        eval_types = sorted(explicit & available)
        missing = sorted(explicit - available)
        if missing:
            logger.warning(
                f"[{dataset_type}] explicit eval objects not loaded "
                f"(filtered out or missing motion files): {missing}"
            )
        if not eval_types:
            logger.warning(
                f"[{dataset_type}] explicit eval list given but no matching "
                f"object types found in loaded data; skipping eval split"
            )
            return set()
        # Refuse to drain a dataset: at least one object_type must remain in
        # train, otherwise per-dataset stats and sampling break downstream.
        if len(eval_types) >= len(obj_to_clips):
            raise ValueError(
                f"[{dataset_type}] explicit test_objects.txt holds out ALL "
                f"{len(obj_to_clips)} loaded object types — would leave 0 "
                f"train clips for this dataset."
            )
        eval_clips = set()
        for ot in eval_types:
            eval_clips.update(obj_to_clips[ot])
        logger.info(
            f"[{dataset_type}] explicit object_type split: holding out "
            f"{len(eval_types)}/{len(obj_to_clips)} types as eval ({eval_types})"
        )
        return eval_clips

    @staticmethod
    def _eval_clips_by_object_type(dataset_type, obj_to_clips, test_ratio, rng):
        """Eval = all clips of ``round(n_types * test_ratio)`` random types.
        Guarantees at least 1 train type (eval may be 0 when n_types == 1).
        """
        object_types = sorted(obj_to_clips.keys())
        n_types = len(object_types)
        n_eval_types = int(round(n_types * test_ratio))
        n_eval_types = min(n_eval_types, max(0, n_types - 1))
        if n_eval_types == 0:
            return set()
        rng.shuffle(object_types)
        eval_clips = set()
        for ot in object_types[:n_eval_types]:
            eval_clips.update(obj_to_clips[ot])
        logger.info(
            f"[{dataset_type}] object_type split: holding out "
            f"{n_eval_types}/{n_types} types as eval"
        )
        return eval_clips

    @staticmethod
    def _eval_clips_by_clip(obj_to_clips, test_ratio, rng):
        """Eval = ``round(n * test_ratio)`` random clips per object_type."""
        eval_clips = set()
        for obj_type in sorted(obj_to_clips.keys()):
            clips = sorted(obj_to_clips[obj_type])
            n = len(clips)
            n_eval = int(round(n * test_ratio))
            n_eval = min(n_eval, max(0, n - 1))
            if n_eval == 0:
                continue
            rng.shuffle(clips)
            eval_clips.update(clips[:n_eval])
        return eval_clips

    def _get_normalization_stats(self, data, dataset_type):
        """Return (mean, std) arrays shaped (J, D). Raises ``KeyError`` if
        the dataset_type has no recorded stats (no training frames).
        """
        stats = self.dataset_stats[dataset_type]
        J, D = data['motion'].shape[1], data['motion'].shape[2]
        mean = np.zeros((J, D))
        std = np.zeros((J, D))
        mean[0, :], std[0, :] = stats['mean_root'], stats['std_root']
        mean[1:, :], std[1:, :] = stats['mean_local'], stats['std_local']
        return mean, std

    def _calculate_dataset_stats(self):
        """Compute normalization statistics keyed by ``dataset_type``.

        Axes:
          - ``use_dataset_stats``: per-dataset pools (True) vs one global pool
            shared across all dataset_type keys (False).
          - ``balanced_stats``: frame-weighted (False, default) vs object-type
            balanced via the law of total variance (True).

        Root (joint 0) and local (joints 1+) are always split — root encodes
        global velocity/orientation, locals encode parent-relative transforms.

        Short-circuits to disk load when ``stats_path`` is set.
        """
        if self._stats_path is not None:
            logger.info(f"Loading saved dataset stats from {self._stats_path}")
            self.dataset_stats = np.load(self._stats_path, allow_pickle=True).item()
            return

        mode = "object-type balanced" if self.balanced_stats else "frame-weighted"
        scope = "per-dataset" if self.use_dataset_stats else "global"
        logger.info(f"Calculating dataset statistics ({mode}, {scope})...")

        compute = self._stats_balanced if self.balanced_stats else self._stats_frame_weighted
        dataset_types = list(self.dataset_object_count.keys())

        if self.use_dataset_stats:
            jobs = [(d, d) for d in dataset_types]
        else:
            jobs = [('<all>', None)]

        for label, filter_d in jobs:
            mean_root, std_root, mean_local, std_local = compute(filter_d)
            if mean_root is None:
                logger.warning(
                    f"No frames for '{label}'; skipping dataset-level stats."
                )
                continue

            if self.tie_std:
                std_root[:3] = std_root[:3].mean()
                std_root[3:9] = std_root[3:9].mean()
                std_local[:3] = std_local[:3].mean()
                std_local[3:9] = std_local[3:9].mean()
                if self.feature_len >= 12:
                    std_root[9:12] = std_root[9:12].mean()
                    std_local[9:12] = std_local[9:12].mean()

            assert np.all(std_root > 0) and np.all(std_local > 0)

            stats = {
                'mean_root':  mean_root,  'std_root':  std_root,
                'mean_local': mean_local, 'std_local': std_local,
            }
            # Global mode broadcasts the single result across all dataset_type
            # keys so downstream lookups remain uniform.
            targets = [label] if self.use_dataset_stats else dataset_types
            for t in targets:
                self.dataset_stats[t] = stats

            with np.printoptions(precision=4, suppress=True, linewidth=200):
                logger.info(
                    f"[{label}] stats ({mode}):\n"
                    f"  mean_root  = {mean_root}\n"
                    f"  std_root   = {std_root}\n"
                    f"  mean_local = {mean_local}\n"
                    f"  std_local  = {std_local}\n"
                    f"  std_root  range=[{std_root.min():.4g}, {std_root.max():.4g}]"
                    f"  std_local range=[{std_local.min():.4g}, {std_local.max():.4g}]"
                )

        logger.info("Dataset statistics calculation finished.")

    def _stats_frame_weighted(self, d_type=None):
        """Pool all frames equally. ``d_type=None`` pools across all dataset types."""
        sum_root = np.zeros(self.feature_len)
        sq_sum_root = np.zeros(self.feature_len)
        sum_local = np.zeros(self.feature_len)
        sq_sum_local = np.zeros(self.feature_len)
        n_root, n_local = 0, 0

        for m_data in self.train_motion_dict.values():
            if d_type is not None and m_data['dataset_type'] != d_type:
                continue
            motion = m_data['motion']  # (T, J, D)
            T, J, D = motion.shape

            root = motion[:, 0, :]
            sum_root += root.sum(axis=0)
            sq_sum_root += (root ** 2).sum(axis=0)
            n_root += T

            local = motion[:, 1:J, :].reshape(-1, D)
            sum_local += local.sum(axis=0)
            sq_sum_local += (local ** 2).sum(axis=0)
            n_local += T * (J - 1)

        if n_root == 0:
            return None, None, None, None

        mean_root = sum_root / n_root
        std_root = np.maximum(
            np.sqrt(np.maximum(sq_sum_root / n_root - mean_root ** 2, 0.0)),
            1e-8,
        )
        mean_local = sum_local / n_local
        std_local = np.maximum(
            np.sqrt(np.maximum(sq_sum_local / n_local - mean_local ** 2, 0.0)),
            1e-8,
        )
        return mean_root, std_root, mean_local, std_local

    def _stats_balanced(self, d_type=None):
        """Equal-weight per-object-type stats via law of total variance:
        ``Var(X) = E_k[Var_k(X)] + Var_k[E_k(X)]``. ``d_type=None`` pools
        across all dataset types.
        """
        obj_to_motions = defaultdict(list)
        for m_data in self.train_motion_dict.values():
            if d_type is not None and m_data['dataset_type'] != d_type:
                continue
            obj_to_motions[m_data['object_type']].append(m_data['motion'])

        if not obj_to_motions:
            return None, None, None, None

        obj_means_root, obj_vars_root = [], []
        obj_means_local, obj_vars_local = [], []

        for motions in obj_to_motions.values():
            sum_root = np.zeros(self.feature_len)
            sq_sum_root = np.zeros(self.feature_len)
            sum_local = np.zeros(self.feature_len)
            sq_sum_local = np.zeros(self.feature_len)
            total_f, total_local = 0, 0

            for motion in motions:
                T, J, D = motion.shape
                total_f += T
                total_local += T * (J - 1)

                root = motion[:, 0, :]
                sum_root += root.sum(axis=0)
                sq_sum_root += (root ** 2).sum(axis=0)

                local = motion[:, 1:, :].reshape(-1, D)
                sum_local += local.sum(axis=0)
                sq_sum_local += (local ** 2).sum(axis=0)

            mean_r = sum_root / total_f
            mean_l = sum_local / total_local
            obj_means_root.append(mean_r)
            obj_vars_root.append(
                np.maximum(sq_sum_root / total_f - mean_r ** 2, 0.0)
            )
            obj_means_local.append(mean_l)
            obj_vars_local.append(
                np.maximum(sq_sum_local / total_local - mean_l ** 2, 0.0)
            )

        # Law of total variance: E[Var] + Var[E]
        mean_root = np.mean(obj_means_root, axis=0)
        within_var_root = np.mean(obj_vars_root, axis=0)
        between_var_root = np.var(obj_means_root, axis=0)
        std_root = np.maximum(
            np.sqrt(np.maximum(within_var_root + between_var_root, 0.0)),
            1e-8,
        )

        mean_local = np.mean(obj_means_local, axis=0)
        within_var_local = np.mean(obj_vars_local, axis=0)
        between_var_local = np.var(obj_means_local, axis=0)
        std_local = np.maximum(
            np.sqrt(np.maximum(within_var_local + between_var_local, 0.0)),
            1e-8,
        )

        return mean_root, std_root, mean_local, std_local

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_dataset_stats(self, path):
        """Persist ``self.dataset_stats`` as a pickled .npy. Inference
        reloads with ``np.load(path, allow_pickle=True).item()``.
        """
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.dataset_stats, allow_pickle=True)
        logger.info(f"Saved dataset stats to {path}")


# ---------------------------------------------------------------------------
# MixtureSampler — balanced sampling across skeleton types
# ---------------------------------------------------------------------------

class MixtureSampler(data.Sampler):
    """Power-law balanced sampling across skeleton types.

    Per object type k with n_k motions, total sampling probability ∝ n_k^(1-alpha).
    Per sample within type k, weight ∝ n_k^(-alpha).

      alpha=0.0 : uniform per sample (data-rich types dominate)
      alpha=0.5 : square-root balanced (default — good compromise)
      alpha=1.0 : uniform per type (ignores motion count entirely)

    This is a plain Sampler (not DistributedSampler) that yields all indices.
    Accelerate's BatchSamplerShard handles per-rank sharding automatically.
    """

    def __init__(self, data_source, alpha: float = 0.5):
        super().__init__(data_source)

        name_list = data_source.motion_dataset.train_name_list
        motion_dict = data_source.motion_dataset.train_motion_dict
        total_samples = len(name_list)

        indices_by_object = defaultdict(list)
        for i, name in enumerate(name_list):
            indices_by_object[motion_dict[name]['object_type']].append(i)

        # w_i = n_k^(-alpha) for sample i in type k, giving type k total
        # probability ∝ n_k * n_k^(-alpha) = n_k^(1-alpha).
        weights = np.zeros(total_samples)
        for indices in indices_by_object.values():
            n_k = len(indices)
            weights[indices] = n_k ** (-alpha)

        weights /= weights.sum()
        type_probs = {
            object_type: weights[indices].sum()
            for object_type, indices in indices_by_object.items()
        }
        sorted_types = sorted(type_probs.items(), key=lambda x: x[1], reverse=True)
        logger.info(f'MixtureSampler (alpha={alpha}): {len(sorted_types)} object types')
        for name, prob in sorted_types[:5]:
            logger.info(f'  top: {name} — n={len(indices_by_object[name])}, p={prob:.4f}')
        if len(sorted_types) > 10:
            logger.info('  ...')
        for name, prob in sorted_types[-5:]:
            logger.info(f'  bot: {name} — n={len(indices_by_object[name])}, p={prob:.4f}')

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.total_samples = total_samples
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_samples, replacement=True, generator=g
        ).tolist()
        return iter(indices)

    def __len__(self):
        return self.total_samples


# ---------------------------------------------------------------------------
# Mixture — top-level dataset used by the dataloader factory
# ---------------------------------------------------------------------------

class Mixture(data.Dataset):
    """Wrapper that loads condition files, filters subsets, then builds MotionDataset."""

    def __init__(
        self,
        data_configs: dict,
        topology_condition_type: str,
        max_motion_length: int,
        max_joints: int = 65,
        min_joints: int = 8,
        max_depth: int = 0,
        feature_len: int = 12,
        text_encoder_type: str = "t5",
        text_encoder_version: str = "t5-base",
        use_dataset_stats: bool = True,
        balanced_stats: bool = False,
        tie_std: bool = False,
        use_addition_aug: bool = False,
        use_removal_aug: bool = False,
        use_pooling_aug: bool = False,
        use_perturbation_aug: bool = False,
        test_split_ratio: float = 0.0,
        split_seed: int = 42,
        max_freqs: int = 8,
        debug_validate_aug: bool = False,
        ground_motion_height: bool = True,
        realign_feature: bool = True,
        prebuilt_max_joints: int = 0,
        prebuilt_max_depth: int = 0,
        target_object_types: Optional[Set[str]] = None,
        target_clip_stems: Optional[Set[str]] = None,
        stats_path: Optional[str] = None,
    ):
        logger.info(f'Initializing Mixture Dataset with datasets: {list(data_configs.keys())}')
        data_dict = {
            dataset_config.type: self._build_dataset_entry(
                dataset_config, min_joints, max_joints,
            )
            for _, dataset_config in data_configs.items()
        }

        self.motion_dataset = MotionDataset(
            data_dict=data_dict,
            topology_condition_type=topology_condition_type,
            max_motion_length=max_motion_length,
            config_max_depth=max_depth,
            feature_len=feature_len,
            text_encoder_type=text_encoder_type,
            text_encoder_version=text_encoder_version,
            use_dataset_stats=use_dataset_stats,
            balanced_stats=balanced_stats,
            tie_std=tie_std,
            use_addition_aug=use_addition_aug,
            use_removal_aug=use_removal_aug,
            use_pooling_aug=use_pooling_aug,
            use_perturbation_aug=use_perturbation_aug,
            test_split_ratio=test_split_ratio,
            split_seed=split_seed,
            max_freqs=max_freqs,
            debug_validate_aug=debug_validate_aug,
            ground_motion_height=ground_motion_height,
            realign_feature=realign_feature,
            prebuilt_max_joints=prebuilt_max_joints,
            prebuilt_max_depth=prebuilt_max_depth,
            target_object_types=target_object_types,
            target_clip_stems=target_clip_stems,
            stats_path=stats_path,
        )
        if target_object_types is None:
            # Training only: inference can produce a tiny (even empty) split.
            assert len(self.motion_dataset) > 1, 'You loaded an empty dataset!'

        self.max_joints = self.motion_dataset.max_joints
        self.max_depth = self.motion_dataset.max_depth

    # ------------------------------------------------------------------
    # Per-dataset entry construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dataset_entry(dataset_config, min_joints, max_joints):
        """Build the ``data_dict`` entry for one dataset: load cond / captions
        / test_objects, apply dataset-specific and joint-count filters.
        """
        root_dir = dataset_config.path
        dataset_type = dataset_config.type
        motion_dir = pjoin(root_dir, 'motions')
        assert os.path.exists(motion_dir), f"Motion directory {motion_dir} does not exist."

        cond_dict = Mixture._load_cond_dict(root_dir)
        captions = Mixture._load_captions(root_dir, dataset_type)
        test_objects = Mixture._load_test_objects(root_dir, dataset_type)

        cond_dict = Mixture._apply_dataset_filters(cond_dict, dataset_config, root_dir)
        cond_dict = Mixture._filter_by_joint_count(cond_dict, dataset_type, min_joints, max_joints)

        return {
            'root_dir': root_dir,          # where the text-embedding cache lives
            'motion_dir': motion_dir,
            'cond_dict': cond_dict,
            'captions': captions,
            'test_objects': test_objects,
        }

    @staticmethod
    def _load_cond_dict(root_dir):
        cond_file = pjoin(root_dir, 'cond.npy')
        assert os.path.exists(cond_file), f"Cond file {cond_file} does not exist."
        return np.load(cond_file, allow_pickle=True).item()

    @staticmethod
    def _load_captions(root_dir, dataset_type):
        """Load the flat ``{clip_stem: caption}`` dict from ``captions.json``.

        The feature-extraction stage writes it whenever any clip has a
        caption; a missing file means the dataset is caption-free.
        """
        captions_file = pjoin(root_dir, 'captions.json')
        if not os.path.exists(captions_file):
            logger.info(f'{dataset_type.capitalize()}: no captions.json in '
                        f'{root_dir} (caption-free dataset)')
            return {}
        with open(captions_file) as f:
            captions = json.load(f)
        logger.info(
            f'{dataset_type.capitalize()}: loaded captions from '
            f'{captions_file} ({len(captions)} entries)'
        )
        return captions

    @staticmethod
    def _load_test_objects(root_dir, dataset_type):
        """Explicit eval object_type list from ``test_objects.txt``.
        Skipped for mixamo (single object_type → empty train).
        """
        if dataset_type == 'mixamo':
            return []
        test_objects_file = pjoin(root_dir, 'test_objects.txt')
        if not os.path.exists(test_objects_file):
            return []
        with open(test_objects_file) as f:
            test_objects = sorted({line.strip() for line in f if line.strip()})
        logger.info(
            f'{dataset_type.capitalize()}: loaded {len(test_objects)} '
            f'test object types from {test_objects_file}'
        )
        return test_objects

    @staticmethod
    def _apply_dataset_filters(cond_dict, dataset_config, root_dir):
        """Dispatch to per-dataset filter (subset / object filter / count cap)."""
        if dataset_config.type == 'truebones':
            return Mixture._filter_truebones(cond_dict, dataset_config, root_dir)
        if dataset_config.type == 'objaverse':
            return Mixture._filter_objaverse(cond_dict, dataset_config, root_dir)
        return cond_dict

    @staticmethod
    def _filter_truebones(cond_dict, dataset_config, root_dir):
        """Keep only object types in ``dataset_config.objects_subset`` (or all)."""
        subset_key = dataset_config.objects_subset
        if subset_key == 'all':
            subset = sorted(cond_dict.keys())
        else:
            category_groups_path = pjoin(root_dir, 'category_groups.json')
            assert os.path.exists(category_groups_path), \
                f"category_groups.json not found at {category_groups_path}"
            with open(category_groups_path) as f:
                category_groups = json.load(f)
            if subset_key not in category_groups:
                raise ValueError(
                    f"Unknown objects_subset: {subset_key!r}, "
                    f"available: {list(category_groups.keys())}")
            subset = category_groups[subset_key]
        cond_dict = {k: cond_dict[k] for k in subset if k in cond_dict}
        logger.info(f'Truebones Dataset has {len(cond_dict)} object types')
        return cond_dict

    @staticmethod
    def _filter_objaverse(cond_dict, dataset_config, root_dir):
        """Apply objaverse-specific filters: filter_object + objects_num cap."""
        if dataset_config.filter_object:
            filter_path = pjoin(root_dir, 'filtered_objects.txt')
            if os.path.exists(filter_path):
                with open(filter_path) as f:
                    excluded = {line.strip() for line in f if line.strip()}
                before = len(cond_dict)
                cond_dict = {k: v for k, v in cond_dict.items() if k not in excluded}
                logger.info(
                    f'Objaverse: filter_object removed {before - len(cond_dict)} '
                    f'of {before} object types via {filter_path}'
                )
            else:
                logger.info(
                    f'Objaverse: filter_object=True but no filtered_objects.txt '
                    f'at {filter_path}; skipping filtering.'
                )
        objects_num = dataset_config.objects_num
        if objects_num > 0:
            keys = sorted(cond_dict.keys())[:objects_num]
            cond_dict = {k: cond_dict[k] for k in keys}
        logger.info(f'Objaverse Dataset has {len(cond_dict)} object types')
        return cond_dict

    @staticmethod
    def _filter_by_joint_count(cond_dict, dataset_type, min_joints, max_joints):
        """Drop object types outside ``[min_joints, max_joints]`` (inclusive)."""
        skipped = {
            k: len(v['parents']) for k, v in cond_dict.items()
            if len(v['parents']) > max_joints or len(v['parents']) < min_joints
        }
        cond_dict = {k: v for k, v in cond_dict.items() if k not in skipped}
        if skipped:
            logger.info(
                f'{dataset_type.capitalize()}: skipped {len(skipped)} object types '
                f'outside [{min_joints}, {max_joints}] joints'
            )
            for name, nj in sorted(skipped.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f'  {name}: {nj} joints')
        return cond_dict

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, item):
        return self.motion_dataset[item]

    def __len__(self):
        return len(self.motion_dataset)

    def save_dataset_stats(self, path):
        self.motion_dataset.save_dataset_stats(path)
