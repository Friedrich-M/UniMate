"""Configuration schema for motion generation.

This module defines the configuration structure for the entire training pipeline.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict
from pathlib import Path
import json


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    output_dir: str


# ---------------------------------------------------------------------------
# Per-dataset configs (top-level JSON keys)
# ---------------------------------------------------------------------------

@dataclass
class TruebonesConfig:
    """Truebones dataset specific configuration.

    ``path`` points at a feature-extraction output directory (see
    ``data_process/feature_extraction/extract_features.py``): ``cond.npy``,
    ``captions.json`` and per-clip NPZs under ``motions/``.
    """
    type: str = "truebones"
    path: str = "dataset/features/truebones"
    objects_subset: str = "all"  # 'all' or a key of {path}/category_groups.json


@dataclass
class MixamoConfig:
    """Mixamo dataset specific configuration."""
    type: str = "mixamo"
    path: str = "dataset/features/mixamo"


@dataclass
class ObjaverseConfig:
    """Objaverse dataset specific configuration."""
    type: str = "objaverse"
    path: str = "dataset/features/objaverse"
    objects_num: int = -1  # Number of object types to use (-1 = all)
    filter_object: bool = False  # If True, exclude object types listed in {path}/filtered_objects.txt


# ---------------------------------------------------------------------------
# Dataset config (shared settings + mixture routing)
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Dataset configuration.

    All datasets are loaded via the unified Mixture pipeline.
    Use ``dataset_list`` to specify which sub-datasets to include
    (e.g. ``["truebones"]`` for single-dataset, ``["truebones", "mixamo"]`` for combined).
    """
    dataset_list: List[str] = field(default_factory=lambda: ["truebones"])

    # Per-dataset configs (populated from top-level JSON keys)
    truebones: Optional[TruebonesConfig] = None
    mixamo: Optional[MixamoConfig] = None
    objaverse: Optional[ObjaverseConfig] = None

    # Topology conditioning
    topology_condition_type: str = "tpos"  # 'tpos' or 'first_frame'

    # Motion representation. Only 'unimate' is supported: 12-dim per joint —
    # RIFKE position (3) + reordered 6D rotation (6) + local velocity (3).
    motion_repr: str = "unimate"
    max_motion_length: Optional[int] = 40  # Maximum frames per clip
    max_joints: int = 100  # Filter cap: objects with > max_joints joints are skipped; the padded width is then auto-computed from what remains
    min_joints: int = 5  # Filter floor: objects with < min_joints joints are skipped
    max_depth: int = 0  # Depth cap: 0 = use auto-computed; >0 = min(auto-computed, max_depth)
    feature_len: int = 12  # Per-joint feature width; always 12 (enforced in __post_init__)

    # Normalization
    use_dataset_stats: bool = True  # True: stats keyed per dataset_type; False: one global pool across all motions
    balanced_stats: bool = False  # True: equal-weight per object type; False: frame-weighted pooling
    tie_std: bool = False  # True: average std within position(0:3) and rotation(3:9) groups

    # Sampling
    sampler_alpha: float = 0.5  # Power-law exponent: 0=uniform-per-sample, 0.5=sqrt-balanced, 1=uniform-per-type

    # Train/eval split (clip-level, stratified per object_type).
    # test_split_ratio=0 disables splitting (all clips → train).
    test_split_ratio: float = 0.0
    split_seed: int = 42

    # Augmentation
    use_addition_aug: bool = False  # Joint addition (ellipsoid) augmentation
    use_removal_aug: bool = False  # Leaf-joint removal augmentation
    use_pooling_aug: bool = False  # Skeleton pooling (merge pass-through joints) augmentation
    use_perturbation_aug: bool = False  # Random per-bone length scaling (positions recomputed via FK)

    # Per-clip Y grounding: shift so the lowest joint sits on Y=0 (also applied to tpos)
    ground_motion_height: bool = True

    # Re-align cropped clips so frame-0 facing is identity (only applies when start_idx > 0)
    realign_feature: bool = True

    # Runtime field — populated in __post_init__, not serialized
    data_configs: Optional[Dict[str, Union[TruebonesConfig, MixamoConfig, ObjaverseConfig]]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.dataset_list:
            raise ValueError("dataset_list must be non-empty.")

        if self.motion_repr != 'unimate':
            raise ValueError(
                f"Unknown motion_repr: {self.motion_repr!r}, must be 'unimate'"
            )
        self.feature_len = 12

        if self.topology_condition_type not in ('tpos', 'first_frame'):
            raise ValueError(
                f"Unknown topology_condition_type: {self.topology_condition_type!r}, "
                "must be 'tpos' or 'first_frame'"
            )

        _config_map = {
            'truebones': lambda: self.truebones or TruebonesConfig(),
            'mixamo': lambda: self.mixamo or MixamoConfig(),
            'objaverse': lambda: self.objaverse or ObjaverseConfig(),
        }
        for ds_type in self.dataset_list:
            factory = _config_map.get(ds_type)
            if factory is None:
                raise ValueError(f"Unknown dataset type in dataset_list: {ds_type}")
            self.data_configs[ds_type] = factory()


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model configuration."""
    # Two orthogonal axes, not one name.
    #   attention  'full'  — one attention over the flattened T*J tokens
    #              'graph' — factored spatial (per-frame) + temporal (per-joint),
    #                        the spatial stage carrying graph-distance and
    #                        edge-type biases
    #   text_cond  'adaln'      — the caption is folded into the adaLN vector
    #              'cross_attn' — the caption enters each block as
    #                             cross-attention K/V
    # The legacy single ``type`` field conflated the two, which hid that
    # 'graph' and 'cross' shared an attention stack and differed only in how the
    # caption enters; ``from_json`` still accepts it and maps it over. Not
    # every pair is implemented — ``models.factory`` owns that table.
    attention: str = "graph"
    text_cond: str = "adaln"

    # Architecture
    latent_dim: int = 512
    ff_size: int = 2048
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.0

    # Positional encoding
    use_spectral_rope: bool = False  # True: spectral RoPE on the joint axis (backend per use_signnet); False: sinusoidal RoPE
    max_freqs: int = 8  # Laplacian eigenvectors per joint (only used when use_spectral_rope=True)
    use_signnet: bool = True  # Spectral encoder backend: True=SignNet (sign-invariant), False=WIRE learnable frequencies

    # Conditioning
    cond_mode: str = "no_cond"  # 'no_cond', 'text' (caption CFG)
    cond_mask_prob: float = 0.0  # Probability of masking cond during training (enables CFG at inference)

    # Embeddings
    use_joint_name_emb: bool = False  # Joint name embedding
    use_graph_emb: bool = False  # Graph structure embedding (GCN)
    use_depth_emb: bool = False  # Kinematic depth embedding per joint
    concat_parent_features: bool = False  # Concatenate parent joint features in input
    num_tpos_queries: int = 0  # 0 = mean-pool tpos, >0 = cross-attention pool with K queries
    inject_tpos_to_adaln: bool = False  # Add pooled tpos embedding into the adaLN conditioning vector

    # Graph attention bias (attention='graph'). True: spatial attention adds learned
    # graph-distance + edge-type biases. False (ablation): vanilla self-attention
    # over joints; the joint validity mask is applied via SDPA's attn_mask.
    use_graph_attn_bias: bool = True
    # False: every block learns its own bias (num_layers copies of the small
    # distance/edge-type tables). True: the model owns ONE bias, computed once
    # per forward and reused by every layer — Graphormer's own arrangement,
    # fewer parameters and one bias build instead of num_layers.
    share_graph_attn_bias: bool = False

    # Memory optimization (attention='graph', either text_cond)
    gradient_checkpointing: bool = False  # Trade compute for memory in attention blocks

    # Text encoder
    text_encoder_type: str = "t5"  # 'clip', 'bert', 't5'
    text_encoder_version: Optional[str] = None  # None = use default for the type

    ATTENTION_CHOICES = ('full', 'graph')
    TEXT_COND_CHOICES = ('adaln', 'cross_attn')

    def __post_init__(self):
        # Fail on the config, not minutes later inside create_model once the
        # dataset has already been loaded.
        for field, value, choices in (
                ('attention', self.attention, self.ATTENTION_CHOICES),
                ('text_cond', self.text_cond, self.TEXT_COND_CHOICES)):
            if value not in choices:
                raise ValueError(
                    f"model.{field}={value!r} is not one of {list(choices)}")

        _defaults = {
            'bert': 'distilbert/distilbert-base-uncased',
            'clip': 'ViT-B/32',
            't5': 'google/flan-t5-base',
        }
        if self.text_encoder_version is None:
            if self.text_encoder_type not in _defaults:
                raise ValueError(f"Unknown text_encoder_type: {self.text_encoder_type}")
            self.text_encoder_version = _defaults[self.text_encoder_type]


# ---------------------------------------------------------------------------
# Scheduler config
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    """Diffusion scheduler configuration."""
    diffusion_steps: int = 100
    timestep_respacing: str = ""
    rescale_timesteps: bool = False

    noise_schedule: str = "cosine"  # "linear", "cosine"
    scale_beta: float = 1.0
    predict_xstart: bool = True
    learn_sigma: bool = False
    sigma_small: bool = True


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    num_workers: int = 8
    seed: int = 10

    # Generation paradigm
    diff_model: str = "diffusion"  # 'diffusion', 'flow'

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    max_grad_norm: Optional[float] = None  # Gradient clipping norm
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients

    # LR schedule
    warmup_ratio: float = 0.02  # Fraction of num_steps used for warmup
    min_lr_ratio: float = 0.01  # Min LR as a fraction of learning_rate

    # Duration
    num_epochs: Optional[int] = None
    num_steps: Optional[int] = 100_000
    log_interval: int = 50
    save_interval: int = 10_000

    # Loss weights
    lambda_geo: float = 0.5
    lambda_smooth: float = 0.1

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Sampling
    balanced: bool = False  # Use balancing sampler for fairness between topologies


# ---------------------------------------------------------------------------
# Sampling config
# ---------------------------------------------------------------------------

@dataclass
class SamplingArgs:
    """Sampling arguments."""
    model_path: Optional[str] = None
    output_dir: Optional[str] = None
    seed: Optional[int] = 10

    # Sampling hyperparameters
    object_type: Optional[List[str]] = None
    num_samples: int = 4
    num_repetitions: int = 3
    device: str = "cuda"

    # CFG
    cfg_scale: float = 1.0


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

@dataclass
class MainConfig:
    """Top-level config loaded from JSON.

    JSON top-level keys: experiment, truebones, mixamo, objaverse,
    dataset, model, scheduler, training, sampling.
    """
    experiment: ExperimentConfig
    truebones: TruebonesConfig
    mixamo: MixamoConfig
    objaverse: ObjaverseConfig
    dataset: DatasetConfig
    model: ModelConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    sampling: SamplingArgs

    @classmethod
    def from_json(cls, config_path: Union[str, Path]) -> 'MainConfig':
        """Load config from JSON file."""
        with open(config_path) as f:
            data = json.load(f)

        experiment = ExperimentConfig(**data['experiment'])

        # Per-dataset configs live at top level in JSON
        truebones = TruebonesConfig(**data.get('truebones', {}))
        mixamo = MixamoConfig(**data.get('mixamo', {}))
        objaverse = ObjaverseConfig(**data.get('objaverse', {}))

        # Inject per-dataset configs into dataset block for __post_init__
        dataset_data = data['dataset']
        dataset_data['truebones'] = truebones
        dataset_data['mixamo'] = mixamo
        dataset_data['objaverse'] = objaverse
        dataset = DatasetConfig(**dataset_data)

        model_data = dict(data['model'])
        # Legacy single-field form: {'type': 'native'|'graph'|'cross'}. Mapped
        # rather than rejected so the config.json inside every existing
        # experiment directory still loads.
        legacy_type = model_data.pop('type', None)
        if legacy_type is not None:
            legacy_map = {'native': ('full', 'adaln'),
                          'graph': ('graph', 'adaln'),
                          'cross': ('graph', 'cross_attn')}
            if legacy_type not in legacy_map:
                raise ValueError(
                    f"Unknown legacy model type {legacy_type!r}; expected one "
                    f"of {sorted(legacy_map)}. New configs should set "
                    f"'attention' and 'text_cond' instead.")
            attention, text_cond = legacy_map[legacy_type]
            # A file carrying both forms is ambiguous: silently preferring one
            # would hand back a model the config does not describe.
            conflicts = {k: (model_data[k], v)
                         for k, v in (('attention', attention),
                                      ('text_cond', text_cond))
                         if k in model_data and model_data[k] != v}
            if conflicts:
                detail = ', '.join(f'{k}={got!r} but type implies {want!r}'
                                   for k, (got, want) in conflicts.items())
                raise ValueError(
                    f"model config sets both the legacy 'type'={legacy_type!r} "
                    f"and a conflicting axis: {detail}. Drop 'type'.")
            model_data['attention'] = attention
            model_data['text_cond'] = text_cond
        model = ModelConfig(**model_data)
        scheduler = SchedulerConfig(**data['scheduler'])
        training = TrainingConfig(**data['training'])
        sampling = SamplingArgs(**data['sampling'])

        return cls(
            experiment=experiment,
            truebones=truebones,
            mixamo=mixamo,
            objaverse=objaverse,
            dataset=dataset,
            model=model,
            scheduler=scheduler,
            training=training,
            sampling=sampling,
        )

    def to_json(self, config_path: Union[str, Path]):
        """Save config to JSON file."""
        data = dataclasses.asdict(self)
        # Remove runtime-only fields that are reconstructed in __post_init__
        data['dataset'].pop('data_configs', None)
        # Remove nested sub-configs from dataset (they live at top level)
        for key in ('truebones', 'mixamo', 'objaverse'):
            data['dataset'].pop(key, None)
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=4)
