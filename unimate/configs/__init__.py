"""Re-export the config dataclasses so ``from unimate.configs import X`` works."""

from unimate.configs.schema import (
    MainConfig,
    TruebonesConfig,
    MixamoConfig,
    ObjaverseConfig,
    ExperimentConfig,
    DatasetConfig,
    ModelConfig,
    SchedulerConfig,
    TrainingConfig,
    SamplingArgs,
)

__all__ = [
    "MainConfig",
    "TruebonesConfig",
    "MixamoConfig",
    "ObjaverseConfig",
    "ExperimentConfig",
    "DatasetConfig",
    "ModelConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "SamplingArgs",
]
