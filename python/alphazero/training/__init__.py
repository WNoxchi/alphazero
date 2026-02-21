"""Training utilities and loss functions."""

from alphazero.training.lr_schedule import (
    DEFAULT_STEP_DECAY_SCHEDULE,
    StepDecayLRSchedule,
    load_lr_schedule_from_config,
    load_lr_schedule_from_yaml,
    normalize_step_decay_schedule,
)

__all__ = [
    "DEFAULT_STEP_DECAY_SCHEDULE",
    "StepDecayLRSchedule",
    "load_lr_schedule_from_config",
    "load_lr_schedule_from_yaml",
    "normalize_step_decay_schedule",
]

try:
    from alphazero.training.loss import (
        DEFAULT_L2_WEIGHT,
        LossComponents,
        ValueHeadType,
        compute_loss,
        compute_loss_components,
        l2_regularization_loss,
        policy_cross_entropy_loss,
        scalar_value_loss,
        wdl_value_loss,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional torch install
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "DEFAULT_L2_WEIGHT",
            "LossComponents",
            "ValueHeadType",
            "compute_loss",
            "compute_loss_components",
            "policy_cross_entropy_loss",
            "scalar_value_loss",
            "wdl_value_loss",
            "l2_regularization_loss",
        ]
    )

try:
    from alphazero.training.trainer import (
        CheckpointPaths,
        TrainingConfig,
        TrainingLoopResult,
        TrainingStepMetrics,
        apply_random_go_symmetry,
        build_training_components_from_config,
        build_training_components_from_yaml,
        create_optimizer,
        load_training_checkpoint,
        load_training_config_from_config,
        load_training_config_from_yaml,
        prepare_replay_batch,
        save_training_checkpoint,
        train_one_step,
        training_loop,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional torch install
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "CheckpointPaths",
            "TrainingConfig",
            "TrainingLoopResult",
            "TrainingStepMetrics",
            "apply_random_go_symmetry",
            "build_training_components_from_config",
            "build_training_components_from_yaml",
            "create_optimizer",
            "load_training_checkpoint",
            "load_training_config_from_config",
            "load_training_config_from_yaml",
            "prepare_replay_batch",
            "save_training_checkpoint",
            "train_one_step",
            "training_loop",
        ]
    )
