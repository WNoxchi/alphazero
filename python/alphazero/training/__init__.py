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
