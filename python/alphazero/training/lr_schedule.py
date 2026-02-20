"""Learning-rate scheduling utilities for AlphaZero training."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from alphazero.config import load_yaml_config


StepDecayPoint = tuple[int, float]
StepDecayTable = tuple[StepDecayPoint, ...]

DEFAULT_STEP_DECAY_SCHEDULE: Final[StepDecayTable] = (
    (0, 0.2),
    (200_000, 0.02),
    (400_000, 0.002),
    (600_000, 0.0002),
)


def _coerce_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"schedule step must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"schedule step must be non-negative, got {value}")
    return value


def _coerce_lr(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"learning rate must be numeric, got {type(value).__name__}")
    lr = float(value)
    if lr <= 0.0:
        raise ValueError(f"learning rate must be > 0, got {lr}")
    return lr


def normalize_step_decay_schedule(entries: object) -> StepDecayTable:
    """Validate and normalize a piecewise-constant step-decay schedule."""

    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError("lr_schedule must be a sequence of [step, lr] pairs")
    if len(entries) == 0:
        raise ValueError("lr_schedule must contain at least one [step, lr] pair")

    normalized: list[StepDecayPoint] = []
    previous_step = -1

    for index, entry in enumerate(entries):
        if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)):
            raise TypeError(
                f"lr_schedule entry at index {index} must be a 2-item sequence [step, lr]"
            )
        if len(entry) != 2:
            raise ValueError(
                f"lr_schedule entry at index {index} must contain exactly 2 values"
            )

        step = _coerce_step(entry[0])
        lr = _coerce_lr(entry[1])

        if index == 0 and step != 0:
            raise ValueError("lr_schedule must start at step 0")
        if step <= previous_step:
            raise ValueError("lr_schedule steps must be strictly increasing")

        normalized.append((step, lr))
        previous_step = step

    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class StepDecayLRSchedule:
    """Piecewise-constant LR schedule: use the latest rate whose step <= current step."""

    entries: StepDecayTable = DEFAULT_STEP_DECAY_SCHEDULE
    _steps: tuple[int, ...] = field(init=False, repr=False)
    _rates: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = normalize_step_decay_schedule(self.entries)
        object.__setattr__(self, "entries", normalized)
        object.__setattr__(self, "_steps", tuple(step for step, _ in normalized))
        object.__setattr__(self, "_rates", tuple(rate for _, rate in normalized))

    @property
    def initial_lr(self) -> float:
        """Return the rate active at training step 0."""

        return self._rates[0]

    @property
    def milestones(self) -> tuple[int, ...]:
        """Return decay boundaries excluding step 0 (compatible with scheduler APIs)."""

        return self._steps[1:]

    def lr_at_step(self, step: int) -> float:
        """Return the active learning rate for a global training step."""

        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError(f"step must be an integer, got {type(step).__name__}")
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")

        index = bisect_right(self._steps, step) - 1
        return self._rates[index]


def load_lr_schedule_from_config(config: Mapping[str, Any]) -> StepDecayLRSchedule:
    """Load step-decay configuration from a parsed pipeline config mapping.

    Expected shape:
      {"training": {"lr_schedule": [[0, 0.2], [200000, 0.02], ...]}}
    """

    if not isinstance(config, Mapping):
        raise TypeError(f"config must be a mapping, got {type(config).__name__}")

    training_config = config.get("training")
    if training_config is None:
        return StepDecayLRSchedule()
    if not isinstance(training_config, Mapping):
        raise ValueError("'training' section must be a mapping")

    raw_schedule = training_config.get("lr_schedule", DEFAULT_STEP_DECAY_SCHEDULE)
    return StepDecayLRSchedule(entries=normalize_step_decay_schedule(raw_schedule))


def load_lr_schedule_from_yaml(config_path: str | Path) -> StepDecayLRSchedule:
    """Load step-decay schedule directly from a pipeline YAML file."""

    config = load_yaml_config(config_path)
    return load_lr_schedule_from_config(config)


__all__ = [
    "DEFAULT_STEP_DECAY_SCHEDULE",
    "StepDecayLRSchedule",
    "StepDecayPoint",
    "StepDecayTable",
    "load_lr_schedule_from_config",
    "load_lr_schedule_from_yaml",
    "normalize_step_decay_schedule",
]
