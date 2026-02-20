"""Training loop utilities for AlphaZero."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch import nn

from alphazero.config import GameConfig, load_yaml_config
from alphazero.network.bn_fold import export_folded_model
from alphazero.training.loss import DEFAULT_L2_WEIGHT, ValueHeadType, compute_loss_components
from alphazero.training.lr_schedule import (
    StepDecayLRSchedule,
    load_lr_schedule_from_config,
)


def _coerce_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _coerce_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _coerce_positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    converted = float(value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be > 0, got {converted}")
    return converted


def _coerce_non_negative_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    converted = float(value)
    if converted < 0.0:
        raise ValueError(f"{name} must be >= 0, got {converted}")
    return converted


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


DEFAULT_BATCH_SIZE = 1024
DEFAULT_MAX_STEPS = 700_000
DEFAULT_MOMENTUM = 0.9
DEFAULT_CHECKPOINT_INTERVAL = 1_000
DEFAULT_MILESTONE_INTERVAL = 50_000
DEFAULT_LOG_INTERVAL = 100
DEFAULT_MIN_BUFFER_SIZE = 10_000
DEFAULT_WAIT_FOR_BUFFER_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Runtime knobs for the Python training loop."""

    batch_size: int = DEFAULT_BATCH_SIZE
    max_steps: int = DEFAULT_MAX_STEPS
    momentum: float = DEFAULT_MOMENTUM
    l2_reg: float = DEFAULT_L2_WEIGHT
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    milestone_interval: int = DEFAULT_MILESTONE_INTERVAL
    log_interval: int = DEFAULT_LOG_INTERVAL
    min_buffer_size: int = DEFAULT_MIN_BUFFER_SIZE
    wait_for_buffer_seconds: float = DEFAULT_WAIT_FOR_BUFFER_SECONDS
    use_mixed_precision: bool = True
    device: str | torch.device | None = None
    checkpoint_dir: Path | None = None
    export_folded_checkpoints: bool = True

    def __post_init__(self) -> None:
        _coerce_positive_int("batch_size", self.batch_size)
        _coerce_positive_int("max_steps", self.max_steps)
        _coerce_positive_float("momentum", self.momentum)
        _coerce_non_negative_float("l2_reg", self.l2_reg)
        _coerce_positive_int("checkpoint_interval", self.checkpoint_interval)
        _coerce_positive_int("milestone_interval", self.milestone_interval)
        _coerce_positive_int("log_interval", self.log_interval)
        _coerce_non_negative_int("min_buffer_size", self.min_buffer_size)
        _coerce_positive_float("wait_for_buffer_seconds", self.wait_for_buffer_seconds)

        if self.checkpoint_dir is not None and not isinstance(self.checkpoint_dir, Path):
            object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))


@dataclass(frozen=True, slots=True)
class CheckpointPaths:
    """Checkpoint artifacts emitted at a specific training step."""

    step: int
    checkpoint_path: Path
    folded_weights_path: Path | None
    is_milestone: bool


@dataclass(frozen=True, slots=True)
class TrainingLoopResult:
    """Summary returned after a training loop run."""

    final_step: int
    checkpoints: tuple[CheckpointPaths, ...]


@dataclass(frozen=True, slots=True)
class TrainingStepMetrics:
    """Per-step metrics used for logging and testing."""

    step: int
    loss_total: float
    loss_policy: float
    loss_value: float
    loss_l2: float
    lr: float
    grad_global_norm: float
    grad_nonzero_param_count: int
    buffer_size: int
    train_steps_per_second: float

    def as_dict(self) -> dict[str, float]:
        return {
            "loss/total": self.loss_total,
            "loss/policy": self.loss_policy,
            "loss/value": self.loss_value,
            "loss/l2": self.loss_l2,
            "lr": self.lr,
            "grad/global_norm": self.grad_global_norm,
            "grad/nonzero_param_count": float(self.grad_nonzero_param_count),
            "buffer/size": float(self.buffer_size),
            "throughput/train_steps_per_sec": self.train_steps_per_second,
        }


class ReplayBufferLike(Protocol):
    """Protocol for the C++ replay buffer bindings used by training."""

    def size(self) -> int:
        ...

    def sample(self, batch_size: int) -> Sequence[object]:
        ...


StepLogger = Callable[[int, Mapping[str, float]], None]


def _extract_position_field(position: object, field_name: str) -> object:
    if isinstance(position, Mapping):
        if field_name not in position:
            raise KeyError(f"Replay sample is missing required field {field_name!r}")
        return position[field_name]
    if hasattr(position, field_name):
        return getattr(position, field_name)
    raise AttributeError(
        f"Replay sample is missing required attribute {field_name!r}"
    )


def _coerce_replay_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Replay sample {name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"Replay sample {name} must be positive, got {value}")
    return value


def _coerce_replay_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Replay sample {name} must be numeric, got {type(value).__name__}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"Replay sample {name} must be finite")
    return converted


def _as_float_tensor(values: object, *, device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float32)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _normalize_policy_targets(policy: torch.Tensor) -> torch.Tensor:
    policy = policy.clamp_min(0.0)
    row_sums = policy.sum(dim=1, keepdim=True)
    if torch.any(row_sums <= 0.0):
        raise ValueError("Replay sample contains an all-zero policy target")
    return policy / row_sums


def prepare_replay_batch(
    sampled_positions: Sequence[object],
    game_config: GameConfig,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert replay samples into dense training tensors."""

    if not sampled_positions:
        raise ValueError("sampled_positions must be non-empty")

    rows, cols = game_config.board_shape
    encoded_state_size = game_config.input_channels * rows * cols
    action_space_size = game_config.action_space_size
    batch_size = len(sampled_positions)

    states = torch.empty(
        (batch_size, game_config.input_channels, rows, cols),
        dtype=torch.float32,
        device=device,
    )
    target_policy = torch.empty(
        (batch_size, action_space_size),
        dtype=torch.float32,
        device=device,
    )

    if game_config.value_head_type == "scalar":
        target_value = torch.empty((batch_size,), dtype=torch.float32, device=device)
    else:
        target_value = torch.empty((batch_size, 3), dtype=torch.float32, device=device)

    for sample_index, position in enumerate(sampled_positions):
        raw_state = _extract_position_field(position, "encoded_state")
        raw_policy = _extract_position_field(position, "policy")

        has_encoded_state_size = (isinstance(position, Mapping) and "encoded_state_size" in position) or hasattr(
            position,
            "encoded_state_size",
        )
        has_policy_size = (isinstance(position, Mapping) and "policy_size" in position) or hasattr(
            position,
            "policy_size",
        )
        position_state_size = (
            _coerce_replay_int("encoded_state_size", _extract_position_field(position, "encoded_state_size"))
            if has_encoded_state_size
            else encoded_state_size
        )
        position_policy_size = (
            _coerce_replay_int("policy_size", _extract_position_field(position, "policy_size"))
            if has_policy_size
            else action_space_size
        )

        if position_state_size != encoded_state_size:
            raise ValueError(
                "Replay sample encoded_state_size does not match game config: "
                f"expected {encoded_state_size}, got {position_state_size}"
            )
        if position_policy_size != action_space_size:
            raise ValueError(
                "Replay sample policy_size does not match game config: "
                f"expected {action_space_size}, got {position_policy_size}"
            )

        state_tensor = _as_float_tensor(raw_state, device=device).flatten()
        policy_tensor = _as_float_tensor(raw_policy, device=device).flatten()

        if state_tensor.numel() < encoded_state_size:
            raise ValueError(
                "Replay sample encoded_state is smaller than encoded_state_size"
            )
        if policy_tensor.numel() < action_space_size:
            raise ValueError("Replay sample policy is smaller than policy_size")

        states[sample_index] = state_tensor[:encoded_state_size].reshape(
            game_config.input_channels,
            rows,
            cols,
        )
        target_policy[sample_index] = policy_tensor[:action_space_size]

        if game_config.value_head_type == "scalar":
            target_value[sample_index] = _coerce_replay_float(
                "value",
                _extract_position_field(position, "value"),
            )
        else:
            raw_wdl = _extract_position_field(position, "value_wdl")
            wdl_tensor = _as_float_tensor(raw_wdl, device=device).flatten()
            if wdl_tensor.numel() < 3:
                raise ValueError("Replay sample value_wdl must contain at least three values")
            target_value[sample_index] = wdl_tensor[:3]

    return states, _normalize_policy_targets(target_policy), target_value


def _apply_dihedral_transform(tensor: torch.Tensor, symmetry_index: int) -> torch.Tensor:
    transformed = torch.rot90(tensor, k=symmetry_index % 4, dims=(-2, -1))
    if symmetry_index >= 4:
        transformed = torch.flip(transformed, dims=(-1,))
    return transformed


def apply_random_go_symmetry(
    states: torch.Tensor,
    target_policy: torch.Tensor,
    *,
    symmetry_indices: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a random D4 transform to each Go sample and matching policy target."""

    if states.ndim != 4:
        raise ValueError(f"states must have shape (batch, C, H, W), got {tuple(states.shape)}")
    if target_policy.ndim != 2:
        raise ValueError(
            f"target_policy must have shape (batch, action_space_size), got {tuple(target_policy.shape)}"
        )
    batch_size, _, rows, cols = states.shape
    if rows != cols:
        raise ValueError("Go symmetry augmentation requires square board tensors")
    board_area = rows * cols
    if target_policy.shape[0] != batch_size:
        raise ValueError("states and target_policy batch dimensions must match")
    if target_policy.shape[1] != board_area + 1:
        raise ValueError(
            "Go policy target must include board actions plus pass action; "
            f"expected {board_area + 1}, got {target_policy.shape[1]}"
        )

    if symmetry_indices is None:
        symmetry_indices = torch.randint(
            low=0,
            high=8,
            size=(batch_size,),
            generator=generator,
            device=states.device,
        )
    else:
        if symmetry_indices.shape != (batch_size,):
            raise ValueError(
                f"symmetry_indices must have shape ({batch_size},), got {tuple(symmetry_indices.shape)}"
            )
        symmetry_indices = symmetry_indices.to(device=states.device, dtype=torch.int64)

    board_policy = target_policy[:, :board_area].reshape(batch_size, rows, cols)
    transformed_states = states.clone()
    transformed_policy_board = board_policy.clone()

    for symmetry_index in range(8):
        sample_mask = symmetry_indices == symmetry_index
        if not bool(sample_mask.any()):
            continue
        transformed_states[sample_mask] = _apply_dihedral_transform(
            states[sample_mask],
            symmetry_index,
        )
        transformed_policy_board[sample_mask] = _apply_dihedral_transform(
            board_policy[sample_mask],
            symmetry_index,
        )

    transformed_policy = target_policy.clone()
    transformed_policy[:, :board_area] = transformed_policy_board.reshape(batch_size, board_area)
    return transformed_states, transformed_policy


def _create_grad_scaler(*, device: torch.device, enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler(device=device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def _set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = lr


def _gradient_statistics(model: nn.Module) -> tuple[float, int]:
    squared_norm_sum = 0.0
    nonzero_grad_parameters = 0
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Encountered non-finite gradients during training")

        gradient_float = gradient.detach().to(dtype=torch.float32)
        squared_norm_sum += float((gradient_float * gradient_float).sum().item())
        if bool(torch.count_nonzero(gradient_float)):
            nonzero_grad_parameters += 1

    return math.sqrt(squared_norm_sum), nonzero_grad_parameters


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    states: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    game_config: GameConfig,
    lr_schedule: StepDecayLRSchedule,
    global_step: int,
    l2_reg: float,
    scaler: Any,
    use_mixed_precision: bool,
) -> TrainingStepMetrics:
    """Run one optimization step and return decomposed metrics."""

    step_start_time = time.perf_counter()
    lr = lr_schedule.lr_at_step(global_step)
    _set_optimizer_learning_rate(optimizer, lr)

    optimizer.zero_grad(set_to_none=True)
    device_type = states.device.type
    value_type: ValueHeadType
    if game_config.value_head_type == "scalar":
        value_type = "scalar"
    elif game_config.value_head_type == "wdl":
        value_type = "wdl"
    else:
        raise ValueError(
            f"Unsupported value_head_type {game_config.value_head_type!r}"
        )

    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=use_mixed_precision,
    ):
        policy_logits, predicted_value = model(states)
        loss_components = compute_loss_components(
            policy_logits,
            predicted_value,
            target_policy,
            target_value,
            value_type=value_type,
            l2_weight=l2_reg,
            model=model,
        )

    if not torch.isfinite(loss_components.total_loss):
        raise FloatingPointError("Encountered non-finite total loss")
    if not torch.isfinite(loss_components.policy_loss):
        raise FloatingPointError("Encountered non-finite policy loss")
    if not torch.isfinite(loss_components.value_loss):
        raise FloatingPointError("Encountered non-finite value loss")

    scaler.scale(loss_components.total_loss).backward()
    scaler.unscale_(optimizer)
    grad_global_norm, nonzero_grad_parameters = _gradient_statistics(model)
    scaler.step(optimizer)
    scaler.update()

    step_duration = max(time.perf_counter() - step_start_time, 1e-8)
    return TrainingStepMetrics(
        step=global_step + 1,
        loss_total=float(loss_components.total_loss.detach().item()),
        loss_policy=float(loss_components.policy_loss.detach().item()),
        loss_value=float(loss_components.value_loss.detach().item()),
        loss_l2=float(loss_components.l2_loss.detach().item()),
        lr=lr,
        grad_global_norm=grad_global_norm,
        grad_nonzero_param_count=nonzero_grad_parameters,
        buffer_size=0,
        train_steps_per_second=1.0 / step_duration,
    )


def save_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_schedule: StepDecayLRSchedule,
    *,
    step: int,
    checkpoint_dir: Path,
    is_milestone: bool = False,
    export_folded_weights: bool = True,
) -> CheckpointPaths:
    """Persist model/optimizer/schedule state and optional folded export."""

    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = "milestone" if is_milestone else "checkpoint"
    checkpoint_path = checkpoint_dir / f"{checkpoint_stem}_{step:08d}.pt"

    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_schedule": [list(entry) for entry in lr_schedule.entries],
        "is_milestone": is_milestone,
    }
    torch.save(payload, checkpoint_path)

    folded_weights_path: Path | None = None
    if export_folded_weights:
        folded_model = export_folded_model(model).eval()
        folded_weights_path = checkpoint_dir / f"{checkpoint_stem}_{step:08d}_folded.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": folded_model.state_dict(),
                "source_checkpoint": str(checkpoint_path),
            },
            folded_weights_path,
        )

    return CheckpointPaths(
        step=step,
        checkpoint_path=checkpoint_path,
        folded_weights_path=folded_weights_path,
        is_milestone=is_milestone,
    )


def load_training_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[int, StepDecayLRSchedule]:
    """Load a previously saved training checkpoint into model/optimizer state."""

    payload = torch.load(Path(checkpoint_path), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping")
    if "model_state_dict" not in payload or "step" not in payload:
        raise ValueError("Checkpoint payload is missing required keys: 'model_state_dict' and 'step'")

    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    step = int(payload["step"])
    if step < 0:
        raise ValueError(f"Checkpoint step must be non-negative, got {step}")

    schedule_entries = payload.get("lr_schedule")
    if schedule_entries is None:
        schedule = StepDecayLRSchedule()
    else:
        schedule = StepDecayLRSchedule(entries=tuple(tuple(entry) for entry in schedule_entries))
    return step, schedule


def create_optimizer(
    model: nn.Module,
    *,
    lr_schedule: StepDecayLRSchedule,
    momentum: float,
) -> torch.optim.SGD:
    """Create the SGD optimizer configured for AlphaZero training."""

    return torch.optim.SGD(
        model.parameters(),
        lr=lr_schedule.initial_lr,
        momentum=momentum,
        weight_decay=0.0,
    )


def load_training_config_from_config(config: Mapping[str, Any]) -> TrainingConfig:
    """Load a :class:`TrainingConfig` from a parsed pipeline configuration mapping."""

    if not isinstance(config, Mapping):
        raise TypeError(f"config must be a mapping, got {type(config).__name__}")

    training = config.get("training", {})
    if training is None:
        training = {}
    if not isinstance(training, Mapping):
        raise ValueError("'training' section must be a mapping")

    system = config.get("system", {})
    if system is None:
        system = {}
    if not isinstance(system, Mapping):
        raise ValueError("'system' section must be a mapping")

    checkpoint_dir_raw = system.get("checkpoint_dir")
    checkpoint_dir = Path(checkpoint_dir_raw) if isinstance(checkpoint_dir_raw, (str, Path)) else None

    return TrainingConfig(
        batch_size=int(training.get("batch_size", DEFAULT_BATCH_SIZE)),
        max_steps=int(training.get("max_steps", DEFAULT_MAX_STEPS)),
        momentum=float(training.get("momentum", DEFAULT_MOMENTUM)),
        l2_reg=float(training.get("l2_reg", DEFAULT_L2_WEIGHT)),
        checkpoint_interval=int(training.get("checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL)),
        milestone_interval=int(training.get("milestone_interval", DEFAULT_MILESTONE_INTERVAL)),
        log_interval=int(training.get("log_interval", DEFAULT_LOG_INTERVAL)),
        min_buffer_size=int(training.get("min_buffer_size", DEFAULT_MIN_BUFFER_SIZE)),
        checkpoint_dir=checkpoint_dir,
    )


def load_training_config_from_yaml(config_path: str | Path) -> TrainingConfig:
    """Load :class:`TrainingConfig` from a pipeline YAML file."""

    return load_training_config_from_config(load_yaml_config(config_path))


def training_loop(
    model: nn.Module,
    replay_buffer: ReplayBufferLike,
    game_config: GameConfig,
    training_config: TrainingConfig,
    *,
    lr_schedule: StepDecayLRSchedule | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    step_logger: StepLogger | None = None,
    start_step: int = 0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> TrainingLoopResult:
    """Run the AlphaZero training loop until ``training_config.max_steps`` is reached."""

    if start_step < 0:
        raise ValueError(f"start_step must be non-negative, got {start_step}")
    if start_step > training_config.max_steps:
        raise ValueError(
            f"start_step ({start_step}) cannot exceed max_steps ({training_config.max_steps})"
        )

    device = _resolve_device(training_config.device)
    model = model.to(device=device)
    model.train()

    active_schedule = lr_schedule
    if active_schedule is None:
        active_schedule = StepDecayLRSchedule()

    active_optimizer = optimizer
    if active_optimizer is None:
        active_optimizer = create_optimizer(
            model,
            lr_schedule=active_schedule,
            momentum=training_config.momentum,
        )

    scaler = _create_grad_scaler(
        device=device,
        enabled=training_config.use_mixed_precision,
    )

    emitted_checkpoints: list[CheckpointPaths] = []
    step = start_step

    while step < training_config.max_steps:
        current_buffer_size = int(replay_buffer.size())
        if current_buffer_size < training_config.min_buffer_size:
            sleep_fn(training_config.wait_for_buffer_seconds)
            continue

        sampled_positions = replay_buffer.sample(training_config.batch_size)
        states, target_policy, target_value = prepare_replay_batch(
            sampled_positions,
            game_config,
            device=device,
        )

        if game_config.supports_symmetry:
            states, target_policy = apply_random_go_symmetry(states, target_policy)

        step_metrics = train_one_step(
            model,
            active_optimizer,
            states=states,
            target_policy=target_policy,
            target_value=target_value,
            game_config=game_config,
            lr_schedule=active_schedule,
            global_step=step,
            l2_reg=training_config.l2_reg,
            scaler=scaler,
            use_mixed_precision=training_config.use_mixed_precision,
        )

        step += 1
        updated_metrics = TrainingStepMetrics(
            step=step,
            loss_total=step_metrics.loss_total,
            loss_policy=step_metrics.loss_policy,
            loss_value=step_metrics.loss_value,
            loss_l2=step_metrics.loss_l2,
            lr=step_metrics.lr,
            grad_global_norm=step_metrics.grad_global_norm,
            grad_nonzero_param_count=step_metrics.grad_nonzero_param_count,
            buffer_size=current_buffer_size,
            train_steps_per_second=step_metrics.train_steps_per_second,
        )

        if step_logger is not None and step % training_config.log_interval == 0:
            step_logger(step, updated_metrics.as_dict())

        checkpoint_dir = training_config.checkpoint_dir
        if checkpoint_dir is not None and step % training_config.checkpoint_interval == 0:
            emitted_checkpoints.append(
                save_training_checkpoint(
                    model,
                    active_optimizer,
                    active_schedule,
                    step=step,
                    checkpoint_dir=checkpoint_dir,
                    is_milestone=False,
                    export_folded_weights=training_config.export_folded_checkpoints,
                )
            )

            if step % training_config.milestone_interval == 0:
                emitted_checkpoints.append(
                    save_training_checkpoint(
                        model,
                        active_optimizer,
                        active_schedule,
                        step=step,
                        checkpoint_dir=checkpoint_dir,
                        is_milestone=True,
                        export_folded_weights=training_config.export_folded_checkpoints,
                    )
                )

    return TrainingLoopResult(
        final_step=step,
        checkpoints=tuple(emitted_checkpoints),
    )


def build_training_components_from_config(
    config: Mapping[str, Any],
) -> tuple[TrainingConfig, StepDecayLRSchedule]:
    """Build training runtime configuration and LR schedule from parsed config."""

    return load_training_config_from_config(config), load_lr_schedule_from_config(config)


def build_training_components_from_yaml(
    config_path: str | Path,
) -> tuple[TrainingConfig, StepDecayLRSchedule]:
    """Build training runtime configuration and LR schedule from pipeline YAML."""

    config = load_yaml_config(config_path)
    return build_training_components_from_config(config)


__all__ = [
    "CheckpointPaths",
    "ReplayBufferLike",
    "StepLogger",
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
