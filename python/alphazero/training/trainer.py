"""Training loop utilities for AlphaZero."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
import torch.nn.functional as functional
from torch import nn

from alphazero.config import GameConfig, load_yaml_config
from alphazero.training.loss import (
    DEFAULT_L2_WEIGHT,
    ValueHeadType,
    l2_regularization_loss,
    ownership_loss,
)
from alphazero.training.lr_schedule import (
    StepDecayLRSchedule,
    load_lr_schedule_from_config,
)
from alphazero.utils.checkpoint import (
    DEFAULT_ROLLING_CHECKPOINT_KEEP_LAST,
    CheckpointPaths,
    extract_replay_buffer_metadata,
    load_checkpoint,
    save_checkpoint,
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
DEFAULT_CHECKPOINT_KEEP_LAST = DEFAULT_ROLLING_CHECKPOINT_KEEP_LAST
DEFAULT_OWNERSHIP_LOSS_WEIGHT = 0.0
_LOSS_EPSILON = 1.0e-8


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Runtime knobs for the Python training loop."""

    batch_size: int = DEFAULT_BATCH_SIZE
    max_steps: int = DEFAULT_MAX_STEPS
    momentum: float = DEFAULT_MOMENTUM
    l2_reg: float = DEFAULT_L2_WEIGHT
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    checkpoint_keep_last: int = DEFAULT_CHECKPOINT_KEEP_LAST
    milestone_interval: int = DEFAULT_MILESTONE_INTERVAL
    log_interval: int = DEFAULT_LOG_INTERVAL
    min_buffer_size: int = DEFAULT_MIN_BUFFER_SIZE
    wait_for_buffer_seconds: float = DEFAULT_WAIT_FOR_BUFFER_SECONDS
    ownership_loss_weight: float = DEFAULT_OWNERSHIP_LOSS_WEIGHT
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
        _coerce_positive_int("checkpoint_keep_last", self.checkpoint_keep_last)
        _coerce_positive_int("milestone_interval", self.milestone_interval)
        _coerce_positive_int("log_interval", self.log_interval)
        _coerce_non_negative_int("min_buffer_size", self.min_buffer_size)
        _coerce_positive_float("wait_for_buffer_seconds", self.wait_for_buffer_seconds)
        _coerce_non_negative_float("ownership_loss_weight", self.ownership_loss_weight)

        if self.checkpoint_dir is not None and not isinstance(self.checkpoint_dir, Path):
            object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))

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
    loss_ownership: float
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
            "loss/ownership": self.loss_ownership,
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


def _coerce_replay_weight(value: object) -> float:
    converted = _coerce_replay_float("training_weight", value)
    if converted < 0.0:
        raise ValueError(f"Replay sample training_weight must be non-negative, got {converted}")
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


def _value_dim_for_game_config(game_config: GameConfig) -> int:
    if game_config.value_head_type == "scalar":
        return 1
    if game_config.value_head_type == "wdl":
        return 3
    raise ValueError(f"Unsupported value_head_type {game_config.value_head_type!r}")


def _prepare_replay_batch_from_positions(
    sampled_positions: Sequence[object],
    game_config: GameConfig,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
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

    value_dim = _value_dim_for_game_config(game_config)
    if value_dim == 1:
        target_value = torch.empty((batch_size,), dtype=torch.float32, device=device)
    else:
        target_value = torch.empty((batch_size, value_dim), dtype=torch.float32, device=device)
    sample_weights = torch.ones((batch_size,), dtype=torch.float32, device=device)
    board_area = rows * cols
    ownership_target = torch.zeros((batch_size, board_area), dtype=torch.float32, device=device)
    ownership_presence: list[bool] = []

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

        if value_dim == 1:
            target_value[sample_index] = _coerce_replay_float(
                "value",
                _extract_position_field(position, "value"),
            )
        else:
            raw_wdl = _extract_position_field(position, "value_wdl")
            wdl_tensor = _as_float_tensor(raw_wdl, device=device).flatten()
            if wdl_tensor.numel() < value_dim:
                raise ValueError("Replay sample value_wdl must contain at least three values")
            target_value[sample_index] = wdl_tensor[:value_dim]

        has_training_weight = (isinstance(position, Mapping) and "training_weight" in position) or hasattr(
            position,
            "training_weight",
        )
        sample_weights[sample_index] = (
            _coerce_replay_weight(_extract_position_field(position, "training_weight"))
            if has_training_weight
            else 1.0
        )

        has_ownership = (isinstance(position, Mapping) and "ownership" in position) or hasattr(position, "ownership")
        if not has_ownership:
            ownership_presence.append(False)
            continue

        has_ownership_size = (isinstance(position, Mapping) and "ownership_size" in position) or hasattr(
            position,
            "ownership_size",
        )
        ownership_size = (
            int(_extract_position_field(position, "ownership_size"))
            if has_ownership_size
            else board_area
        )
        if ownership_size == 0:
            ownership_presence.append(False)
            continue
        if ownership_size != board_area:
            raise ValueError(
                "Replay sample ownership_size does not match board area: "
                f"expected {board_area}, got {ownership_size}"
            )

        ownership_values = _as_float_tensor(
            _extract_position_field(position, "ownership"),
            device=device,
        ).flatten()
        if ownership_values.numel() < board_area:
            raise ValueError(
                "Replay sample ownership is smaller than board area"
            )
        ownership_target[sample_index] = ownership_values[:board_area]
        ownership_presence.append(True)

    ownership_tensor: torch.Tensor | None
    any_ownership = any(ownership_presence)
    if any_ownership and not all(ownership_presence):
        raise ValueError("Replay sample batch mixes rows with and without ownership targets")
    if any_ownership:
        if not bool(torch.isfinite(ownership_target).all()):
            raise ValueError("Replay sample ownership targets must be finite")
        ownership_tensor = ownership_target
    else:
        ownership_tensor = None

    return (
        states,
        _normalize_policy_targets(target_policy),
        target_value,
        sample_weights,
        ownership_tensor,
    )


def _extract_packed_batch_fields(
    packed_batch: object,
) -> tuple[object, object, object, object | None, object | None]:
    if isinstance(packed_batch, Mapping):
        required = ("states", "policies", "values")
        missing = [field for field in required if field not in packed_batch]
        if missing:
            raise KeyError(f"Replay packed batch is missing required fields: {', '.join(missing)}")
        return (
            packed_batch["states"],
            packed_batch["policies"],
            packed_batch["values"],
            packed_batch.get("weights"),
            packed_batch.get("ownership"),
        )

    if isinstance(packed_batch, Sequence) and not isinstance(packed_batch, (str, bytes)):
        if len(packed_batch) == 3:
            return packed_batch[0], packed_batch[1], packed_batch[2], None, None
        if len(packed_batch) == 4:
            return packed_batch[0], packed_batch[1], packed_batch[2], packed_batch[3], None
        if len(packed_batch) == 5:
            return packed_batch[0], packed_batch[1], packed_batch[2], packed_batch[3], packed_batch[4]
        if len(packed_batch) != 3:
            raise ValueError(
                "Replay packed batch sequence must contain exactly three, four, or five entries: "
                "(states, policies, values[, weights[, ownership]])"
            )
        return packed_batch[0], packed_batch[1], packed_batch[2], None, None

    raise TypeError(
        "Replay packed batch must be either a mapping with keys "
        "{'states', 'policies', 'values'} or a 3/4/5-tuple/list"
    )


def _prepare_replay_batch_from_packed_arrays(
    packed_batch: object,
    game_config: GameConfig,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    rows, cols = game_config.board_shape
    encoded_state_size = game_config.input_channels * rows * cols
    action_space_size = game_config.action_space_size
    value_dim = _value_dim_for_game_config(game_config)

    raw_states, raw_policies, raw_values, raw_weights, raw_ownership = _extract_packed_batch_fields(packed_batch)
    flat_states = _as_float_tensor(raw_states, device=device)
    flat_policy = _as_float_tensor(raw_policies, device=device)
    flat_values = _as_float_tensor(raw_values, device=device)

    expected_state_shape = (batch_size, encoded_state_size)
    if tuple(flat_states.shape) != expected_state_shape:
        raise ValueError(
            "Replay packed batch states has incorrect shape: "
            f"expected {expected_state_shape}, got {tuple(flat_states.shape)}"
        )

    expected_policy_shape = (batch_size, action_space_size)
    if tuple(flat_policy.shape) != expected_policy_shape:
        raise ValueError(
            "Replay packed batch policies has incorrect shape: "
            f"expected {expected_policy_shape}, got {tuple(flat_policy.shape)}"
        )

    if value_dim == 1:
        if flat_values.ndim == 1:
            if tuple(flat_values.shape) != (batch_size,):
                raise ValueError(
                    "Replay packed batch scalar values has incorrect shape: "
                    f"expected {(batch_size,)}, got {tuple(flat_values.shape)}"
                )
            target_value = flat_values
        elif flat_values.ndim == 2:
            if tuple(flat_values.shape) != (batch_size, 1):
                raise ValueError(
                    "Replay packed batch scalar values has incorrect shape: "
                    f"expected {(batch_size, 1)}, got {tuple(flat_values.shape)}"
                )
            target_value = flat_values.reshape(batch_size)
        else:
            raise ValueError(
                "Replay packed batch scalar values must be rank 1 or 2, "
                f"got rank {flat_values.ndim}"
            )
        if not bool(torch.isfinite(target_value).all()):
            raise ValueError("Replay packed batch scalar values must be finite")
    else:
        expected_value_shape = (batch_size, value_dim)
        if tuple(flat_values.shape) != expected_value_shape:
            raise ValueError(
                "Replay packed batch WDL values has incorrect shape: "
                f"expected {expected_value_shape}, got {tuple(flat_values.shape)}"
            )
        target_value = flat_values

    if raw_weights is None:
        sample_weights = torch.ones((batch_size,), dtype=torch.float32, device=device)
    else:
        sample_weights_tensor = _as_float_tensor(raw_weights, device=device)
        if sample_weights_tensor.ndim == 1:
            if tuple(sample_weights_tensor.shape) != (batch_size,):
                raise ValueError(
                    "Replay packed batch weights has incorrect shape: "
                    f"expected {(batch_size,)}, got {tuple(sample_weights_tensor.shape)}"
                )
            sample_weights = sample_weights_tensor
        elif sample_weights_tensor.ndim == 2:
            if tuple(sample_weights_tensor.shape) != (batch_size, 1):
                raise ValueError(
                    "Replay packed batch weights has incorrect shape: "
                    f"expected {(batch_size, 1)}, got {tuple(sample_weights_tensor.shape)}"
                )
            sample_weights = sample_weights_tensor.reshape(batch_size)
        else:
            raise ValueError(
                "Replay packed batch weights must be rank 1 or 2, "
                f"got rank {sample_weights_tensor.ndim}"
            )
        if not bool(torch.isfinite(sample_weights).all()):
            raise ValueError("Replay packed batch weights must be finite")
        if bool((sample_weights < 0.0).any()):
            raise ValueError("Replay packed batch weights must be non-negative")

    states = flat_states.reshape(batch_size, game_config.input_channels, rows, cols)
    target_policy = _normalize_policy_targets(flat_policy)
    ownership_target: torch.Tensor | None = None
    if raw_ownership is not None:
        ownership_tensor = _as_float_tensor(raw_ownership, device=device)
        board_area = rows * cols
        if ownership_tensor.ndim == 1:
            if ownership_tensor.numel() == 0:
                ownership_target = None
            elif tuple(ownership_tensor.shape) == (batch_size * board_area,):
                ownership_target = ownership_tensor.reshape(batch_size, board_area)
            else:
                raise ValueError(
                    "Replay packed batch ownership has incorrect shape: "
                    f"expected {(batch_size * board_area,)} or {(batch_size, board_area)}, "
                    f"got {tuple(ownership_tensor.shape)}"
                )
        elif ownership_tensor.ndim == 2:
            expected_ownership_shape = (batch_size, board_area)
            if tuple(ownership_tensor.shape) != expected_ownership_shape:
                raise ValueError(
                    "Replay packed batch ownership has incorrect shape: "
                    f"expected {expected_ownership_shape}, got {tuple(ownership_tensor.shape)}"
                )
            ownership_target = ownership_tensor
        else:
            raise ValueError(
                "Replay packed batch ownership must be rank 1 or 2, "
                f"got rank {ownership_tensor.ndim}"
            )
        if ownership_target is not None and not bool(torch.isfinite(ownership_target).all()):
            raise ValueError("Replay packed batch ownership targets must be finite")
    return states, target_policy, target_value, sample_weights, ownership_target


def prepare_replay_batch(
    sampled_positions: Sequence[object],
    game_config: GameConfig,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert replay samples into dense training tensors."""

    states, target_policy, target_value, _sample_weights, _ownership_target = _prepare_replay_batch_from_positions(
        sampled_positions,
        game_config,
        device=device,
    )
    return states, target_policy, target_value


def sample_replay_batch_tensors(
    replay_buffer: ReplayBufferLike,
    game_config: GameConfig,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Sample one mini-batch from replay and convert it into dense training tensors."""

    rows, cols = game_config.board_shape
    encoded_state_size = game_config.input_channels * rows * cols
    action_space_size = game_config.action_space_size
    value_dim = _value_dim_for_game_config(game_config)

    sample_batch_fn = getattr(replay_buffer, "sample_batch", None)
    if callable(sample_batch_fn):
        packed_batch = sample_batch_fn(
            batch_size,
            encoded_state_size,
            action_space_size,
            value_dim,
        )
        return _prepare_replay_batch_from_packed_arrays(
            packed_batch,
            game_config,
            batch_size=batch_size,
            device=device,
        )

    sampled_positions = replay_buffer.sample(batch_size)
    return _prepare_replay_batch_from_positions(sampled_positions, game_config, device=device)


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


def _apply_go_symmetry_to_ownership_targets(
    ownership_target: torch.Tensor,
    *,
    board_shape: tuple[int, int],
    symmetry_indices: torch.Tensor,
) -> torch.Tensor:
    if ownership_target.ndim != 2:
        raise ValueError(
            "ownership_target must have shape (batch, board_area), "
            f"got {tuple(ownership_target.shape)}"
        )
    batch_size = ownership_target.shape[0]
    rows, cols = board_shape
    board_area = rows * cols
    if tuple(ownership_target.shape) != (batch_size, board_area):
        raise ValueError(
            f"ownership_target must have shape ({batch_size}, {board_area}), "
            f"got {tuple(ownership_target.shape)}"
        )
    if tuple(symmetry_indices.shape) != (batch_size,):
        raise ValueError(
            f"symmetry_indices must have shape ({batch_size},), got {tuple(symmetry_indices.shape)}"
        )

    ownership_board = ownership_target.reshape(batch_size, rows, cols)
    transformed_ownership_board = ownership_board.clone()
    for symmetry_index in range(8):
        sample_mask = symmetry_indices == symmetry_index
        if not bool(sample_mask.any()):
            continue
        transformed_ownership_board[sample_mask] = _apply_dihedral_transform(
            ownership_board[sample_mask],
            symmetry_index,
        )
    return transformed_ownership_board.reshape(batch_size, board_area)


def _create_grad_scaler(*, device: torch.device, enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler(device=device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def _set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = lr


def _gradient_statistics(model: nn.Module) -> tuple[float, int]:
    gradients: list[torch.Tensor] = []
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is not None:
            gradients.append(gradient.detach())

    if not gradients:
        return 0.0, 0

    finite_checks = torch.stack([torch.isfinite(gradient).all() for gradient in gradients])
    if not bool(finite_checks.all()):
        raise FloatingPointError("Encountered non-finite gradients during training")

    per_parameter_norms = torch.stack(
        [
            torch.linalg.vector_norm(
                gradient,
                ord=2.0,
                dtype=torch.float32,
            )
            for gradient in gradients
        ]
    )
    global_norm = float(torch.linalg.vector_norm(per_parameter_norms, ord=2.0).item())

    nonzero_flags = torch.stack([torch.count_nonzero(gradient) > 0 for gradient in gradients])
    nonzero_grad_parameters = int(nonzero_flags.sum().item())

    return global_norm, nonzero_grad_parameters


def _coerce_sample_weights(
    sample_weights: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if sample_weights is None:
        return torch.ones((batch_size,), dtype=torch.float32, device=device)

    weights = _as_float_tensor(sample_weights, device=device)
    if weights.ndim == 1:
        if tuple(weights.shape) != (batch_size,):
            raise ValueError(
                "sample_weights must have shape (batch_size,), "
                f"expected {(batch_size,)}, got {tuple(weights.shape)}"
            )
    elif weights.ndim == 2:
        if tuple(weights.shape) != (batch_size, 1):
            raise ValueError(
                "sample_weights must have shape (batch_size, 1) when rank 2, "
                f"expected {(batch_size, 1)}, got {tuple(weights.shape)}"
            )
        weights = weights.reshape(batch_size)
    else:
        raise ValueError(
            "sample_weights must be rank 1 or 2, "
            f"got rank {weights.ndim}"
        )

    if not bool(torch.isfinite(weights).all()):
        raise ValueError("sample_weights must be finite")
    if bool((weights < 0.0).any()):
        raise ValueError("sample_weights must be non-negative")
    return weights


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    states: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    ownership_target: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    ownership_loss_weight: float = DEFAULT_OWNERSHIP_LOSS_WEIGHT,
    game_config: GameConfig,
    lr_schedule: StepDecayLRSchedule,
    global_step: int,
    l2_reg: float,
    scaler: Any,
    use_mixed_precision: bool,
    compute_gradient_stats: bool = True,
) -> TrainingStepMetrics:
    """Run one optimization step and return decomposed metrics."""

    step_start_time = time.perf_counter()
    lr = lr_schedule.lr_at_step(global_step)
    _set_optimizer_learning_rate(optimizer, lr)

    optimizer.zero_grad(set_to_none=True)
    device_type = states.device.type
    if states.ndim < 1:
        raise ValueError("states must include a batch dimension")
    batch_size = int(states.shape[0])
    weights = _coerce_sample_weights(
        sample_weights,
        batch_size=batch_size,
        device=states.device,
    )
    if ownership_loss_weight < 0.0:
        raise ValueError(f"ownership_loss_weight must be non-negative, got {ownership_loss_weight}")

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
        model_output = model(states)
        if not isinstance(model_output, (tuple, list)):
            raise TypeError(
                "model(states) must return a tuple/list containing policy/value tensors "
                f"(and optional ownership), got {type(model_output).__name__}"
            )
        if len(model_output) == 2:
            policy_logits, predicted_value = model_output
            predicted_ownership_logits = None
        elif len(model_output) == 3:
            policy_logits, predicted_value, predicted_ownership_logits = model_output
        else:
            raise ValueError(
                "model(states) must return a 2-tuple (policy, value) or "
                "3-tuple (policy, value, ownership)"
            )
        if policy_logits.ndim != 2:
            raise ValueError(
                "policy_logits must have shape (batch, action_space_size), "
                f"got {tuple(policy_logits.shape)}"
            )
        if target_policy.shape != policy_logits.shape:
            raise ValueError(
                "target_policy must match policy_logits shape; "
                f"got {tuple(target_policy.shape)} vs {tuple(policy_logits.shape)}"
            )
        policy_log_probabilities = functional.log_softmax(policy_logits.to(dtype=torch.float32), dim=-1)
        policy_targets = target_policy.to(dtype=torch.float32, device=policy_log_probabilities.device)
        policy_per_sample = -(policy_targets * policy_log_probabilities).sum(dim=-1)
        if tuple(policy_per_sample.shape) != (batch_size,):
            raise ValueError(
                "policy loss must produce one value per sample; "
                f"expected {(batch_size,)}, got {tuple(policy_per_sample.shape)}"
            )
        policy_loss = (policy_per_sample * weights).mean()

        if value_type == "scalar":
            if predicted_value.ndim == 2 and predicted_value.shape[1] == 1:
                predicted_values = predicted_value.squeeze(dim=1).to(dtype=torch.float32)
            elif predicted_value.ndim == 1:
                predicted_values = predicted_value.to(dtype=torch.float32)
            else:
                raise ValueError(
                    "value must have shape (batch,) or (batch, 1) for scalar heads, "
                    f"got {tuple(predicted_value.shape)}"
                )

            if target_value.ndim == 2 and target_value.shape[1] == 1:
                target_values = target_value.squeeze(dim=1).to(dtype=torch.float32, device=predicted_values.device)
            elif target_value.ndim == 1:
                target_values = target_value.to(dtype=torch.float32, device=predicted_values.device)
            else:
                raise ValueError(
                    "target_value must have shape (batch,) or (batch, 1) for scalar heads, "
                    f"got {tuple(target_value.shape)}"
                )

            if predicted_values.shape != target_values.shape:
                raise ValueError(
                    "scalar value and target shapes must match, "
                    f"got {tuple(predicted_values.shape)} and {tuple(target_values.shape)}"
                )
            value_per_sample = (predicted_values - target_values).pow(2)
        else:
            if predicted_value.ndim != 2 or predicted_value.shape[1] != 3:
                raise ValueError(
                    "value must have shape (batch, 3) for WDL heads, "
                    f"got {tuple(predicted_value.shape)}"
                )
            if target_value.shape != predicted_value.shape:
                raise ValueError(
                    "target_value must match WDL value shape; "
                    f"got {tuple(target_value.shape)} vs {tuple(predicted_value.shape)}"
                )
            predicted_probabilities = predicted_value.to(dtype=torch.float32).clamp_min(_LOSS_EPSILON)
            target_probabilities = target_value.to(dtype=torch.float32, device=predicted_probabilities.device)
            value_per_sample = -(target_probabilities * torch.log(predicted_probabilities)).sum(dim=-1)

        if tuple(value_per_sample.shape) != (batch_size,):
            raise ValueError(
                "value loss must produce one value per sample; "
                f"expected {(batch_size,)}, got {tuple(value_per_sample.shape)}"
            )
        value_loss = (value_per_sample * weights).mean()

        loss_ownership = torch.zeros((), dtype=torch.float32, device=policy_loss.device)
        if ownership_loss_weight > 0.0:
            if predicted_ownership_logits is None:
                raise ValueError(
                    "ownership_loss_weight > 0 requires model(states) to return ownership logits "
                    "as a third output tensor."
                )
            if ownership_target is None:
                raise ValueError(
                    "ownership_loss_weight > 0 requires ownership_target values in replay batches; "
                    "ensure mcts.compute_ownership=true and replay data includes ownership labels."
                )
            loss_ownership = ownership_loss(
                predicted_ownership_logits,
                ownership_target.to(device=policy_loss.device),
                weights.to(device=policy_loss.device),
            )

        l2_loss = l2_regularization_loss(model)
        if l2_loss.device != policy_loss.device:
            l2_loss = l2_loss.to(device=policy_loss.device)
        total_loss = (
            policy_loss
            + value_loss
            + (float(ownership_loss_weight) * loss_ownership)
            + (float(l2_reg) * l2_loss)
        )

    if not torch.isfinite(total_loss):
        raise FloatingPointError("Encountered non-finite total loss")
    if not torch.isfinite(policy_loss):
        raise FloatingPointError("Encountered non-finite policy loss")
    if not torch.isfinite(value_loss):
        raise FloatingPointError("Encountered non-finite value loss")
    if not torch.isfinite(loss_ownership):
        raise FloatingPointError("Encountered non-finite ownership loss")

    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    if compute_gradient_stats:
        grad_global_norm, nonzero_grad_parameters = _gradient_statistics(model)
    else:
        grad_global_norm = 0.0
        nonzero_grad_parameters = 0
    scaler.step(optimizer)
    scaler.update()

    step_duration = max(time.perf_counter() - step_start_time, 1e-8)
    # Transfer all loss components to CPU in one operation to avoid
    # multiple GPU->CPU synchronization points from repeated `.item()` calls.
    loss_values = torch.stack(
        [
            total_loss.detach(),
            policy_loss.detach(),
            value_loss.detach(),
            loss_ownership.detach(),
            l2_loss.detach(),
        ]
    ).cpu().tolist()
    return TrainingStepMetrics(
        step=global_step + 1,
        loss_total=loss_values[0],
        loss_policy=loss_values[1],
        loss_value=loss_values[2],
        loss_ownership=loss_values[3],
        loss_l2=loss_values[4],
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
    replay_buffer: ReplayBufferLike | None = None,
    keep_last: int = DEFAULT_CHECKPOINT_KEEP_LAST,
    is_milestone: bool = False,
    export_folded_weights: bool = True,
    game_config: Any | None = None,
) -> CheckpointPaths:
    """Persist model/optimizer/schedule state with replay metadata and buffer contents."""

    replay_metadata = extract_replay_buffer_metadata(replay_buffer)
    result = save_checkpoint(
        model,
        optimizer,
        step=step,
        checkpoint_dir=checkpoint_dir,
        lr_schedule_entries=lr_schedule.entries,
        replay_buffer_metadata=replay_metadata,
        is_milestone=is_milestone,
        export_folded_weights=export_folded_weights,
        keep_last=keep_last,
    )

    if replay_buffer is not None and game_config is not None:
        from alphazero.utils.checkpoint import save_replay_buffer_state

        rows, cols = game_config.board_shape
        encoded_state_size = game_config.input_channels * rows * cols
        save_replay_buffer_state(
            replay_buffer,
            result.checkpoint_path,
            encoded_state_size=encoded_state_size,
            policy_size=game_config.action_space_size,
        )

    return result


def load_training_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[int, StepDecayLRSchedule]:
    """Load a previously saved training checkpoint into model/optimizer state."""

    loaded = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        map_location=map_location,
    )

    if loaded.lr_schedule_entries:
        schedule = StepDecayLRSchedule(entries=loaded.lr_schedule_entries)
    else:
        schedule = StepDecayLRSchedule()
    return loaded.step, schedule


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
        checkpoint_keep_last=int(training.get("checkpoint_keep_last", DEFAULT_CHECKPOINT_KEEP_LAST)),
        milestone_interval=int(training.get("milestone_interval", DEFAULT_MILESTONE_INTERVAL)),
        log_interval=int(training.get("log_interval", DEFAULT_LOG_INTERVAL)),
        min_buffer_size=int(training.get("min_buffer_size", DEFAULT_MIN_BUFFER_SIZE)),
        ownership_loss_weight=float(
            training.get("ownership_loss_weight", DEFAULT_OWNERSHIP_LOSS_WEIGHT)
        ),
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

        states, target_policy, target_value, sample_weights, ownership_target = sample_replay_batch_tensors(
            replay_buffer,
            game_config,
            batch_size=training_config.batch_size,
            device=device,
        )

        if game_config.supports_symmetry:
            symmetry_indices = torch.randint(
                low=0,
                high=8,
                size=(states.shape[0],),
                device=states.device,
            )
            states, target_policy = apply_random_go_symmetry(
                states,
                target_policy,
                symmetry_indices=symmetry_indices,
            )
            if ownership_target is not None:
                ownership_target = _apply_go_symmetry_to_ownership_targets(
                    ownership_target,
                    board_shape=game_config.board_shape,
                    symmetry_indices=symmetry_indices,
                )

        step_metrics = train_one_step(
            model,
            active_optimizer,
            states=states,
            target_policy=target_policy,
            target_value=target_value,
            ownership_target=ownership_target,
            sample_weights=sample_weights,
            ownership_loss_weight=training_config.ownership_loss_weight,
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
            loss_ownership=step_metrics.loss_ownership,
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
                    replay_buffer=replay_buffer,
                    keep_last=training_config.checkpoint_keep_last,
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
                        replay_buffer=replay_buffer,
                        keep_last=training_config.checkpoint_keep_last,
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
    "sample_replay_batch_tensors",
    "save_training_checkpoint",
    "train_one_step",
    "training_loop",
]
