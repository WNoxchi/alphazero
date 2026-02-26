"""Checkpoint save/load utilities for AlphaZero training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional torch install.
    if exc.name != "torch":
        raise
    torch = None  # type: ignore[assignment]
    nn = Any  # type: ignore[assignment]


DEFAULT_ROLLING_CHECKPOINT_KEEP_LAST = 10
_CHECKPOINT_FILE_RE = re.compile(r"^(checkpoint|milestone)_(\d{8})\.pt$")


@dataclass(frozen=True, slots=True)
class CheckpointPaths:
    """Checkpoint artifacts emitted at a specific training step."""

    step: int
    checkpoint_path: Path
    folded_weights_path: Path | None
    is_milestone: bool


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Training state loaded from a checkpoint artifact."""

    step: int
    lr_schedule_entries: tuple[tuple[int, float], ...]
    replay_buffer_metadata: dict[str, int]
    is_milestone: bool


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - exercised only when torch missing.
        raise ModuleNotFoundError(
            "torch is required for checkpoint operations"
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


def _coerce_lr_entry(entry: Sequence[object], index: int) -> tuple[int, float]:
    if len(entry) != 2:
        raise ValueError(
            f"lr_schedule entry at index {index} must contain exactly 2 values [step, lr]"
        )
    step = _coerce_non_negative_int("lr_schedule step", entry[0])
    lr_raw = entry[1]
    if isinstance(lr_raw, bool) or not isinstance(lr_raw, (int, float)):
        raise TypeError(
            f"lr_schedule rate at index {index} must be numeric, got {type(lr_raw).__name__}"
        )
    lr = float(lr_raw)
    if lr <= 0.0:
        raise ValueError(f"lr_schedule rate at index {index} must be > 0, got {lr}")
    return step, lr


def normalize_lr_schedule_entries(
    entries: Sequence[Sequence[object]] | None,
) -> tuple[tuple[int, float], ...]:
    """Validate schedule entries for checkpoint payload stability."""

    if entries is None:
        return tuple()
    normalized: list[tuple[int, float]] = []
    previous_step = -1
    for index, entry in enumerate(entries):
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence):
            raise TypeError(
                f"lr_schedule entry at index {index} must be a 2-item sequence [step, lr]"
            )
        step, lr = _coerce_lr_entry(entry, index)
        if step <= previous_step:
            raise ValueError("lr_schedule steps must be strictly increasing")
        normalized.append((step, lr))
        previous_step = step
    return tuple(normalized)


def normalize_replay_buffer_metadata(metadata: Mapping[str, object] | None) -> dict[str, int]:
    """Normalize replay metadata to non-negative integer fields."""

    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"replay_buffer_metadata must be a mapping, got {type(metadata).__name__}"
        )

    normalized: dict[str, int] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("replay_buffer_metadata keys must be strings")
        normalized[key] = _coerce_non_negative_int(f"replay_buffer_metadata[{key!r}]", value)
    return normalized


def _try_read_replay_field(replay_buffer: object, field_name: str) -> int | None:
    if not hasattr(replay_buffer, field_name):
        return None
    value = getattr(replay_buffer, field_name)
    if callable(value):
        value = value()
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def extract_replay_buffer_metadata(replay_buffer: object | None) -> dict[str, int]:
    """Extract replay metadata if a replay buffer exposes it.

    This function is tolerant of partially implemented replay buffer bindings;
    missing fields produce an empty or partial mapping rather than failing.
    """

    if replay_buffer is None:
        return {}

    metadata_accessor = getattr(replay_buffer, "checkpoint_metadata", None)
    if callable(metadata_accessor):
        accessor_metadata = metadata_accessor()
        if isinstance(accessor_metadata, Mapping):
            return normalize_replay_buffer_metadata(accessor_metadata)

    metadata: dict[str, int] = {}
    for field_name in ("write_head", "count", "games_total"):
        value = _try_read_replay_field(replay_buffer, field_name)
        if value is not None:
            metadata[field_name] = value

    if "count" not in metadata and hasattr(replay_buffer, "size"):
        size_fn = getattr(replay_buffer, "size")
        if callable(size_fn):
            size_value = size_fn()
            if isinstance(size_value, int) and not isinstance(size_value, bool) and size_value >= 0:
                metadata["count"] = size_value

    return metadata


def _parse_checkpoint_file(path: Path) -> tuple[int, bool] | None:
    match = _CHECKPOINT_FILE_RE.match(path.name)
    if match is None:
        return None
    if path.name.endswith("_folded.pt"):
        return None
    kind, step_digits = match.groups()
    step = int(step_digits)
    return step, kind == "milestone"


def list_checkpoints(
    checkpoint_dir: str | Path,
    *,
    include_milestones: bool = True,
) -> tuple[Path, ...]:
    """List non-folded checkpoints sorted by ascending training step."""

    checkpoint_root = Path(checkpoint_dir)
    if not checkpoint_root.exists():
        return tuple()

    parsed: list[tuple[int, int, Path]] = []
    for path in checkpoint_root.glob("*.pt"):
        parsed_entry = _parse_checkpoint_file(path)
        if parsed_entry is None:
            continue
        step, is_milestone = parsed_entry
        if not include_milestones and is_milestone:
            continue
        # For identical steps, prefer regular checkpoints over milestones.
        kind_priority = 1 if is_milestone else 2
        parsed.append((step, kind_priority, path))

    parsed.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in parsed)


def find_latest_checkpoint(
    checkpoint_dir: str | Path,
    *,
    include_milestones: bool = True,
) -> Path | None:
    """Return the newest checkpoint path, or ``None`` when none exist."""

    checkpoints = list_checkpoints(
        checkpoint_dir,
        include_milestones=include_milestones,
    )
    if not checkpoints:
        return None
    return checkpoints[-1]


def replay_buffer_path_for_checkpoint(checkpoint_path: Path) -> Path:
    """Derive the companion .replay.npz path for a given checkpoint."""
    return checkpoint_path.with_suffix(".replay.npz")


def save_replay_buffer_state(
    replay_buffer: object,
    checkpoint_path: str | Path,
    *,
    encoded_state_size: int,
    policy_size: int,
) -> Path | None:
    """Save replay buffer contents as a companion .replay.npz alongside a checkpoint.

    Returns the path written, or None if the buffer is empty or lacks export_buffer.
    """
    import numpy as np

    export_fn = getattr(replay_buffer, "export_buffer", None)
    if not callable(export_fn):
        return None
    buf_size = replay_buffer.size()
    if buf_size == 0:
        return None

    states, policies, values_wdl, game_ids, move_numbers = export_fn(
        encoded_state_size, policy_size,
    )
    replay_path = replay_buffer_path_for_checkpoint(Path(checkpoint_path))
    np.savez(
        replay_path,
        states=states,
        policies=policies,
        values_wdl=values_wdl,
        game_ids=game_ids,
        move_numbers=move_numbers,
        encoded_state_size=np.array(encoded_state_size, dtype=np.int64),
        policy_size=np.array(policy_size, dtype=np.int64),
    )
    return replay_path


def load_replay_buffer_state(
    replay_buffer: object,
    checkpoint_path: str | Path,
    *,
    encoded_state_size: int,
    policy_size: int,
) -> int:
    """Load replay buffer state from a companion .replay.npz file.

    Returns the number of positions loaded, or 0 if the file doesn't exist.
    """
    import numpy as np

    replay_path = replay_buffer_path_for_checkpoint(Path(checkpoint_path))
    if not replay_path.exists():
        return 0

    import_fn = getattr(replay_buffer, "import_buffer", None)
    if not callable(import_fn):
        return 0

    data = np.load(replay_path)
    import_fn(
        data["states"],
        data["policies"],
        data["values_wdl"],
        data["game_ids"],
        data["move_numbers"],
        encoded_state_size,
        policy_size,
    )
    return int(data["states"].shape[0])


def _prune_rolling_checkpoints(checkpoint_dir: Path, *, keep_last: int) -> None:
    regular_checkpoints = list(
        list_checkpoints(checkpoint_dir, include_milestones=False)
    )
    if len(regular_checkpoints) <= keep_last:
        return

    prune_count = len(regular_checkpoints) - keep_last
    for obsolete_checkpoint in regular_checkpoints[:prune_count]:
        if obsolete_checkpoint.exists():
            obsolete_checkpoint.unlink()
        parsed = _parse_checkpoint_file(obsolete_checkpoint)
        if parsed is None:
            continue
        step, _ = parsed
        folded = checkpoint_dir / f"checkpoint_{step:08d}_folded.pt"
        if folded.exists():
            folded.unlink()
        replay = checkpoint_dir / f"checkpoint_{step:08d}.replay.npz"
        if replay.exists():
            replay.unlink()


def save_checkpoint(
    model: nn.Module,
    optimizer: Any,
    *,
    step: int,
    checkpoint_dir: str | Path,
    lr_schedule_entries: Sequence[Sequence[object]] | None = None,
    replay_buffer_metadata: Mapping[str, object] | None = None,
    is_milestone: bool = False,
    export_folded_weights: bool = True,
    keep_last: int = DEFAULT_ROLLING_CHECKPOINT_KEEP_LAST,
) -> CheckpointPaths:
    """Persist full training state and optional BN-folded inference weights."""

    _require_torch()
    _coerce_positive_int("step", step)
    _coerce_positive_int("keep_last", keep_last)

    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    normalized_schedule = normalize_lr_schedule_entries(lr_schedule_entries)
    normalized_replay_metadata = normalize_replay_buffer_metadata(replay_buffer_metadata)

    checkpoint_stem = "milestone" if is_milestone else "checkpoint"
    checkpoint_path = checkpoint_root / f"{checkpoint_stem}_{step:08d}.pt"
    # Unwrap torch.compile wrapper to save clean state_dict keys
    # (compiled models prefix keys with "_orig_mod.")
    raw_model = getattr(model, "_orig_mod", model)
    payload = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_schedule": [list(entry) for entry in normalized_schedule],
        "replay_buffer_metadata": normalized_replay_metadata,
        "is_milestone": bool(is_milestone),
    }
    torch.save(payload, checkpoint_path)

    folded_weights_path: Path | None = None
    if export_folded_weights:
        from alphazero.network.bn_fold import export_folded_model

        folded_model = export_folded_model(model).eval()
        folded_weights_path = checkpoint_root / f"{checkpoint_stem}_{step:08d}_folded.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": folded_model.state_dict(),
                "source_checkpoint": str(checkpoint_path),
                "is_milestone": bool(is_milestone),
            },
            folded_weights_path,
        )

    if not is_milestone:
        _prune_rolling_checkpoints(checkpoint_root, keep_last=keep_last)

    return CheckpointPaths(
        step=step,
        checkpoint_path=checkpoint_path,
        folded_weights_path=folded_weights_path,
        is_milestone=bool(is_milestone),
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Any | None = None,
    *,
    map_location: str | Any | None = None,
) -> LoadedCheckpoint:
    """Load model, optimizer, and scheduler metadata from a checkpoint."""

    _require_torch()
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping")
    if "model_state_dict" not in payload or "step" not in payload:
        raise ValueError(
            "Checkpoint payload is missing required keys: 'model_state_dict' and 'step'"
        )

    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    step = int(payload["step"])
    if step < 0:
        raise ValueError(f"Checkpoint step must be non-negative, got {step}")

    raw_schedule = payload.get("lr_schedule")
    if raw_schedule is None:
        schedule_entries: tuple[tuple[int, float], ...] = tuple()
    else:
        if isinstance(raw_schedule, (str, bytes)) or not isinstance(raw_schedule, Sequence):
            raise TypeError("lr_schedule stored in checkpoint must be a sequence")
        schedule_entries = normalize_lr_schedule_entries(raw_schedule)

    replay_metadata_raw = payload.get("replay_buffer_metadata")
    replay_metadata = (
        normalize_replay_buffer_metadata(replay_metadata_raw)
        if isinstance(replay_metadata_raw, Mapping)
        else {}
    )
    is_milestone = bool(payload.get("is_milestone", path.name.startswith("milestone_")))

    return LoadedCheckpoint(
        step=step,
        lr_schedule_entries=schedule_entries,
        replay_buffer_metadata=replay_metadata,
        is_milestone=is_milestone,
    )


def load_latest_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: Any | None = None,
    *,
    include_milestones: bool = True,
    map_location: str | Any | None = None,
) -> LoadedCheckpoint | None:
    """Load the newest checkpoint from a directory if one exists."""

    latest = find_latest_checkpoint(
        checkpoint_dir,
        include_milestones=include_milestones,
    )
    if latest is None:
        return None
    return load_checkpoint(
        latest,
        model,
        optimizer,
        map_location=map_location,
    )


__all__ = [
    "DEFAULT_ROLLING_CHECKPOINT_KEEP_LAST",
    "CheckpointPaths",
    "LoadedCheckpoint",
    "extract_replay_buffer_metadata",
    "find_latest_checkpoint",
    "list_checkpoints",
    "load_checkpoint",
    "load_latest_checkpoint",
    "load_replay_buffer_state",
    "normalize_lr_schedule_entries",
    "normalize_replay_buffer_metadata",
    "replay_buffer_path_for_checkpoint",
    "save_checkpoint",
    "save_replay_buffer_state",
]
