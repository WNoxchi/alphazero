#!/usr/bin/env python3
"""Export trained AlphaZero checkpoints into deployment artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from alphazero.config import GameConfig, get_game_config, load_yaml_config


def _coerce_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key!r} section must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeDependencies:
    """Runtime-callable dependencies for model export operations."""

    torch: Any
    ResNetSE: Any
    load_checkpoint: Callable[..., Any]
    export_folded_model: Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class NetworkShape:
    """Network architecture dimensions needed to reconstruct a checkpoint model."""

    num_blocks: int
    num_filters: int
    se_reduction: int


@dataclass(frozen=True, slots=True)
class ExportRuntime:
    """Resolved runtime state for one export invocation."""

    dependencies: RuntimeDependencies
    checkpoint_path: Path
    checkpoint_step: int
    game_config: GameConfig
    model: Any
    sample_input: Any
    output_path: Path
    export_format: str
    fold_batch_norm: bool
    dynamic_batch: bool


@dataclass(frozen=True, slots=True)
class ExportSummary:
    """Terminal summary returned by the export script."""

    checkpoint_path: Path
    checkpoint_step: int
    output_path: Path
    export_format: str
    fold_batch_norm: bool


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for export operations."""

    parser = argparse.ArgumentParser(description="Export AlphaZero model checkpoints.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a training checkpoint (*.pt).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination artifact path. Defaults beside checkpoint with format-specific suffix.",
    )
    parser.add_argument(
        "--format",
        choices=("torchscript", "onnx"),
        default="torchscript",
        help="Export format.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config used to resolve game/network architecture.",
    )
    parser.add_argument(
        "--game",
        choices=("chess", "go"),
        default=None,
        help="Game override when --config is omitted.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="ResNet block count override.",
    )
    parser.add_argument(
        "--num-filters",
        type=int,
        default=None,
        help="ResNet channel width override.",
    )
    parser.add_argument(
        "--se-reduction",
        type=int,
        default=None,
        help="SE reduction ratio override.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Example input batch size used for export tracing (default: 1).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used during export (default: cpu).",
    )
    parser.add_argument(
        "--fold-bn",
        action="store_true",
        help="Fold Conv+BatchNorm pairs before exporting.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with dynamic batch axis metadata (ONNX only).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output artifact if it already exists.",
    )
    return parser


def load_runtime_dependencies() -> RuntimeDependencies:
    """Import export runtime dependencies lazily."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for scripts/export_model.py") from exc

    from alphazero.network import ResNetSE, export_folded_model
    from alphazero.utils.checkpoint import load_checkpoint

    return RuntimeDependencies(
        torch=torch,
        ResNetSE=ResNetSE,
        load_checkpoint=load_checkpoint,
        export_folded_model=export_folded_model,
    )


def _resolve_config_mapping(config_path: str | None) -> Mapping[str, Any] | None:
    if config_path is None:
        return None
    return load_yaml_config(config_path)


def _resolve_game_config(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any] | None,
) -> GameConfig:
    game_from_config: str | None = None
    if config is not None:
        raw_game = config.get("game")
        if raw_game is not None:
            if not isinstance(raw_game, str) or not raw_game.strip():
                raise ValueError("Config key 'game' must be a non-empty string")
            game_from_config = raw_game.strip().lower()

    game_from_args: str | None = None
    if args.game is not None:
        game_from_args = str(args.game).strip().lower()

    if game_from_args is not None and game_from_config is not None and game_from_args != game_from_config:
        raise ValueError(
            "--game and --config disagree on game selection: "
            f"--game={game_from_args!r}, config.game={game_from_config!r}"
        )

    resolved_game = game_from_args or game_from_config
    if resolved_game is None:
        raise ValueError("Either --game or --config with a 'game' key is required")

    return get_game_config(resolved_game)


def _resolve_network_shape(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any] | None,
) -> NetworkShape:
    default_num_blocks = 20
    default_num_filters = 256
    default_se_reduction = 4

    if config is not None:
        network = _section(config, "network")
        architecture = network.get("architecture", "resnet_se")
        if not isinstance(architecture, str) or not architecture.strip():
            raise ValueError("network.architecture must be a non-empty string")
        if architecture.strip().lower() != "resnet_se":
            raise ValueError(
                f"Unsupported network architecture {architecture!r}; only 'resnet_se' is supported"
            )

        default_num_blocks = int(network.get("num_blocks", default_num_blocks))
        default_num_filters = int(network.get("num_filters", default_num_filters))
        default_se_reduction = int(network.get("se_reduction", default_se_reduction))

    num_blocks = default_num_blocks if args.num_blocks is None else int(args.num_blocks)
    num_filters = default_num_filters if args.num_filters is None else int(args.num_filters)
    se_reduction = default_se_reduction if args.se_reduction is None else int(args.se_reduction)

    return NetworkShape(
        num_blocks=_coerce_positive_int("num_blocks", num_blocks),
        num_filters=_coerce_positive_int("num_filters", num_filters),
        se_reduction=_coerce_positive_int("se_reduction", se_reduction),
    )


def _resolve_output_path(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> Path:
    export_format = str(args.format).strip().lower()
    if export_format == "torchscript":
        default_suffix = ".ts"
    elif export_format == "onnx":
        default_suffix = ".onnx"
    else:
        raise ValueError(f"Unsupported export format {args.format!r}")

    if args.output is None:
        return checkpoint_path.parent / f"{checkpoint_path.stem}_{export_format}{default_suffix}"

    output_path = Path(args.output)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(default_suffix)
    return output_path


def _resolve_torch_device(torch_module: Any, device_spec: str | None) -> Any:
    if device_spec is None:
        return torch_module.device("cpu")
    return torch_module.device(device_spec)


def _build_sample_input(
    *,
    torch_module: Any,
    game_config: GameConfig,
    batch_size: int,
    device: Any,
) -> Any:
    rows, cols = game_config.board_shape
    return torch_module.zeros(
        (batch_size, game_config.input_channels, rows, cols),
        dtype=torch_module.float32,
        device=device,
    )


def build_export_runtime(
    *,
    args: argparse.Namespace,
    dependencies: RuntimeDependencies | None = None,
) -> ExportRuntime:
    """Resolve runtime state for model export."""

    active_dependencies = dependencies or load_runtime_dependencies()
    batch_size = _coerce_positive_int("batch_size", int(args.batch_size))
    opset_version = _coerce_positive_int("opset", int(args.opset))
    _ = opset_version

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    config = _resolve_config_mapping(args.config)
    game_config = _resolve_game_config(args=args, config=config)
    network_shape = _resolve_network_shape(args=args, config=config)

    model = active_dependencies.ResNetSE(
        game_config,
        num_blocks=network_shape.num_blocks,
        num_filters=network_shape.num_filters,
        se_reduction=network_shape.se_reduction,
    )

    loaded_checkpoint = active_dependencies.load_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        map_location="cpu",
    )

    checkpoint_step = int(getattr(loaded_checkpoint, "step", 0))
    if checkpoint_step < 0:
        raise ValueError(f"Loaded checkpoint step must be non-negative, got {checkpoint_step}")

    fold_batch_norm = bool(args.fold_bn)
    if fold_batch_norm:
        model = active_dependencies.export_folded_model(model)

    torch_module = active_dependencies.torch
    resolved_device = _resolve_torch_device(torch_module, args.device)
    model = model.to(device=resolved_device)
    model.eval()

    sample_input = _build_sample_input(
        torch_module=torch_module,
        game_config=game_config,
        batch_size=batch_size,
        device=resolved_device,
    )

    output_path = _resolve_output_path(args=args, checkpoint_path=checkpoint_path)
    if output_path.exists() and not bool(args.overwrite):
        raise FileExistsError(
            f"Output path already exists: {output_path} (pass --overwrite to replace)"
        )

    return ExportRuntime(
        dependencies=active_dependencies,
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
        game_config=game_config,
        model=model,
        sample_input=sample_input,
        output_path=output_path,
        export_format=str(args.format).strip().lower(),
        fold_batch_norm=fold_batch_norm,
        dynamic_batch=bool(args.dynamic_batch),
    )


def _export_torchscript(runtime: ExportRuntime) -> None:
    torch_module = runtime.dependencies.torch

    runtime.output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch_module.no_grad():
        traced = torch_module.jit.trace(runtime.model, runtime.sample_input, strict=True)
        try:
            exported = torch_module.jit.freeze(traced.eval())
        except Exception:
            exported = traced
        exported.save(str(runtime.output_path))


def _export_onnx(
    runtime: ExportRuntime,
    *,
    opset: int,
) -> None:
    torch_module = runtime.dependencies.torch
    dynamic_axes = None
    if runtime.dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        }

    runtime.output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch_module.no_grad():
        torch_module.onnx.export(
            runtime.model,
            runtime.sample_input,
            str(runtime.output_path),
            input_names=["input"],
            output_names=["policy_logits", "value"],
            dynamic_axes=dynamic_axes,
            export_params=True,
            do_constant_folding=True,
            opset_version=opset,
        )


def run_from_args(
    args: argparse.Namespace,
    *,
    dependencies: RuntimeDependencies | None = None,
) -> ExportSummary:
    """Export a checkpoint artifact from parsed CLI arguments."""

    runtime = build_export_runtime(args=args, dependencies=dependencies)
    if runtime.export_format == "torchscript":
        _export_torchscript(runtime)
    elif runtime.export_format == "onnx":
        _export_onnx(runtime, opset=_coerce_positive_int("opset", int(args.opset)))
    else:
        raise ValueError(f"Unsupported export format {runtime.export_format!r}")

    return ExportSummary(
        checkpoint_path=runtime.checkpoint_path,
        checkpoint_step=runtime.checkpoint_step,
        output_path=runtime.output_path,
        export_format=runtime.export_format,
        fold_batch_norm=runtime.fold_batch_norm,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_from_args(args)
    except (FileNotFoundError, FileExistsError, ModuleNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Exported {summary.export_format} artifact to {summary.output_path} "
        f"(checkpoint step {summary.checkpoint_step}, fold_bn={summary.fold_batch_norm})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
