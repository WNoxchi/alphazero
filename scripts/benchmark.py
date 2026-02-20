#!/usr/bin/env python3
"""Benchmark inference, training, and MCTS throughput for AlphaZero."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from alphazero.config import GameConfig, get_game_config, load_yaml_config


DEFAULT_INFERENCE_BATCH_SIZES: tuple[int, ...] = (32, 64, 128, 256, 512)
DEFAULT_TRAINING_BATCH_SIZES: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
DEFAULT_MCTS_GAMES: tuple[int, ...] = (16, 32, 64)
DEFAULT_MCTS_THREADS: tuple[int, ...] = (4, 8, 16)


def _coerce_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _coerce_non_negative_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    converted = float(value)
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative, got {converted}")
    return converted


def _parse_positive_int_csv(name: str, raw: str) -> tuple[int, ...]:
    if not isinstance(raw, str):
        raise TypeError(f"{name} must be a comma-separated string")

    tokens = [token.strip() for token in raw.split(",")]
    if any(token == "" for token in tokens):
        raise ValueError(f"{name} must not contain empty elements")

    parsed: list[int] = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"{name} contains non-integer value {token!r}") from exc
        parsed.append(_coerce_positive_int(f"{name} element", value))

    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class NetworkShapeConfig:
    """Network size controls used by benchmark model construction."""

    num_blocks: int
    num_filters: int
    se_reduction: int


@dataclass(frozen=True, slots=True)
class InferenceBenchmarkResult:
    """One inference benchmark datapoint for a batch size."""

    batch_size: int
    positions_per_second: float
    batch_latency_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingBenchmarkResult:
    """One training benchmark datapoint for a batch size."""

    batch_size: int
    steps_per_second: float
    positions_per_second: float
    step_latency_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MctsBenchmarkResult:
    """One MCTS benchmark datapoint for a (games, threads) configuration."""

    concurrent_games: int
    threads_per_game: int
    simulations_per_move: int
    simulations_per_second: float
    moves_per_second: float
    games_per_hour: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Terminal benchmark report across requested benchmark modes."""

    mode: str
    game: str
    device: str | None
    inference: tuple[InferenceBenchmarkResult, ...]
    training: tuple[TrainingBenchmarkResult, ...]
    mcts: tuple[MctsBenchmarkResult, ...]


@dataclass(frozen=True, slots=True)
class RuntimeDependencies:
    """Runtime-callable dependencies for benchmark operations."""

    torch: Any | None = None
    ResNetSE: Any | None = None
    create_optimizer: Callable[..., Any] | None = None
    StepDecayLRSchedule: Any | None = None
    train_one_step: Callable[..., Any] | None = None
    cpp: Any | None = None
    perf_counter: Callable[[], float] = time.perf_counter
    sleep: Callable[[float], None] = time.sleep


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for benchmark runs."""

    parser = argparse.ArgumentParser(description="Benchmark AlphaZero throughput.")
    parser.add_argument(
        "--mode",
        choices=("inference", "training", "mcts", "all"),
        default="all",
        help="Benchmark mode to run.",
    )
    parser.add_argument("--game", required=True, choices=("chess", "go"), help="Game to benchmark.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config used to source default network dimensions.",
    )

    parser.add_argument(
        "--batch-sizes",
        default=None,
        help="Shared comma-separated batch sizes for inference/training (for example: 64,128,256).",
    )
    parser.add_argument(
        "--inference-batch-sizes",
        default=None,
        help="Optional inference-specific batch sizes override.",
    )
    parser.add_argument(
        "--training-batch-sizes",
        default=None,
        help="Optional training-specific batch sizes override.",
    )
    parser.add_argument(
        "--games",
        default=",".join(str(value) for value in DEFAULT_MCTS_GAMES),
        help="Comma-separated concurrent game counts for MCTS benchmark.",
    )
    parser.add_argument(
        "--threads",
        default=",".join(str(value) for value in DEFAULT_MCTS_THREADS),
        help="Comma-separated thread counts per game for MCTS benchmark.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Timed iterations per datapoint for inference/training benchmarks.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Warmup iterations per datapoint for inference/training benchmarks.",
    )
    parser.add_argument(
        "--mcts-warmup-seconds",
        type=float,
        default=2.0,
        help="Warmup duration (seconds) before MCTS throughput sampling.",
    )
    parser.add_argument(
        "--mcts-duration-seconds",
        type=float,
        default=8.0,
        help="Sampling duration (seconds) for MCTS throughput measurement.",
    )
    parser.add_argument(
        "--mcts-simulations-per-move",
        type=int,
        default=800,
        help="MCTS simulations per move for MCTS benchmark workers.",
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Network residual block count override.",
    )
    parser.add_argument(
        "--num-filters",
        type=int,
        default=None,
        help="Network channel width override.",
    )
    parser.add_argument(
        "--se-reduction",
        type=int,
        default=None,
        help="Network SE reduction ratio override.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override for inference/training benchmarks (for example: 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable mixed precision and run inference/training in FP32.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0xC0FFEE,
        help="Random seed for benchmark reproducibility.",
    )
    parser.add_argument(
        "--node-arena-capacity",
        type=int,
        default=262_144,
        help="Node arena capacity per game for MCTS benchmark workers.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Fail immediately on the first datapoint error.",
    )
    return parser


def _import_cpp_bindings() -> Any:
    try:
        import alphazero_cpp  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "alphazero_cpp extension is unavailable. Build the project first with "
            "`cmake --build build` before running scripts/benchmark.py in MCTS mode."
        ) from exc
    return alphazero_cpp


def load_runtime_dependencies(mode: str) -> RuntimeDependencies:
    """Import runtime dependencies lazily based on requested benchmark mode."""

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"inference", "training", "mcts", "all"}:
        raise ValueError(f"Unsupported benchmark mode {mode!r}")

    needs_python_stack = normalized_mode in {"inference", "training", "all"}
    needs_training_stack = normalized_mode in {"training", "all"}
    needs_cpp = normalized_mode in {"mcts", "all"}

    torch_module: Any | None = None
    resnet_cls: Any | None = None
    create_optimizer: Callable[..., Any] | None = None
    schedule_cls: Any | None = None
    train_one_step: Callable[..., Any] | None = None
    cpp_module: Any | None = None

    if needs_python_stack:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("torch is required for inference/training benchmarks") from exc

        from alphazero.network import ResNetSE

        torch_module = torch
        resnet_cls = ResNetSE

    if needs_training_stack:
        from alphazero.training.lr_schedule import StepDecayLRSchedule
        from alphazero.training.trainer import create_optimizer as _create_optimizer
        from alphazero.training.trainer import train_one_step as _train_one_step

        create_optimizer = _create_optimizer
        schedule_cls = StepDecayLRSchedule
        train_one_step = _train_one_step

    if needs_cpp:
        cpp_module = _import_cpp_bindings()

    return RuntimeDependencies(
        torch=torch_module,
        ResNetSE=resnet_cls,
        create_optimizer=create_optimizer,
        StepDecayLRSchedule=schedule_cls,
        train_one_step=train_one_step,
        cpp=cpp_module,
    )


def _resolve_torch_device(torch_module: Any, device_spec: str | None) -> Any:
    if device_spec is None:
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device_spec)


def _cuda_synchronize(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", "cpu") == "cuda":
        torch_module.cuda.synchronize(device=device)


def _create_grad_scaler(torch_module: Any, *, device: Any, enabled: bool) -> Any:
    try:
        return torch_module.amp.GradScaler(device=device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch_module.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def _resolve_config_mapping(config_path: str | None) -> Mapping[str, Any] | None:
    if config_path is None:
        return None
    return load_yaml_config(config_path)


def _resolve_game_config(args: argparse.Namespace, config: Mapping[str, Any] | None) -> GameConfig:
    game_config = get_game_config(str(args.game))
    if config is None:
        return game_config

    raw_game = config.get("game")
    if raw_game is None:
        return game_config
    if not isinstance(raw_game, str) or not raw_game.strip():
        raise ValueError("Config file 'game' must be a non-empty string when provided")

    config_game = raw_game.strip().lower()
    if config_game != game_config.name:
        raise ValueError(
            "--game and --config disagree on game selection: "
            f"--game={game_config.name!r}, config.game={config_game!r}"
        )
    return game_config


def _resolve_network_shape(
    args: argparse.Namespace,
    config: Mapping[str, Any] | None,
) -> NetworkShapeConfig:
    default_num_blocks = 20
    default_num_filters = 256
    default_se_reduction = 4

    if config is not None:
        network = config.get("network", {})
        if network is None:
            network = {}
        if not isinstance(network, Mapping):
            raise ValueError("'network' section must be a mapping when --config is provided")

        default_num_blocks = int(network.get("num_blocks", default_num_blocks))
        default_num_filters = int(network.get("num_filters", default_num_filters))
        default_se_reduction = int(network.get("se_reduction", default_se_reduction))

    num_blocks = default_num_blocks if args.num_blocks is None else int(args.num_blocks)
    num_filters = default_num_filters if args.num_filters is None else int(args.num_filters)
    se_reduction = default_se_reduction if args.se_reduction is None else int(args.se_reduction)

    return NetworkShapeConfig(
        num_blocks=_coerce_positive_int("num_blocks", num_blocks),
        num_filters=_coerce_positive_int("num_filters", num_filters),
        se_reduction=_coerce_positive_int("se_reduction", se_reduction),
    )


def _resolve_batch_sizes(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shared = _parse_positive_int_csv("batch_sizes", args.batch_sizes) if args.batch_sizes else None

    if args.inference_batch_sizes:
        inference = _parse_positive_int_csv("inference_batch_sizes", args.inference_batch_sizes)
    elif shared is not None:
        inference = shared
    else:
        inference = DEFAULT_INFERENCE_BATCH_SIZES

    if args.training_batch_sizes:
        training = _parse_positive_int_csv("training_batch_sizes", args.training_batch_sizes)
    elif shared is not None:
        training = shared
    else:
        training = DEFAULT_TRAINING_BATCH_SIZES

    return inference, training


def _build_cpp_game_config(cpp: Any, game_name: str) -> Any:
    normalized = game_name.strip().lower()
    if normalized == "chess":
        return cpp.chess_game_config()
    if normalized == "go":
        return cpp.go_game_config()
    raise ValueError(f"Unsupported game {game_name!r}")


def _measure_loop(
    *,
    warmup_iterations: int,
    timed_iterations: int,
    run_iteration: Callable[[], None],
    synchronize: Callable[[], None] | None,
    perf_counter: Callable[[], float],
) -> float:
    for _ in range(warmup_iterations):
        run_iteration()
    if synchronize is not None:
        synchronize()

    start = perf_counter()
    for _ in range(timed_iterations):
        run_iteration()
    if synchronize is not None:
        synchronize()

    return max(perf_counter() - start, 1.0e-9)


def benchmark_inference(
    *,
    dependencies: RuntimeDependencies,
    game_config: GameConfig,
    network_shape: NetworkShapeConfig,
    batch_sizes: Sequence[int],
    iterations: int,
    warmup_iterations: int,
    use_mixed_precision: bool,
    device: str | None,
    random_seed: int,
    fail_fast: bool,
) -> tuple[InferenceBenchmarkResult, ...]:
    """Benchmark model forward throughput in positions/second."""

    if dependencies.torch is None or dependencies.ResNetSE is None:
        raise RuntimeError("inference benchmark requires torch + ResNetSE runtime dependencies")

    torch_module = dependencies.torch
    resolved_device = _resolve_torch_device(torch_module, device)
    torch_module.manual_seed(int(random_seed))

    model = dependencies.ResNetSE(
        game_config,
        num_blocks=network_shape.num_blocks,
        num_filters=network_shape.num_filters,
        se_reduction=network_shape.se_reduction,
    ).to(device=resolved_device)
    model.eval()

    rows, cols = game_config.board_shape
    synchronize = (
        (lambda: _cuda_synchronize(torch_module, resolved_device))
        if getattr(resolved_device, "type", "cpu") == "cuda"
        else None
    )

    results: list[InferenceBenchmarkResult] = []
    for batch_size in batch_sizes:
        _coerce_positive_int("batch_size", int(batch_size))

        try:
            inputs = torch_module.randn(
                int(batch_size),
                game_config.input_channels,
                rows,
                cols,
                dtype=torch_module.float32,
                device=resolved_device,
            )

            def run_iteration() -> None:
                with torch_module.no_grad():
                    with torch_module.autocast(
                        device_type=resolved_device.type,
                        dtype=torch_module.bfloat16,
                        enabled=use_mixed_precision and resolved_device.type == "cuda",
                    ):
                        model(inputs)

            elapsed = _measure_loop(
                warmup_iterations=warmup_iterations,
                timed_iterations=iterations,
                run_iteration=run_iteration,
                synchronize=synchronize,
                perf_counter=dependencies.perf_counter,
            )

            positions_per_second = (float(batch_size) * float(iterations)) / elapsed
            batch_latency_ms = (elapsed / float(iterations)) * 1000.0
            results.append(
                InferenceBenchmarkResult(
                    batch_size=int(batch_size),
                    positions_per_second=positions_per_second,
                    batch_latency_ms=batch_latency_ms,
                )
            )
        except Exception as exc:
            if fail_fast:
                raise
            if getattr(resolved_device, "type", "cpu") == "cuda":
                torch_module.cuda.empty_cache()
            results.append(
                InferenceBenchmarkResult(
                    batch_size=int(batch_size),
                    positions_per_second=0.0,
                    batch_latency_ms=0.0,
                    error=str(exc),
                )
            )

    return tuple(results)


def benchmark_training(
    *,
    dependencies: RuntimeDependencies,
    game_config: GameConfig,
    network_shape: NetworkShapeConfig,
    batch_sizes: Sequence[int],
    iterations: int,
    warmup_iterations: int,
    use_mixed_precision: bool,
    device: str | None,
    random_seed: int,
    fail_fast: bool,
) -> tuple[TrainingBenchmarkResult, ...]:
    """Benchmark one-step training throughput in steps/second."""

    if (
        dependencies.torch is None
        or dependencies.ResNetSE is None
        or dependencies.create_optimizer is None
        or dependencies.StepDecayLRSchedule is None
        or dependencies.train_one_step is None
    ):
        raise RuntimeError(
            "training benchmark requires torch + ResNetSE + trainer runtime dependencies"
        )

    torch_module = dependencies.torch
    resnet_cls = dependencies.ResNetSE
    create_optimizer = dependencies.create_optimizer
    schedule_cls = dependencies.StepDecayLRSchedule
    train_one_step = dependencies.train_one_step

    resolved_device = _resolve_torch_device(torch_module, device)
    torch_module.manual_seed(int(random_seed))

    model = resnet_cls(
        game_config,
        num_blocks=network_shape.num_blocks,
        num_filters=network_shape.num_filters,
        se_reduction=network_shape.se_reduction,
    ).to(device=resolved_device)
    model.train()

    lr_schedule = schedule_cls()
    optimizer = create_optimizer(model, lr_schedule=lr_schedule, momentum=0.9)
    scaler = _create_grad_scaler(
        torch_module,
        device=resolved_device,
        enabled=use_mixed_precision and resolved_device.type == "cuda",
    )

    rows, cols = game_config.board_shape
    action_space = game_config.action_space_size
    synchronize = (
        (lambda: _cuda_synchronize(torch_module, resolved_device))
        if getattr(resolved_device, "type", "cpu") == "cuda"
        else None
    )

    results: list[TrainingBenchmarkResult] = []
    global_step = 0

    for batch_size in batch_sizes:
        _coerce_positive_int("batch_size", int(batch_size))

        try:
            states = torch_module.randn(
                int(batch_size),
                game_config.input_channels,
                rows,
                cols,
                dtype=torch_module.float32,
                device=resolved_device,
            )
            policy_target = torch_module.rand(
                int(batch_size),
                action_space,
                dtype=torch_module.float32,
                device=resolved_device,
            )
            policy_target = policy_target / policy_target.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

            if game_config.value_head_type == "scalar":
                value_target = torch_module.empty(
                    int(batch_size),
                    dtype=torch_module.float32,
                    device=resolved_device,
                ).uniform_(-1.0, 1.0)
            else:
                value_target = torch_module.rand(
                    int(batch_size),
                    3,
                    dtype=torch_module.float32,
                    device=resolved_device,
                )
                value_target = value_target / value_target.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

            def run_iteration() -> None:
                nonlocal global_step
                train_one_step(
                    model,
                    optimizer,
                    states=states,
                    target_policy=policy_target,
                    target_value=value_target,
                    game_config=game_config,
                    lr_schedule=lr_schedule,
                    global_step=global_step,
                    l2_reg=1.0e-4,
                    scaler=scaler,
                    use_mixed_precision=use_mixed_precision and resolved_device.type == "cuda",
                )
                global_step += 1

            elapsed = _measure_loop(
                warmup_iterations=warmup_iterations,
                timed_iterations=iterations,
                run_iteration=run_iteration,
                synchronize=synchronize,
                perf_counter=dependencies.perf_counter,
            )

            steps_per_second = float(iterations) / elapsed
            positions_per_second = (float(batch_size) * float(iterations)) / elapsed
            step_latency_ms = (elapsed / float(iterations)) * 1000.0
            results.append(
                TrainingBenchmarkResult(
                    batch_size=int(batch_size),
                    steps_per_second=steps_per_second,
                    positions_per_second=positions_per_second,
                    step_latency_ms=step_latency_ms,
                )
            )
        except Exception as exc:
            if fail_fast:
                raise
            if getattr(resolved_device, "type", "cpu") == "cuda":
                torch_module.cuda.empty_cache()
            results.append(
                TrainingBenchmarkResult(
                    batch_size=int(batch_size),
                    steps_per_second=0.0,
                    positions_per_second=0.0,
                    step_latency_ms=0.0,
                    error=str(exc),
                )
            )

    return tuple(results)


def benchmark_mcts(
    *,
    dependencies: RuntimeDependencies,
    game_name: str,
    games: Sequence[int],
    threads: Sequence[int],
    simulations_per_move: int,
    node_arena_capacity: int,
    warmup_seconds: float,
    sample_seconds: float,
    random_seed: int,
    fail_fast: bool,
) -> tuple[MctsBenchmarkResult, ...]:
    """Benchmark MCTS throughput in simulations/second over games×threads grids."""

    if dependencies.cpp is None:
        raise RuntimeError("MCTS benchmark requires alphazero_cpp runtime dependencies")

    cpp = dependencies.cpp
    simulations_per_move = _coerce_positive_int("mcts_simulations_per_move", int(simulations_per_move))
    node_arena_capacity = _coerce_positive_int("node_arena_capacity", int(node_arena_capacity))
    warmup_seconds = _coerce_non_negative_float("mcts_warmup_seconds", warmup_seconds)
    sample_seconds = _coerce_non_negative_float("mcts_duration_seconds", sample_seconds)

    cpp_game_config = _build_cpp_game_config(cpp, game_name)
    action_space_size = int(cpp_game_config.action_space_size)
    policy_template = [0.0] * action_space_size

    def evaluator(_state: object) -> dict[str, object]:
        return {
            "policy": policy_template,
            "value": 0.0,
            "policy_is_logits": True,
        }

    results: list[MctsBenchmarkResult] = []

    for concurrent_games in games:
        _coerce_positive_int("games element", int(concurrent_games))
        for threads_per_game in threads:
            _coerce_positive_int("threads element", int(threads_per_game))

            try:
                replay_capacity = max(4096, int(concurrent_games) * 64)
                replay_buffer = cpp.ReplayBuffer(replay_capacity, int(random_seed))

                game_config = cpp.SelfPlayGameConfig()
                game_config.simulations_per_move = int(simulations_per_move)
                game_config.mcts_threads = int(threads_per_game)
                game_config.node_arena_capacity = int(node_arena_capacity)
                game_config.enable_dirichlet_noise = False
                game_config.temperature = 0.0
                game_config.temperature_moves = 0

                # Use always-on resignation with threshold 1.0 so each game completes quickly
                # while still exercising a full simulation batch per game.
                game_config.enable_resignation = True
                game_config.resign_threshold = 1.0
                game_config.resign_disable_fraction = 0.0
                game_config.random_seed = int(random_seed)

                manager_config = cpp.SelfPlayManagerConfig()
                manager_config.concurrent_games = int(concurrent_games)
                manager_config.max_games_per_slot = 0
                manager_config.initial_game_id = 1
                manager_config.random_seed = int(random_seed)
                manager_config.game_config = game_config

                manager = cpp.SelfPlayManager(cpp_game_config, replay_buffer, evaluator, manager_config)
                manager.start()
                try:
                    if warmup_seconds > 0.0:
                        dependencies.sleep(warmup_seconds)

                    start_metrics = manager.metrics()
                    start_time = dependencies.perf_counter()
                    dependencies.sleep(sample_seconds)
                    elapsed_seconds = max(dependencies.perf_counter() - start_time, 1.0e-9)
                    end_metrics = manager.metrics()
                finally:
                    manager.stop()

                if bool(getattr(end_metrics, "worker_failed", False)):
                    raise RuntimeError("SelfPlayManager worker failed during MCTS benchmark")

                delta_simulations = max(
                    0,
                    int(end_metrics.total_simulations) - int(start_metrics.total_simulations),
                )
                delta_moves = max(
                    0,
                    int(end_metrics.total_moves) - int(start_metrics.total_moves),
                )
                delta_games = max(
                    0,
                    int(end_metrics.games_completed) - int(start_metrics.games_completed),
                )

                results.append(
                    MctsBenchmarkResult(
                        concurrent_games=int(concurrent_games),
                        threads_per_game=int(threads_per_game),
                        simulations_per_move=int(simulations_per_move),
                        simulations_per_second=float(delta_simulations) / elapsed_seconds,
                        moves_per_second=float(delta_moves) / elapsed_seconds,
                        games_per_hour=(float(delta_games) / elapsed_seconds) * 3600.0,
                    )
                )
            except Exception as exc:
                if fail_fast:
                    raise
                results.append(
                    MctsBenchmarkResult(
                        concurrent_games=int(concurrent_games),
                        threads_per_game=int(threads_per_game),
                        simulations_per_move=int(simulations_per_move),
                        simulations_per_second=0.0,
                        moves_per_second=0.0,
                        games_per_hour=0.0,
                        error=str(exc),
                    )
                )

    return tuple(results)


def _format_number(value: float) -> str:
    return f"{value:,.2f}"


def _format_error(error: str | None) -> str:
    if error is None:
        return ""
    normalized = " ".join(error.split())
    return normalized if len(normalized) <= 80 else f"{normalized[:77]}..."


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return lines


def format_report(report: BenchmarkReport) -> str:
    """Render benchmark results as a human-readable table report."""

    lines = [
        f"Benchmark mode: {report.mode}",
        f"Game: {report.game}",
    ]
    if report.device is not None:
        lines.append(f"Device: {report.device}")

    if report.inference:
        lines.append("")
        lines.append("Inference Throughput (positions/sec)")
        inference_rows = [
            (
                str(result.batch_size),
                _format_number(result.positions_per_second) if result.error is None else "0.00",
                _format_number(result.batch_latency_ms) if result.error is None else "0.00",
                _format_error(result.error),
            )
            for result in report.inference
        ]
        lines.extend(
            _render_table(
                ("batch", "positions/sec", "latency_ms", "error"),
                inference_rows,
            )
        )

    if report.training:
        lines.append("")
        lines.append("Training Throughput (steps/sec)")
        training_rows = [
            (
                str(result.batch_size),
                _format_number(result.steps_per_second) if result.error is None else "0.00",
                _format_number(result.positions_per_second) if result.error is None else "0.00",
                _format_number(result.step_latency_ms) if result.error is None else "0.00",
                _format_error(result.error),
            )
            for result in report.training
        ]
        lines.extend(
            _render_table(
                ("batch", "steps/sec", "positions/sec", "latency_ms", "error"),
                training_rows,
            )
        )

    if report.mcts:
        lines.append("")
        lines.append("MCTS Throughput (simulations/sec)")
        mcts_rows = [
            (
                str(result.concurrent_games),
                str(result.threads_per_game),
                str(result.simulations_per_move),
                _format_number(result.simulations_per_second) if result.error is None else "0.00",
                _format_number(result.moves_per_second) if result.error is None else "0.00",
                _format_number(result.games_per_hour) if result.error is None else "0.00",
                _format_error(result.error),
            )
            for result in report.mcts
        ]
        lines.extend(
            _render_table(
                (
                    "games",
                    "threads",
                    "sims/move",
                    "sims/sec",
                    "moves/sec",
                    "games/hour",
                    "error",
                ),
                mcts_rows,
            )
        )

    return "\n".join(lines)


def run_from_args(
    args: argparse.Namespace,
    *,
    dependencies: RuntimeDependencies | None = None,
) -> BenchmarkReport:
    """Execute requested benchmark mode(s) from parsed CLI arguments."""

    mode = str(args.mode).strip().lower()
    if mode not in {"inference", "training", "mcts", "all"}:
        raise ValueError(f"Unsupported benchmark mode {mode!r}")

    active_dependencies = dependencies or load_runtime_dependencies(mode)

    iterations = _coerce_positive_int("iterations", int(args.iterations))
    warmup_iterations = _coerce_positive_int("warmup_iterations", int(args.warmup_iterations))
    random_seed = int(args.random_seed)
    fail_fast = bool(args.fail_fast)

    config = _resolve_config_mapping(args.config)
    game_config = _resolve_game_config(args, config)

    inference_results: tuple[InferenceBenchmarkResult, ...] = ()
    training_results: tuple[TrainingBenchmarkResult, ...] = ()
    mcts_results: tuple[MctsBenchmarkResult, ...] = ()
    resolved_device: str | None = None

    if mode in {"inference", "training", "all"}:
        network_shape = _resolve_network_shape(args, config)
        inference_batch_sizes, training_batch_sizes = _resolve_batch_sizes(args)
        use_mixed_precision = not bool(args.fp32)

        if mode in {"inference", "all"}:
            inference_results = benchmark_inference(
                dependencies=active_dependencies,
                game_config=game_config,
                network_shape=network_shape,
                batch_sizes=inference_batch_sizes,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
                use_mixed_precision=use_mixed_precision,
                device=args.device,
                random_seed=random_seed,
                fail_fast=fail_fast,
            )

        if mode in {"training", "all"}:
            training_results = benchmark_training(
                dependencies=active_dependencies,
                game_config=game_config,
                network_shape=network_shape,
                batch_sizes=training_batch_sizes,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
                use_mixed_precision=use_mixed_precision,
                device=args.device,
                random_seed=random_seed,
                fail_fast=fail_fast,
            )

        if active_dependencies.torch is not None:
            torch_module = active_dependencies.torch
            resolved_device = str(_resolve_torch_device(torch_module, args.device))

    if mode in {"mcts", "all"}:
        games = _parse_positive_int_csv("games", str(args.games))
        threads = _parse_positive_int_csv("threads", str(args.threads))

        mcts_results = benchmark_mcts(
            dependencies=active_dependencies,
            game_name=game_config.name,
            games=games,
            threads=threads,
            simulations_per_move=int(args.mcts_simulations_per_move),
            node_arena_capacity=int(args.node_arena_capacity),
            warmup_seconds=float(args.mcts_warmup_seconds),
            sample_seconds=float(args.mcts_duration_seconds),
            random_seed=random_seed,
            fail_fast=fail_fast,
        )

    return BenchmarkReport(
        mode=mode,
        game=game_config.name,
        device=resolved_device,
        inference=inference_results,
        training=training_results,
        mcts=mcts_results,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        report = run_from_args(args)
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
