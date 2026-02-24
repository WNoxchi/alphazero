#!/usr/bin/env python3
"""Main training entry point for AlphaZero."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
import signal
import sys
from typing import Any, Callable, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))
BUILD_SRC = ROOT / "build" / "src"
if str(BUILD_SRC) not in sys.path:
    sys.path.insert(0, str(BUILD_SRC))

from alphazero.config import GameConfig, get_game_config, load_yaml_config


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


def _coerce_numeric(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    return float(value)


def _coerce_positive_float(name: str, value: object) -> float:
    numeric = _coerce_numeric(name, value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be > 0, got {numeric}")
    return numeric


def _coerce_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
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
    """Runtime-callable dependencies for the training script."""

    cpp: Any
    ResNetSE: Any
    create_optimizer: Callable[..., Any]
    load_lr_schedule_from_config: Callable[[Mapping[str, Any]], Any]
    load_pipeline_config_from_config: Callable[[Mapping[str, Any]], Any]
    load_training_checkpoint: Callable[..., tuple[int, Any]]
    load_training_config_from_config: Callable[[Mapping[str, Any]], Any]
    make_eval_queue_batch_evaluator: Callable[..., Any]
    make_selfplay_evaluator_from_eval_queue: Callable[..., Any]
    run_parallel_pipeline: Callable[..., Any]
    save_training_checkpoint: Callable[..., Any]
    build_run_name: Callable[[str], str]
    create_metrics_logger: Callable[..., Any]


@dataclass(slots=True)
class TrainingRuntime:
    """Constructed runtime state required for one training run."""

    config_path: Path
    config: Mapping[str, Any]
    game_config: GameConfig
    model: Any
    replay_buffer: Any
    eval_queue: Any
    self_play_manager: Any
    training_config: Any
    pipeline_config: Any
    lr_schedule: Any
    optimizer: Any
    logger: Any
    start_step: int


@dataclass(frozen=True, slots=True)
class TrainingRunSummary:
    """Terminal summary from a train.py invocation."""

    final_step: int
    interrupted: bool
    games_completed: int
    final_checkpoint_path: Path | None


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for training runs."""

    parser = argparse.ArgumentParser(description="Run AlphaZero training.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    return parser


def _import_cpp_bindings() -> Any:
    try:
        import alphazero_cpp  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "alphazero_cpp extension is unavailable. Build the project first with "
            "`cmake --build build` before running scripts/train.py."
        ) from exc
    return alphazero_cpp


def load_runtime_dependencies() -> RuntimeDependencies:
    """Import runtime dependencies lazily to keep module import lightweight."""

    from alphazero.network import ResNetSE
    from alphazero.pipeline.orchestrator import (
        load_pipeline_config_from_config,
        make_eval_queue_batch_evaluator,
        make_selfplay_evaluator_from_eval_queue,
        run_parallel_pipeline,
    )
    from alphazero.training.lr_schedule import load_lr_schedule_from_config
    from alphazero.training.trainer import (
        create_optimizer,
        load_training_checkpoint,
        load_training_config_from_config,
        save_training_checkpoint,
    )
    from alphazero.utils.logging import build_run_name, create_metrics_logger

    return RuntimeDependencies(
        cpp=_import_cpp_bindings(),
        ResNetSE=ResNetSE,
        create_optimizer=create_optimizer,
        load_lr_schedule_from_config=load_lr_schedule_from_config,
        load_pipeline_config_from_config=load_pipeline_config_from_config,
        load_training_checkpoint=load_training_checkpoint,
        load_training_config_from_config=load_training_config_from_config,
        make_eval_queue_batch_evaluator=make_eval_queue_batch_evaluator,
        make_selfplay_evaluator_from_eval_queue=make_selfplay_evaluator_from_eval_queue,
        run_parallel_pipeline=run_parallel_pipeline,
        save_training_checkpoint=save_training_checkpoint,
        build_run_name=build_run_name,
        create_metrics_logger=create_metrics_logger,
    )


def _resolve_game_config(config: Mapping[str, Any]) -> GameConfig:
    raw_game = config.get("game")
    if not isinstance(raw_game, str) or not raw_game.strip():
        raise ValueError("Config must define a non-empty 'game' string")
    return get_game_config(raw_game)


def _build_model(
    dependencies: RuntimeDependencies,
    config: Mapping[str, Any],
    game_config: GameConfig,
) -> Any:
    network = _section(config, "network")
    architecture = network.get("architecture", "resnet_se")
    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError("network.architecture must be a non-empty string")
    if architecture.strip().lower() != "resnet_se":
        raise ValueError(
            f"Unsupported network architecture {architecture!r}; only 'resnet_se' is supported"
        )

    return dependencies.ResNetSE(
        game_config,
        num_blocks=int(network.get("num_blocks", 20)),
        num_filters=int(network.get("num_filters", 256)),
        se_reduction=int(network.get("se_reduction", 4)),
    )


def _resolve_training_config(
    dependencies: RuntimeDependencies,
    config: Mapping[str, Any],
) -> Any:
    training_config = dependencies.load_training_config_from_config(config)
    training = _section(config, "training")
    system = _section(config, "system")

    precision = system.get("precision", "bf16")
    if not isinstance(precision, str) or not precision.strip():
        raise ValueError("system.precision must be a non-empty string")
    normalized_precision = precision.strip().lower()
    if normalized_precision not in {"bf16", "fp32"}:
        raise ValueError(
            f"Unsupported system.precision {precision!r}; expected 'bf16' or 'fp32'"
        )
    use_mixed_precision = normalized_precision == "bf16"

    wait_for_buffer_seconds = training_config.wait_for_buffer_seconds
    if "wait_for_buffer_seconds" in training:
        wait_for_buffer_seconds = _coerce_positive_float(
            "training.wait_for_buffer_seconds",
            training["wait_for_buffer_seconds"],
        )

    export_folded_checkpoints = training_config.export_folded_checkpoints
    if "export_folded_checkpoints" in training:
        export_folded_checkpoints = _coerce_bool(
            "training.export_folded_checkpoints",
            training["export_folded_checkpoints"],
        )

    device = training_config.device
    if "device" in system:
        raw_device = system["device"]
        if raw_device is not None and not isinstance(raw_device, str):
            raise TypeError(
                f"system.device must be a string or null, got {type(raw_device).__name__}"
            )
        device = raw_device

    return replace(
        training_config,
        use_mixed_precision=use_mixed_precision,
        wait_for_buffer_seconds=wait_for_buffer_seconds,
        export_folded_checkpoints=export_folded_checkpoints,
        device=device,
    )


def _resolve_compile_model(config: Mapping[str, Any]) -> bool:
    system = _section(config, "system")
    return _coerce_bool("system.compile", system.get("compile", True))


def _warmup_compiled_model(
    model: Any,
    game_config: GameConfig,
    *,
    training_batch_size: int,
    eval_batch_size: int,
    device: Any,
    use_mixed_precision: bool,
) -> None:
    """Pre-compile forward/backward graphs for both inference and training shapes.

    torch.compile compiles lazily on the first forward pass for each distinct
    input shape and mode.  Running warmup passes before the pipeline threads
    start avoids holding the GIL during lazy compilation while self-play is
    running, which would block inference and stall the entire pipeline.
    """
    import torch

    rows, cols = game_config.board_shape
    channels = game_config.input_channels

    print(
        f"Warming up compiled model "
        f"(eval batch={eval_batch_size}, train batch={training_batch_size})..."
    )

    # The model stays in train() mode for the entire pipeline lifetime so
    # that torch.compile only ever sees one set of dynamo guards per shape.
    # Toggling eval()/train() would invalidate guards and trigger expensive
    # recompilation that holds the GIL and stalls the pipeline.

    # 1. Inference shape — forward only, train mode with no_grad
    #    (matches the evaluator's execution context).
    model.train()
    dummy_inf = torch.zeros(eval_batch_size, channels, rows, cols, device=device)
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_mixed_precision,
        ):
            model(dummy_inf)

    # 2. Training shape — forward + backward, train mode.
    dummy_train = torch.zeros(
        training_batch_size, channels, rows, cols, device=device
    )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_mixed_precision,
    ):
        policy, value = model(dummy_train)
        loss = policy.sum() + value.sum()
        loss.backward()
    model.zero_grad(set_to_none=True)

    print("Model warmup complete.")


def _resolve_run_name(
    dependencies: RuntimeDependencies,
    config: Mapping[str, Any],
    game_name: str,
) -> str:
    system = _section(config, "system")
    run_name = system.get("run_name")
    if run_name is None:
        return dependencies.build_run_name(game_name)
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("system.run_name must be a non-empty string when provided")
    return run_name.strip()


def _build_cpp_game_config(cpp: Any, game_name: str) -> Any:
    normalized = game_name.strip().lower()
    if normalized == "chess":
        return cpp.chess_game_config()
    if normalized == "go":
        return cpp.go_game_config()
    raise ValueError(f"Unsupported game {game_name!r}")


def _build_replay_buffer(cpp: Any, config: Mapping[str, Any]) -> Any:
    replay = _section(config, "replay_buffer")
    capacity = _coerce_positive_int(
        "replay_buffer.capacity",
        int(replay.get("capacity", 1_000_000)),
    )
    random_seed = _coerce_non_negative_int(
        "replay_buffer.random_seed",
        int(replay.get("random_seed", 0x9E3779B97F4A7C15)),
    )
    return cpp.ReplayBuffer(capacity=capacity, random_seed=random_seed)


def _build_eval_queue_config(cpp: Any, config: Mapping[str, Any]) -> Any:
    mcts = _section(config, "mcts")
    queue_config = cpp.EvalQueueConfig()
    queue_config.batch_size = _coerce_positive_int(
        "mcts.batch_size",
        int(mcts.get("batch_size", queue_config.batch_size)),
    )
    if "eval_queue_flush_timeout_us" in mcts:
        queue_config.flush_timeout_us = _coerce_non_negative_int(
            "mcts.eval_queue_flush_timeout_us",
            int(mcts["eval_queue_flush_timeout_us"]),
        )
    return queue_config


def _build_selfplay_manager_config(cpp: Any, config: Mapping[str, Any]) -> Any:
    mcts = _section(config, "mcts")
    manager_config = cpp.SelfPlayManagerConfig()

    manager_config.concurrent_games = _coerce_positive_int(
        "mcts.concurrent_games",
        int(mcts.get("concurrent_games", manager_config.concurrent_games)),
    )
    manager_config.max_games_per_slot = _coerce_non_negative_int(
        "mcts.max_games_per_slot",
        int(mcts.get("max_games_per_slot", manager_config.max_games_per_slot)),
    )
    manager_config.initial_game_id = _coerce_positive_int(
        "mcts.initial_game_id",
        int(mcts.get("initial_game_id", manager_config.initial_game_id)),
    )
    manager_config.random_seed = _coerce_non_negative_int(
        "mcts.random_seed",
        int(mcts.get("random_seed", manager_config.random_seed)),
    )

    game_config = manager_config.game_config
    game_config.simulations_per_move = _coerce_positive_int(
        "mcts.simulations_per_move",
        int(mcts.get("simulations_per_move", game_config.simulations_per_move)),
    )
    game_config.mcts_threads = _coerce_positive_int(
        "mcts.threads_per_game",
        int(mcts.get("threads_per_game", game_config.mcts_threads)),
    )
    game_config.node_arena_capacity = _coerce_positive_int(
        "mcts.node_arena_capacity",
        int(mcts.get("node_arena_capacity", game_config.node_arena_capacity)),
    )
    game_config.c_puct = _coerce_positive_float(
        "mcts.c_puct",
        float(mcts.get("c_puct", game_config.c_puct)),
    )
    game_config.c_fpu = _coerce_numeric(
        "mcts.c_fpu",
        float(mcts.get("c_fpu", game_config.c_fpu)),
    )
    game_config.enable_dirichlet_noise = _coerce_bool(
        "mcts.enable_dirichlet_noise",
        mcts.get("enable_dirichlet_noise", game_config.enable_dirichlet_noise),
    )
    game_config.dirichlet_epsilon = _coerce_numeric(
        "mcts.dirichlet_epsilon",
        float(mcts.get("dirichlet_epsilon", game_config.dirichlet_epsilon)),
    )
    if "dirichlet_alpha" in mcts:
        game_config.dirichlet_alpha_override = _coerce_positive_float(
            "mcts.dirichlet_alpha",
            float(mcts["dirichlet_alpha"]),
        )
    game_config.temperature = _coerce_positive_float(
        "mcts.temperature",
        float(mcts.get("temperature", game_config.temperature)),
    )
    game_config.temperature_moves = _coerce_non_negative_int(
        "mcts.temperature_moves",
        int(mcts.get("temperature_moves", game_config.temperature_moves)),
    )
    game_config.enable_resignation = _coerce_bool(
        "mcts.enable_resignation",
        mcts.get("enable_resignation", game_config.enable_resignation),
    )
    game_config.resign_threshold = _coerce_numeric(
        "mcts.resign_threshold",
        float(mcts.get("resign_threshold", game_config.resign_threshold)),
    )
    game_config.resign_disable_fraction = _coerce_numeric(
        "mcts.resign_disable_fraction",
        float(mcts.get("resign_disable_fraction", game_config.resign_disable_fraction)),
    )
    game_config.random_seed = _coerce_non_negative_int(
        "mcts.search_random_seed",
        int(mcts.get("search_random_seed", game_config.random_seed)),
    )

    if game_config.resign_disable_fraction < 0.0 or game_config.resign_disable_fraction > 1.0:
        raise ValueError("mcts.resign_disable_fraction must be in [0, 1]")
    return manager_config


def build_training_runtime(
    *,
    config_path: str | Path,
    resume_path: str | Path | None,
    dependencies: RuntimeDependencies | None = None,
    config_override: Mapping[str, Any] | None = None,
) -> TrainingRuntime:
    """Build all runtime objects needed for a training run."""

    active_dependencies = dependencies or load_runtime_dependencies()
    resolved_config_path = Path(config_path)
    config = config_override if config_override is not None else load_yaml_config(resolved_config_path)
    if not isinstance(config, Mapping):
        raise TypeError(f"Config must be a mapping, got {type(config).__name__}")

    game_config = _resolve_game_config(config)
    model = _build_model(active_dependencies, config, game_config)
    training_config = _resolve_training_config(active_dependencies, config)
    pipeline_config = active_dependencies.load_pipeline_config_from_config(config)
    lr_schedule = active_dependencies.load_lr_schedule_from_config(config)
    optimizer = active_dependencies.create_optimizer(
        model,
        lr_schedule=lr_schedule,
        momentum=float(training_config.momentum),
    )

    start_step = 0
    if resume_path is not None:
        resolved_resume = Path(resume_path)
        if not resolved_resume.exists():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {resolved_resume}"
            )
        start_step, lr_schedule = active_dependencies.load_training_checkpoint(
            resolved_resume,
            model,
            optimizer,
            map_location="cpu",
        )

    cpp = active_dependencies.cpp
    replay_buffer = _build_replay_buffer(cpp, config)
    eval_queue_config = _build_eval_queue_config(cpp, config)
    selfplay_manager_config = _build_selfplay_manager_config(cpp, config)

    if _resolve_compile_model(config):
        import torch

        model = torch.compile(model, mode="default")
        device = training_config.device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        model.to(device=device)
        _warmup_compiled_model(
            model,
            game_config,
            training_batch_size=int(training_config.batch_size),
            eval_batch_size=int(eval_queue_config.batch_size),
            device=device,
            use_mixed_precision=bool(training_config.use_mixed_precision),
        )

    rows, cols = game_config.board_shape
    encoded_state_size = game_config.input_channels * rows * cols
    eval_batch_evaluator = active_dependencies.make_eval_queue_batch_evaluator(
        model,
        game_config,
        device=training_config.device,
        use_mixed_precision=bool(training_config.use_mixed_precision),
        max_batch_size=eval_queue_config.batch_size,
    )
    eval_queue = cpp.EvalQueue(
        evaluator=eval_batch_evaluator,
        encoded_state_size=encoded_state_size,
        config=eval_queue_config,
    )
    self_play_manager = cpp.SelfPlayManager(
        _build_cpp_game_config(cpp, game_config.name),
        replay_buffer,
        eval_queue,
        selfplay_manager_config,
    )

    run_name = _resolve_run_name(active_dependencies, config, game_config.name)
    logger = active_dependencies.create_metrics_logger(run_name=run_name, config=config)

    return TrainingRuntime(
        config_path=resolved_config_path,
        config=config,
        game_config=game_config,
        model=model,
        replay_buffer=replay_buffer,
        eval_queue=eval_queue,
        self_play_manager=self_play_manager,
        training_config=training_config,
        pipeline_config=pipeline_config,
        lr_schedule=lr_schedule,
        optimizer=optimizer,
        logger=logger,
        start_step=start_step,
    )


@contextmanager
def _raise_keyboard_interrupt_on_signal() -> Iterator[None]:
    previous_handlers: dict[signal.Signals, Any] = {}

    def _handle_signal(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    handled_signals: list[signal.Signals] = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        handled_signals.append(signal.SIGTERM)

    for handled_signal in handled_signals:
        previous_handlers[handled_signal] = signal.getsignal(handled_signal)
        signal.signal(handled_signal, _handle_signal)
    try:
        yield
    finally:
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)


def _maybe_save_final_checkpoint(
    runtime: TrainingRuntime,
    dependencies: RuntimeDependencies,
    *,
    final_step: int,
) -> Path | None:
    checkpoint_dir = runtime.training_config.checkpoint_dir
    if checkpoint_dir is None or final_step <= 0:
        return None

    saved = dependencies.save_training_checkpoint(
        runtime.model,
        runtime.optimizer,
        runtime.lr_schedule,
        step=final_step,
        checkpoint_dir=Path(checkpoint_dir),
        replay_buffer=runtime.replay_buffer,
        keep_last=int(runtime.training_config.checkpoint_keep_last),
        is_milestone=False,
        export_folded_weights=bool(runtime.training_config.export_folded_checkpoints),
    )
    return Path(saved.checkpoint_path)


def run_training_session(
    runtime: TrainingRuntime,
    *,
    dependencies: RuntimeDependencies | None = None,
) -> TrainingRunSummary:
    """Execute one training session and ensure graceful finalization."""

    from datetime import datetime, timedelta

    from tqdm import tqdm

    active_dependencies = dependencies or load_runtime_dependencies()

    latest_step = runtime.start_step
    pipeline_result: Any | None = None
    interrupted = False
    final_checkpoint_path: Path | None = None
    pipeline_error: BaseException | None = None

    max_steps = int(runtime.training_config.max_steps)
    start_step = runtime.start_step
    session_start = datetime.now()

    # Persistent progress bar pinned to the bottom of the terminal.
    # leave=False ensures the bar is cleared on close (no trailing artifacts).
    pbar = tqdm(
        total=max_steps,
        initial=start_step,
        unit="step",
        bar_format=(
            "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
        ),
        desc="Training",
        dynamic_ncols=True,
        leave=False,
        smoothing=0.05,
    )
    pbar.set_postfix_str(f" Start {session_start:%H:%M}")

    # Route console summaries through tqdm.write() so the progress bar stays
    # pinned at the bottom while step logs scroll above it.
    console_interval = runtime.logger._console_summary_interval_steps

    def _update_bar(step: int) -> None:
        pbar.n = step
        elapsed = (datetime.now() - session_start).total_seconds()
        progress = step - start_step
        if progress > 0 and elapsed > 0:
            remaining_steps = max_steps - step
            remaining_secs = remaining_steps * elapsed / progress
            end_time = datetime.now() + timedelta(seconds=remaining_secs)
            pbar.set_postfix_str(
                f" Start {session_start:%H:%M} | End ~{end_time:%H:%M}"
            )
        pbar.refresh()

    def games_total_getter() -> int:
        snapshot = runtime.self_play_manager.metrics()
        return int(snapshot.games_completed)

    base_step_logger = runtime.logger.make_training_step_logger(
        games_total_getter=games_total_getter,
        emit_console=False,
    )

    def step_logger(step: int, metrics: Mapping[str, float]) -> None:
        nonlocal latest_step
        latest_step = max(latest_step, int(step))
        base_step_logger(step, metrics)
        _update_bar(step)
        if step % console_interval == 0:
            try:
                summary = runtime.logger.render_console_summary(step)
                tqdm.write(summary)
            except RuntimeError:
                pass

    def cycle_logger(_cycle: int, metrics: Mapping[str, float]) -> None:
        nonlocal latest_step
        maybe_step = metrics.get("pipeline/global_step", latest_step)
        latest_step = max(latest_step, int(maybe_step))
        runtime.logger.log_selfplay_snapshot(latest_step, runtime.self_play_manager.metrics())
        for tag, value in metrics.items():
            runtime.logger.log_scalar(tag, value, latest_step)
        _update_bar(latest_step)

    try:
        with _raise_keyboard_interrupt_on_signal():
            pipeline_result = active_dependencies.run_parallel_pipeline(
                runtime.model,
                runtime.replay_buffer,
                runtime.game_config,
                runtime.eval_queue,
                runtime.self_play_manager,
                runtime.training_config,
                runtime.pipeline_config,
                lr_schedule=runtime.lr_schedule,
                optimizer=runtime.optimizer,
                step_logger=step_logger,
                cycle_logger=cycle_logger,
                start_step=runtime.start_step,
            )
    except KeyboardInterrupt:
        interrupted = True
    except BaseException as exc:  # pragma: no cover - exercised in runtime failure conditions.
        pipeline_error = exc
    finally:
        pbar.close()
        if pipeline_result is not None:
            latest_step = max(latest_step, int(pipeline_result.final_step))

        if pipeline_result is not None or interrupted:
            final_checkpoint_path = _maybe_save_final_checkpoint(
                runtime,
                active_dependencies,
                final_step=latest_step,
            )

        try:
            runtime.logger.log_selfplay_snapshot(latest_step, runtime.self_play_manager.metrics())
        except Exception:
            # Metrics snapshots are best-effort during teardown.
            pass
        runtime.logger.flush()
        runtime.logger.close()

    if pipeline_error is not None:
        raise pipeline_error

    games_completed = int(runtime.self_play_manager.metrics().games_completed)
    return TrainingRunSummary(
        final_step=latest_step,
        interrupted=interrupted,
        games_completed=games_completed,
        final_checkpoint_path=final_checkpoint_path,
    )


def run_from_args(args: argparse.Namespace) -> TrainingRunSummary:
    dependencies = load_runtime_dependencies()
    runtime = build_training_runtime(
        config_path=args.config,
        resume_path=args.resume,
        dependencies=dependencies,
    )
    return run_training_session(runtime, dependencies=dependencies)


def _print_summary(summary: TrainingRunSummary) -> None:
    run_state = "interrupted" if summary.interrupted else "completed"
    print(f"Training {run_state} at step {summary.final_step}.")
    print(f"Self-play games completed: {summary.games_completed}.")
    if summary.final_checkpoint_path is not None:
        print(f"Final checkpoint: {summary.final_checkpoint_path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_from_args(args)
    except (FileNotFoundError, ModuleNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.__cause__ is not None:
            import traceback
            traceback.print_exception(type(exc.__cause__), exc.__cause__, exc.__cause__.__traceback__, file=sys.stderr)
        return 2

    _print_summary(summary)
    return 130 if summary.interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
