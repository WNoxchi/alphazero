"""Interleaved GPU scheduling for AlphaZero self-play and training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from alphazero.config import GameConfig, load_yaml_config


DEFAULT_INFERENCE_BATCHES_PER_CYCLE = 50
DEFAULT_TRAINING_STEPS_PER_CYCLE = 1


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


def _coerce_non_negative_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    converted = float(value)
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative, got {converted}")
    return converted


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Runtime controls for the interleaved inference/training scheduler."""

    inference_batches_per_cycle: int = DEFAULT_INFERENCE_BATCHES_PER_CYCLE
    training_steps_per_cycle: int = DEFAULT_TRAINING_STEPS_PER_CYCLE
    max_cycles: int | None = None

    def __post_init__(self) -> None:
        _coerce_positive_int(
            "inference_batches_per_cycle",
            self.inference_batches_per_cycle,
        )
        _coerce_positive_int(
            "training_steps_per_cycle",
            self.training_steps_per_cycle,
        )
        if self.max_cycles is not None:
            _coerce_positive_int("max_cycles", self.max_cycles)


@dataclass(frozen=True, slots=True)
class InterleavedCycleMetrics:
    """Per-cycle metrics emitted by the interleaved scheduler."""

    cycle: int
    global_step: int
    inference_batches: int
    training_steps: int
    inference_seconds: float
    training_seconds: float
    cycle_seconds: float

    def as_dict(self) -> dict[str, float]:
        return {
            "pipeline/cycle": float(self.cycle),
            "pipeline/global_step": float(self.global_step),
            "pipeline/inference_batches": float(self.inference_batches),
            "pipeline/training_steps": float(self.training_steps),
            "pipeline/inference_seconds": self.inference_seconds,
            "pipeline/training_seconds": self.training_seconds,
            "pipeline/cycle_seconds": self.cycle_seconds,
        }


@dataclass(frozen=True, slots=True)
class InterleavedScheduleResult:
    """Summary of an interleaved schedule execution."""

    final_step: int
    cycles_completed: int
    inference_batches_processed: int
    training_steps_completed: int
    terminated_early: bool


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Pipeline execution summary including checkpoints created during training."""

    final_step: int
    cycles_completed: int
    inference_batches_processed: int
    training_steps_completed: int
    checkpoints: tuple[Any, ...]
    terminated_early: bool


class EvalQueueLike(Protocol):
    """Runtime protocol for batched inference queue implementations."""

    def process_batch(self) -> None:
        ...

    def stop(self) -> None:
        ...


class SelfPlayManagerLike(Protocol):
    """Runtime protocol for self-play manager implementations."""

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


class ReplayBufferLike(Protocol):
    """Runtime protocol for replay buffers consumed by training."""

    def size(self) -> int:
        ...

    def sample(self, batch_size: int) -> Sequence[object]:
        ...


class TrainingConfigLike(Protocol):
    """Subset of training config attributes required by pipeline orchestration."""

    batch_size: int
    max_steps: int
    momentum: float
    l2_reg: float
    checkpoint_interval: int
    checkpoint_keep_last: int
    milestone_interval: int
    log_interval: int
    min_buffer_size: int
    wait_for_buffer_seconds: float
    use_mixed_precision: bool
    device: str | Any | None
    checkpoint_dir: Path | None
    export_folded_checkpoints: bool


StepLogger = Callable[[int, Mapping[str, float]], None]
CycleLogger = Callable[[int, Mapping[str, float]], None]


def load_pipeline_config_from_config(config: Mapping[str, Any]) -> PipelineConfig:
    """Load :class:`PipelineConfig` from a parsed pipeline configuration mapping."""

    if not isinstance(config, Mapping):
        raise TypeError(f"config must be a mapping, got {type(config).__name__}")

    pipeline = config.get("pipeline", {})
    if pipeline is None:
        pipeline = {}
    if not isinstance(pipeline, Mapping):
        raise ValueError("'pipeline' section must be a mapping")

    max_cycles = pipeline.get("max_cycles")
    normalized_max_cycles: int | None
    if max_cycles is None:
        normalized_max_cycles = None
    else:
        normalized_max_cycles = int(max_cycles)

    return PipelineConfig(
        inference_batches_per_cycle=int(
            pipeline.get(
                "inference_batches_per_cycle",
                DEFAULT_INFERENCE_BATCHES_PER_CYCLE,
            )
        ),
        training_steps_per_cycle=int(
            pipeline.get(
                "training_steps_per_cycle",
                DEFAULT_TRAINING_STEPS_PER_CYCLE,
            )
        ),
        max_cycles=normalized_max_cycles,
    )


def load_pipeline_config_from_yaml(config_path: str | Path) -> PipelineConfig:
    """Load :class:`PipelineConfig` directly from a pipeline YAML file."""

    return load_pipeline_config_from_config(load_yaml_config(config_path))


def run_interleaved_schedule(
    *,
    inference_batch_fn: Callable[[], None],
    training_step_fn: Callable[[int], bool],
    max_steps: int,
    pipeline_config: PipelineConfig,
    start_step: int = 0,
    cycle_logger: CycleLogger | None = None,
) -> InterleavedScheduleResult:
    """Execute an S:T interleaving schedule until max steps or cycle limit is reached."""

    _coerce_non_negative_int("max_steps", max_steps)
    _coerce_non_negative_int("start_step", start_step)
    if start_step > max_steps:
        raise ValueError(f"start_step ({start_step}) cannot exceed max_steps ({max_steps})")

    step = start_step
    cycle = 0
    total_inference_batches = 0
    total_training_steps = 0

    while step < max_steps:
        if pipeline_config.max_cycles is not None and cycle >= pipeline_config.max_cycles:
            break

        cycle += 1
        cycle_start = time.perf_counter()

        inference_start = time.perf_counter()
        for _ in range(pipeline_config.inference_batches_per_cycle):
            inference_batch_fn()
            total_inference_batches += 1
        inference_seconds = max(time.perf_counter() - inference_start, 0.0)

        training_start = time.perf_counter()
        training_steps_this_cycle = 0
        for _ in range(pipeline_config.training_steps_per_cycle):
            if step >= max_steps:
                break
            executed = training_step_fn(step)
            if not isinstance(executed, bool):
                raise TypeError(
                    "training_step_fn must return bool "
                    f"(whether a step was executed), got {type(executed).__name__}"
                )
            if not executed:
                break
            step += 1
            training_steps_this_cycle += 1
            total_training_steps += 1
        training_seconds = max(time.perf_counter() - training_start, 0.0)
        cycle_seconds = max(time.perf_counter() - cycle_start, 0.0)

        if cycle_logger is not None:
            cycle_metrics = InterleavedCycleMetrics(
                cycle=cycle,
                global_step=step,
                inference_batches=pipeline_config.inference_batches_per_cycle,
                training_steps=training_steps_this_cycle,
                inference_seconds=inference_seconds,
                training_seconds=training_seconds,
                cycle_seconds=cycle_seconds,
            )
            cycle_logger(cycle, cycle_metrics.as_dict())

    terminated_early = step < max_steps and pipeline_config.max_cycles is not None and cycle >= pipeline_config.max_cycles
    return InterleavedScheduleResult(
        final_step=step,
        cycles_completed=cycle,
        inference_batches_processed=total_inference_batches,
        training_steps_completed=total_training_steps,
        terminated_early=terminated_early,
    )


def _resolve_torch_device(device: str | Any | None) -> Any:
    import torch

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _create_grad_scaler(*, device: Any, enabled: bool) -> Any:
    import torch

    try:
        return torch.amp.GradScaler(device=device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def make_eval_queue_batch_evaluator(
    model: Any,
    game_config: GameConfig,
    *,
    device: str | Any | None = None,
    use_mixed_precision: bool = True,
    max_batch_size: int | None = None,
) -> Callable[[Any], tuple[Any, Any]]:
    """Build a contiguous-batch evaluator callback compatible with ``alphazero_cpp.EvalQueue``."""

    import torch

    rows, cols = game_config.board_shape
    encoded_state_size = game_config.input_channels * rows * cols
    resolved_device = _resolve_torch_device(device)
    model = model.to(device=resolved_device)
    # Dedicated inference stream allows eval-queue batches to overlap with
    # training kernels launched on the default stream.
    inference_stream = (
        torch.cuda.Stream(device=resolved_device)
        if resolved_device.type == "cuda"
        else None
    )

    def evaluator(encoded_states: Any) -> tuple[Any, Any]:
        from contextlib import nullcontext
        import numpy as np

        encoded_states_array = np.asarray(encoded_states, dtype=np.float32)
        if encoded_states_array.ndim != 2:
            raise ValueError(
                "EvalQueue evaluator expects rank-2 state input "
                f"(batch, encoded_state_size), got shape {tuple(encoded_states_array.shape)}"
            )
        if int(encoded_states_array.shape[1]) != encoded_state_size:
            raise ValueError(
                "EvalQueue evaluator received incorrect encoded-state size: "
                f"expected {encoded_state_size}, got {int(encoded_states_array.shape[1])}"
            )
        if not encoded_states_array.flags.c_contiguous:
            encoded_states_array = np.ascontiguousarray(encoded_states_array)

        batch_size = int(encoded_states_array.shape[0])
        if batch_size == 0:
            return (
                np.empty((0, game_config.action_space_size), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        # Pad to fixed batch size so torch.compile/CUDAGraph only records one graph.
        if max_batch_size is not None and batch_size < max_batch_size:
            pad_rows = max_batch_size - batch_size
            encoded_states_array = np.pad(
                encoded_states_array, ((0, pad_rows), (0, 0)), mode="constant"
            )

        stream_context = (
            torch.cuda.stream(inference_stream)
            if inference_stream is not None
            else nullcontext()
        )
        # The model stays in train() mode for the entire pipeline lifetime.
        # Toggling eval()/train() on a torch.compile'd model invalidates
        # dynamo guards and triggers expensive recompilation, which holds the
        # GIL and stalls the pipeline.  With torch.no_grad() the only
        # behavioural difference is BatchNorm (batch stats vs running stats);
        # the slight noise is acceptable for self-play evaluation.
        with stream_context:
            flat_states = torch.from_numpy(encoded_states_array)
            if resolved_device.type != "cpu":
                flat_states = flat_states.to(device=resolved_device, non_blocking=True)
            padded_size = int(flat_states.shape[0])
            model_inputs = flat_states.reshape(padded_size, game_config.input_channels, rows, cols)

            with torch.no_grad():
                with torch.autocast(
                    device_type=resolved_device.type,
                    dtype=torch.bfloat16,
                    enabled=use_mixed_precision,
                ):
                    policy_logits, value = model(model_inputs)

            # Slice off padding before returning results.
            # All GPU ops stay on the inference stream so the subsequent
            # D2H copies are properly ordered.
            policy_logits = policy_logits[:batch_size]
            value = value[:batch_size]

            if policy_logits.ndim != 2 or int(policy_logits.shape[1]) != game_config.action_space_size:
                raise ValueError(
                    "Model policy output has unexpected shape; expected "
                    f"(batch, {game_config.action_space_size}), got {tuple(policy_logits.shape)}"
                )

            if game_config.value_head_type == "scalar":
                value_scalars = value.reshape(batch_size, -1)[:, 0]
            elif game_config.value_head_type == "wdl":
                if value.ndim != 2 or int(value.shape[1]) < 3:
                    raise ValueError(
                        "Model WDL value output must have shape (batch, 3), "
                        f"got {tuple(value.shape)}"
                    )
                value_scalars = value[:, 0] - value[:, 2]
            else:
                raise ValueError(
                    f"Unsupported value_head_type {game_config.value_head_type!r}"
                )

            policy_cpu = policy_logits.detach().to(
                device="cpu",
                dtype=torch.float32,
                non_blocking=True,
            )
            value_cpu = value_scalars.detach().to(
                device="cpu",
                dtype=torch.float32,
                non_blocking=True,
            )

        if inference_stream is not None:
            # synchronize() releases the GIL while waiting, letting the training
            # thread continue launching work on the default CUDA stream.
            inference_stream.synchronize()

        return policy_cpu.contiguous().numpy(), value_cpu.contiguous().numpy()

    return evaluator


def make_selfplay_evaluator_from_eval_queue(eval_queue: Any) -> Callable[[object], dict[str, object]]:
    """Deprecated bridge from EvalQueue to the legacy per-state Python evaluator API.

    The training pipeline now passes ``EvalQueue`` directly to ``SelfPlayManager`` so
    MCTS leaf evaluation stays in C++. Keep this adapter for standalone APIs and tests
    that still require a Python callable evaluator.
    """

    def evaluator(state: object) -> dict[str, object]:
        if not hasattr(state, "encode"):
            raise TypeError("state must provide an encode() method for EvalQueue submission")
        encoded_state = state.encode()  # type: ignore[attr-defined]
        # encode() may return a multi-dimensional numpy array; flatten for the C++ EvalQueue.
        if hasattr(encoded_state, "ravel"):
            encoded_state = encoded_state.ravel().tolist()
        eval_result = eval_queue.submit_and_wait(encoded_state)
        return {
            "policy": list(eval_result.policy_logits),
            "value": float(eval_result.value),
            "policy_is_logits": True,
        }

    return evaluator


def run_parallel_pipeline(
    model: Any,
    replay_buffer: ReplayBufferLike,
    game_config: GameConfig,
    eval_queue: EvalQueueLike,
    self_play_manager: SelfPlayManagerLike,
    training_config: TrainingConfigLike,
    pipeline_config: PipelineConfig,
    *,
    lr_schedule: Any = None,
    optimizer: Any = None,
    step_logger: StepLogger | None = None,
    cycle_logger: CycleLogger | None = None,
    start_step: int = 0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> PipelineRunResult:
    """Run the full self-play/training pipeline with concurrent inference and training workers."""

    import torch
    from alphazero.training.lr_schedule import StepDecayLRSchedule
    from alphazero.training.trainer import (
        TrainingStepMetrics,
        apply_random_go_symmetry,
        create_optimizer,
        sample_replay_batch_tensors,
        save_training_checkpoint,
        train_one_step,
    )

    _coerce_non_negative_int("start_step", start_step)
    _coerce_non_negative_float(
        "training_config.wait_for_buffer_seconds",
        training_config.wait_for_buffer_seconds,
    )
    max_steps = int(training_config.max_steps)
    if start_step > max_steps:
        raise ValueError(
            f"start_step ({start_step}) cannot exceed max_steps ({training_config.max_steps})"
        )

    device = _resolve_torch_device(training_config.device)
    model = model.to(device=device)

    # Move optimizer state (e.g. SGD momentum buffers) to match the model device,
    # since checkpoints are loaded with map_location="cpu".
    if optimizer is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device)

    active_schedule = lr_schedule or StepDecayLRSchedule()
    active_optimizer = optimizer
    if active_optimizer is None:
        active_optimizer = create_optimizer(
            model,
            lr_schedule=active_schedule,
            momentum=float(training_config.momentum),
        )

    scaler = _create_grad_scaler(
        device=device,
        enabled=bool(training_config.use_mixed_precision),
    )

    if start_step == max_steps:
        return PipelineRunResult(
            final_step=start_step,
            cycles_completed=0,
            inference_batches_processed=0,
            training_steps_completed=0,
            checkpoints=tuple(),
            terminated_early=False,
        )

    emitted_checkpoints: list[Any] = []

    pipeline_stop_event = threading.Event()
    progress_condition = threading.Condition()

    current_step = start_step
    cycles_completed = 0
    inference_batches_processed = 0
    training_steps_completed = 0
    cumulative_inference_seconds = 0.0
    cumulative_training_seconds = 0.0
    terminated_early = False

    inference_worker_error: BaseException | None = None
    training_worker_error: BaseException | None = None

    def inference_worker() -> None:
        nonlocal inference_worker_error
        nonlocal inference_batches_processed
        nonlocal cumulative_inference_seconds
        while not pipeline_stop_event.is_set():
            batch_start = time.perf_counter()
            try:
                eval_queue.process_batch()
            except BaseException as exc:  # pragma: no cover - surfaced in pipeline loop.
                if pipeline_stop_event.is_set():
                    return
                with progress_condition:
                    inference_worker_error = exc
                    pipeline_stop_event.set()
                    progress_condition.notify_all()
                return
            batch_seconds = max(time.perf_counter() - batch_start, 0.0)
            with progress_condition:
                inference_batches_processed += 1
                cumulative_inference_seconds += batch_seconds
                progress_condition.notify_all()

    def training_worker() -> None:
        nonlocal current_step
        nonlocal training_steps_completed
        nonlocal cumulative_training_seconds
        nonlocal training_worker_error
        while not pipeline_stop_event.is_set():
            with progress_condition:
                if current_step >= max_steps:
                    progress_condition.notify_all()
                    return
                step_to_train = current_step

            current_buffer_size = int(replay_buffer.size())
            if current_buffer_size < training_config.min_buffer_size:
                sleep_fn(float(training_config.wait_for_buffer_seconds))
                continue

            try:
                states, target_policy, target_value = sample_replay_batch_tensors(
                    replay_buffer,
                    game_config,
                    batch_size=training_config.batch_size,
                    device=device,
                )
                if game_config.supports_symmetry:
                    states, target_policy = apply_random_go_symmetry(states, target_policy)

                step_start = time.perf_counter()
                step_metrics = train_one_step(
                    model,
                    active_optimizer,
                    states=states,
                    target_policy=target_policy,
                    target_value=target_value,
                    game_config=game_config,
                    lr_schedule=active_schedule,
                    global_step=step_to_train,
                    l2_reg=float(training_config.l2_reg),
                    scaler=scaler,
                    use_mixed_precision=bool(training_config.use_mixed_precision),
                )
                training_step_seconds = max(time.perf_counter() - step_start, 0.0)

                next_step = step_to_train + 1
                updated_metrics = TrainingStepMetrics(
                    step=next_step,
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
                if step_logger is not None and next_step % training_config.log_interval == 0:
                    step_logger(next_step, updated_metrics.as_dict())

                checkpoint_dir = training_config.checkpoint_dir
                if checkpoint_dir is not None and next_step % training_config.checkpoint_interval == 0:
                    emitted_checkpoints.append(
                        save_training_checkpoint(
                            model,
                            active_optimizer,
                            active_schedule,
                            step=next_step,
                            checkpoint_dir=Path(checkpoint_dir),
                            replay_buffer=replay_buffer,
                            keep_last=int(training_config.checkpoint_keep_last),
                            is_milestone=False,
                            export_folded_weights=bool(training_config.export_folded_checkpoints),
                        )
                    )
                    if next_step % training_config.milestone_interval == 0:
                        emitted_checkpoints.append(
                            save_training_checkpoint(
                                model,
                                active_optimizer,
                                active_schedule,
                                step=next_step,
                                checkpoint_dir=Path(checkpoint_dir),
                                replay_buffer=replay_buffer,
                                keep_last=int(training_config.checkpoint_keep_last),
                                is_milestone=True,
                                export_folded_weights=bool(training_config.export_folded_checkpoints),
                            )
                        )

                with progress_condition:
                    current_step = next_step
                    training_steps_completed += 1
                    cumulative_training_seconds += training_step_seconds
                    progress_condition.notify_all()
            except BaseException as exc:  # pragma: no cover - surfaced in pipeline loop.
                with progress_condition:
                    training_worker_error = exc
                    pipeline_stop_event.set()
                    progress_condition.notify_all()
                return

    inference_thread = threading.Thread(
        target=inference_worker,
        name="eval-queue-inference-worker",
        daemon=True,
    )
    training_thread = threading.Thread(
        target=training_worker,
        name="pipeline-training-worker",
        daemon=True,
    )

    def raise_if_worker_failed(context: str) -> None:
        if inference_worker_error is not None:
            raise RuntimeError(
                f"Inference worker failed while {context}"
            ) from inference_worker_error
        if training_worker_error is not None:
            raise RuntimeError(
                f"Training worker failed while {context}"
            ) from training_worker_error

    def emit_cycle_metrics(
        *,
        cycle: int,
        global_step: int,
        inference_batches: int,
        training_steps: int,
        inference_seconds: float,
        training_seconds: float,
        cycle_seconds: float,
    ) -> None:
        if cycle_logger is None:
            return
        enriched_metrics = InterleavedCycleMetrics(
            cycle=cycle,
            global_step=global_step,
            inference_batches=inference_batches,
            training_steps=training_steps,
            inference_seconds=inference_seconds,
            training_seconds=training_seconds,
            cycle_seconds=cycle_seconds,
        ).as_dict()
        enriched_metrics["buffer/size"] = float(replay_buffer.size())
        cycle_logger(cycle, enriched_metrics)

    pipeline_error: BaseException | None = None
    self_play_started = False
    try:
        self_play_manager.start()
        self_play_started = True
        inference_thread.start()
        training_thread.start()

        cycle_start_wall = time.perf_counter()
        last_cycle_inference_batches = 0
        last_cycle_training_steps = 0
        last_cycle_inference_seconds = 0.0
        last_cycle_training_seconds = 0.0

        while True:
            with progress_condition:
                target_inference_batches = (
                    inference_batches_processed + pipeline_config.inference_batches_per_cycle
                )
                target_training_steps = (
                    training_steps_completed + pipeline_config.training_steps_per_cycle
                )
                while (
                    not pipeline_stop_event.is_set()
                    and inference_worker_error is None
                    and training_worker_error is None
                    and current_step < max_steps
                    and inference_batches_processed < target_inference_batches
                    and training_steps_completed < target_training_steps
                ):
                    progress_condition.wait(timeout=0.05)

                snapshot_step = current_step
                snapshot_inference_batches = inference_batches_processed
                snapshot_training_steps = training_steps_completed
                snapshot_inference_seconds = cumulative_inference_seconds
                snapshot_training_seconds = cumulative_training_seconds

            raise_if_worker_failed("running the parallel pipeline")

            progress_made = (
                snapshot_inference_batches > last_cycle_inference_batches
                or snapshot_training_steps > last_cycle_training_steps
            )
            if progress_made:
                cycles_completed += 1
                emit_cycle_metrics(
                    cycle=cycles_completed,
                    global_step=snapshot_step,
                    inference_batches=(
                        snapshot_inference_batches - last_cycle_inference_batches
                    ),
                    training_steps=(
                        snapshot_training_steps - last_cycle_training_steps
                    ),
                    inference_seconds=(
                        snapshot_inference_seconds - last_cycle_inference_seconds
                    ),
                    training_seconds=(
                        snapshot_training_seconds - last_cycle_training_seconds
                    ),
                    cycle_seconds=max(time.perf_counter() - cycle_start_wall, 0.0),
                )
                cycle_start_wall = time.perf_counter()
                last_cycle_inference_batches = snapshot_inference_batches
                last_cycle_training_steps = snapshot_training_steps
                last_cycle_inference_seconds = snapshot_inference_seconds
                last_cycle_training_seconds = snapshot_training_seconds

            if snapshot_step >= max_steps:
                break

            if (
                pipeline_config.max_cycles is not None
                and cycles_completed >= pipeline_config.max_cycles
            ):
                terminated_early = snapshot_step < max_steps
                pipeline_stop_event.set()
                break
    except BaseException as exc:  # pragma: no cover - exercised under failure conditions.
        pipeline_error = exc
        raise
    finally:
        pipeline_stop_event.set()

        # Stop eval_queue first to unblock any workers waiting in submit_and_wait,
        # then stop self_play_manager to join the (now-unblocked) worker threads.
        shutdown_error: BaseException | None = None
        try:
            eval_queue.stop()
        except Exception as exc:
            if pipeline_error is None:
                shutdown_error = exc

        inference_thread.join(timeout=5.0)
        if inference_thread.is_alive() and pipeline_error is None and shutdown_error is None:
            shutdown_error = RuntimeError("Inference worker thread did not stop cleanly")

        training_thread.join(timeout=5.0)
        if training_thread.is_alive() and pipeline_error is None and shutdown_error is None:
            shutdown_error = RuntimeError("Training worker thread did not stop cleanly")

        if self_play_started:
            try:
                self_play_manager.stop()
            except Exception as exc:
                if pipeline_error is None and shutdown_error is None:
                    shutdown_error = exc

        if shutdown_error is not None:
            raise shutdown_error

    with progress_condition:
        final_step = current_step
        final_inference_batches_processed = inference_batches_processed
        final_training_steps_completed = training_steps_completed

    if (
        pipeline_config.max_cycles is not None
        and cycles_completed >= pipeline_config.max_cycles
        and final_step < max_steps
    ):
        terminated_early = True

    return PipelineRunResult(
        final_step=final_step,
        cycles_completed=cycles_completed,
        inference_batches_processed=final_inference_batches_processed,
        training_steps_completed=final_training_steps_completed,
        checkpoints=tuple(emitted_checkpoints),
        terminated_early=terminated_early,
    )


def run_interleaved_pipeline(
    model: Any,
    replay_buffer: ReplayBufferLike,
    game_config: GameConfig,
    eval_queue: EvalQueueLike,
    self_play_manager: SelfPlayManagerLike,
    training_config: TrainingConfigLike,
    pipeline_config: PipelineConfig,
    *,
    lr_schedule: Any = None,
    optimizer: Any = None,
    step_logger: StepLogger | None = None,
    cycle_logger: CycleLogger | None = None,
    start_step: int = 0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> PipelineRunResult:
    """Backward-compatible wrapper for the parallel pipeline runtime."""

    return run_parallel_pipeline(
        model,
        replay_buffer,
        game_config,
        eval_queue,
        self_play_manager,
        training_config,
        pipeline_config,
        lr_schedule=lr_schedule,
        optimizer=optimizer,
        step_logger=step_logger,
        cycle_logger=cycle_logger,
        start_step=start_step,
        sleep_fn=sleep_fn,
    )


__all__ = [
    "DEFAULT_INFERENCE_BATCHES_PER_CYCLE",
    "DEFAULT_TRAINING_STEPS_PER_CYCLE",
    "PipelineConfig",
    "InterleavedCycleMetrics",
    "InterleavedScheduleResult",
    "PipelineRunResult",
    "load_pipeline_config_from_config",
    "load_pipeline_config_from_yaml",
    "run_interleaved_schedule",
    "run_parallel_pipeline",
    "run_interleaved_pipeline",
    "make_eval_queue_batch_evaluator",
    "make_selfplay_evaluator_from_eval_queue",
]
