"""Tests for interleaved pipeline orchestration."""

from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import pathlib
import sys
import tempfile
from typing import Any, Sequence, cast
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from alphazero.config import CHESS_CONFIG, GO_CONFIG  # noqa: E402
from alphazero.pipeline.orchestrator import (  # noqa: E402
    PipelineConfig,
    load_pipeline_config_from_config,
    load_pipeline_config_from_yaml,
    make_eval_queue_batch_evaluator,
    make_selfplay_evaluator_from_eval_queue,
    run_interleaved_schedule,
)

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_YAML_AVAILABLE = importlib.util.find_spec("yaml") is not None
if _TORCH_AVAILABLE:
    import torch
    from torch import nn


class PipelineConfigLoadingTests(unittest.TestCase):
    def test_load_pipeline_config_uses_spec_defaults_when_pipeline_section_is_missing(self) -> None:
        """Protects default scheduler behavior so missing YAML keys still map to S=50, T=1."""
        config = load_pipeline_config_from_config({"game": "chess"})

        self.assertEqual(config.inference_batches_per_cycle, 50)
        self.assertEqual(config.training_steps_per_cycle, 1)
        self.assertIsNone(config.max_cycles)
        self.assertEqual(config.max_replay_ratio, 0)

    def test_load_pipeline_config_reads_overrides_from_mapping(self) -> None:
        """Guards S:T override parsing so tuning values in config mappings are applied exactly."""
        parsed = load_pipeline_config_from_config(
            {
                "pipeline": {
                    "inference_batches_per_cycle": 7,
                    "training_steps_per_cycle": 3,
                    "max_cycles": 9,
                }
            }
        )
        self.assertEqual(parsed, PipelineConfig(7, 3, 9))

    def test_load_pipeline_config_reads_max_replay_ratio_from_mapping(self) -> None:
        """Guards replay ratio cap parsing so the training worker gate activates at the configured ratio."""
        parsed = load_pipeline_config_from_config(
            {
                "pipeline": {
                    "max_replay_ratio": 8,
                }
            }
        )
        self.assertEqual(parsed.max_replay_ratio, 8)

    def test_pipeline_config_rejects_negative_max_replay_ratio(self) -> None:
        """Prevents misconfiguration where a negative replay ratio would invert the gate logic."""
        with self.assertRaises(ValueError):
            PipelineConfig(max_replay_ratio=-1)

    @unittest.skipUnless(_YAML_AVAILABLE, "PyYAML is required to parse nested pipeline YAML overrides")
    def test_load_pipeline_config_reads_overrides_from_yaml(self) -> None:
        """Verifies YAML loading path honors nested pipeline values when a full YAML parser is available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = pathlib.Path(temp_dir) / "pipeline.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "game: chess",
                        "pipeline:",
                        "  inference_batches_per_cycle: 11",
                        "  training_steps_per_cycle: 4",
                        "  max_cycles: 5",
                    ]
                ),
                encoding="utf-8",
            )
            loaded = load_pipeline_config_from_yaml(config_path)

        self.assertEqual(loaded, PipelineConfig(11, 4, 5))


class InterleavedScheduleTests(unittest.TestCase):
    def test_schedule_runs_inference_then_training_by_configured_ratio(self) -> None:
        """Ensures each cycle executes exactly S inference batches before T training attempts."""
        events: list[str] = []

        def inference_batch() -> None:
            events.append("I")

        def training_step(step: int) -> bool:
            events.append(f"T{step}")
            return True

        result = run_interleaved_schedule(
            inference_batch_fn=inference_batch,
            training_step_fn=training_step,
            max_steps=3,
            pipeline_config=PipelineConfig(
                inference_batches_per_cycle=2,
                training_steps_per_cycle=1,
            ),
        )

        self.assertEqual(events, ["I", "I", "T0", "I", "I", "T1", "I", "I", "T2"])
        self.assertEqual(result.final_step, 3)
        self.assertEqual(result.cycles_completed, 3)
        self.assertEqual(result.inference_batches_processed, 6)
        self.assertEqual(result.training_steps_completed, 3)
        self.assertFalse(result.terminated_early)

    def test_schedule_breaks_training_phase_when_training_step_is_not_executed(self) -> None:
        """Protects underfilled-buffer handling so one failed train attempt ends that cycle's train phase."""
        events: list[str] = []
        decisions = iter([False, True, True])

        def inference_batch() -> None:
            events.append("I")

        def training_step(step: int) -> bool:
            decision = next(decisions)
            events.append(f"T{step}:{int(decision)}")
            return decision

        result = run_interleaved_schedule(
            inference_batch_fn=inference_batch,
            training_step_fn=training_step,
            max_steps=2,
            pipeline_config=PipelineConfig(
                inference_batches_per_cycle=1,
                training_steps_per_cycle=2,
                max_cycles=4,
            ),
        )

        self.assertEqual(events, ["I", "T0:0", "I", "T0:1", "T1:1"])
        self.assertEqual(result.final_step, 2)
        self.assertEqual(result.cycles_completed, 2)
        self.assertEqual(result.inference_batches_processed, 2)
        self.assertEqual(result.training_steps_completed, 2)
        self.assertFalse(result.terminated_early)

    def test_schedule_reports_early_termination_when_max_cycle_cap_is_hit(self) -> None:
        """Prevents silent infinite waits by signaling when max_cycles stops progress before max_steps."""

        def inference_batch() -> None:
            return None

        def training_step(_step: int) -> bool:
            return False

        result = run_interleaved_schedule(
            inference_batch_fn=inference_batch,
            training_step_fn=training_step,
            max_steps=5,
            pipeline_config=PipelineConfig(
                inference_batches_per_cycle=3,
                training_steps_per_cycle=1,
                max_cycles=2,
            ),
        )

        self.assertEqual(result.final_step, 0)
        self.assertEqual(result.cycles_completed, 2)
        self.assertEqual(result.inference_batches_processed, 6)
        self.assertEqual(result.training_steps_completed, 0)
        self.assertTrue(result.terminated_early)


class EvalQueueAdapterTests(unittest.TestCase):
    def test_selfplay_evaluator_adapter_forwards_state_encoding_and_logits_flag(self) -> None:
        """Ensures self-play receives policy logits/value from EvalQueue without shape semantics drift."""

        class _FakeState:
            def encode(self) -> list[float]:
                return [1.0, 2.0, 3.0]

        class _FakeEvalResult:
            def __init__(self) -> None:
                self.policy_logits = [0.25, 0.75]
                self.value = -0.4

        class _FakeEvalQueue:
            def __init__(self) -> None:
                self.requests: list[list[float]] = []

            def submit_and_wait(self, encoded_state: list[float]) -> _FakeEvalResult:
                self.requests.append(list(encoded_state))
                return _FakeEvalResult()

        queue = _FakeEvalQueue()
        evaluator = make_selfplay_evaluator_from_eval_queue(queue)
        output = evaluator(_FakeState())

        policy = cast(Sequence[float], output["policy"])
        value = cast(float, output["value"])
        policy_is_logits = cast(bool, output["policy_is_logits"])

        self.assertEqual(queue.requests, [[1.0, 2.0, 3.0]])
        self.assertEqual(list(policy), [0.25, 0.75])
        self.assertAlmostEqual(float(value), -0.4)
        self.assertTrue(bool(policy_is_logits))


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required for model-to-eval-queue adapter tests")
class EvalQueueModelAdapterTests(unittest.TestCase):
    def test_scalar_value_head_adapter_preserves_policy_shape_and_scalar_values(self) -> None:
        """Guards Go inference adaptation so eval queue callbacks preserve policy size and scalar outputs."""

        class _ScalarModel(nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                batch_size = x.shape[0]
                policy = torch.arange(
                    GO_CONFIG.action_space_size,
                    dtype=torch.float32,
                    device=x.device,
                ).repeat(batch_size, 1)
                value = torch.linspace(
                    -0.5,
                    0.5,
                    steps=batch_size,
                    dtype=torch.float32,
                    device=x.device,
                ).unsqueeze(1)
                return policy, value

        model = _ScalarModel()
        evaluator = make_eval_queue_batch_evaluator(
            model,
            GO_CONFIG,
            device="cpu",
            use_mixed_precision=False,
        )
        import numpy as np

        encoded_size = GO_CONFIG.input_channels * GO_CONFIG.board_shape[0] * GO_CONFIG.board_shape[1]
        policy_logits, value_scalars = evaluator(
            np.asarray(
                [
                    [0.0] * encoded_size,
                    [1.0] * encoded_size,
                ],
                dtype=np.float32,
            )
        )

        self.assertEqual(tuple(policy_logits.shape), (2, GO_CONFIG.action_space_size))
        self.assertEqual(tuple(value_scalars.shape), (2,))
        self.assertAlmostEqual(float(value_scalars[0]), -0.5, places=6)
        self.assertAlmostEqual(float(value_scalars[1]), 0.5, places=6)

    def test_adapter_ignores_optional_ownership_output_from_three_head_model(self) -> None:
        """WHY: Go ownership head is auxiliary-only; eval queue must still consume policy/value without unpack errors."""

        class _ScalarOwnershipModel(nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                batch_size = x.shape[0]
                policy = torch.zeros(
                    (batch_size, GO_CONFIG.action_space_size),
                    dtype=torch.float32,
                    device=x.device,
                )
                value = torch.full((batch_size, 1), 0.25, dtype=torch.float32, device=x.device)
                ownership = torch.ones(
                    (batch_size, GO_CONFIG.board_shape[0] * GO_CONFIG.board_shape[1]),
                    dtype=torch.float32,
                    device=x.device,
                )
                return policy, value, ownership

        model = _ScalarOwnershipModel()
        evaluator = make_eval_queue_batch_evaluator(
            model,
            GO_CONFIG,
            device="cpu",
            use_mixed_precision=False,
        )
        import numpy as np

        encoded_size = GO_CONFIG.input_channels * GO_CONFIG.board_shape[0] * GO_CONFIG.board_shape[1]
        policy_logits, value_scalars = evaluator(
            np.asarray([[0.0] * encoded_size], dtype=np.float32)
        )

        self.assertEqual(tuple(policy_logits.shape), (1, GO_CONFIG.action_space_size))
        self.assertEqual(tuple(value_scalars.shape), (1,))
        self.assertAlmostEqual(float(value_scalars[0]), 0.25, places=6)

    def test_wdl_value_head_adapter_maps_wdl_to_scalar_win_minus_loss(self) -> None:
        """Protects chess value conversion contract used by MCTS backup."""

        class _WDLModel(nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                batch_size = x.shape[0]
                policy = torch.zeros(
                    (batch_size, CHESS_CONFIG.action_space_size),
                    dtype=torch.float32,
                    device=x.device,
                )
                value = torch.tensor(
                    [[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]],
                    dtype=torch.float32,
                    device=x.device,
                )
                return policy, value[:batch_size]

        model = _WDLModel()
        evaluator = make_eval_queue_batch_evaluator(
            model,
            CHESS_CONFIG,
            device="cpu",
            use_mixed_precision=False,
        )
        import numpy as np

        encoded_size = CHESS_CONFIG.input_channels * CHESS_CONFIG.board_shape[0] * CHESS_CONFIG.board_shape[1]
        policy_logits, value_scalars = evaluator(
            np.asarray(
                [
                    [0.0] * encoded_size,
                    [0.1] * encoded_size,
                ],
                dtype=np.float32,
            )
        )

        self.assertEqual(tuple(policy_logits.shape), (2, CHESS_CONFIG.action_space_size))
        self.assertEqual(tuple(value_scalars.shape), (2,))
        self.assertAlmostEqual(float(value_scalars[0]), 0.6, places=6)
        self.assertAlmostEqual(float(value_scalars[1]), -0.3, places=6)

    def test_cpu_evaluator_skips_cuda_stream_initialization(self) -> None:
        """WHY: Guards CPU fallback so stream-specific CUDA logic never runs on CPU-only evaluators."""

        class _ScalarModel(nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                batch_size = x.shape[0]
                policy = torch.zeros(
                    (batch_size, GO_CONFIG.action_space_size),
                    dtype=torch.float32,
                    device=x.device,
                )
                value = torch.zeros(
                    (batch_size, 1),
                    dtype=torch.float32,
                    device=x.device,
                )
                return policy, value

        model = _ScalarModel()
        encoded_size = GO_CONFIG.input_channels * GO_CONFIG.board_shape[0] * GO_CONFIG.board_shape[1]
        import numpy as np

        with patch("torch.cuda.Stream") as stream_ctor, patch("torch.cuda.stream") as stream_ctx:
            evaluator = make_eval_queue_batch_evaluator(
                model,
                GO_CONFIG,
                device="cpu",
                use_mixed_precision=False,
            )
            evaluator(np.zeros((1, encoded_size), dtype=np.float32))

        stream_ctor.assert_not_called()
        stream_ctx.assert_not_called()
        # The model stays in train() mode throughout the pipeline to avoid
        # torch.compile guard invalidation from eval()/train() toggling.
        self.assertTrue(model.training)

    def test_cuda_evaluator_uses_non_blocking_transfers_and_stream_sync(self) -> None:
        """WHY: Prevents GIL-holding GPU sync regressions by requiring non-blocking copies and explicit stream sync."""
        import numpy as np

        class _FakeDevice:
            type = "cuda"

        class _FakeTensor:
            def __init__(self, array: Any, to_calls: list[tuple[Any, dict[str, Any]]]) -> None:
                self._array = np.asarray(array, dtype=np.float32)
                self._to_calls = to_calls

            @property
            def ndim(self) -> int:
                return int(self._array.ndim)

            @property
            def shape(self) -> tuple[int, ...]:
                return tuple(int(dim) for dim in self._array.shape)

            def to(self, *args: Any, **kwargs: Any) -> "_FakeTensor":
                device_arg = kwargs.get("device")
                if device_arg is None and args:
                    device_arg = args[0]
                self._to_calls.append((device_arg, dict(kwargs)))
                return self

            def reshape(self, *shape: int) -> "_FakeTensor":
                return _FakeTensor(self._array.reshape(*shape), self._to_calls)

            def detach(self) -> "_FakeTensor":
                return self

            def contiguous(self) -> "_FakeTensor":
                return self

            def numpy(self) -> Any:
                return self._array

            def __getitem__(self, key: Any) -> "_FakeTensor":
                return _FakeTensor(self._array[key], self._to_calls)

            def __sub__(self, other: "_FakeTensor") -> "_FakeTensor":
                return _FakeTensor(self._array - other._array, self._to_calls)

        class _FakeModel:
            def __init__(self, to_calls: list[tuple[Any, dict[str, Any]]]) -> None:
                self.training = True
                self.eval_calls = 0
                self._to_calls = to_calls

            def to(self, *, device: Any) -> "_FakeModel":
                self.device = device
                return self

            def eval(self) -> "_FakeModel":
                self.training = False
                self.eval_calls += 1
                return self

            def __call__(self, model_inputs: _FakeTensor) -> tuple[_FakeTensor, _FakeTensor]:
                batch_size = int(model_inputs.shape[0])
                policy = _FakeTensor(
                    np.zeros((batch_size, GO_CONFIG.action_space_size), dtype=np.float32),
                    self._to_calls,
                )
                value = _FakeTensor(
                    np.linspace(-0.3, 0.3, num=batch_size, dtype=np.float32).reshape(batch_size, 1),
                    self._to_calls,
                )
                return policy, value

        class _FakeStream:
            def __init__(self) -> None:
                self.synchronize_calls = 0

            def synchronize(self) -> None:
                self.synchronize_calls += 1

        to_calls: list[tuple[Any, dict[str, Any]]] = []
        fake_model = _FakeModel(to_calls)
        fake_device = _FakeDevice()
        fake_stream = _FakeStream()
        stream_context_calls: list[_FakeStream] = []

        def _stream_context(stream: _FakeStream) -> Any:
            stream_context_calls.append(stream)
            return nullcontext()

        encoded_size = GO_CONFIG.input_channels * GO_CONFIG.board_shape[0] * GO_CONFIG.board_shape[1]
        encoded_states = np.zeros((2, encoded_size), dtype=np.float32)
        with (
            patch("alphazero.pipeline.orchestrator._resolve_torch_device", return_value=fake_device),
            patch("torch.cuda.Stream", return_value=fake_stream) as stream_ctor,
            patch("torch.cuda.stream", side_effect=_stream_context),
            patch("torch.from_numpy", side_effect=lambda array: _FakeTensor(array, to_calls)),
            patch("torch.no_grad", side_effect=lambda: nullcontext()),
            patch("torch.autocast", side_effect=lambda **_kwargs: nullcontext()),
        ):
            evaluator = make_eval_queue_batch_evaluator(
                fake_model,
                GO_CONFIG,
                device="cuda:0",
                use_mixed_precision=False,
            )
            policy_logits, value_scalars = evaluator(encoded_states)

        stream_ctor.assert_called_once_with(device=fake_device)
        self.assertEqual(stream_context_calls, [fake_stream])
        self.assertEqual(fake_stream.synchronize_calls, 1)
        # The evaluator no longer toggles eval()/train() to avoid
        # torch.compile guard invalidation; the model stays in train mode.
        self.assertEqual(fake_model.eval_calls, 0)
        self.assertEqual(tuple(policy_logits.shape), (2, GO_CONFIG.action_space_size))
        self.assertEqual(tuple(value_scalars.shape), (2,))
        self.assertAlmostEqual(float(value_scalars[0]), -0.3, places=6)
        self.assertAlmostEqual(float(value_scalars[1]), 0.3, places=6)

        cuda_transfers = [
            kwargs
            for device_arg, kwargs in to_calls
            if device_arg is fake_device
        ]
        cpu_transfers = [
            kwargs
            for device_arg, kwargs in to_calls
            if device_arg == "cpu"
        ]
        self.assertEqual(len(cuda_transfers), 1)
        self.assertTrue(bool(cuda_transfers[0].get("non_blocking")))
        self.assertEqual(len(cpu_transfers), 2)
        self.assertTrue(all(bool(kwargs.get("non_blocking")) for kwargs in cpu_transfers))


if __name__ == "__main__":
    unittest.main()
