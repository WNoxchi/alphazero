"""Integration smoke tests for the interleaved self-play/training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence
import unittest


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if _TORCH_AVAILABLE:
    import torch
    from torch import nn

    from alphazero.config import GameConfig
    from alphazero.pipeline.orchestrator import (
        PipelineConfig,
        make_eval_queue_batch_evaluator,
        run_interleaved_pipeline,
    )
    from alphazero.training.lr_schedule import StepDecayLRSchedule
    from alphazero.training.trainer import TrainingConfig, create_optimizer


if _TORCH_AVAILABLE:

    @dataclass(slots=True)
    class _ReplayPosition:
        encoded_state: list[float]
        policy: list[float]
        value: float
        value_wdl: list[float]
        game_id: int
        move_number: int
        encoded_state_size: int
        policy_size: int


if _TORCH_AVAILABLE:

    class _InMemoryReplayBuffer:
        """Minimal replay-buffer test double with ring-buffer semantics."""

        def __init__(self, *, capacity: int, random_seed: int) -> None:
            self._capacity = capacity
            self._positions: list[_ReplayPosition] = []
            self._rng = random.Random(random_seed)
            self.write_head = 0
            self.games_total = 0

        def size(self) -> int:
            return len(self._positions)

        def sample(self, batch_size: int) -> list[_ReplayPosition]:
            if not self._positions:
                raise ValueError("Cannot sample from an empty replay buffer")
            return [self._rng.choice(self._positions) for _ in range(batch_size)]

        def append_generated_position(self, position: _ReplayPosition) -> None:
            if len(self._positions) < self._capacity:
                self._positions.append(position)
            else:
                self._positions[self.write_head] = position
            self.write_head = (self.write_head + 1) % self._capacity

        def checkpoint_metadata(self) -> dict[str, int]:
            return {
                "write_head": self.write_head,
                "count": len(self._positions),
                "games_total": self.games_total,
            }


if _TORCH_AVAILABLE:

    class _SyntheticEvalQueue:
        """Feeds batched synthetic states through the real model evaluator."""

        def __init__(
            self,
            *,
            evaluator: Callable[[Sequence[Sequence[float]]], list[dict[str, object]]],
            replay_buffer: _InMemoryReplayBuffer,
            game_config: GameConfig,
            positions_per_batch: int,
        ) -> None:
            rows, cols = game_config.board_shape
            self._encoded_state_size = game_config.input_channels * rows * cols
            self._action_space_size = game_config.action_space_size
            self._evaluator = evaluator
            self._replay_buffer = replay_buffer
            self._positions_per_batch = positions_per_batch
            self._cursor = 0
            self.stop_called = False
            self.processed_batches = 0

        def process_batch(self) -> None:
            encoded_states: list[list[float]] = []
            for local_index in range(self._positions_per_batch):
                state = [0.0] * self._encoded_state_size
                hot_index = (self._cursor + local_index) % self._encoded_state_size
                state[hot_index] = 1.0
                state[(hot_index + 1) % self._encoded_state_size] = 0.5
                encoded_states.append(state)

            evaluations = self._evaluator(encoded_states)
            if len(evaluations) != len(encoded_states):
                raise RuntimeError("Evaluator output count does not match input batch size")

            for state, evaluation in zip(encoded_states, evaluations):
                raw_logits = evaluation.get("policy_logits")
                if not isinstance(raw_logits, Sequence):
                    raise TypeError("policy_logits must be a sequence")
                logits = [float(value) for value in raw_logits]
                if len(logits) != self._action_space_size:
                    raise ValueError(
                        "policy_logits size does not match action space: "
                        f"expected {self._action_space_size}, got {len(logits)}"
                    )

                best_action = max(range(self._action_space_size), key=logits.__getitem__)
                policy = [0.0] * self._action_space_size
                policy[best_action] = 1.0

                raw_value = evaluation.get("value")
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    raise TypeError("value must be numeric")
                value = float(raw_value)

                self._replay_buffer.append_generated_position(
                    _ReplayPosition(
                        encoded_state=state,
                        policy=policy,
                        value=value,
                        value_wdl=[0.0, 1.0, 0.0],
                        game_id=(self._cursor // self._positions_per_batch) + 1,
                        move_number=(self._cursor % 200) + 1,
                        encoded_state_size=self._encoded_state_size,
                        policy_size=self._action_space_size,
                    )
                )

            self._cursor += self._positions_per_batch
            self.processed_batches += 1
            # Model each processed batch as one completed game to emulate live self-play progress.
            self._replay_buffer.games_total += 1

        def stop(self) -> None:
            self.stop_called = True


if _TORCH_AVAILABLE:

    class _SyntheticSelfPlayManager:
        """Minimal manager surface required by run_interleaved_pipeline."""

        def __init__(self, replay_buffer: _InMemoryReplayBuffer) -> None:
            self._replay_buffer = replay_buffer
            self.started = False
            self.stopped = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def metrics(self) -> Any:
            return SimpleNamespace(games_completed=self._replay_buffer.games_total)


if _TORCH_AVAILABLE:

    class _TinyPolicyValueNetwork(nn.Module):
        """Tiny network to keep smoke-test runtime low while exercising real training math."""

        def __init__(self, game_config: GameConfig) -> None:
            super().__init__()
            rows, cols = game_config.board_shape
            features = game_config.input_channels * rows * cols
            self._game_config = game_config
            self._flatten = nn.Flatten()
            self._hidden = nn.Linear(features, 32)
            self._policy = nn.Linear(32, game_config.action_space_size)
            self._value = nn.Linear(32, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = torch.tanh(self._hidden(self._flatten(x)))
            policy_logits = self._policy(hidden)
            value = torch.tanh(self._value(hidden))
            return policy_logits, value


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required for pipeline integration smoke tests")
class PipelineIntegrationSmokeTests(unittest.TestCase):
    def test_pipeline_smoke_runs_100_steps_and_emits_artifacts(self) -> None:
        """WHY: Guards TASK-092 by proving the full interleaved pipeline can run 100 steps without crashing."""
        torch.manual_seed(7)

        game_config = GameConfig(
            name="toy-smoke",
            board_shape=(3, 3),
            input_channels=2,
            action_space_size=10,
            value_head_type="scalar",
            supports_symmetry=False,
            num_symmetries=1,
        )
        model = _TinyPolicyValueNetwork(game_config)
        replay_buffer = _InMemoryReplayBuffer(capacity=2048, random_seed=1234)

        evaluator = make_eval_queue_batch_evaluator(
            model,
            game_config,
            device="cpu",
            use_mixed_precision=False,
        )
        eval_queue = _SyntheticEvalQueue(
            evaluator=evaluator,
            replay_buffer=replay_buffer,
            game_config=game_config,
            positions_per_batch=12,
        )
        self_play_manager = _SyntheticSelfPlayManager(replay_buffer)

        pipeline_config = PipelineConfig(
            inference_batches_per_cycle=2,
            training_steps_per_cycle=1,
        )
        lr_schedule = StepDecayLRSchedule(entries=((0, 0.05), (60, 0.01)))

        logged_steps: list[int] = []
        logged_cycle_metrics: list[dict[str, float]] = []

        def step_logger(step: int, metrics: Mapping[str, float]) -> None:
            logged_steps.append(step)
            logged_cycle_metrics.append(dict(metrics))

        cycle_metrics: list[dict[str, float]] = []

        def cycle_logger(_cycle: int, metrics: Mapping[str, float]) -> None:
            cycle_metrics.append(dict(metrics))

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            training_config = TrainingConfig(
                batch_size=16,
                max_steps=100,
                momentum=0.9,
                l2_reg=1e-4,
                checkpoint_interval=25,
                checkpoint_keep_last=10,
                milestone_interval=100,
                log_interval=10,
                min_buffer_size=16,
                wait_for_buffer_seconds=0.0001,
                use_mixed_precision=False,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                export_folded_checkpoints=False,
            )
            optimizer = create_optimizer(
                model,
                lr_schedule=lr_schedule,
                momentum=training_config.momentum,
            )

            result = run_interleaved_pipeline(
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
                sleep_fn=lambda _seconds: None,
            )

            self.assertEqual(result.final_step, 100)
            self.assertEqual(result.training_steps_completed, 100)
            self.assertFalse(result.terminated_early)
            self.assertGreater(result.cycles_completed, 0)

            self.assertTrue(self_play_manager.started)
            self.assertTrue(self_play_manager.stopped)
            self.assertTrue(eval_queue.stop_called)
            self.assertEqual(eval_queue.processed_batches, result.inference_batches_processed)

            self.assertGreaterEqual(replay_buffer.size(), training_config.min_buffer_size)
            self.assertGreater(replay_buffer.games_total, 0)

            self.assertEqual(logged_steps, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            self.assertTrue(logged_cycle_metrics)
            self.assertIn("loss/total", logged_cycle_metrics[0])
            self.assertIn("throughput/train_steps_per_sec", logged_cycle_metrics[0])

            self.assertTrue(cycle_metrics)
            self.assertIn("pipeline/cycle", cycle_metrics[0])
            self.assertIn("pipeline/global_step", cycle_metrics[0])
            self.assertIn("pipeline/inference_batches", cycle_metrics[0])
            self.assertIn("buffer/size", cycle_metrics[0])

            regular_steps = sorted(
                checkpoint.step for checkpoint in result.checkpoints if not checkpoint.is_milestone
            )
            milestone_steps = sorted(
                checkpoint.step for checkpoint in result.checkpoints if checkpoint.is_milestone
            )
            self.assertEqual(regular_steps, [25, 50, 75, 100])
            self.assertEqual(milestone_steps, [100])
            for checkpoint in result.checkpoints:
                self.assertTrue(checkpoint.checkpoint_path.exists())


if __name__ == "__main__":
    unittest.main()
