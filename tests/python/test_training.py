"""Tests for the Python training loop."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
import pathlib
import random
import sys
import tempfile
from typing import Mapping
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if _TORCH_AVAILABLE:
    import torch
    from torch import nn

    from alphazero.config import GameConfig
    from alphazero.training.loss import compute_loss_components
    from alphazero.training.lr_schedule import StepDecayLRSchedule
    from alphazero.training.trainer import (
        TrainingConfig,
        _gradient_statistics,
        apply_random_go_symmetry,
        create_optimizer,
        load_training_checkpoint,
        prepare_replay_batch,
        save_training_checkpoint,
        train_one_step,
        training_loop,
    )


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

    class _ReplayBuffer:
        def __init__(self, positions: list[_ReplayPosition], *, gate_calls: int = 0) -> None:
            self._positions = positions
            self._gate_calls_remaining = gate_calls
            self.sample_calls = 0
            self._rng = random.Random(1234)

        def size(self) -> int:
            if self._gate_calls_remaining > 0:
                self._gate_calls_remaining -= 1
                return 0
            return len(self._positions)

        def sample(self, batch_size: int) -> list[_ReplayPosition]:
            self.sample_calls += 1
            return [self._rng.choice(self._positions) for _ in range(batch_size)]


if _TORCH_AVAILABLE:

    class _PackedReplayBuffer(_ReplayBuffer):
        def __init__(self, positions: list[_ReplayPosition], *, gate_calls: int = 0) -> None:
            super().__init__(positions, gate_calls=gate_calls)
            self.sample_batch_calls = 0

        def sample(self, batch_size: int) -> list[_ReplayPosition]:
            raise AssertionError("training_loop should use sample_batch when available")

        def sample_batch(
            self,
            batch_size: int,
            encoded_state_size: int,
            policy_size: int,
            value_dim: int,
        ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
            self.sample_batch_calls += 1
            sampled = [self._rng.choice(self._positions) for _ in range(batch_size)]
            states: list[list[float]] = []
            policies: list[list[float]] = []
            values: list[list[float]] = []
            for position in sampled:
                if position.encoded_state_size != encoded_state_size:
                    raise ValueError("encoded_state_size mismatch")
                if position.policy_size != policy_size:
                    raise ValueError("policy_size mismatch")
                states.append(position.encoded_state[:encoded_state_size])
                policies.append(position.policy[:policy_size])
                if value_dim == 1:
                    values.append([position.value])
                elif value_dim == 3:
                    values.append(position.value_wdl[:3])
                else:
                    raise ValueError("value_dim must be 1 or 3")
            return states, policies, values


if _TORCH_AVAILABLE:

    class _TinyPolicyValueNetwork(nn.Module):
        def __init__(self, game_config: GameConfig) -> None:
            super().__init__()
            rows, cols = game_config.board_shape
            self._game_config = game_config
            self._flatten = nn.Flatten()
            self._hidden = nn.Linear(game_config.input_channels * rows * cols, 64)
            self._policy = nn.Linear(64, game_config.action_space_size)
            value_dim = 1 if game_config.value_head_type == "scalar" else 3
            self._value = nn.Linear(64, value_dim)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            features = torch.tanh(self._hidden(self._flatten(x)))
            policy_logits = self._policy(features)
            raw_value = self._value(features)
            if self._game_config.value_head_type == "scalar":
                value = torch.tanh(raw_value)
            else:
                value = torch.softmax(raw_value, dim=-1)
            return policy_logits, value


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required for training-loop tests")
class TrainingLoopTests(unittest.TestCase):
    def _toy_game_config(self, *, supports_symmetry: bool) -> GameConfig:
        num_symmetries = 8 if supports_symmetry else 1
        return GameConfig(
            name="toy",
            board_shape=(3, 3),
            input_channels=2,
            action_space_size=10,
            value_head_type="scalar",
            supports_symmetry=supports_symmetry,
            num_symmetries=num_symmetries,
        )

    def _make_replay_positions(self, game_config: GameConfig, *, count: int = 8) -> list[_ReplayPosition]:
        rows, cols = game_config.board_shape
        encoded_size = game_config.input_channels * rows * cols
        base_state = [0.0] * encoded_size
        base_state[0] = 1.0
        base_policy = [0.0] * game_config.action_space_size
        base_policy[2] = 1.0

        return [
            _ReplayPosition(
                encoded_state=base_state.copy(),
                policy=base_policy.copy(),
                value=1.0,
                value_wdl=[1.0, 0.0, 0.0],
                game_id=1,
                move_number=index,
                encoded_state_size=encoded_size,
                policy_size=game_config.action_space_size,
            )
            for index in range(count)
        ]

    def _make_scaler(self) -> object:
        try:
            return torch.amp.GradScaler(device="cpu", enabled=False)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=False)

    def _reference_gradient_statistics(self, model: nn.Module) -> tuple[float, int]:
        """Preserves the original metric math so the optimized path can be validated against it."""
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

    def test_gradient_statistics_matches_reference_and_skips_none_gradients(self) -> None:
        """Guards TensorBoard gradient metrics by keeping optimized stats numerically aligned with legacy logic."""
        model = nn.Sequential(
            nn.Linear(4, 3, bias=True),
            nn.Linear(3, 2, bias=False),
        )
        first_layer_weight = model[0].weight
        first_layer_bias = model[0].bias
        second_layer_weight = model[1].weight

        first_layer_weight.grad = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10.0
        first_layer_bias.grad = torch.zeros(3, dtype=torch.float32)
        second_layer_weight.grad = None

        expected_norm, expected_nonzero = self._reference_gradient_statistics(model)
        actual_norm, actual_nonzero = _gradient_statistics(model)

        self.assertTrue(math.isclose(actual_norm, expected_norm, rel_tol=1e-6, abs_tol=1e-7))
        self.assertEqual(actual_nonzero, expected_nonzero)

    def test_gradient_statistics_returns_zero_when_no_gradients_present(self) -> None:
        """Protects no-op optimizer steps by requiring stable zero-valued gradient metrics."""
        model = nn.Linear(2, 2)

        global_norm, nonzero_count = _gradient_statistics(model)

        self.assertEqual(global_norm, 0.0)
        self.assertEqual(nonzero_count, 0)

    def test_gradient_statistics_raises_for_non_finite_gradients(self) -> None:
        """Ensures training still fails fast when NaN/Inf gradients appear before optimizer.step."""
        model = nn.Linear(2, 2, bias=False)
        model.weight.grad = torch.tensor(
            [[1.0, float("nan")], [0.0, 1.0]],
            dtype=torch.float32,
        )

        with self.assertRaises(FloatingPointError):
            _gradient_statistics(model)

    def test_prepare_replay_batch_builds_dense_tensors_and_normalizes_policy_rows(self) -> None:
        """Protects replay deserialization so training sees correctly-shaped tensors and valid distributions."""
        config = self._toy_game_config(supports_symmetry=False)
        positions = self._make_replay_positions(config, count=2)
        positions[0].policy[2] = 5.0
        positions[0].policy[3] = 5.0

        states, target_policy, target_value = prepare_replay_batch(
            positions,
            config,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(states.shape), (2, 2, 3, 3))
        self.assertEqual(tuple(target_policy.shape), (2, 10))
        self.assertEqual(tuple(target_value.shape), (2,))
        self.assertTrue(
            torch.allclose(
                target_policy.sum(dim=1),
                torch.ones(2, dtype=target_policy.dtype),
            )
        )

    def test_training_loop_prefers_sample_batch_when_replay_buffer_supports_it(self) -> None:
        """Guards the C++ packed replay path by requiring training_loop to use sample_batch when exposed."""
        config = self._toy_game_config(supports_symmetry=False)
        model = _TinyPolicyValueNetwork(config)
        schedule = StepDecayLRSchedule(entries=((0, 0.1),))
        optimizer = create_optimizer(model, lr_schedule=schedule, momentum=0.9)
        replay_buffer = _PackedReplayBuffer(self._make_replay_positions(config, count=8))
        training_config = TrainingConfig(
            batch_size=4,
            max_steps=3,
            momentum=0.9,
            l2_reg=0.0,
            checkpoint_interval=1000,
            milestone_interval=50000,
            log_interval=10,
            min_buffer_size=4,
            wait_for_buffer_seconds=0.001,
            use_mixed_precision=False,
            device="cpu",
            checkpoint_dir=None,
        )

        result = training_loop(
            model,
            replay_buffer,
            config,
            training_config,
            lr_schedule=schedule,
            optimizer=optimizer,
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(result.final_step, training_config.max_steps)
        self.assertEqual(replay_buffer.sample_batch_calls, training_config.max_steps)

    def test_go_symmetry_augmentation_preserves_pass_action_and_policy_consistency(self) -> None:
        """Ensures data augmentation never corrupts pass probability or board-policy alignment."""
        states = torch.arange(18, dtype=torch.float32).reshape(2, 1, 3, 3)
        policy = torch.zeros(2, 10, dtype=torch.float32)
        policy[0, :9] = torch.arange(9, dtype=torch.float32)
        policy[1, :9] = torch.arange(10, 19, dtype=torch.float32)
        policy[:, 9] = torch.tensor([0.3, 0.7], dtype=torch.float32)

        symmetry_indices = torch.tensor([1, 6], dtype=torch.int64)
        augmented_states, augmented_policy = apply_random_go_symmetry(
            states,
            policy,
            symmetry_indices=symmetry_indices,
        )

        expected_board_0 = torch.rot90(states[0], k=1, dims=(-2, -1))
        expected_board_1 = torch.flip(torch.rot90(states[1], k=2, dims=(-2, -1)), dims=(-1,))
        self.assertTrue(torch.equal(augmented_states[0], expected_board_0))
        self.assertTrue(torch.equal(augmented_states[1], expected_board_1))
        self.assertTrue(torch.equal(augmented_policy[:, 9], policy[:, 9]))

        policy_board = policy[:, :9].reshape(2, 3, 3)
        expected_policy_0 = torch.rot90(policy_board[0], k=1, dims=(-2, -1)).reshape(-1)
        expected_policy_1 = torch.flip(
            torch.rot90(policy_board[1], k=2, dims=(-2, -1)),
            dims=(-1,),
        ).reshape(-1)
        self.assertTrue(torch.equal(augmented_policy[0, :9], expected_policy_0))
        self.assertTrue(torch.equal(augmented_policy[1, :9], expected_policy_1))

    def test_single_training_step_reduces_loss_with_finite_mixed_precision_metrics(self) -> None:
        """Covers TASK-091 directly: one synthetic mixed-precision step should improve loss with live gradients."""
        torch.manual_seed(7)
        config = self._toy_game_config(supports_symmetry=False)
        model = _TinyPolicyValueNetwork(config)
        schedule = StepDecayLRSchedule(entries=((0, 0.1),))
        optimizer = create_optimizer(model, lr_schedule=schedule, momentum=0.9)
        scaler = self._make_scaler()

        positions = self._make_replay_positions(config, count=4)
        states, target_policy, target_value = prepare_replay_batch(
            positions,
            config,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            policy_logits_before, value_before = model(states)
            loss_before = compute_loss_components(
                policy_logits_before,
                value_before,
                target_policy,
                target_value,
                value_type="scalar",
                l2_weight=0.0,
                model=model,
            ).total_loss.item()

        metrics = train_one_step(
            model,
            optimizer,
            states=states,
            target_policy=target_policy,
            target_value=target_value,
            game_config=config,
            lr_schedule=schedule,
            global_step=0,
            l2_reg=0.0,
            scaler=scaler,
            use_mixed_precision=True,
        )
        with torch.no_grad():
            policy_logits_after, value_after = model(states)
            loss_after = compute_loss_components(
                policy_logits_after,
                value_after,
                target_policy,
                target_value,
                value_type="scalar",
                l2_weight=0.0,
                model=model,
            ).total_loss.item()

        self.assertTrue(torch.isfinite(torch.tensor(metrics.loss_total)))
        self.assertTrue(torch.isfinite(torch.tensor(metrics.loss_policy)))
        self.assertTrue(torch.isfinite(torch.tensor(metrics.loss_value)))
        self.assertLess(loss_after, loss_before)
        self.assertGreater(metrics.grad_nonzero_param_count, 0)
        self.assertGreater(metrics.grad_global_norm, 0.0)

    def test_train_one_step_batches_loss_metric_transfer_with_single_four_scalar_stack(self) -> None:
        """Guards TASK-04's GPU sync reduction by requiring one dedicated 4-loss stack in train_one_step."""
        torch.manual_seed(17)
        config = self._toy_game_config(supports_symmetry=False)
        model = _TinyPolicyValueNetwork(config)
        schedule = StepDecayLRSchedule(entries=((0, 0.1),))
        optimizer = create_optimizer(model, lr_schedule=schedule, momentum=0.9)
        scaler = self._make_scaler()

        positions = self._make_replay_positions(config, count=4)
        states, target_policy, target_value = prepare_replay_batch(
            positions,
            config,
            device=torch.device("cpu"),
        )

        original_stack = torch.stack
        stack_signatures: list[tuple[int, tuple[int, ...]]] = []

        def _recording_stack(tensors: object, *args: object, **kwargs: object) -> torch.Tensor:
            if isinstance(tensors, (list, tuple)) and all(
                isinstance(tensor, torch.Tensor) for tensor in tensors
            ):
                tensor_sequence = tuple(tensors)
                stack_signatures.append(
                    (
                        len(tensor_sequence),
                        tuple(tensor.ndim for tensor in tensor_sequence),
                    )
                )
                return original_stack(tensor_sequence, *args, **kwargs)
            return original_stack(tensors, *args, **kwargs)

        with mock.patch("alphazero.training.trainer.torch.stack", side_effect=_recording_stack):
            metrics = train_one_step(
                model,
                optimizer,
                states=states,
                target_policy=target_policy,
                target_value=target_value,
                game_config=config,
                lr_schedule=schedule,
                global_step=0,
                l2_reg=0.0,
                scaler=scaler,
                use_mixed_precision=False,
            )

        four_scalar_stacks = [signature for signature in stack_signatures if signature == (4, (0, 0, 0, 0))]
        self.assertEqual(len(four_scalar_stacks), 1)
        self.assertIsInstance(metrics.loss_total, float)
        self.assertIsInstance(metrics.loss_policy, float)
        self.assertIsInstance(metrics.loss_value, float)
        self.assertIsInstance(metrics.loss_l2, float)

    def test_training_loop_waits_for_min_buffer_then_reduces_loss_and_logs(self) -> None:
        """Validates end-to-end loop behavior: buffer gate, optimization progress, and periodic metric callbacks."""
        config = self._toy_game_config(supports_symmetry=False)
        model = _TinyPolicyValueNetwork(config)
        schedule = StepDecayLRSchedule(entries=((0, 0.2),))
        optimizer = create_optimizer(model, lr_schedule=schedule, momentum=0.9)

        positions = self._make_replay_positions(config, count=8)
        replay_buffer = _ReplayBuffer(positions, gate_calls=2)
        training_config = TrainingConfig(
            batch_size=8,
            max_steps=20,
            momentum=0.9,
            l2_reg=0.0,
            checkpoint_interval=1000,
            milestone_interval=50000,
            log_interval=5,
            min_buffer_size=8,
            wait_for_buffer_seconds=0.001,
            use_mixed_precision=True,
            device="cpu",
            checkpoint_dir=None,
        )

        states, target_policy, target_value = prepare_replay_batch(
            replay_buffer.sample(8),
            config,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            policy_logits_before, value_before = model(states)
            loss_before = compute_loss_components(
                policy_logits_before,
                value_before,
                target_policy,
                target_value,
                value_type="scalar",
                l2_weight=0.0,
                model=model,
            ).total_loss.item()

        sleep_calls: list[float] = []
        logged_steps: list[int] = []

        def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        def logger(step: int, _metrics: Mapping[str, float]) -> None:
            logged_steps.append(step)

        result = training_loop(
            model,
            replay_buffer,
            config,
            training_config,
            lr_schedule=schedule,
            optimizer=optimizer,
            step_logger=logger,
            sleep_fn=fake_sleep,
        )

        with torch.no_grad():
            policy_logits_after, value_after = model(states)
            loss_after = compute_loss_components(
                policy_logits_after,
                value_after,
                target_policy,
                target_value,
                value_type="scalar",
                l2_weight=0.0,
                model=model,
            ).total_loss.item()

        self.assertEqual(result.final_step, training_config.max_steps)
        self.assertGreaterEqual(len(sleep_calls), 2)
        self.assertEqual(logged_steps, [5, 10, 15, 20])
        self.assertLess(loss_after, loss_before)

    def test_checkpoint_round_trip_restores_model_optimizer_step_and_schedule(self) -> None:
        """Protects resume reliability by ensuring checkpoint save/load exactly restores train state."""
        config = self._toy_game_config(supports_symmetry=False)
        model = _TinyPolicyValueNetwork(config)
        schedule = StepDecayLRSchedule(entries=((0, 0.2), (3, 0.02)))
        optimizer = create_optimizer(model, lr_schedule=schedule, momentum=0.9)
        scaler = self._make_scaler()

        positions = self._make_replay_positions(config, count=4)
        states, target_policy, target_value = prepare_replay_batch(
            positions,
            config,
            device=torch.device("cpu"),
        )

        train_one_step(
            model,
            optimizer,
            states=states,
            target_policy=target_policy,
            target_value=target_value,
            game_config=config,
            lr_schedule=schedule,
            global_step=0,
            l2_reg=0.0,
            scaler=scaler,
            use_mixed_precision=False,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = pathlib.Path(tmp_dir)
            saved = save_training_checkpoint(
                model,
                optimizer,
                schedule,
                step=4,
                checkpoint_dir=checkpoint_dir,
                is_milestone=False,
                export_folded_weights=True,
            )
            expected_parameters = {
                name: parameter.detach().clone()
                for name, parameter in model.state_dict().items()
            }

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10.0)

            loaded_step, loaded_schedule = load_training_checkpoint(
                saved.checkpoint_path,
                model,
                optimizer,
                map_location="cpu",
            )

            self.assertEqual(loaded_step, 4)
            self.assertEqual(loaded_schedule.entries, schedule.entries)
            self.assertTrue(saved.checkpoint_path.exists())
            self.assertIsNotNone(saved.folded_weights_path)
            self.assertTrue(saved.folded_weights_path is not None and saved.folded_weights_path.exists())

            for name, parameter in model.state_dict().items():
                self.assertTrue(torch.allclose(parameter, expected_parameters[name]))


if __name__ == "__main__":
    unittest.main()
