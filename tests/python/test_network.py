"""Tests for the Python AlphaZero network base interface."""

from __future__ import annotations

import pathlib
import sys
import unittest

import importlib.util


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from alphazero.config import CHESS_CONFIG, GO_CONFIG, GameConfig  # noqa: E402

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if _TORCH_AVAILABLE:
    import torch
    from torch import nn

    from alphazero.network import AlphaZeroNetwork  # noqa: E402


if _TORCH_AVAILABLE:

    class _ShapeCheckingDummyNetwork(AlphaZeroNetwork):
        """Concrete test double that exercises the base class contract."""

        def __init__(self, game_config: GameConfig) -> None:
            super().__init__(game_config)
            rows, cols = game_config.board_shape
            flattened_size = game_config.input_channels * rows * cols
            value_dim = 1 if game_config.value_head_type == "scalar" else 3
            self.policy_head = nn.Linear(flattened_size, game_config.action_space_size)
            self.value_head = nn.Linear(flattened_size, value_dim)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.validate_input_shape(x)
            flattened = x.flatten(start_dim=1)
            return self.policy_head(flattened), self.value_head(flattened)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required for network base tests")
class AlphaZeroNetworkBaseTests(unittest.TestCase):
    def test_base_class_is_abstract_and_cannot_be_instantiated(self) -> None:
        """Prevents incomplete network implementations from silently bypassing forward()."""
        with self.assertRaises(TypeError):
            AlphaZeroNetwork(CHESS_CONFIG)

    def test_constructor_rejects_non_game_config_values(self) -> None:
        """Fails fast when callers wire invalid config objects into model construction."""

        class _InvalidConfigNetwork(AlphaZeroNetwork):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return x, x

        with self.assertRaisesRegex(TypeError, "game_config"):
            _InvalidConfigNetwork(object())  # type: ignore[arg-type]

    def test_subclass_instantiation_supports_chess_and_go_configs(self) -> None:
        """Guards the cross-game constructor contract required by future architectures."""
        chess_model = _ShapeCheckingDummyNetwork(CHESS_CONFIG)
        go_model = _ShapeCheckingDummyNetwork(GO_CONFIG)

        self.assertEqual(chess_model.game_config, CHESS_CONFIG)
        self.assertEqual(go_model.game_config, GO_CONFIG)

    def test_forward_contract_produces_expected_output_shapes_per_game(self) -> None:
        """Ensures downstream MCTS/training code can rely on stable policy/value tensor shapes."""
        chess_model = _ShapeCheckingDummyNetwork(CHESS_CONFIG)
        chess_input = torch.randn(2, 119, 8, 8, dtype=torch.float32)
        chess_policy, chess_value = chess_model(chess_input)
        self.assertEqual(tuple(chess_policy.shape), (2, 4672))
        self.assertEqual(tuple(chess_value.shape), (2, 3))

        go_model = _ShapeCheckingDummyNetwork(GO_CONFIG)
        go_input = torch.randn(3, 17, 19, 19, dtype=torch.float32)
        go_policy, go_value = go_model(go_input)
        self.assertEqual(tuple(go_policy.shape), (3, 362))
        self.assertEqual(tuple(go_value.shape), (3, 1))

    def test_validate_input_shape_rejects_mismatched_dimensions(self) -> None:
        """Catches invalid tensors early so later layers fail with clear contract errors."""
        model = _ShapeCheckingDummyNetwork(CHESS_CONFIG)

        with self.assertRaisesRegex(ValueError, "input channels"):
            model.validate_input_shape(torch.randn(1, 17, 8, 8, dtype=torch.float32))

        with self.assertRaisesRegex(ValueError, "board shape"):
            model.validate_input_shape(torch.randn(1, 119, 7, 8, dtype=torch.float32))

        with self.assertRaisesRegex(ValueError, "shape"):
            model.validate_input_shape(torch.randn(119, 8, 8, dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
