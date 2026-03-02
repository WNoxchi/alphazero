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

    from alphazero.network import (  # noqa: E402
        AlphaZeroNetwork,
        OwnershipHead,
        PolicyHead,
        ResNetSE,
        ScalarValueHead,
        WDLValueHead,
    )


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


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required for network head tests")
class NetworkHeadTests(unittest.TestCase):
    def test_policy_head_outputs_logits_with_expected_action_dimension(self) -> None:
        """Guards the policy contract consumed by move selection and loss computation."""
        head = PolicyHead(
            num_filters=128,
            board_shape=GO_CONFIG.board_shape,
            action_space_size=GO_CONFIG.action_space_size,
        )
        features = torch.randn(4, 128, *GO_CONFIG.board_shape, dtype=torch.float32)
        logits = head(features)
        self.assertEqual(tuple(logits.shape), (4, GO_CONFIG.action_space_size))

    def test_scalar_value_head_outputs_tanh_bounded_values(self) -> None:
        """Prevents regressions where scalar heads emit unbounded values outside [-1, 1]."""
        head = ScalarValueHead(num_filters=256, board_shape=GO_CONFIG.board_shape)
        features = torch.randn(3, 256, *GO_CONFIG.board_shape, dtype=torch.float32)
        values = head(features)
        self.assertEqual(tuple(values.shape), (3, 1))
        self.assertTrue(torch.all(values >= -1.0).item())
        self.assertTrue(torch.all(values <= 1.0).item())

    def test_wdl_value_head_outputs_normalized_probabilities(self) -> None:
        """Ensures chess value output remains a valid categorical distribution."""
        head = WDLValueHead(num_filters=256, board_shape=CHESS_CONFIG.board_shape)
        features = torch.randn(5, 256, *CHESS_CONFIG.board_shape, dtype=torch.float32)
        probabilities = head(features)
        self.assertEqual(tuple(probabilities.shape), (5, 3))
        self.assertTrue(torch.all(probabilities >= 0.0).item())
        self.assertTrue(torch.all(probabilities <= 1.0).item())
        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=1),
                torch.ones(probabilities.shape[0], dtype=probabilities.dtype),
                atol=1e-6,
            )
        )

    def test_ownership_head_outputs_unbounded_board_logits(self) -> None:
        """WHY: Go ownership supervision requires one raw logit per board point for BCE-with-logits training."""
        head = OwnershipHead(num_filters=256)
        features = torch.randn(2, 256, *GO_CONFIG.board_shape, dtype=torch.float32)
        logits = head(features)
        self.assertEqual(tuple(logits.shape), (2, GO_CONFIG.board_shape[0] * GO_CONFIG.board_shape[1]))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_ownership_head_initialization_uses_small_weights_and_zero_bias(self) -> None:
        """WHY: small auxiliary-head init prevents ownership loss from overwhelming main heads at startup."""
        torch.manual_seed(5)
        head = OwnershipHead(num_filters=128)
        weight_std = float(head.conv.weight.std().item())
        self.assertTrue(0.005 <= weight_std <= 0.02)
        self.assertTrue(torch.allclose(head.conv.bias, torch.zeros_like(head.conv.bias)))


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required for ResNet+SE tests")
class ResNetSETests(unittest.TestCase):
    def test_small_medium_and_large_profiles_construct_with_expected_depth_and_width(self) -> None:
        """Guards config profile wiring so YAML profile choices map to intended model sizes."""
        model = ResNetSE.small(GO_CONFIG)
        self.assertEqual(model.num_blocks, 10)
        self.assertEqual(model.num_filters, 128)
        self.assertEqual(len(model.residual_blocks), 10)
        self.assertEqual(model.input_conv.out_channels, 128)

        model = ResNetSE.medium(GO_CONFIG)
        self.assertEqual(model.num_blocks, 20)
        self.assertEqual(model.num_filters, 256)
        self.assertEqual(len(model.residual_blocks), 20)
        self.assertEqual(model.input_conv.out_channels, 256)

        model = ResNetSE.large(GO_CONFIG)
        self.assertEqual(model.num_blocks, 40)
        self.assertEqual(model.num_filters, 256)
        self.assertEqual(len(model.residual_blocks), 40)
        self.assertEqual(model.input_conv.out_channels, 256)

    def test_parameter_counts_follow_spec_scale_bands(self) -> None:
        """Catches accidental architecture drift that would materially change compute cost."""

        def count_parameters(model: nn.Module) -> int:
            return sum(parameter.numel() for parameter in model.parameters())

        small_params = count_parameters(ResNetSE.small(GO_CONFIG))
        medium_params = count_parameters(ResNetSE.medium(GO_CONFIG))
        large_params = count_parameters(ResNetSE.large(GO_CONFIG))

        self.assertGreater(small_params, 4_000_000)
        self.assertLess(small_params, 9_000_000)
        self.assertGreater(medium_params, 22_000_000)
        self.assertLess(medium_params, 32_000_000)
        self.assertGreater(large_params, 45_000_000)
        self.assertLess(large_params, 60_000_000)
        self.assertLess(small_params, medium_params)
        self.assertLess(medium_params, large_params)

    def test_initialization_applies_required_zero_and_batch_norm_defaults(self) -> None:
        """Prevents training instability from initialization regressions in critical layers."""
        model = ResNetSE.small(CHESS_CONFIG)

        self.assertTrue(
            torch.allclose(model.policy_head.linear.weight, torch.zeros_like(model.policy_head.linear.weight))
        )
        self.assertTrue(torch.allclose(model.policy_head.linear.bias, torch.zeros_like(model.policy_head.linear.bias)))
        self.assertTrue(torch.allclose(model.value_head.linear.weight, torch.zeros_like(model.value_head.linear.weight)))
        self.assertTrue(torch.allclose(model.value_head.linear.bias, torch.zeros_like(model.value_head.linear.bias)))

        self.assertFalse(torch.allclose(model.input_conv.weight, torch.zeros_like(model.input_conv.weight)))

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertTrue(torch.allclose(module.weight, torch.ones_like(module.weight)))
                self.assertTrue(torch.allclose(module.bias, torch.zeros_like(module.bias)))

    def test_resnet_selects_value_head_variant_for_each_game(self) -> None:
        """Protects game-specific value semantics by wiring scalar vs WDL heads correctly."""
        chess_model = ResNetSE.small(CHESS_CONFIG)
        go_model = ResNetSE.small(GO_CONFIG)

        self.assertIsInstance(chess_model.value_head, WDLValueHead)
        self.assertIsInstance(go_model.value_head, ScalarValueHead)
        self.assertIsNone(chess_model.ownership_head)
        self.assertIsInstance(go_model.ownership_head, OwnershipHead)

    def test_forward_shapes_match_contract_for_chess_and_go(self) -> None:
        """Protects the policy/value output contract consumed by MCTS and training."""
        chess_model = ResNetSE.small(CHESS_CONFIG)
        chess_input = torch.randn(2, CHESS_CONFIG.input_channels, *CHESS_CONFIG.board_shape, dtype=torch.float32)
        chess_policy, chess_value = chess_model(chess_input)

        self.assertEqual(tuple(chess_policy.shape), (2, CHESS_CONFIG.action_space_size))
        self.assertEqual(tuple(chess_value.shape), (2, 3))
        self.assertTrue(torch.all(chess_value >= 0.0).item())
        self.assertTrue(torch.all(chess_value <= 1.0).item())
        self.assertTrue(
            torch.allclose(
                chess_value.sum(dim=1),
                torch.ones(chess_value.shape[0], dtype=chess_value.dtype),
                atol=1e-6,
            )
        )

        go_model = ResNetSE.small(GO_CONFIG)
        go_input = torch.randn(3, GO_CONFIG.input_channels, *GO_CONFIG.board_shape, dtype=torch.float32)
        go_output = go_model(go_input)
        self.assertEqual(len(go_output), 3)
        go_policy, go_value, go_ownership = go_output

        self.assertEqual(tuple(go_policy.shape), (3, GO_CONFIG.action_space_size))
        self.assertEqual(tuple(go_value.shape), (3, 1))
        self.assertEqual(tuple(go_ownership.shape), (3, GO_CONFIG.board_shape[0] * GO_CONFIG.board_shape[1]))
        self.assertTrue(torch.all(go_value >= -1.0).item())
        self.assertTrue(torch.all(go_value <= 1.0).item())


if __name__ == "__main__":
    unittest.main()
