"""Tests for AlphaZero training loss computation."""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if _TORCH_AVAILABLE:
    import torch
    from torch import nn

    from alphazero.training.loss import (  # noqa: E402
        compute_loss,
        compute_loss_components,
        l2_regularization_loss,
        policy_cross_entropy_loss,
        scalar_value_loss,
        wdl_value_loss,
    )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required for loss tests")
class LossFunctionTests(unittest.TestCase):
    def test_policy_cross_entropy_matches_hand_computed_reference(self) -> None:
        """Protects policy-loss math because it is the primary training supervision signal."""
        policy_logits = torch.tensor(
            [[1.0, 0.0, -1.0], [0.5, -0.5, 0.0]],
            dtype=torch.float32,
        )
        target_policy = torch.tensor(
            [[0.75, 0.25, 0.0], [0.1, 0.2, 0.7]],
            dtype=torch.float32,
        )

        observed = policy_cross_entropy_loss(policy_logits, target_policy)

        expected_rows: list[float] = []
        for logits_row, target_row in zip(policy_logits.tolist(), target_policy.tolist(), strict=True):
            log_sum_exp = math.log(sum(math.exp(value) for value in logits_row))
            cross_entropy = -sum(
                probability * (logit - log_sum_exp)
                for probability, logit in zip(target_row, logits_row, strict=True)
            )
            expected_rows.append(cross_entropy)
        expected = sum(expected_rows) / len(expected_rows)

        self.assertAlmostEqual(observed.item(), expected, places=6)

    def test_policy_cross_entropy_uses_legal_action_mask(self) -> None:
        """Ensures illegal actions are excluded from softmax normalization when a mask is provided."""
        policy_logits = torch.tensor([[0.2, -0.1, 5.0, 0.4]], dtype=torch.float32)
        target_policy = torch.tensor([[0.7, 0.3, 0.0, 0.0]], dtype=torch.float32)
        legal_action_mask = torch.tensor([[True, True, False, True]])

        observed = policy_cross_entropy_loss(
            policy_logits,
            target_policy,
            legal_action_mask=legal_action_mask,
        )
        unmasked = policy_cross_entropy_loss(policy_logits, target_policy)

        legal_logits = [0.2, -0.1, 0.4]
        legal_targets = [0.7, 0.3, 0.0]
        log_sum_exp = math.log(sum(math.exp(value) for value in legal_logits))
        expected = -sum(
            probability * (logit - log_sum_exp)
            for probability, logit in zip(legal_targets, legal_logits, strict=True)
        )

        self.assertAlmostEqual(observed.item(), expected, places=6)
        self.assertGreater(unmasked.item(), observed.item() + 1.0)

    def test_scalar_value_loss_matches_hand_computed_mse(self) -> None:
        """Prevents value-head regression for scalar games where outcome targets are in [-1, 1]."""
        value = torch.tensor([[0.1], [-0.4], [0.8]], dtype=torch.float32)
        target_value = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)

        observed = scalar_value_loss(value, target_value)
        expected = ((1.0 - 0.1) ** 2 + (-1.0 + 0.4) ** 2 + (0.0 - 0.8) ** 2) / 3.0

        self.assertAlmostEqual(observed.item(), expected, places=6)

    def test_wdl_value_loss_matches_hand_computed_cross_entropy(self) -> None:
        """Locks the chess WDL objective to exact cross-entropy math used in training."""
        value = torch.tensor(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.3, 0.6],
            ],
            dtype=torch.float32,
        )
        target_value = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        observed = wdl_value_loss(value, target_value)
        expected = -(math.log(0.7) + math.log(0.6) + math.log(0.6)) / 3.0

        self.assertAlmostEqual(observed.item(), expected, places=6)

    def test_l2_regularization_includes_weights_and_biases(self) -> None:
        """Guards explicit L2 behavior expected by the pipeline loss pseudocode."""
        model = nn.Sequential(nn.Linear(2, 2, bias=True), nn.Linear(2, 1, bias=True))

        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[1.0, -2.0], [0.5, 0.25]], dtype=torch.float32))
            model[0].bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float32))
            model[1].weight.copy_(torch.tensor([[3.0, -4.0]], dtype=torch.float32))
            model[1].bias.copy_(torch.tensor([0.5], dtype=torch.float32))

        observed = l2_regularization_loss(model)
        expected = 30.6125

        self.assertAlmostEqual(observed.item(), expected, places=6)

    def test_compute_loss_components_composes_terms_with_equal_policy_and_value_weights(self) -> None:
        """Ensures the total objective is policy + value + c*L2 with no hidden rescaling."""
        policy_logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        target_policy = torch.tensor([[1.0, 0.0], [0.25, 0.75]], dtype=torch.float32)
        value = torch.tensor([[0.2], [-0.4]], dtype=torch.float32)
        target_value = torch.tensor([1.0, -1.0], dtype=torch.float32)

        model = nn.Linear(1, 1, bias=True)
        with torch.no_grad():
            model.weight.fill_(2.0)
            model.bias.fill_(-1.0)

        l2_weight = 0.1
        components = compute_loss_components(
            policy_logits,
            value,
            target_policy,
            target_value,
            value_type="scalar",
            l2_weight=l2_weight,
            model=model,
        )
        total_loss = compute_loss(
            policy_logits,
            value,
            target_policy,
            target_value,
            value_type="scalar",
            l2_weight=l2_weight,
            model=model,
        )

        row0_policy = -(1.0 * (0.0 - math.log(math.exp(0.0) + math.exp(1.0))))
        row1_policy = -(
            0.25 * (1.0 - math.log(math.exp(1.0) + math.exp(0.0)))
            + 0.75 * (0.0 - math.log(math.exp(1.0) + math.exp(0.0)))
        )
        expected_policy = (row0_policy + row1_policy) / 2.0
        expected_value = ((1.0 - 0.2) ** 2 + (-1.0 + 0.4) ** 2) / 2.0
        expected_l2 = 5.0
        expected_total = expected_policy + expected_value + l2_weight * expected_l2

        self.assertAlmostEqual(components.policy_loss.item(), expected_policy, places=6)
        self.assertAlmostEqual(components.value_loss.item(), expected_value, places=6)
        self.assertAlmostEqual(components.l2_loss.item(), expected_l2, places=6)
        self.assertAlmostEqual(components.total_loss.item(), expected_total, places=6)
        self.assertAlmostEqual(total_loss.item(), expected_total, places=6)

    def test_compute_loss_components_supports_wdl_value_mode(self) -> None:
        """Ensures chess mode routes to WDL cross-entropy and still composes total loss correctly."""
        policy_logits = torch.tensor([[0.2, -0.1]], dtype=torch.float32)
        target_policy = torch.tensor([[0.3, 0.7]], dtype=torch.float32)
        value = torch.tensor([[0.1, 0.8, 0.1]], dtype=torch.float32)
        target_value = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

        components = compute_loss_components(
            policy_logits,
            value,
            target_policy,
            target_value,
            value_type="wdl",
            l2_weight=0.0,
            model=None,
        )

        policy_log_sum_exp = math.log(math.exp(0.2) + math.exp(-0.1))
        expected_policy = -(0.3 * (0.2 - policy_log_sum_exp) + 0.7 * (-0.1 - policy_log_sum_exp))
        expected_value = -math.log(0.8)

        self.assertAlmostEqual(components.policy_loss.item(), expected_policy, places=6)
        self.assertAlmostEqual(components.value_loss.item(), expected_value, places=6)
        self.assertAlmostEqual(components.l2_loss.item(), 0.0, places=6)
        self.assertAlmostEqual(
            components.total_loss.item(),
            expected_policy + expected_value,
            places=6,
        )

    def test_invalid_value_type_is_rejected(self) -> None:
        """Fails fast when callers misconfigure value head type in the training loop."""
        policy_logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        target_policy = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        value = torch.tensor([[0.1]], dtype=torch.float32)
        target_value = torch.tensor([0.0], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "value_type"):
            compute_loss_components(
                policy_logits,
                value,
                target_policy,
                target_value,
                value_type="unsupported",  # type: ignore[arg-type]
            )

    def test_target_probability_on_illegal_action_is_rejected(self) -> None:
        """Catches mismatched legal-mask wiring that would silently corrupt policy supervision."""
        policy_logits = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        target_policy = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        legal_action_mask = torch.tensor([[True, True, False]])

        with self.assertRaisesRegex(ValueError, "illegal"):
            policy_cross_entropy_loss(
                policy_logits,
                target_policy,
                legal_action_mask=legal_action_mask,
            )


if __name__ == "__main__":
    unittest.main()
