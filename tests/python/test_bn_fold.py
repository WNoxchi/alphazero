"""Tests for batch norm folding utilities."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from alphazero.config import CHESS_CONFIG, GO_CONFIG  # noqa: E402

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if _TORCH_AVAILABLE:
    import torch
    from torch import nn

    from alphazero.network.bn_fold import (  # noqa: E402
        export_folded_model,
        fold_batch_norms,
        fold_conv_bn_pair,
        has_batch_norm_layers,
    )
    from alphazero.network.resnet_se import ResNetSE  # noqa: E402


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required for batch norm folding tests")
class BatchNormFoldTests(unittest.TestCase):
    def test_fold_conv_bn_pair_matches_eval_outputs(self) -> None:
        """Protects the core folding math used to remove BN overhead during inference."""
        torch.manual_seed(7)
        conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=1, bias=True)
        bn = nn.BatchNorm2d(5)

        with torch.no_grad():
            bn.weight.copy_(torch.randn_like(bn.weight))
            bn.bias.copy_(torch.randn_like(bn.bias))
            bn.running_mean.copy_(torch.randn_like(bn.running_mean))
            bn.running_var.copy_(torch.rand_like(bn.running_var) + 0.25)

        conv.eval()
        bn.eval()
        inputs = torch.randn(4, 3, 8, 8, dtype=torch.float32)

        with torch.no_grad():
            expected = bn(conv(inputs))
            folded_conv = fold_conv_bn_pair(conv, bn)
            observed = folded_conv(inputs)

        self.assertTrue(torch.allclose(expected, observed, atol=1e-5, rtol=1e-5))

    def test_fold_batch_norms_exports_batchnorm_free_copy_with_matching_outputs(self) -> None:
        """Ensures exported folded models are BN-free while preserving policy/value predictions."""
        torch.manual_seed(13)
        model = ResNetSE.small(CHESS_CONFIG).eval()
        inputs = torch.randn(2, CHESS_CONFIG.input_channels, *CHESS_CONFIG.board_shape, dtype=torch.float32)

        with torch.no_grad():
            expected_policy, expected_value = model(inputs)

        folded_model = export_folded_model(model).eval()
        self.assertIsNot(folded_model, model)
        self.assertTrue(has_batch_norm_layers(model))
        self.assertFalse(has_batch_norm_layers(folded_model))

        with torch.no_grad():
            observed_policy, observed_value = folded_model(inputs)

        self.assertTrue(torch.allclose(expected_policy, observed_policy, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(expected_value, observed_value, atol=1e-5, rtol=1e-5))

    def test_fold_batch_norms_supports_inplace_mode(self) -> None:
        """Guards the in-place path used when callers intentionally avoid an extra model copy."""
        model = ResNetSE.small(GO_CONFIG).eval()
        self.assertTrue(has_batch_norm_layers(model))

        returned = fold_batch_norms(model, inplace=True)
        self.assertIs(returned, model)
        self.assertFalse(has_batch_norm_layers(model))

    def test_fold_conv_bn_pair_requires_running_statistics(self) -> None:
        """Fails fast for non-foldable BN modules so checkpoint export errors are explicit."""
        conv = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1, bias=False)
        bn = nn.BatchNorm2d(4, track_running_stats=False)

        with self.assertRaisesRegex(ValueError, "running statistics"):
            fold_conv_bn_pair(conv, bn)


if __name__ == "__main__":
    unittest.main()
