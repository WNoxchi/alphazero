"""Tests for checkpoint save/load utility behavior."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if _TORCH_AVAILABLE:
    import torch
    from torch import nn

    from alphazero.utils.checkpoint import (
        extract_replay_buffer_metadata,
        find_latest_checkpoint,
        list_checkpoints,
        load_checkpoint,
        load_latest_checkpoint,
        save_checkpoint,
    )


if _TORCH_AVAILABLE:

    class _TinyCheckpointNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(4)
            self.head = nn.Linear(4 * 4 * 4, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.bn(self.conv(x))
            return self.head(torch.flatten(features, 1))


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required for checkpoint utility tests")
class CheckpointUtilityTests(unittest.TestCase):
    def _build_model_and_optimizer(self) -> tuple[nn.Module, torch.optim.Optimizer]:
        model = _TinyCheckpointNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        return model, optimizer

    def _prime_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        x = torch.randn(2, 1, 4, 4)
        y = model(x).sum()
        y.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def test_save_and_load_round_trip_preserves_training_state_and_replay_metadata(self) -> None:
        """Protects warm resume correctness by restoring model, optimizer, schedule, and replay metadata."""
        model, optimizer = self._build_model_and_optimizer()
        self._prime_optimizer_state(model, optimizer)

        expected_state = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        schedule_entries = ((0, 0.2), (1000, 0.02))
        replay_metadata = {"write_head": 17, "count": 11, "games_total": 3}

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = pathlib.Path(temp_dir)
            saved = save_checkpoint(
                model,
                optimizer,
                step=1000,
                checkpoint_dir=checkpoint_dir,
                lr_schedule_entries=schedule_entries,
                replay_buffer_metadata=replay_metadata,
                is_milestone=False,
                export_folded_weights=True,
                keep_last=10,
            )

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(5.0)

            loaded = load_checkpoint(saved.checkpoint_path, model, optimizer, map_location="cpu")

            self.assertEqual(loaded.step, 1000)
            self.assertEqual(loaded.lr_schedule_entries, schedule_entries)
            self.assertEqual(loaded.replay_buffer_metadata, replay_metadata)
            self.assertTrue(saved.checkpoint_path.exists())
            self.assertIsNotNone(saved.folded_weights_path)
            self.assertTrue(saved.folded_weights_path is not None and saved.folded_weights_path.exists())

            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.allclose(tensor, expected_state[name]))

    def test_rolling_checkpoint_prune_keeps_only_last_k_and_preserves_milestones(self) -> None:
        """Guards retention policy so rolling checkpoints do not delete milestone archives."""
        model, optimizer = self._build_model_and_optimizer()
        self._prime_optimizer_state(model, optimizer)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = pathlib.Path(temp_dir)

            for step in (1000, 2000, 3000, 4000):
                save_checkpoint(
                    model,
                    optimizer,
                    step=step,
                    checkpoint_dir=checkpoint_dir,
                    lr_schedule_entries=((0, 0.2),),
                    replay_buffer_metadata={"count": step // 1000},
                    is_milestone=False,
                    export_folded_weights=True,
                    keep_last=2,
                )

            milestone = save_checkpoint(
                model,
                optimizer,
                step=3000,
                checkpoint_dir=checkpoint_dir,
                lr_schedule_entries=((0, 0.2),),
                replay_buffer_metadata={"count": 3},
                is_milestone=True,
                export_folded_weights=False,
                keep_last=2,
            )

            regular = list_checkpoints(checkpoint_dir, include_milestones=False)
            self.assertEqual(
                [path.name for path in regular],
                ["checkpoint_00003000.pt", "checkpoint_00004000.pt"],
            )
            self.assertFalse((checkpoint_dir / "checkpoint_00001000.pt").exists())
            self.assertFalse((checkpoint_dir / "checkpoint_00001000_folded.pt").exists())
            self.assertTrue(milestone.checkpoint_path.exists())

    def test_latest_checkpoint_resolution_prefers_regular_checkpoint_when_steps_tie(self) -> None:
        """Prevents ambiguous resume behavior by deterministically preferring rolling checkpoints on tie."""
        model, optimizer = self._build_model_and_optimizer()
        self._prime_optimizer_state(model, optimizer)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = pathlib.Path(temp_dir)
            save_checkpoint(
                model,
                optimizer,
                step=5000,
                checkpoint_dir=checkpoint_dir,
                is_milestone=True,
                export_folded_weights=False,
            )
            regular = save_checkpoint(
                model,
                optimizer,
                step=5000,
                checkpoint_dir=checkpoint_dir,
                is_milestone=False,
                export_folded_weights=False,
            )

            latest_path = find_latest_checkpoint(checkpoint_dir, include_milestones=True)
            self.assertEqual(latest_path, regular.checkpoint_path)

            latest_loaded = load_latest_checkpoint(
                checkpoint_dir,
                model,
                optimizer,
                include_milestones=True,
                map_location="cpu",
            )
            self.assertIsNotNone(latest_loaded)
            assert latest_loaded is not None
            self.assertEqual(latest_loaded.step, 5000)
            self.assertFalse(latest_loaded.is_milestone)

    def test_extract_replay_buffer_metadata_prefers_checkpoint_metadata_and_falls_back_to_fields(self) -> None:
        """Ensures replay metadata capture remains robust across different binding surfaces."""

        class _ReplayWithAccessor:
            def checkpoint_metadata(self) -> dict[str, int]:
                return {"write_head": 9, "count": 7, "games_total": 2}

        class _ReplayWithFields:
            write_head = 4

            def count(self) -> int:
                return 3

            def size(self) -> int:
                return 10

        from_accessor = extract_replay_buffer_metadata(_ReplayWithAccessor())
        self.assertEqual(from_accessor, {"write_head": 9, "count": 7, "games_total": 2})

        from_fields = extract_replay_buffer_metadata(_ReplayWithFields())
        self.assertEqual(from_fields, {"write_head": 4, "count": 3})


if __name__ == "__main__":
    unittest.main()
