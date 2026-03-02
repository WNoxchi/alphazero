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
        compact_replay_buffer_path_for_checkpoint,
        extract_replay_buffer_metadata,
        find_latest_checkpoint,
        list_checkpoints,
        load_checkpoint,
        load_latest_checkpoint,
        load_replay_buffer_state,
        replay_buffer_path_for_checkpoint,
        save_checkpoint,
        save_replay_buffer_state,
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


def _import_cpp_bindings():
    """Try to import alphazero_cpp (same approach as test_bindings.py)."""
    try:
        import alphazero_cpp  # type: ignore[import-not-found]
        return alphazero_cpp
    except ModuleNotFoundError:
        pass

    search_roots = [ROOT / "build", ROOT]
    suffixes = (".so", ".pyd", ".dylib")
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for suffix in suffixes:
            for candidate in search_root.rglob(f"alphazero_cpp*{suffix}"):
                parent = str(candidate.parent)
                if parent not in sys.path:
                    sys.path.insert(0, parent)
                try:
                    import alphazero_cpp  # type: ignore[import-not-found]
                    return alphazero_cpp
                except ModuleNotFoundError:
                    continue
    return None


_cpp = _import_cpp_bindings()
_CPP_AVAILABLE = _cpp is not None and hasattr(_cpp, "CompactReplayBuffer")


@unittest.skipUnless(
    _TORCH_AVAILABLE and _CPP_AVAILABLE,
    "torch and alphazero_cpp with CompactReplayBuffer required",
)
class CompactReplayBufferCheckpointTests(unittest.TestCase):
    """Tests for compact binary replay buffer checkpoint save/load."""

    _NUM_BINARY_PLANES = 3
    _NUM_FLOAT_PLANES = 1
    _FLOAT_PLANE_INDICES = [1]
    _POLICY_SIZE = 10
    _STATE_SIZE = (_NUM_BINARY_PLANES + _NUM_FLOAT_PLANES) * 64

    def _make_buffer(self, capacity=16):
        return _cpp.CompactReplayBuffer(
            capacity=capacity,
            num_binary_planes=self._NUM_BINARY_PLANES,
            num_float_planes=self._NUM_FLOAT_PLANES,
            float_plane_indices=self._FLOAT_PLANE_INDICES,
            full_policy_size=self._POLICY_SIZE,
        )

    def _make_position(self, game_id, move_number):
        import numpy as np
        state = np.zeros(self._STATE_SIZE, dtype=np.float32)
        # Fill binary planes with a pattern.
        for plane in range(self._NUM_BINARY_PLANES + self._NUM_FLOAT_PLANES):
            base = plane * 64
            if plane in self._FLOAT_PLANE_INDICES:
                state[base:base + 64] = float(game_id % 17) / 16.0
            else:
                for sq in range(64):
                    state[base + sq] = 1.0 if (plane + sq + game_id + move_number) % 3 == 0 else 0.0
        policy = np.zeros(self._POLICY_SIZE, dtype=np.float32)
        policy[game_id % self._POLICY_SIZE] = 0.6
        policy[(game_id + 1) % self._POLICY_SIZE] = 0.4
        value = 1.0 if game_id % 2 == 0 else -1.0
        wdl = [1.0, 0.0, 0.0] if value > 0 else [0.0, 0.0, 1.0]
        return _cpp.ReplayPosition.make(
            state.tolist(), policy.tolist(), value, wdl, game_id, move_number, 1.0,
        )

    def _fill_buffer(self, buf, num_games=5, moves_per_game=3):
        for game_id in range(num_games):
            positions = [
                self._make_position(game_id, move)
                for move in range(moves_per_game)
            ]
            buf.add_game(positions)

    def test_compact_save_produces_bin_file(self):
        """save_replay_buffer_state writes .replay.bin for CompactReplayBuffer."""
        buf = self._make_buffer()
        self._fill_buffer(buf)

        with tempfile.TemporaryDirectory() as td:
            cp_path = pathlib.Path(td) / "checkpoint_00001000.pt"
            cp_path.touch()

            result = save_replay_buffer_state(
                buf, cp_path,
                encoded_state_size=self._STATE_SIZE,
                policy_size=self._POLICY_SIZE,
            )
            self.assertIsNotNone(result)
            self.assertTrue(result.name.endswith(".replay.bin"))
            self.assertTrue(result.exists())
            # Should NOT create a .replay.npz.
            npz_path = replay_buffer_path_for_checkpoint(cp_path)
            self.assertFalse(npz_path.exists())

    def test_compact_roundtrip_via_checkpoint(self):
        """Compact save then load restores identical buffer contents."""
        import numpy as np

        source = self._make_buffer()
        self._fill_buffer(source)
        n = source.size()

        with tempfile.TemporaryDirectory() as td:
            cp_path = pathlib.Path(td) / "checkpoint_00001000.pt"
            cp_path.touch()

            save_replay_buffer_state(
                source, cp_path,
                encoded_state_size=self._STATE_SIZE,
                policy_size=self._POLICY_SIZE,
            )

            restored = self._make_buffer()
            loaded = load_replay_buffer_state(
                restored, cp_path,
                encoded_state_size=self._STATE_SIZE,
                policy_size=self._POLICY_SIZE,
            )
            self.assertEqual(loaded, n)
            self.assertEqual(restored.size(), n)

            # Compare dense exports.
            src_dense = source.export_buffer(self._STATE_SIZE, self._POLICY_SIZE)
            dst_dense = restored.export_buffer(self._STATE_SIZE, self._POLICY_SIZE)
            for i in range(len(src_dense)):
                np.testing.assert_array_equal(src_dense[i], dst_dense[i])

    def test_legacy_npz_fallback_loads_into_compact_buffer(self):
        """CompactReplayBuffer can load a legacy .replay.npz via import_buffer."""
        import numpy as np

        buf = self._make_buffer()
        self._fill_buffer(buf)
        n = buf.size()

        # Manually save as legacy .npz format.
        states, policies, values_wdl, game_ids, move_numbers, ownership = buf.export_buffer(
            self._STATE_SIZE, self._POLICY_SIZE,
        )
        with tempfile.TemporaryDirectory() as td:
            cp_path = pathlib.Path(td) / "checkpoint_00001000.pt"
            cp_path.touch()
            npz_path = replay_buffer_path_for_checkpoint(cp_path)
            np.savez(
                npz_path,
                states=states, policies=policies, values_wdl=values_wdl,
                game_ids=game_ids, move_numbers=move_numbers,
                encoded_state_size=np.array(self._STATE_SIZE, dtype=np.int64),
                policy_size=np.array(self._POLICY_SIZE, dtype=np.int64),
                ownership=ownership,
            )

            restored = self._make_buffer()
            loaded = load_replay_buffer_state(
                restored, cp_path,
                encoded_state_size=self._STATE_SIZE,
                policy_size=self._POLICY_SIZE,
            )
            self.assertEqual(loaded, n)

    def test_compact_bin_preferred_over_legacy_npz(self):
        """When both .replay.bin and .replay.npz exist, the compact format is used."""
        import numpy as np

        buf = self._make_buffer()
        self._fill_buffer(buf, num_games=2, moves_per_game=2)

        with tempfile.TemporaryDirectory() as td:
            cp_path = pathlib.Path(td) / "checkpoint_00001000.pt"
            cp_path.touch()

            # Write compact .bin.
            save_replay_buffer_state(
                buf, cp_path,
                encoded_state_size=self._STATE_SIZE,
                policy_size=self._POLICY_SIZE,
            )
            # Also write a legacy .npz with DIFFERENT data (empty).
            npz_path = replay_buffer_path_for_checkpoint(cp_path)
            np.savez(
                npz_path,
                states=np.zeros((1, self._STATE_SIZE), dtype=np.float32),
                policies=np.zeros((1, self._POLICY_SIZE), dtype=np.float32),
                values_wdl=np.zeros((1, 3), dtype=np.float32),
                game_ids=np.zeros(1, dtype=np.uint32),
                move_numbers=np.zeros(1, dtype=np.uint16),
            )

            restored = self._make_buffer()
            loaded = load_replay_buffer_state(
                restored, cp_path,
                encoded_state_size=self._STATE_SIZE,
                policy_size=self._POLICY_SIZE,
            )
            # Should load from .bin (4 positions), not .npz (1 position).
            self.assertEqual(loaded, buf.size())


if __name__ == "__main__":
    unittest.main()
