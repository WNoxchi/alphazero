"""Contract tests for the optional pybind11 `alphazero_cpp` extension."""

from __future__ import annotations

import pathlib
import re
import sys
import threading
import time
import tempfile
import types
import unittest
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))


def _import_bindings() -> tuple[types.ModuleType | None, str]:
    try:
        import alphazero_cpp  # type: ignore[import-not-found]

        return alphazero_cpp, ""
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

                    return alphazero_cpp, ""
                except ModuleNotFoundError:
                    continue

    return None, "alphazero_cpp extension is not built in this environment"


alphazero_cpp, _IMPORT_SKIP_REASON = _import_bindings()


def _require_bindings() -> Any:
    if alphazero_cpp is None:
        raise AssertionError("alphazero_cpp bindings are unavailable")
    return alphazero_cpp


@unittest.skipIf(alphazero_cpp is None, _IMPORT_SKIP_REASON)
class PythonBindingsTests(unittest.TestCase):
    def test_game_state_interface_is_callable_from_python(self) -> None:
        """Ensures Python-side users can create states and invoke all required `GameState` methods."""
        bindings = _require_bindings()
        chess_config = bindings.chess_game_config()
        state = chess_config.new_game()
        self.assertEqual(chess_config.dirichlet_alpha_reference_moves, 30)

        legal_actions = state.legal_actions()
        self.assertTrue(legal_actions)
        self.assertFalse(state.is_terminal())
        self.assertIn(state.current_player(), (0, 1))
        self.assertIsInstance(state.hash(), int)
        self.assertIsInstance(state.to_string(), str)

        encoded = state.encode()
        self.assertEqual(encoded.shape, (119, 8, 8))

        next_state = state.apply_action(legal_actions[0])
        self.assertIsInstance(next_state, bindings.GameState)

        cloned = state.clone()
        self.assertIsInstance(cloned, bindings.GameState)

        go_state = bindings.GoState()
        self.assertEqual(bindings.go_game_config().dirichlet_alpha_reference_moves, 361)
        go_tensor = go_state.encode()
        self.assertEqual(tuple(go_tensor.shape), (17, 19, 19))

    def test_replay_buffer_round_trips_positions(self) -> None:
        """Verifies that sampled replay entries preserve shape metadata and payload slices."""
        bindings = _require_bindings()
        replay_buffer = bindings.ReplayBuffer(capacity=16, random_seed=12345)

        encoded_state = [0.1, 0.2, 0.3, 0.4]
        policy = [0.6, 0.4]
        value_wdl = [1.0, 0.0, 0.0]
        position = bindings.ReplayPosition.make(
            encoded_state=encoded_state,
            policy=policy,
            value=1.0,
            value_wdl=value_wdl,
            game_id=7,
            move_number=3,
            training_weight=0.625,
        )

        replay_buffer.add_game([position])
        sampled = replay_buffer.sample(1)
        self.assertEqual(len(sampled), 1)
        sample = sampled[0]

        self.assertEqual(sample.encoded_state_size, len(encoded_state))
        self.assertEqual(sample.policy_size, len(policy))
        self.assertAlmostEqual(sample.value, 1.0)
        self.assertAlmostEqual(sample.training_weight, 0.625)
        self.assertEqual(sample.game_id, 7)
        self.assertEqual(sample.move_number, 3)
        import numpy.testing as npt
        npt.assert_allclose(sample.encoded_state, encoded_state, rtol=1e-6)
        npt.assert_allclose(sample.policy, policy, rtol=1e-6)
        npt.assert_allclose(sample.value_wdl, value_wdl, rtol=1e-6)

    def test_replay_buffer_sample_batch_returns_packed_numpy_arrays(self) -> None:
        """Ensures packed replay sampling exposes contiguous numpy tensors with aligned state/policy/value rows."""
        bindings = _require_bindings()
        replay_buffer = bindings.ReplayBuffer(capacity=16, random_seed=777)

        def scalar_for(game_id: int, move_number: int) -> float:
            bucket = (game_id + move_number) % 3
            if bucket == 0:
                return 1.0
            if bucket == 1:
                return 0.0
            return -1.0

        def wdl_for(value: float) -> list[float]:
            if value > 0.0:
                return [1.0, 0.0, 0.0]
            if value < 0.0:
                return [0.0, 0.0, 1.0]
            return [0.0, 1.0, 0.0]

        def weight_for(game_id: int, move_number: int) -> float:
            return 0.5 + ((game_id + move_number) % 4) * 0.125

        positions = []
        for game_id, move_number in ((11, 1), (12, 2), (13, 3)):
            signature = float(game_id * 1000 + move_number)
            scalar_value = scalar_for(game_id, move_number)
            positions.append(
                bindings.ReplayPosition.make(
                    encoded_state=[signature, float(move_number), signature + 2.0, signature + 3.0],
                    policy=[signature + 0.5, float(game_id), float(move_number)],
                    value=scalar_value,
                    value_wdl=wdl_for(scalar_value),
                    game_id=game_id,
                    move_number=move_number,
                    training_weight=weight_for(game_id, move_number),
                )
            )
        replay_buffer.add_game(positions)

        import numpy as np
        import numpy.testing as npt

        states, policies, scalar_values, scalar_weights, scalar_ownership = replay_buffer.sample_batch(
            batch_size=8,
            encoded_state_size=4,
            policy_size=3,
            value_dim=1,
        )
        self.assertEqual(states.shape, (8, 4))
        self.assertEqual(policies.shape, (8, 3))
        self.assertEqual(scalar_values.shape, (8, 1))
        self.assertEqual(states.dtype, np.float32)
        self.assertEqual(policies.dtype, np.float32)
        self.assertEqual(scalar_values.dtype, np.float32)
        self.assertEqual(scalar_weights.shape, (8,))
        self.assertEqual(scalar_weights.dtype, np.float32)
        self.assertEqual(scalar_ownership.size, 0)

        for row in range(8):
            game_id = int(round(float(policies[row, 1])))
            move_number = int(round(float(policies[row, 2])))
            expected_signature = float(game_id * 1000 + move_number)
            self.assertAlmostEqual(float(states[row, 0]), expected_signature)
            self.assertAlmostEqual(float(states[row, 1]), float(move_number))
            self.assertAlmostEqual(float(policies[row, 0]), expected_signature + 0.5)
            self.assertAlmostEqual(float(scalar_values[row, 0]), scalar_for(game_id, move_number))
            self.assertAlmostEqual(float(scalar_weights[row]), weight_for(game_id, move_number))

        _, wdl_policies, wdl_values, wdl_weights, wdl_ownership = replay_buffer.sample_batch(
            batch_size=8,
            encoded_state_size=4,
            policy_size=3,
            value_dim=3,
        )
        self.assertEqual(wdl_policies.shape, (8, 3))
        self.assertEqual(wdl_values.shape, (8, 3))
        self.assertEqual(wdl_weights.shape, (8,))
        self.assertEqual(wdl_ownership.size, 0)
        for row in range(8):
            game_id = int(round(float(wdl_policies[row, 1])))
            move_number = int(round(float(wdl_policies[row, 2])))
            expected = wdl_for(scalar_for(game_id, move_number))
            npt.assert_allclose(wdl_values[row], expected, rtol=1e-6)
            self.assertAlmostEqual(float(wdl_weights[row]), weight_for(game_id, move_number))

    def test_compact_replay_buffer_binding_matches_dense_buffer_contract(self) -> None:
        """Validates that Python can drive the compact buffer with the same add/sample/export/import API surface."""
        bindings = _require_bindings()
        compact_buffer = bindings.CompactReplayBuffer(
            capacity=8,
            num_binary_planes=1,
            num_float_planes=1,
            float_plane_indices=[1],
            full_policy_size=5,
            random_seed=4321,
            squares_per_plane=64,
        )

        import numpy.testing as npt

        binary_plane = [1.0 if (square % 2) == 0 else 0.0 for square in range(64)]
        float_plane = [0.25] * 64
        encoded_state = binary_plane + float_plane
        policy = [0.7, 0.0, 0.0, 0.3, 0.0]
        value_wdl = [1.0, 0.0, 0.0]
        position = bindings.ReplayPosition.make(
            encoded_state=encoded_state,
            policy=policy,
            value=1.0,
            value_wdl=value_wdl,
            game_id=77,
            move_number=9,
        )
        compact_buffer.add_game([position])

        sampled = compact_buffer.sample(1)
        self.assertEqual(len(sampled), 1)
        self.assertEqual(sampled[0].game_id, 77)
        self.assertEqual(sampled[0].move_number, 9)

        states, policies, values_wdl, weights, ownership = compact_buffer.sample_batch(
            batch_size=1,
            encoded_state_size=128,
            policy_size=5,
            value_dim=3,
        )
        npt.assert_allclose(states[0, :64], binary_plane, rtol=1e-6)
        expected_quantized_float = round(float_plane[0] * 255.0) / 255.0
        npt.assert_allclose(states[0, 64:], [expected_quantized_float] * 64, rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(float(policies[0, 0]), 0.7, places=3)
        self.assertAlmostEqual(float(policies[0, 3]), 0.3, places=3)
        npt.assert_allclose(values_wdl[0], value_wdl, rtol=1e-6)
        npt.assert_allclose(weights, [1.0], rtol=1e-6)
        self.assertEqual(ownership.size, 0)

        exported = compact_buffer.export_buffer(encoded_state_size=128, policy_size=5)
        restored = bindings.CompactReplayBuffer(
            capacity=8,
            num_binary_planes=1,
            num_float_planes=1,
            float_plane_indices=[1],
            full_policy_size=5,
            random_seed=9876,
            squares_per_plane=64,
        )
        (
            exported_states,
            exported_policies,
            exported_values_wdl,
            exported_game_ids,
            exported_move_numbers,
            exported_ownership,
        ) = exported
        restored.import_buffer(
            exported_states,
            exported_policies,
            exported_values_wdl,
            exported_game_ids,
            exported_move_numbers,
            encoded_state_size=128,
            policy_size=5,
            ownership=exported_ownership,
        )
        self.assertEqual(restored.size(), 1)
        restored_sample = restored.sample(1)[0]
        self.assertEqual(restored_sample.game_id, 77)
        self.assertEqual(restored_sample.move_number, 9)

    def test_replay_buffer_sample_batch_returns_ownership_when_present(self) -> None:
        """Ensures ownership targets appear as the 5th packed-array output when replay rows include ownership labels."""
        bindings = _require_bindings()
        replay_buffer = bindings.ReplayBuffer(capacity=8, random_seed=17)

        ownership = [0.0] * int(bindings.ReplayPosition.MAX_BOARD_AREA)
        ownership[0] = 1.0
        ownership[1] = -1.0
        ownership[4] = 1.0
        replay_buffer.add_game(
            [
                bindings.ReplayPosition.make(
                    encoded_state=[1.0, 2.0, 3.0, 4.0],
                    policy=[0.5, 0.3, 0.2],
                    value=1.0,
                    value_wdl=[1.0, 0.0, 0.0],
                    game_id=9,
                    move_number=0,
                    ownership=ownership,
                )
            ]
        )

        _states, _policies, _values, _weights, sampled_ownership = replay_buffer.sample_batch(
            batch_size=2,
            encoded_state_size=4,
            policy_size=3,
            value_dim=1,
        )
        self.assertEqual(sampled_ownership.shape, (2, int(bindings.ReplayPosition.MAX_BOARD_AREA)))
        self.assertAlmostEqual(float(sampled_ownership[0, 0]), 1.0)
        self.assertAlmostEqual(float(sampled_ownership[0, 1]), -1.0)
        self.assertAlmostEqual(float(sampled_ownership[0, 2]), 0.0)
        self.assertAlmostEqual(float(sampled_ownership[1, 4]), 1.0)

    def test_export_buffer_rejects_mixed_ownership_payloads(self) -> None:
        """WHY: mixed ownership/non-ownership rows must fail deterministically so checkpoints cannot silently drop ownership labels."""
        bindings = _require_bindings()

        no_ownership = bindings.ReplayPosition.make(
            encoded_state=[1.0, 2.0, 3.0, 4.0],
            policy=[0.5, 0.3, 0.2],
            value=0.0,
            value_wdl=[0.0, 1.0, 0.0],
            game_id=70,
            move_number=0,
        )
        with_ownership = bindings.ReplayPosition.make(
            encoded_state=[4.0, 3.0, 2.0, 1.0],
            policy=[0.2, 0.3, 0.5],
            value=1.0,
            value_wdl=[1.0, 0.0, 0.0],
            game_id=70,
            move_number=1,
            ownership=[1.0] * int(bindings.ReplayPosition.MAX_BOARD_AREA),
        )

        dense_buffer = bindings.ReplayBuffer(capacity=8, random_seed=71)
        dense_buffer.add_game([no_ownership, with_ownership])
        with self.assertRaisesRegex(ValueError, "mixed ownership presence"):
            dense_buffer.export_buffer(encoded_state_size=4, policy_size=3)

        compact_buffer = bindings.CompactReplayBuffer(
            capacity=8,
            num_binary_planes=1,
            num_float_planes=0,
            float_plane_indices=[],
            full_policy_size=4,
            random_seed=72,
            squares_per_plane=bindings.ReplayPosition.MAX_BOARD_AREA,
        )
        compact_buffer.add_game(
            [
                bindings.ReplayPosition.make(
                    encoded_state=[0.0] * int(bindings.ReplayPosition.MAX_BOARD_AREA),
                    policy=[1.0, 0.0, 0.0, 0.0],
                    value=0.0,
                    value_wdl=[0.0, 1.0, 0.0],
                    game_id=71,
                    move_number=0,
                ),
                bindings.ReplayPosition.make(
                    encoded_state=[1.0] * int(bindings.ReplayPosition.MAX_BOARD_AREA),
                    policy=[0.0, 1.0, 0.0, 0.0],
                    value=0.0,
                    value_wdl=[0.0, 1.0, 0.0],
                    game_id=71,
                    move_number=1,
                    ownership=[-1.0] * int(bindings.ReplayPosition.MAX_BOARD_AREA),
                ),
            ]
        )
        with self.assertRaisesRegex(ValueError, "mixed ownership presence"):
            compact_buffer.export_buffer(
                encoded_state_size=int(bindings.ReplayPosition.MAX_BOARD_AREA),
                policy_size=4,
            )

    def test_compact_replay_buffer_binding_exposes_recency_sampling_controls(self) -> None:
        """WHY: Python training entrypoints must be able to select recency-weighted replay sampling when configured."""
        bindings = _require_bindings()
        self.assertTrue(hasattr(bindings, "ReplaySamplingStrategy"))
        strategy = bindings.ReplaySamplingStrategy.RECENCY_WEIGHTED

        compact_buffer = bindings.CompactReplayBuffer(
            capacity=16,
            num_binary_planes=1,
            num_float_planes=1,
            float_plane_indices=[1],
            full_policy_size=5,
            random_seed=2026,
            sampling_strategy=strategy,
            recency_weight_lambda=2.0,
        )

        binary_plane = [1.0 if (square % 2) == 0 else 0.0 for square in range(64)]
        float_plane = [0.5] * 64
        encoded_state = binary_plane + float_plane
        compact_buffer.add_game(
            [
                bindings.ReplayPosition.make(
                    encoded_state=encoded_state,
                    policy=[1.0, 0.0, 0.0, 0.0, 0.0],
                    value=1.0,
                    value_wdl=[1.0, 0.0, 0.0],
                    game_id=1,
                    move_number=0,
                ),
                bindings.ReplayPosition.make(
                    encoded_state=encoded_state,
                    policy=[0.0, 1.0, 0.0, 0.0, 0.0],
                    value=-1.0,
                    value_wdl=[0.0, 0.0, 1.0],
                    game_id=2,
                    move_number=0,
                ),
            ]
        )

        sampled = compact_buffer.sample(4)
        self.assertEqual(len(sampled), 4)
        for position in sampled:
            self.assertIn(position.game_id, {1, 2})

    def test_compact_buffer_loads_dense_replay_checkpoint_without_format_changes(self) -> None:
        """WHY: dense replay checkpoints must stay loadable after switching training to compact replay storage."""
        bindings = _require_bindings()
        from alphazero.utils.checkpoint import load_replay_buffer_state, save_replay_buffer_state

        dense_buffer = bindings.ReplayBuffer(capacity=8, random_seed=1337)

        binary_even = [1.0 if (square % 2) == 0 else 0.0 for square in range(64)]
        binary_odd = [1.0 if (square % 2) == 1 else 0.0 for square in range(64)]
        encoded_a = binary_even + ([1.0] * 64)
        encoded_b = binary_odd + ([0.0] * 64)
        dense_buffer.add_game(
            [
                bindings.ReplayPosition.make(
                    encoded_state=encoded_a,
                    policy=[0.0, 0.6, 0.0, 0.4, 0.0],
                    value=1.0,
                    value_wdl=[1.0, 0.0, 0.0],
                    game_id=11,
                    move_number=3,
                ),
                bindings.ReplayPosition.make(
                    encoded_state=encoded_b,
                    policy=[0.125, 0.0, 0.875, 0.0, 0.0],
                    value=-1.0,
                    value_wdl=[0.0, 0.0, 1.0],
                    game_id=12,
                    move_number=4,
                ),
            ]
        )

        compact_buffer = bindings.CompactReplayBuffer(
            capacity=8,
            num_binary_planes=1,
            num_float_planes=1,
            float_plane_indices=[1],
            full_policy_size=5,
            random_seed=2024,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = pathlib.Path(temp_dir) / "checkpoint_00000042.pt"
            checkpoint_path.write_text("placeholder", encoding="utf-8")

            replay_path = save_replay_buffer_state(
                dense_buffer,
                checkpoint_path,
                encoded_state_size=128,
                policy_size=5,
            )
            self.assertIsNotNone(replay_path)
            assert replay_path is not None
            self.assertTrue(replay_path.exists())

            loaded = load_replay_buffer_state(
                compact_buffer,
                checkpoint_path,
                encoded_state_size=128,
                policy_size=5,
            )
            self.assertEqual(loaded, 2)

        import numpy.testing as npt

        dense_export = dense_buffer.export_buffer(encoded_state_size=128, policy_size=5)
        compact_export = compact_buffer.export_buffer(encoded_state_size=128, policy_size=5)

        dense_states, dense_policies, dense_values_wdl, dense_game_ids, dense_move_numbers, dense_ownership = dense_export
        compact_states, compact_policies, compact_values_wdl, compact_game_ids, compact_move_numbers, compact_ownership = compact_export

        npt.assert_allclose(compact_states[:, :64], dense_states[:, :64], rtol=0.0, atol=0.0)
        npt.assert_allclose(compact_states[:, 64:], dense_states[:, 64:], rtol=0.0, atol=(1.0 / 255.0) + 1e-7)
        npt.assert_allclose(compact_policies, dense_policies, rtol=0.0, atol=1e-3)
        npt.assert_allclose(compact_values_wdl, dense_values_wdl, rtol=0.0, atol=0.0)
        npt.assert_array_equal(compact_game_ids, dense_game_ids)
        npt.assert_array_equal(compact_move_numbers, dense_move_numbers)
        self.assertEqual(dense_ownership.size, 0)
        self.assertEqual(compact_ownership.size, 0)

    def test_chess_uci_helpers_round_trip_legal_actions(self) -> None:
        """Ensures play-mode move I/O remains stable for chess UCI text entry and engine integration."""
        bindings = _require_bindings()
        state = bindings.ChessState()

        legal_pairs = state.legal_actions_uci()
        self.assertTrue(legal_pairs)

        first_action, first_uci = legal_pairs[0]
        self.assertEqual(state.action_to_uci(first_action), first_uci)
        self.assertEqual(state.uci_to_action(first_uci), first_action)

    def test_mcts_search_binding_runs_simulations_and_selects_legal_action(self) -> None:
        """Verifies Python can drive standalone MCTS (outside SelfPlayManager) for interactive play flows."""
        bindings = _require_bindings()
        game_config = bindings.go_game_config()
        search_config = bindings.SearchConfig()
        search_config.simulations_per_move = 8
        search_config.enable_dirichlet_noise = False
        search_config.temperature = 0.0
        search_config.temperature_moves = 0
        search_config.enable_resignation = False

        search = bindings.MctsSearch(game_config, search_config, node_arena_capacity=8192)
        state = bindings.GoState()
        legal_actions = state.legal_actions()
        preferred_action = legal_actions[0]

        def evaluator(_state: object) -> dict[str, object]:
            policy = [-1000.0] * game_config.action_space_size
            policy[preferred_action] = 1000.0
            return {
                "policy": policy,
                "value": 0.0,
                "policy_is_logits": True,
            }

        search.set_root_state(state)
        search.run_simulations(evaluator, simulation_count=8)
        selected_action = search.select_action(1)

        self.assertIn(selected_action, legal_actions)
        self.assertFalse(search.should_resign())

    def test_mcts_search_binding_runs_for_chess_configs(self) -> None:
        """Guards runtime node dispatch so chess search remains functional through the shared Python API."""
        bindings = _require_bindings()
        game_config = bindings.chess_game_config()
        search_config = bindings.SearchConfig()
        search_config.simulations_per_move = 4
        search_config.enable_dirichlet_noise = False
        search_config.temperature = 0.0
        search_config.temperature_moves = 0
        search_config.enable_resignation = False

        search = bindings.MctsSearch(game_config, search_config, node_arena_capacity=2048)
        state = bindings.ChessState()
        legal_actions = state.legal_actions()
        preferred_action = legal_actions[0]

        def evaluator(_state: object) -> dict[str, object]:
            policy = [-1000.0] * game_config.action_space_size
            policy[preferred_action] = 1000.0
            return {
                "policy": policy,
                "value": 0.0,
                "policy_is_logits": True,
            }

        search.set_root_state(state)
        search.run_simulations(evaluator, simulation_count=4)
        selected_action = search.select_action(1)

        self.assertIn(selected_action, legal_actions)
        self.assertFalse(search.should_resign())

    def test_self_play_game_config_exposes_playout_cap_fields(self) -> None:
        """Ensures Python can configure playout-cap, root-FPU, and Dirichlet knobs before launching workers."""
        bindings = _require_bindings()
        game_config = bindings.SelfPlayGameConfig()
        search_config = bindings.SearchConfig()

        game_config.enable_playout_cap = True
        game_config.reduced_simulations = 37
        game_config.full_playout_probability = 0.2
        game_config.c_fpu_root = 0.0
        game_config.randomize_dirichlet_epsilon = True
        game_config.dirichlet_epsilon_min = 0.15
        game_config.dirichlet_epsilon_max = 0.35
        game_config.dynamic_dirichlet_alpha = True
        game_config.compute_ownership = True
        search_config.c_fpu_root = 0.0
        search_config.dynamic_dirichlet_alpha = True

        self.assertTrue(game_config.enable_playout_cap)
        self.assertEqual(game_config.reduced_simulations, 37)
        self.assertAlmostEqual(game_config.full_playout_probability, 0.2)
        self.assertAlmostEqual(game_config.c_fpu_root, 0.0)
        self.assertTrue(game_config.randomize_dirichlet_epsilon)
        self.assertAlmostEqual(game_config.dirichlet_epsilon_min, 0.15)
        self.assertAlmostEqual(game_config.dirichlet_epsilon_max, 0.35)
        self.assertTrue(game_config.dynamic_dirichlet_alpha)
        self.assertTrue(game_config.compute_ownership)
        self.assertAlmostEqual(search_config.c_fpu_root, 0.0)
        self.assertTrue(search_config.dynamic_dirichlet_alpha)

    def test_self_play_manager_bindings_release_gil_for_lifecycle_and_metrics_calls(self) -> None:
        """WHY: removing these call guards can reintroduce Python-thread stalls during self-play startup and runtime control calls."""
        bindings_source = (ROOT / "src" / "bindings" / "python_bindings.cpp").read_text(encoding="utf-8")

        self.assertRegex(
            bindings_source,
            re.compile(
                r'\.def\(\s*"start"\s*,\s*&SelfPlayManager::start\s*,\s*'
                r"py::call_guard<py::gil_scoped_release>\(\)\s*\)"
            ),
        )
        self.assertRegex(
            bindings_source,
            re.compile(
                r'\.def\(\s*"update_simulations_per_move"\s*,\s*'
                r"&SelfPlayManager::update_simulations_per_move\s*,\s*"
                r'py::arg\("new_sims"\)\s*,\s*'
                r"py::call_guard<py::gil_scoped_release>\(\)\s*\)"
            ),
        )
        self.assertRegex(
            bindings_source,
            re.compile(
                r'\.def\(\s*"metrics"\s*,\s*&SelfPlayManager::metrics\s*,\s*'
                r"py::call_guard<py::gil_scoped_release>\(\)\s*\)"
            ),
        )

    def test_game_state_bindings_release_gil_for_hot_paths(self) -> None:
        """WHY: state transition/encoding helpers can be CPU-heavy and should not monopolize the Python GIL."""
        bindings_source = (ROOT / "src" / "bindings" / "python_bindings.cpp").read_text(encoding="utf-8")

        required_patterns = (
            r'\.def\(\s*"apply_action"\s*,\s*&GameState::apply_action\s*,\s*py::arg\("action"\)\s*,\s*'
            r"py::call_guard<py::gil_scoped_release>\(\)\s*\)",
            r'\.def\(\s*"legal_actions"\s*,\s*&GameState::legal_actions\s*,\s*'
            r"py::call_guard<py::gil_scoped_release>\(\)\s*\)",
            r'\.def\(\s*"clone"\s*,\s*&GameState::clone\s*,\s*'
            r"py::call_guard<py::gil_scoped_release>\(\)\s*\)",
            r'\.def_static\(\s*"from_fen"\s*,\s*&ChessState::from_fen\s*,\s*py::arg\("fen"\)\s*,\s*'
            r"py::call_guard<py::gil_scoped_release>\(\)\s*\)",
            r'\.def\(\s*"legal_actions_uci"\s*,\s*&chess_legal_actions_uci\s*,\s*'
            r"py::call_guard<py::gil_scoped_release>\(\)\s*\)",
            r'\.def_static\(\s*"from_sgf"\s*,\s*&GoState::from_sgf\s*,\s*py::arg\("sgf"\)\s*,\s*'
            r"py::call_guard<py::gil_scoped_release>\(\)\s*\)",
            r'\.def\(\s*"legal_actions"\s*,\s*&GoState::legal_actions\s*,\s*'
            r"py::call_guard<py::gil_scoped_release>\(\)\s*\)",
        )
        for required_pattern in required_patterns:
            self.assertRegex(bindings_source, re.compile(required_pattern))

        self.assertRegex(
            bindings_source,
            re.compile(
                r"std::vector<float>\s+encode_state_flat\([^)]*\)\s*\{"
                r".*?py::gil_scoped_release\s+release_gil;"
                r".*?state\.encode\(encoded\.data\(\)\);",
                re.DOTALL,
            ),
        )
        self.assertRegex(
            bindings_source,
            re.compile(
                r"py::array_t<float>\s+encode_state_tensor\([^)]*\)\s*\{"
                r".*?py::gil_scoped_release\s+release_gil;"
                r".*?state\.encode\(encoded\.mutable_data\(\)\);",
                re.DOTALL,
            ),
        )

    def test_replay_buffer_bindings_release_gil_for_hot_paths(self) -> None:
        """WHY: replay hot paths should release the GIL either at the binding edge or around heavy native buffer operations."""
        bindings_source = (ROOT / "src" / "bindings" / "python_bindings.cpp").read_text(encoding="utf-8")

        gil_guard = r"py::call_guard<py::gil_scoped_release>\(\)\s*"
        required_patterns = (
            r'\.def\(\s*"add_game"\s*,\s*&alphazero::selfplay::ReplayBuffer::add_game\s*,\s*'
            r'py::arg\("positions"\)\s*,\s*'
            + gil_guard
            + r"\)",
            r'\.def\(\s*"sample"\s*,\s*&alphazero::selfplay::ReplayBuffer::sample\s*,\s*'
            r'py::arg\("batch_size"\)\s*,\s*'
            + gil_guard
            + r"\)",
            r'\.def\(\s*"add_game"\s*,\s*&CompactReplayBuffer::add_game\s*,\s*'
            r'py::arg\("positions"\)\s*,\s*'
            + gil_guard
            + r"\)",
            r'\.def\(\s*"sample"\s*,\s*&CompactReplayBuffer::sample\s*,\s*'
            r'py::arg\("batch_size"\)\s*,\s*'
            + gil_guard
            + r"\)",
            r'\.def\(\s*"save_to_file"\s*,\s*&CompactReplayBuffer::save_to_file\s*,\s*'
            r'py::arg\("path"\)\s*,\s*'
            + gil_guard
            + r"\)",
            r'\.def\(\s*"load_from_file"\s*,\s*&CompactReplayBuffer::load_from_file\s*,\s*'
            r'py::arg\("path"\)\s*,\s*'
            + gil_guard
            + r"\)",
        )

        for required_pattern in required_patterns:
            self.assertRegex(bindings_source, re.compile(required_pattern))

        self.assertIn("replay_buffer_sample_batch_numpy_impl", bindings_source)
        self.assertIn("replay_buffer_export_numpy_impl", bindings_source)
        self.assertIn("replay_buffer_import_numpy_impl", bindings_source)
        self.assertGreaterEqual(bindings_source.count("py::gil_scoped_release release_gil;"), 3)

    def test_sample_batch_numpy_capsule_owner_transfer_is_exception_safe(self) -> None:
        """WHY: sample_batch NumPy views must transfer capsule ownership without raw-new leak windows."""
        bindings_source = (ROOT / "src" / "bindings" / "python_bindings.cpp").read_text(encoding="utf-8")

        self.assertNotIn("new std::shared_ptr<SampledBatch>(batch)", bindings_source)
        self.assertIn("std::make_unique<std::shared_ptr<SampledBatch>>(batch)", bindings_source)
        self.assertIn("owner_guard.release()", bindings_source)
        self.assertGreaterEqual(bindings_source.count("sampled_batch_owner_capsule(batch)"), 2)

    def test_self_play_manager_exposes_simulation_budget_update_api(self) -> None:
        """WHY: train.py must be able to retune self-play simulation budgets at runtime for phase-4 scheduling."""
        bindings = _require_bindings()
        go_config = bindings.go_game_config()
        replay_buffer = bindings.ReplayBuffer(capacity=32, random_seed=123)

        pass_favoring_policy = [-100.0] * go_config.action_space_size
        pass_favoring_policy[-1] = 100.0

        def evaluator(_state: object) -> dict[str, object]:
            return {
                "policy": pass_favoring_policy,
                "value": 0.0,
                "policy_is_logits": True,
            }

        manager_config = bindings.SelfPlayManagerConfig()
        manager = bindings.SelfPlayManager(go_config, replay_buffer, evaluator, manager_config)

        manager.update_simulations_per_move(5)
        with self.assertRaisesRegex(
            ValueError,
            "SelfPlayManager simulations-per-move update must be greater than zero",
        ):
            manager.update_simulations_per_move(0)

    def test_eval_queue_processes_requests_with_python_batch_callback(self) -> None:
        """Protects the CPU↔Python batching bridge so each submitter gets the correct per-request result."""
        bindings = _require_bindings()
        eval_config = bindings.EvalQueueConfig()
        eval_config.batch_size = 4
        eval_config.flush_timeout_us = 10_000

        def evaluator(batch: object) -> tuple[object, object]:
            import numpy as np

            batch_array = np.asarray(batch, dtype=np.float32)
            first_values = batch_array[:, 0]
            policy_logits = np.stack(
                (first_values, first_values + 1.0),
                axis=1,
            ).astype(np.float32, copy=False)
            values = (first_values * 2.0).astype(np.float32, copy=False)
            return policy_logits, values

        queue = bindings.EvalQueue(evaluator=evaluator, encoded_state_size=1, config=eval_config)
        stop_event = threading.Event()

        def consumer() -> None:
            while not stop_event.is_set():
                queue.process_batch()

        thread = threading.Thread(target=consumer)
        thread.start()
        try:
            result = queue.submit_and_wait([3.0])
            self.assertEqual(result.policy_logits, [3.0, 4.0])
            self.assertAlmostEqual(result.value, 6.0)
        finally:
            stop_event.set()
            queue.stop()
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())

    def test_eval_queue_stop_unblocks_waiting_submitters_without_consumer(self) -> None:
        """WHY: shutdown may stop EvalQueue before a consumer drains pending requests, so waiting submitters must be released."""
        bindings = _require_bindings()
        eval_config = bindings.EvalQueueConfig()
        eval_config.batch_size = 8
        eval_config.flush_timeout_us = 10_000

        def evaluator(_batch: object) -> tuple[object, object]:
            raise AssertionError("consumer should not run in this regression test")

        queue = bindings.EvalQueue(evaluator=evaluator, encoded_state_size=1, config=eval_config)
        started = threading.Event()
        results: dict[str, BaseException | None] = {"first": None, "second": None}

        def submitter(slot: str, value: float) -> None:
            started.set()
            try:
                queue.submit_and_wait([value])
            except BaseException as exc:  # pragma: no cover - exercised by assertions below
                results[slot] = exc

        first = threading.Thread(target=submitter, args=("first", 1.0))
        second = threading.Thread(target=submitter, args=("second", 2.0))
        first.start()
        second.start()

        self.assertTrue(started.wait(timeout=1.0))
        time.sleep(0.05)
        queue.stop()

        first.join(timeout=2.0)
        second.join(timeout=2.0)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertIsInstance(results["first"], RuntimeError)
        self.assertIsInstance(results["second"], RuntimeError)

        first_message = str(results["first"])
        second_message = str(results["second"])
        self.assertTrue("stopped" in first_message.lower())
        self.assertTrue("stopped" in second_message.lower())

    def test_self_play_manager_starts_and_stops_from_python(self) -> None:
        """Validates that Python can control C++ self-play lifecycle and collect resulting replay data."""
        bindings = _require_bindings()
        go_config = bindings.go_game_config()
        replay_buffer = bindings.ReplayBuffer(capacity=64, random_seed=99)

        pass_favoring_policy = [-100.0] * go_config.action_space_size
        pass_favoring_policy[-1] = 100.0

        def evaluator(_state: object) -> dict[str, object]:
            return {
                "policy": pass_favoring_policy,
                "value": 0.0,
                "policy_is_logits": True,
            }

        manager_config = bindings.SelfPlayManagerConfig()
        manager_config.concurrent_games = 1
        manager_config.max_games_per_slot = 1
        manager_config.game_config.simulations_per_move = 1
        manager_config.game_config.mcts_threads = 1
        manager_config.game_config.enable_dirichlet_noise = False
        manager_config.game_config.temperature = 0.0
        manager_config.game_config.temperature_moves = 0
        manager_config.game_config.enable_resignation = False
        manager_config.game_config.resign_disable_fraction = 0.0

        manager = bindings.SelfPlayManager(go_config, replay_buffer, evaluator, manager_config)
        manager.start()

        timeout_seconds = 10.0
        deadline = time.monotonic() + timeout_seconds
        while manager.is_running() and time.monotonic() < deadline:
            time.sleep(0.01)

        manager.stop()
        metrics = manager.metrics()
        self.assertGreaterEqual(metrics.games_completed, 1)
        self.assertGreaterEqual(replay_buffer.size(), 1)

    def test_self_play_manager_accepts_eval_queue_constructor(self) -> None:
        """Guards the new binding overload so self-play can bypass per-leaf Python evaluator callbacks."""
        bindings = _require_bindings()
        go_config = bindings.go_game_config()
        replay_buffer = bindings.ReplayBuffer(capacity=64, random_seed=99)

        eval_config = bindings.EvalQueueConfig()
        eval_config.batch_size = 8
        eval_config.flush_timeout_us = 1_000
        encoded_state_size = go_config.total_input_channels * go_config.board_rows * go_config.board_cols

        def evaluator(batch: object) -> tuple[object, object]:
            import numpy as np

            batch_array = np.asarray(batch, dtype=np.float32)
            batch_size = int(batch_array.shape[0])
            policy_logits = np.full(
                (batch_size, go_config.action_space_size),
                -100.0,
                dtype=np.float32,
            )
            policy_logits[:, -1] = 100.0
            values = np.zeros((batch_size,), dtype=np.float32)
            return policy_logits, values

        queue = bindings.EvalQueue(
            evaluator=evaluator,
            encoded_state_size=encoded_state_size,
            config=eval_config,
        )
        stop_event = threading.Event()

        def consumer() -> None:
            while not stop_event.is_set():
                queue.process_batch()

        thread = threading.Thread(target=consumer)
        thread.start()
        try:
            manager_config = bindings.SelfPlayManagerConfig()
            manager_config.concurrent_games = 1
            manager_config.max_games_per_slot = 1
            manager_config.game_config.simulations_per_move = 1
            manager_config.game_config.mcts_threads = 1
            manager_config.game_config.enable_dirichlet_noise = False
            manager_config.game_config.temperature = 0.0
            manager_config.game_config.temperature_moves = 0
            manager_config.game_config.enable_resignation = False
            manager_config.game_config.resign_disable_fraction = 0.0

            manager = bindings.SelfPlayManager(go_config, replay_buffer, queue, manager_config)
            manager.start()

            timeout_seconds = 10.0
            deadline = time.monotonic() + timeout_seconds
            while manager.is_running() and time.monotonic() < deadline:
                time.sleep(0.01)

            manager.stop()
            metrics = manager.metrics()
            self.assertGreaterEqual(metrics.games_completed, 1)
            self.assertGreaterEqual(replay_buffer.size(), 1)
        finally:
            stop_event.set()
            queue.stop()
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
