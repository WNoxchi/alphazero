"""Contract tests for the optional pybind11 `alphazero_cpp` extension."""

from __future__ import annotations

import pathlib
import sys
import threading
import time
import types
import unittest
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]


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
        )

        replay_buffer.add_game([position])
        sampled = replay_buffer.sample(1)
        self.assertEqual(len(sampled), 1)
        sample = sampled[0]

        self.assertEqual(sample.encoded_state_size, len(encoded_state))
        self.assertEqual(sample.policy_size, len(policy))
        self.assertAlmostEqual(sample.value, 1.0)
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
                )
            )
        replay_buffer.add_game(positions)

        import numpy as np
        import numpy.testing as npt

        states, policies, scalar_values = replay_buffer.sample_batch(
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

        for row in range(8):
            game_id = int(round(float(policies[row, 1])))
            move_number = int(round(float(policies[row, 2])))
            expected_signature = float(game_id * 1000 + move_number)
            self.assertAlmostEqual(float(states[row, 0]), expected_signature)
            self.assertAlmostEqual(float(states[row, 1]), float(move_number))
            self.assertAlmostEqual(float(policies[row, 0]), expected_signature + 0.5)
            self.assertAlmostEqual(float(scalar_values[row, 0]), scalar_for(game_id, move_number))

        _, wdl_policies, wdl_values = replay_buffer.sample_batch(
            batch_size=8,
            encoded_state_size=4,
            policy_size=3,
            value_dim=3,
        )
        self.assertEqual(wdl_policies.shape, (8, 3))
        self.assertEqual(wdl_values.shape, (8, 3))
        for row in range(8):
            game_id = int(round(float(wdl_policies[row, 1])))
            move_number = int(round(float(wdl_policies[row, 2])))
            expected = wdl_for(scalar_for(game_id, move_number))
            npt.assert_allclose(wdl_values[row], expected, rtol=1e-6)

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

        states, policies, values_wdl = compact_buffer.sample_batch(
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

        exported = compact_buffer.export_buffer(encoded_state_size=128, policy_size=5)
        restored = bindings.CompactReplayBuffer(
            capacity=8,
            num_binary_planes=1,
            num_float_planes=1,
            float_plane_indices=[1],
            full_policy_size=5,
            random_seed=9876,
        )
        restored.import_buffer(*exported, encoded_state_size=128, policy_size=5)
        self.assertEqual(restored.size(), 1)
        restored_sample = restored.sample(1)[0]
        self.assertEqual(restored_sample.game_id, 77)
        self.assertEqual(restored_sample.move_number, 9)

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
