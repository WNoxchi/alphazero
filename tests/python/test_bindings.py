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
        self.assertEqual(len(encoded), 119 * 8 * 8)

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
        self.assertEqual(sample.encoded_state.tolist(), encoded_state)
        self.assertEqual(sample.policy.tolist(), policy)
        self.assertEqual(sample.value_wdl.tolist(), value_wdl)

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

    def test_eval_queue_processes_requests_with_python_batch_callback(self) -> None:
        """Protects the CPU↔Python batching bridge so each submitter gets the correct per-request result."""
        bindings = _require_bindings()
        eval_config = bindings.EvalQueueConfig()
        eval_config.batch_size = 4
        eval_config.flush_timeout_us = 10_000

        def evaluator(batch: list[object]) -> list[dict[str, object]]:
            outputs: list[dict[str, object]] = []
            for state in batch:
                state_values = state
                first_value = float(state_values[0])  # type: ignore[index]
                outputs.append(
                    {
                        "policy_logits": [first_value, first_value + 1.0],
                        "value": first_value * 2.0,
                    }
                )
            return outputs

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


if __name__ == "__main__":
    unittest.main()
