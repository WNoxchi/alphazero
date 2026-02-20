"""Tests for scripts/play.py interactive and engine play behavior."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, cast
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "play.py"

_SPEC = importlib.util.spec_from_file_location("alphazero_play_script", SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import bootstrap guard.
    raise RuntimeError(f"Unable to load play script module from {SCRIPT_PATH}")
play_script = cast(Any, importlib.util.module_from_spec(_SPEC))
sys.modules[_SPEC.name] = play_script
_SPEC.loader.exec_module(play_script)


class _FakeDevice:
    def __init__(self, device_type: str) -> None:
        self.type = device_type


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorch:
    float32 = object()
    bfloat16 = object()
    cuda = _FakeCuda()

    @staticmethod
    def device(spec: str) -> _FakeDevice:
        return _FakeDevice(spec)


class _FakeModel:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.device = None
        self.eval_called = False

    def to(self, *, device: Any) -> "_FakeModel":
        self.device = device
        return self

    def eval(self) -> "_FakeModel":
        self.eval_called = True
        return self


class _FakeSearchConfig:
    def __init__(self) -> None:
        self.simulations_per_move = 800
        self.c_puct = 2.5
        self.c_fpu = 0.25
        self.enable_dirichlet_noise = True
        self.dirichlet_epsilon = 0.25
        self.dirichlet_alpha_override = 0.0
        self.temperature = 1.0
        self.temperature_moves = 30
        self.enable_resignation = True
        self.resign_threshold = -0.9
        self.random_seed = 0


class _FakeChessState:
    _ACTION_TO_UCI = {1: "e2e4", 2: "e7e5"}
    _UCI_TO_ACTION = {value: key for key, value in _ACTION_TO_UCI.items()}

    def __init__(self, ply: int = 0, history: tuple[int, ...] = ()) -> None:
        self.ply = ply
        self._history = history

    @staticmethod
    def actions_to_pgn(action_history: list[int], result: str, starting_fen: str = "") -> str:
        del starting_fen
        return f"PGN[{result}] {' '.join(str(action) for action in action_history)}"

    def clone(self) -> "_FakeChessState":
        return _FakeChessState(self.ply, self._history)

    def encode(self) -> list[float]:
        return [0.0]

    def current_player(self) -> int:
        return self.ply % 2

    def legal_actions(self) -> list[int]:
        if self.ply == 0:
            return [1]
        if self.ply == 1:
            return [2]
        return []

    def legal_actions_uci(self) -> list[tuple[int, str]]:
        return [(action, self._ACTION_TO_UCI[action]) for action in self.legal_actions()]

    def action_to_uci(self, action: int) -> str:
        return self._ACTION_TO_UCI[action]

    def uci_to_action(self, move: str) -> int:
        normalized = move.strip().lower()
        try:
            action = self._UCI_TO_ACTION[normalized]
        except KeyError as exc:
            raise ValueError("invalid UCI") from exc
        if action not in self.legal_actions():
            raise ValueError("illegal UCI")
        return action

    def apply_action(self, action: int) -> "_FakeChessState":
        if action not in self.legal_actions():
            raise ValueError("illegal action")
        return _FakeChessState(self.ply + 1, self._history + (action,))

    def is_terminal(self) -> bool:
        return self.ply >= 2

    def outcome(self, player: int) -> float:
        if not self.is_terminal():
            raise ValueError("outcome requested for non-terminal state")
        winner = 0  # White always wins in this deterministic harness.
        if player == winner:
            return 1.0
        return -1.0

    def to_string(self) -> str:
        return f"FakeChessState(ply={self.ply})"


class _FakeGoState:
    def __init__(self) -> None:
        self._unused = True


class _FakeMctsSearch:
    def __init__(self, game_config: Any, config: Any, node_arena_capacity: int) -> None:
        del game_config
        self.config = config
        self.node_arena_capacity = node_arena_capacity
        self.root_state: _FakeChessState | None = None
        self.run_calls = 0

    def set_root_state(self, state: _FakeChessState) -> None:
        self.root_state = state

    def run_simulations(self, evaluator: Any, simulation_count: int | None = None) -> None:
        del evaluator, simulation_count
        self.run_calls += 1

    def should_resign(self) -> bool:
        return False

    def select_action(self, move_number: int) -> int:
        del move_number
        if self.root_state is None:
            raise AssertionError("root_state was not set before select_action")
        legal = self.root_state.legal_actions()
        if not legal:
            raise AssertionError("no legal actions available")
        return legal[0]


class _FakeCpp:
    def __init__(self) -> None:
        self.SearchConfig = _FakeSearchConfig
        self.ChessState = _FakeChessState
        self.GoState = _FakeGoState
        self.created_searches: list[_FakeMctsSearch] = []

        parent = self

        class _BoundMctsSearch(_FakeMctsSearch):
            def __init__(self, game_config: Any, config: Any, node_arena_capacity: int) -> None:
                super().__init__(game_config, config, node_arena_capacity)
                parent.created_searches.append(self)

        self.MctsSearch = _BoundMctsSearch

    def chess_game_config(self) -> Any:
        return SimpleNamespace(action_space_size=4672)

    def go_game_config(self) -> Any:
        return SimpleNamespace(action_space_size=362)


@dataclass(frozen=True, slots=True)
class _FakeLoadCheckpoint:
    calls: list[tuple[Path, Any, Any, str]]

    def __call__(self, path: Path, model: Any, optimizer: Any, *, map_location: str) -> None:
        self.calls.append((Path(path), model, optimizer, map_location))


class PlayScriptTests(unittest.TestCase):
    def _make_args(self, *, model_path: Path, opponent: str = "human", games: int = 1) -> Any:
        return SimpleNamespace(
            game="chess",
            model=str(model_path),
            simulations=128,
            opponent=opponent,
            games=games,
            human_color="white",
            engine_time_ms=50,
            device=None,
            num_blocks=20,
            num_filters=256,
            se_reduction=4,
            fp32=True,
            c_puct=2.5,
            c_fpu=0.25,
            resign_threshold=-0.9,
            search_random_seed=123,
            node_arena_capacity=2048,
        )

    def _make_dependencies(self, *, cpp: _FakeCpp, loader: _FakeLoadCheckpoint) -> Any:
        return play_script.RuntimeDependencies(
            cpp=cpp,
            ResNetSE=_FakeModel,
            load_checkpoint=loader,
            torch=_FakeTorch,
        )

    def test_build_runtime_sets_deterministic_play_search_config(self) -> None:
        """WHY: Play mode must force deterministic/no-noise MCTS regardless of training-time defaults."""
        cpp = _FakeCpp()
        loader = _FakeLoadCheckpoint(calls=[])

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "checkpoint_00010000.pt"
            model_path.write_text("placeholder", encoding="utf-8")
            args = self._make_args(model_path=model_path)

            runtime = play_script.build_play_runtime(
                args=args,
                dependencies=self._make_dependencies(cpp=cpp, loader=loader),
            )

        self.assertEqual(len(loader.calls), 1)
        loaded_path, _loaded_model, loaded_optimizer, map_location = loader.calls[0]
        self.assertEqual(loaded_path, model_path)
        self.assertIsNone(loaded_optimizer)
        self.assertEqual(map_location, "cpu")

        self.assertEqual(runtime.search_config.simulations_per_move, 128)
        self.assertFalse(runtime.search_config.enable_dirichlet_noise)
        self.assertEqual(runtime.search_config.temperature, 0.0)
        self.assertEqual(runtime.search_config.temperature_moves, 0)
        self.assertTrue(runtime.search_config.enable_resignation)

    def test_human_game_consumes_uci_input_and_ai_replies(self) -> None:
        """WHY: Interactive mode must parse human UCI moves and still run deterministic AI selection."""
        cpp = _FakeCpp()
        runtime = play_script.PlayRuntime(
            dependencies=self._make_dependencies(cpp=cpp, loader=_FakeLoadCheckpoint(calls=[])),
            game_config=play_script.get_game_config("chess"),
            cpp_game_config=cpp.chess_game_config(),
            model=_FakeModel(),
            evaluator=lambda _state: {"policy": [1.0], "value": 0.0, "policy_is_logits": True},
            search_config=_FakeSearchConfig(),
        )
        args = self._make_args(model_path=Path("/tmp/not-used.pt"))

        supplied_inputs = iter(["e2e4"])
        output_lines: list[str] = []
        summary = play_script.play_human_game(
            runtime,
            args=args,
            input_fn=lambda _prompt: next(supplied_inputs),
            output_fn=output_lines.append,
        )

        self.assertEqual(summary.result, "1-0")
        self.assertEqual(summary.move_count, 2)
        self.assertEqual(summary.action_history, (1, 2))
        self.assertIn("PGN[1-0] 1 2", summary.transcript)

        self.assertEqual(len(cpp.created_searches), 1)
        self.assertEqual(cpp.created_searches[0].run_calls, 1)

    def test_engine_match_aggregates_wins_and_losses_across_colors(self) -> None:
        """WHY: Engine mode must support alternating colors and report match-level W/L/D correctly."""
        cpp = _FakeCpp()
        runtime = play_script.PlayRuntime(
            dependencies=self._make_dependencies(cpp=cpp, loader=_FakeLoadCheckpoint(calls=[])),
            game_config=play_script.get_game_config("chess"),
            cpp_game_config=cpp.chess_game_config(),
            model=_FakeModel(),
            evaluator=lambda _state: {"policy": [1.0], "value": 0.0, "policy_is_logits": True},
            search_config=_FakeSearchConfig(),
        )
        args = self._make_args(model_path=Path("/tmp/not-used.pt"), opponent="stockfish", games=2)

        class _FakeEngine:
            def __init__(self, command: list[str]) -> None:
                self.command = command
                self.closed = False

            def initialize(self) -> None:
                return

            def new_game(self) -> None:
                return

            def bestmove(self, played_uci_moves: list[str], *, movetime_ms: int) -> str:
                del movetime_ms
                if len(played_uci_moves) == 0:
                    return "e2e4"
                return "e7e5"

            def close(self) -> None:
                self.closed = True

        original_engine_client = play_script.UciEngineClient
        play_script.UciEngineClient = _FakeEngine
        try:
            summary = play_script.play_engine_match(runtime, args=args, output_fn=lambda _line: None)
        finally:
            play_script.UciEngineClient = original_engine_client

        self.assertEqual(summary.games, 2)
        self.assertEqual(summary.wins, 1)
        self.assertEqual(summary.losses, 1)
        self.assertEqual(summary.draws, 0)


if __name__ == "__main__":
    unittest.main()
