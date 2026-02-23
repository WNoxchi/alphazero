"""AlphaZero chess engine wrapper for web play."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in [ROOT, ROOT / "python", ROOT / "scripts", ROOT / "build" / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@dataclass(slots=True)
class MoveResult:
    """Result of applying a move (human or AI)."""

    uci: str
    fen: str
    legal_moves: list[str]
    is_terminal: bool
    result: str | None
    eval_score: float | None
    move_number: int
    current_player: int


class ChessEngine:
    """Wraps AlphaZero MCTS for serving web play sessions."""

    def __init__(
        self,
        model_path: str,
        *,
        simulations: int = 800,
        num_blocks: int = 20,
        num_filters: int = 256,
        se_reduction: int = 4,
        device: str | None = None,
        node_arena_capacity: int = 262_144,
        fp32: bool = False,
    ) -> None:
        # Lazy import to keep module importable without heavy deps
        from scripts.play import build_play_runtime, build_argument_parser

        self._node_arena_capacity = node_arena_capacity
        self.model_name: str | None = None

        args = argparse.Namespace(
            game="chess",
            model=model_path,
            simulations=simulations,
            games=1,
            human_color="white",
            engine_time_ms=1000,
            device=device,
            num_blocks=num_blocks,
            num_filters=num_filters,
            se_reduction=se_reduction,
            fp32=fp32,
            c_puct=2.5,
            c_fpu=0.25,
            resign_threshold=-0.9,
            search_random_seed=0xC0FFEE1234567890,
            node_arena_capacity=node_arena_capacity,
            opponent="human",
        )
        self._runtime = build_play_runtime(args=args)
        self._cpp = self._runtime.dependencies.cpp

        self._state: Any = None
        self._action_history: list[int] = []
        self._state_history: list[Any] = []
        self._search: Any = None
        self.reset()

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        model_name: str | None = None,
        node_arena_capacity: int = 262_144,
    ) -> "ChessEngine":
        """Construct a ChessEngine from an existing PlayRuntime."""
        engine = object.__new__(cls)
        engine._runtime = runtime
        engine._cpp = runtime.dependencies.cpp
        engine._node_arena_capacity = node_arena_capacity
        engine.model_name = model_name
        engine._state = None
        engine._action_history = []
        engine._state_history = []
        engine._search = None
        engine.reset()
        return engine

    def reset(self) -> dict[str, Any]:
        """Start a new game. Returns initial board state."""
        self._state = self._cpp.ChessState()
        self._action_history = []
        self._state_history = []
        self._search = self._cpp.MctsSearch(
            self._runtime.cpp_game_config,
            self._runtime.search_config,
            self._node_arena_capacity,
        )
        return self._board_state()

    def _board_state(self, eval_score: float | None = None, last_uci: str | None = None) -> dict[str, Any]:
        """Build a serializable snapshot of the current position."""
        is_terminal = bool(self._state.is_terminal())
        result = None
        if is_terminal:
            outcome = float(self._state.outcome(0))
            if outcome > 0:
                result = "1-0"
            elif outcome < 0:
                result = "0-1"
            else:
                result = "1/2-1/2"

        legal_uci = []
        legal_dests: dict[str, list[str]] = {}
        if not is_terminal:
            for action_id, uci_str in self._state.legal_actions_uci():
                legal_uci.append(uci_str)
                src = uci_str[:2]
                dst = uci_str[2:4]
                legal_dests.setdefault(src, []).append(dst)

        return {
            "fen": str(self._state.to_fen()),
            "legal_moves": legal_uci,
            "legal_dests": legal_dests,
            "is_terminal": is_terminal,
            "result": result,
            "eval": eval_score,
            "last_move": last_uci,
            "move_number": len(self._action_history),
            "current_player": int(self._state.current_player()),
            "can_undo": len(self._action_history) > 0,
        }

    def play_human_move(self, uci: str) -> dict[str, Any]:
        """Apply a human move given in UCI notation. Returns updated state."""
        try:
            action = int(self._state.uci_to_action(uci))
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"Illegal move: {uci} ({exc})") from exc

        legal = set(self._state.legal_actions())
        if action not in legal:
            raise ValueError(f"Illegal move: {uci}")

        self._state_history.append(self._state)
        self._state = self._state.apply_action(action)
        self._action_history.append(action)
        return self._board_state(last_uci=uci)

    def get_ai_move(self) -> dict[str, Any]:
        """Run MCTS and return the AI's chosen move."""
        if bool(self._state.is_terminal()):
            return self._board_state()

        self._search.set_root_state(self._state)
        self._search.run_simulations(self._runtime.evaluator)

        if bool(self._search.should_resign()):
            return {
                **self._board_state(),
                "resigned": True,
                "result": "0-1" if int(self._state.current_player()) == 0 else "1-0",
            }

        move_number = len(self._action_history) + 1
        action = int(self._search.select_action(move_number))
        uci = str(self._state.action_to_uci(action))

        # Get root value as eval score (from current player's perspective)
        eval_score = None
        try:
            stats = self._search.root_edge_stats(action)
            if stats is not None:
                eval_score = float(stats.mean_value)
        except Exception:
            pass

        self._state_history.append(self._state)
        self._state = self._state.apply_action(action)
        self._action_history.append(action)
        return self._board_state(eval_score=eval_score, last_uci=uci)

    def undo(self) -> dict[str, Any]:
        """Undo the last move (or last two if the last was an AI move)."""
        if not self._state_history:
            return self._board_state()

        # Undo two moves to get back to the human's turn
        for _ in range(2):
            if self._state_history:
                self._state = self._state_history.pop()
                self._action_history.pop()

        return self._board_state()

    def get_pgn(self) -> str:
        """Export the current game as PGN."""
        if not self._action_history:
            return ""
        is_terminal = bool(self._state.is_terminal())
        if is_terminal:
            outcome = float(self._state.outcome(0))
            if outcome > 0:
                result = "1-0"
            elif outcome < 0:
                result = "0-1"
            else:
                result = "1/2-1/2"
        else:
            result = "*"
        return str(self._cpp.ChessState.actions_to_pgn(self._action_history, result))
