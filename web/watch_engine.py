"""Dual-model chess watch engine for automated self-play in the web UI."""

from __future__ import annotations

from typing import Any


class WatchEngine:
    """Drive a single chess game between white and black PlayRuntime objects."""

    def __init__(
        self,
        white_runtime: Any,
        black_runtime: Any,
        *,
        node_arena_capacity: int = 262_144,
    ) -> None:
        self._white_runtime = white_runtime
        self._black_runtime = black_runtime
        self._node_arena_capacity = node_arena_capacity
        self._cpp = white_runtime.dependencies.cpp
        self._state: Any = None
        self._white_search: Any = None
        self._black_search: Any = None
        self._action_history: list[int] = []

    def reset(self) -> dict[str, Any]:
        """Start a new game and return the initial board state."""
        self._state = self._cpp.ChessState()
        self._action_history = []
        self._white_search = self._cpp.MctsSearch(
            self._white_runtime.cpp_game_config,
            self._white_runtime.search_config,
            self._node_arena_capacity,
        )
        self._black_search = self._cpp.MctsSearch(
            self._black_runtime.cpp_game_config,
            self._black_runtime.search_config,
            self._node_arena_capacity,
        )
        return self._board_state()

    def get_next_move(self) -> dict[str, Any]:
        """Run MCTS for the side to move and apply one action."""
        if bool(self._state.is_terminal()):
            return self._board_state()

        current_player = int(self._state.current_player())
        if current_player == 0:
            search, runtime = self._white_search, self._white_runtime
        else:
            search, runtime = self._black_search, self._black_runtime

        search.set_root_state(self._state)
        search.run_simulations(runtime.evaluator)

        if bool(search.should_resign()):
            result = "0-1" if current_player == 0 else "1-0"
            return {**self._board_state(), "resigned": True, "result": result}

        move_number = len(self._action_history) + 1
        action = int(search.select_action(move_number))
        uci = str(self._state.action_to_uci(action))

        eval_score = None
        try:
            stats = search.root_edge_stats(action)
            if stats is not None:
                eval_score = float(stats.mean_value)
        except Exception:
            pass

        self._state = self._state.apply_action(action)
        self._action_history.append(action)
        return self._board_state(eval_score=eval_score, last_uci=uci)

    def _board_state(
        self,
        eval_score: float | None = None,
        last_uci: str | None = None,
    ) -> dict[str, Any]:
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

    def get_pgn(self, *, white_name: str = "White", black_name: str = "Black") -> str:
        """Export the current game PGN with model-name headers."""
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

        pgn_body = str(self._cpp.ChessState.actions_to_pgn(self._action_history, result))
        headers = f'[White "{white_name}"]\n[Black "{black_name}"]\n\n'
        return headers + pgn_body
