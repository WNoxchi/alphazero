#!/usr/bin/env python3
"""Interactive play and UCI-engine matches against a trained AlphaZero model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import select
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from alphazero.config import GameConfig, get_game_config


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]

_GO_COLUMNS = "ABCDEFGHJKLMNOPQRST"


def _coerce_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeDependencies:
    """Runtime-callable dependencies for the play script."""

    cpp: Any
    ResNetSE: Any
    load_checkpoint: Callable[..., Any]
    torch: Any


@dataclass(frozen=True, slots=True)
class PlayRuntime:
    """Runtime objects required to run interactive or engine play sessions."""

    dependencies: RuntimeDependencies
    game_config: GameConfig
    cpp_game_config: Any
    model: Any
    evaluator: Callable[[Any], dict[str, object]]
    search_config: Any


@dataclass(frozen=True, slots=True)
class PlayGameSummary:
    """Terminal summary for one game."""

    result: str
    winner: int | None
    resigned_by: int | None
    move_count: int
    action_history: tuple[int, ...]
    transcript: str | None


@dataclass(frozen=True, slots=True)
class MatchSummary:
    """Aggregate summary for multi-game engine matches."""

    games: int
    wins: int
    losses: int
    draws: int


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for play sessions."""

    parser = argparse.ArgumentParser(description="Play against a trained AlphaZero model.")
    parser.add_argument("--game", required=True, choices=("chess", "go"), help="Game to play.")
    parser.add_argument("--model", required=True, help="Path to a training checkpoint (*.pt).")
    parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="MCTS simulations per AI move (default: 800).",
    )
    parser.add_argument(
        "--opponent",
        default="human",
        help="'human' for interactive play, or a UCI engine command (for chess).",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of games (mainly for engine matches).",
    )
    parser.add_argument(
        "--human-color",
        choices=("white", "black", "random"),
        default="white",
        help="Human color for interactive mode.",
    )
    parser.add_argument(
        "--engine-time-ms",
        type=int,
        default=1000,
        help="Per-move engine time in milliseconds for UCI matches.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (for example: 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=20,
        help="ResNet block count used to construct the model architecture.",
    )
    parser.add_argument(
        "--num-filters",
        type=int,
        default=256,
        help="ResNet channel count used to construct the model architecture.",
    )
    parser.add_argument(
        "--se-reduction",
        type=int,
        default=4,
        help="Squeeze-Excitation reduction ratio.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable mixed-precision inference and run in FP32.",
    )
    parser.add_argument("--c-puct", type=float, default=2.5, help="PUCT exploration constant.")
    parser.add_argument("--c-fpu", type=float, default=0.25, help="First-play urgency penalty.")
    parser.add_argument(
        "--resign-threshold",
        type=float,
        default=-0.9,
        help="Resignation threshold used by MCTS.",
    )
    parser.add_argument(
        "--search-random-seed",
        type=int,
        default=0xC0FFEE1234567890,
        help="Random seed for MCTS tie-breaking.",
    )
    parser.add_argument(
        "--node-arena-capacity",
        type=int,
        default=262_144,
        help="MCTS node arena capacity.",
    )
    return parser


def _import_cpp_bindings() -> Any:
    try:
        import alphazero_cpp  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "alphazero_cpp extension is unavailable. Build the project first with "
            "`cmake --build build` before running scripts/play.py."
        ) from exc
    return alphazero_cpp


def load_runtime_dependencies() -> RuntimeDependencies:
    """Import runtime dependencies lazily to keep module import lightweight."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for scripts/play.py") from exc

    from alphazero.network import ResNetSE
    from alphazero.utils.checkpoint import load_checkpoint

    return RuntimeDependencies(
        cpp=_import_cpp_bindings(),
        ResNetSE=ResNetSE,
        load_checkpoint=load_checkpoint,
        torch=torch,
    )


def _build_cpp_game_config(cpp: Any, game_name: str) -> Any:
    normalized = game_name.strip().lower()
    if normalized == "chess":
        return cpp.chess_game_config()
    if normalized == "go":
        return cpp.go_game_config()
    raise ValueError(f"Unsupported game {game_name!r}")


def _initial_state(cpp: Any, game_name: str) -> Any:
    normalized = game_name.strip().lower()
    if normalized == "chess":
        return cpp.ChessState()
    if normalized == "go":
        return cpp.GoState()
    raise ValueError(f"Unsupported game {game_name!r}")


def _build_evaluator(
    *,
    model: Any,
    game_config: GameConfig,
    torch: Any,
    device: Any,
    use_mixed_precision: bool,
) -> Callable[[Any], dict[str, object]]:
    rows, cols = game_config.board_shape
    encoded_state_size = game_config.input_channels * rows * cols

    def evaluator(state: Any) -> dict[str, object]:
        encoded_state = torch.as_tensor(state.encode(), dtype=torch.float32, device=device).flatten()
        if int(encoded_state.numel()) != encoded_state_size:
            raise ValueError(
                "State encoding has unexpected size: "
                f"expected {encoded_state_size}, got {int(encoded_state.numel())}"
            )

        model_input = encoded_state.reshape(1, game_config.input_channels, rows, cols)
        model.eval()
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_mixed_precision and device.type == "cuda",
            ):
                policy_logits, value = model(model_input)

        if game_config.value_head_type == "scalar":
            scalar_value = float(value.reshape(1, -1)[0, 0].detach().item())
        elif game_config.value_head_type == "wdl":
            reshaped = value.reshape(1, -1)
            if int(reshaped.shape[1]) < 3:
                raise ValueError(
                    "WDL value head output must have at least three channels, "
                    f"got shape {tuple(value.shape)}"
                )
            scalar_value = float((reshaped[0, 0] - reshaped[0, 2]).detach().item())
        else:
            raise ValueError(f"Unsupported value_head_type {game_config.value_head_type!r}")

        policy = policy_logits.reshape(1, -1).detach().to(device="cpu", dtype=torch.float32)
        if int(policy.shape[1]) != game_config.action_space_size:
            raise ValueError(
                "Policy output has unexpected size: "
                f"expected {game_config.action_space_size}, got {int(policy.shape[1])}"
            )

        return {
            "policy": policy[0].tolist(),
            "value": scalar_value,
            "policy_is_logits": True,
        }

    return evaluator


def build_play_runtime(
    *,
    args: argparse.Namespace,
    dependencies: RuntimeDependencies | None = None,
) -> PlayRuntime:
    """Build all runtime objects required for play mode."""

    _coerce_positive_int("--simulations", int(args.simulations))
    _coerce_positive_int("--games", int(args.games))
    _coerce_positive_int("--engine-time-ms", int(args.engine_time_ms))
    _coerce_positive_int("--num-blocks", int(args.num_blocks))
    _coerce_positive_int("--num-filters", int(args.num_filters))
    _coerce_positive_int("--se-reduction", int(args.se_reduction))
    _coerce_positive_int("--node-arena-capacity", int(args.node_arena_capacity))

    active_dependencies = dependencies or load_runtime_dependencies()
    game_config = get_game_config(str(args.game))
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    model = active_dependencies.ResNetSE(
        game_config,
        num_blocks=int(args.num_blocks),
        num_filters=int(args.num_filters),
        se_reduction=int(args.se_reduction),
    )

    active_dependencies.load_checkpoint(model_path, model, optimizer=None, map_location="cpu")

    torch = active_dependencies.torch
    resolved_device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device=resolved_device)
    model.eval()

    evaluator = _build_evaluator(
        model=model,
        game_config=game_config,
        torch=torch,
        device=resolved_device,
        use_mixed_precision=not bool(args.fp32),
    )

    cpp = active_dependencies.cpp
    search_config = cpp.SearchConfig()
    search_config.simulations_per_move = int(args.simulations)
    search_config.c_puct = float(args.c_puct)
    search_config.c_fpu = float(args.c_fpu)
    search_config.enable_dirichlet_noise = False
    search_config.dirichlet_epsilon = 0.0
    search_config.temperature = 0.0
    search_config.temperature_moves = 0
    search_config.enable_resignation = True
    search_config.resign_threshold = float(args.resign_threshold)
    search_config.random_seed = int(args.search_random_seed)

    return PlayRuntime(
        dependencies=active_dependencies,
        game_config=game_config,
        cpp_game_config=_build_cpp_game_config(cpp, game_config.name),
        model=model,
        evaluator=evaluator,
        search_config=search_config,
    )


class UciEngineClient:
    """Minimal UCI protocol client for chess-engine matches."""

    def __init__(self, command: Sequence[str]) -> None:
        if len(command) == 0:
            raise ValueError("UCI engine command must be non-empty")

        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self._process.kill()
            raise RuntimeError("Failed to open stdin/stdout pipes for UCI engine process")

        self._stdin = self._process.stdin
        self._stdout = self._process.stdout

    def _send(self, command: str) -> None:
        self._stdin.write(command + "\n")
        self._stdin.flush()

    def _read_until(self, prefix: str, *, timeout_seconds: float) -> str:
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"Timed out waiting for UCI response starting with {prefix!r}")

            ready, _, _ = select.select([self._stdout], [], [], remaining)
            if not ready:
                continue

            line = self._stdout.readline()
            if line == "":
                raise RuntimeError("UCI engine terminated unexpectedly")

            stripped = line.strip()
            if stripped.startswith(prefix):
                return stripped

    def initialize(self) -> None:
        self._send("uci")
        self._read_until("uciok", timeout_seconds=10.0)
        self._send("isready")
        self._read_until("readyok", timeout_seconds=10.0)

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok", timeout_seconds=10.0)

    def bestmove(self, played_uci_moves: Sequence[str], *, movetime_ms: int) -> str:
        if played_uci_moves:
            self._send("position startpos moves " + " ".join(played_uci_moves))
        else:
            self._send("position startpos")
        self._send(f"go movetime {movetime_ms}")
        bestmove_line = self._read_until("bestmove", timeout_seconds=max(5.0, (movetime_ms / 1000.0) + 10.0))

        tokens = bestmove_line.split()
        if len(tokens) < 2:
            raise RuntimeError(f"Malformed bestmove response: {bestmove_line!r}")
        return tokens[1]

    def close(self) -> None:
        try:
            if self._process.poll() is None:
                self._send("quit")
                self._process.wait(timeout=2.0)
        except Exception:
            self._process.kill()


def _player_name(game: str, player: int) -> str:
    if game == "chess":
        return "White" if player == 0 else "Black"
    return "Black" if player == 0 else "White"


def _go_action_to_text(action: int) -> str:
    if action == 361:
        return "pass"
    if action < 0 or action >= 361:
        raise ValueError(f"Go action out of range: {action}")
    row = action // 19
    col = action % 19
    label_row = 19 - row
    return f"{_GO_COLUMNS[col]}{label_row}"


def _go_text_to_action(text: str) -> int:
    stripped = text.strip().upper()
    if stripped == "PASS":
        return 361
    if len(stripped) < 2:
        raise ValueError("Go move must be 'pass' or a coordinate like D4")

    column = stripped[0]
    row_text = stripped[1:]
    if column not in _GO_COLUMNS:
        raise ValueError("Go column must be in A-T excluding I")
    if not row_text.isdigit():
        raise ValueError("Go coordinate row must be numeric")

    row_label = int(row_text)
    if row_label < 1 or row_label > 19:
        raise ValueError("Go coordinate row must be in [1, 19]")

    col_index = _GO_COLUMNS.index(column)
    row_index = 19 - row_label
    return row_index * 19 + col_index


def _action_to_text(state: Any, *, game: str, action: int) -> str:
    if game == "chess":
        return str(state.action_to_uci(action))
    return _go_action_to_text(action)


def _parse_human_action(state: Any, *, game: str, raw_input: str) -> int:
    stripped = raw_input.strip()
    if game == "chess":
        if stripped.isdigit():
            action = int(stripped)
            legal = set(state.legal_actions())
            if action not in legal:
                raise ValueError("Action index is not legal in the current position")
            return action
        return int(state.uci_to_action(stripped))

    action = _go_text_to_action(stripped)
    legal = set(state.legal_actions())
    if action not in legal:
        raise ValueError("Move is illegal in the current position")
    return action


def _serialize_game(cpp: Any, *, game: str, action_history: Sequence[int], result: str) -> str | None:
    if game == "chess":
        return str(cpp.ChessState.actions_to_pgn(list(action_history), result))
    if game == "go":
        return str(cpp.GoState.actions_to_sgf(list(action_history), result))
    return None


def _determine_result(state: Any, *, game: str, resigned_by: int | None) -> tuple[str, int | None]:
    if resigned_by is not None:
        winner_player = 1 - resigned_by
        if game == "chess":
            return ("1-0" if winner_player == 0 else "0-1"), winner_player
        return ("B+R" if winner_player == 0 else "W+R"), winner_player

    winner: int | None = None
    outcome_player0 = float(state.outcome(0))
    if outcome_player0 > 0.0:
        winner = 0
    elif outcome_player0 < 0.0:
        winner = 1
    else:
        winner = None

    if game == "chess":
        if winner == 0:
            return "1-0", winner
        if winner == 1:
            return "0-1", winner
        return "1/2-1/2", None

    if winner == 0:
        return "B+?", winner
    if winner == 1:
        return "W+?", winner
    return "0", None


def _resolve_human_player(game: str, human_color: str, rng: random.Random) -> int:
    normalized = human_color.strip().lower()
    if normalized == "random":
        return rng.randint(0, 1)

    if game == "chess":
        return 0 if normalized == "white" else 1

    return 1 if normalized == "white" else 0


def _create_search(runtime: PlayRuntime, *, node_arena_capacity: int) -> Any:
    cpp = runtime.dependencies.cpp
    return cpp.MctsSearch(runtime.cpp_game_config, runtime.search_config, node_arena_capacity)


def play_human_game(
    runtime: PlayRuntime,
    *,
    args: argparse.Namespace,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> PlayGameSummary:
    """Run one human-vs-AI interactive game."""

    cpp = runtime.dependencies.cpp
    state = _initial_state(cpp, runtime.game_config.name)
    search = _create_search(runtime, node_arena_capacity=int(args.node_arena_capacity))
    rng = random.Random(int(args.search_random_seed))
    human_player = _resolve_human_player(runtime.game_config.name, str(args.human_color), rng)

    action_history: list[int] = []
    resigned_by: int | None = None

    output_fn(
        f"Interactive {runtime.game_config.name} game started. Human plays "
        f"{_player_name(runtime.game_config.name, human_player)}."
    )

    while not bool(state.is_terminal()):
        output_fn("\n" + str(state.to_string()))
        current_player = int(state.current_player())

        if current_player == human_player:
            while True:
                prompt = f"{_player_name(runtime.game_config.name, current_player)} to move> "
                raw = input_fn(prompt).strip()
                lowered = raw.lower()
                if lowered in {"quit", "exit", "resign"}:
                    resigned_by = current_player
                    break
                if lowered in {"help", "?"}:
                    output_fn("Enter a move (chess UCI like e2e4, or Go coordinate like D4), or 'resign'.")
                    continue
                if lowered in {"legal", "moves"}:
                    if runtime.game_config.name == "chess":
                        legal_moves = [move for _, move in state.legal_actions_uci()]
                    else:
                        legal_moves = [_go_action_to_text(action) for action in state.legal_actions()]
                    output_fn("Legal moves: " + " ".join(legal_moves))
                    continue

                try:
                    action = _parse_human_action(state, game=runtime.game_config.name, raw_input=raw)
                except ValueError as exc:
                    output_fn(f"Invalid move: {exc}")
                    continue
                break

            if resigned_by is not None:
                break
        else:
            output_fn("AlphaZero thinking...")
            search.set_root_state(state)
            search.run_simulations(runtime.evaluator)
            if bool(search.should_resign()):
                resigned_by = current_player
                output_fn("AlphaZero resigns.")
                break

            move_number_for_temperature = len(action_history) + 1
            action = int(search.select_action(move_number_for_temperature))
            output_fn(f"AlphaZero plays { _action_to_text(state, game=runtime.game_config.name, action=action) }")

        state = state.apply_action(action)
        action_history.append(action)

    result, winner = _determine_result(state, game=runtime.game_config.name, resigned_by=resigned_by)
    transcript = _serialize_game(
        cpp,
        game=runtime.game_config.name,
        action_history=action_history,
        result=result,
    )

    output_fn("\nGame over.")
    output_fn(f"Result: {result}")
    if winner is None:
        output_fn("Winner: draw")
    else:
        output_fn(f"Winner: {_player_name(runtime.game_config.name, winner)}")
    if transcript:
        output_fn("\nGame record:\n" + transcript)

    return PlayGameSummary(
        result=result,
        winner=winner,
        resigned_by=resigned_by,
        move_count=len(action_history),
        action_history=tuple(action_history),
        transcript=transcript,
    )


def play_engine_match(
    runtime: PlayRuntime,
    *,
    args: argparse.Namespace,
    output_fn: OutputFn = print,
) -> MatchSummary:
    """Run AlphaZero vs UCI engine matches (chess only)."""

    if runtime.game_config.name != "chess":
        raise ValueError("UCI engine mode is only supported for chess")

    command = shlex.split(str(args.opponent))
    engine = UciEngineClient(command)
    engine.initialize()

    wins = 0
    losses = 0
    draws = 0

    try:
        for game_index in range(int(args.games)):
            engine.new_game()
            az_player = 0 if game_index % 2 == 0 else 1

            state = runtime.dependencies.cpp.ChessState()
            search = _create_search(runtime, node_arena_capacity=int(args.node_arena_capacity))

            action_history: list[int] = []
            uci_history: list[str] = []
            resigned_by: int | None = None

            output_fn(
                f"\nGame {game_index + 1}/{int(args.games)}: "
                f"AlphaZero as {_player_name('chess', az_player)}"
            )

            while not bool(state.is_terminal()):
                current_player = int(state.current_player())

                if current_player == az_player:
                    search.set_root_state(state)
                    search.run_simulations(runtime.evaluator)
                    if bool(search.should_resign()):
                        resigned_by = current_player
                        output_fn("AlphaZero resigns.")
                        break

                    move_number_for_temperature = len(action_history) + 1
                    action = int(search.select_action(move_number_for_temperature))
                    uci_move = str(state.action_to_uci(action))
                    output_fn(f"AlphaZero: {uci_move}")
                else:
                    bestmove = engine.bestmove(uci_history, movetime_ms=int(args.engine_time_ms))
                    if bestmove in {"(none)", "0000"}:
                        resigned_by = current_player
                        output_fn(f"Engine has no move ({bestmove}); treated as resignation.")
                        break
                    action = int(state.uci_to_action(bestmove))
                    uci_move = bestmove
                    output_fn(f"Engine: {uci_move}")

                state = state.apply_action(action)
                action_history.append(action)
                uci_history.append(uci_move)

            result, winner = _determine_result(state, game="chess", resigned_by=resigned_by)
            _ = _serialize_game(runtime.dependencies.cpp, game="chess", action_history=action_history, result=result)

            if winner is None:
                draws += 1
            elif winner == az_player:
                wins += 1
            else:
                losses += 1

            output_fn(f"Game result: {result}")

    finally:
        engine.close()

    output_fn("\nMatch summary:")
    output_fn(f"Games: {int(args.games)}")
    output_fn(f"AlphaZero W/L/D: {wins}/{losses}/{draws}")

    return MatchSummary(
        games=int(args.games),
        wins=wins,
        losses=losses,
        draws=draws,
    )


def run_from_args(args: argparse.Namespace) -> PlayGameSummary | MatchSummary:
    """Dispatch the requested play mode from parsed CLI arguments."""

    runtime = build_play_runtime(args=args)
    if str(args.opponent).strip().lower() == "human":
        summary: PlayGameSummary | None = None
        for _ in range(int(args.games)):
            summary = play_human_game(runtime, args=args)
        if summary is None:
            raise RuntimeError("No games were played")
        return summary

    return play_engine_match(runtime, args=args)


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        _ = run_from_args(args)
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
