"""Periodic Elo estimation helpers for AlphaZero training monitoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import threading
import time as _time
from typing import Any, Callable, Mapping, Protocol

from alphazero.config import load_yaml_config


DEFAULT_EVAL_INTERVAL_STEPS = 10_000
DEFAULT_EVAL_NUM_GAMES = 50
DEFAULT_EVAL_SIMULATIONS_PER_MOVE = 100
DEFAULT_EVAL_CHECKPOINT_DIR = Path("checkpoints")
_MILESTONE_CHECKPOINT_RE = re.compile(r"^milestone_(\d{8})\.pt$")


def _coerce_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _coerce_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _coerce_unit_interval_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if converted < 0.0 or converted > 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive, got {converted}")
    return converted


def _normalize_checkpoint_dir(path_like: str | Path) -> Path:
    if isinstance(path_like, Path):
        return path_like
    if isinstance(path_like, str):
        stripped = path_like.strip()
        if not stripped:
            raise ValueError("checkpoint_dir must not be empty")
        return Path(stripped)
    raise TypeError(
        f"checkpoint_dir must be a string or pathlib.Path, got {type(path_like).__name__}"
    )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Runtime configuration for periodic Elo estimation."""

    interval_steps: int = DEFAULT_EVAL_INTERVAL_STEPS
    num_games: int = DEFAULT_EVAL_NUM_GAMES
    simulations_per_move: int = DEFAULT_EVAL_SIMULATIONS_PER_MOVE
    checkpoint_dir: Path = DEFAULT_EVAL_CHECKPOINT_DIR

    def __post_init__(self) -> None:
        _coerce_positive_int("interval_steps", self.interval_steps)
        _coerce_positive_int("num_games", self.num_games)
        _coerce_positive_int("simulations_per_move", self.simulations_per_move)
        normalized_checkpoint_dir = _normalize_checkpoint_dir(self.checkpoint_dir)
        if normalized_checkpoint_dir != self.checkpoint_dir:
            object.__setattr__(self, "checkpoint_dir", normalized_checkpoint_dir)


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    """Aggregate result of a fixed-size evaluation match."""

    wins: int
    draws: int
    losses: int

    def __post_init__(self) -> None:
        _coerce_non_negative_int("wins", self.wins)
        _coerce_non_negative_int("draws", self.draws)
        _coerce_non_negative_int("losses", self.losses)
        if self.total_games == 0:
            raise ValueError("MatchOutcome must include at least one game")

    @property
    def total_games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / float(self.total_games)


@dataclass(frozen=True, slots=True)
class EloEvaluationResult:
    """Evaluation payload emitted whenever a periodic match is completed."""

    step: int
    milestone_step: int
    milestone_path: Path
    outcome: MatchOutcome
    score: float
    elo_difference: float

    @property
    def metric_tag(self) -> str:
        return f"eval/elo_vs_step_{self.milestone_step}"


class MatchRunner(Protocol):
    """Callable used by :class:`PeriodicEloEvaluator` to execute matches."""

    def __call__(
        self,
        *,
        current_network: object,
        milestone_checkpoint: Path,
        num_games: int,
        simulations_per_move: int,
    ) -> MatchOutcome | Mapping[str, object]:
        ...


class ScalarLoggerLike(Protocol):
    """Minimal logger contract used for scalar metric emission."""

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        ...


ScalarLogCallback = Callable[[str, float, int], None]


def load_evaluation_config_from_config(config: Mapping[str, Any]) -> EvaluationConfig:
    """Load :class:`EvaluationConfig` from a parsed pipeline config mapping."""

    if not isinstance(config, Mapping):
        raise TypeError(f"config must be a mapping, got {type(config).__name__}")

    evaluation = config.get("evaluation", {})
    if evaluation is None:
        evaluation = {}
    if not isinstance(evaluation, Mapping):
        raise ValueError("'evaluation' section must be a mapping")

    system = config.get("system", {})
    if system is None:
        system = {}
    if not isinstance(system, Mapping):
        raise ValueError("'system' section must be a mapping")

    checkpoint_dir = system.get("checkpoint_dir", DEFAULT_EVAL_CHECKPOINT_DIR)
    return EvaluationConfig(
        interval_steps=int(evaluation.get("interval_steps", DEFAULT_EVAL_INTERVAL_STEPS)),
        num_games=int(evaluation.get("num_games", DEFAULT_EVAL_NUM_GAMES)),
        simulations_per_move=int(
            evaluation.get(
                "simulations_per_move",
                DEFAULT_EVAL_SIMULATIONS_PER_MOVE,
            )
        ),
        checkpoint_dir=_normalize_checkpoint_dir(checkpoint_dir),
    )


def load_evaluation_config_from_yaml(config_path: str | Path) -> EvaluationConfig:
    """Load :class:`EvaluationConfig` from a YAML configuration file."""

    return load_evaluation_config_from_config(load_yaml_config(config_path))


def parse_milestone_step(checkpoint_path: str | Path) -> int | None:
    """Parse the step number from a milestone checkpoint filename."""

    path = Path(checkpoint_path)
    match = _MILESTONE_CHECKPOINT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def list_milestone_checkpoints(checkpoint_dir: str | Path) -> tuple[Path, ...]:
    """List milestone checkpoints sorted by ascending step."""

    checkpoint_root = _normalize_checkpoint_dir(checkpoint_dir)
    if not checkpoint_root.exists():
        return tuple()

    parsed: list[tuple[int, Path]] = []
    for path in checkpoint_root.glob("milestone_*.pt"):
        step = parse_milestone_step(path)
        if step is None:
            continue
        parsed.append((step, path))

    parsed.sort(key=lambda item: item[0])
    return tuple(item[1] for item in parsed)


def find_latest_milestone_checkpoint(
    checkpoint_dir: str | Path,
    *,
    max_step: int | None = None,
) -> Path | None:
    """Find the newest milestone checkpoint, optionally bounded by step."""

    if max_step is not None:
        _coerce_non_negative_int("max_step", max_step)

    latest: Path | None = None
    for checkpoint in list_milestone_checkpoints(checkpoint_dir):
        checkpoint_step = parse_milestone_step(checkpoint)
        if checkpoint_step is None:
            continue
        if max_step is not None and checkpoint_step > max_step:
            break
        latest = checkpoint
    return latest


def estimate_elo_from_score(score: float) -> float:
    """Estimate Elo difference from expected score in [0, 1]."""

    normalized_score = _coerce_unit_interval_float("score", score)
    if normalized_score == 0.0:
        return float("-inf")
    if normalized_score == 1.0:
        return float("inf")
    return -400.0 * math.log10((1.0 / normalized_score) - 1.0)


def estimate_elo_difference(outcome: MatchOutcome) -> float:
    """Estimate Elo difference from a completed match outcome."""

    if not isinstance(outcome, MatchOutcome):
        raise TypeError(f"outcome must be MatchOutcome, got {type(outcome).__name__}")
    return estimate_elo_from_score(outcome.score)


def _coerce_match_outcome(raw_outcome: MatchOutcome | Mapping[str, object]) -> MatchOutcome:
    if isinstance(raw_outcome, MatchOutcome):
        return raw_outcome
    if not isinstance(raw_outcome, Mapping):
        raise TypeError(
            "match_runner must return MatchOutcome or mapping with wins/draws/losses"
        )
    try:
        wins = _coerce_non_negative_int("wins", raw_outcome["wins"])
        draws = _coerce_non_negative_int("draws", raw_outcome["draws"])
        losses = _coerce_non_negative_int("losses", raw_outcome["losses"])
    except KeyError as exc:
        raise KeyError(
            "match_runner mapping result must include 'wins', 'draws', and 'losses'"
        ) from exc
    return MatchOutcome(wins=wins, draws=draws, losses=losses)


def _next_due_step(*, interval_steps: int, current_step: int) -> int:
    quotient = current_step // interval_steps
    return (quotient + 1) * interval_steps


class PeriodicEloEvaluator:
    """Stateful periodic evaluator for non-gating Elo monitoring."""

    def __init__(
        self,
        config: EvaluationConfig,
        *,
        match_runner: MatchRunner,
        scalar_logger: ScalarLogCallback | ScalarLoggerLike | None = None,
        start_step: int = 0,
        console_writer: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be EvaluationConfig, got {type(config).__name__}")
        _coerce_non_negative_int("start_step", start_step)
        if start_step % config.interval_steps == 0:
            self._next_due_step = start_step + config.interval_steps
        else:
            self._next_due_step = _next_due_step(
                interval_steps=config.interval_steps,
                current_step=start_step,
            )

        self._config = config
        self._match_runner = match_runner
        self._scalar_logger = scalar_logger
        self._console_writer = console_writer or print
        self._last_evaluated_step: int | None = None

    @property
    def config(self) -> EvaluationConfig:
        return self._config

    @property
    def next_due_step(self) -> int:
        return self._next_due_step

    def should_evaluate(self, step: int) -> bool:
        normalized_step = _coerce_non_negative_int("step", step)
        return normalized_step >= self._next_due_step

    def maybe_evaluate(
        self,
        *,
        step: int,
        current_network: object,
    ) -> EloEvaluationResult | None:
        normalized_step = _coerce_non_negative_int("step", step)
        if normalized_step == 0:
            return None
        if self._last_evaluated_step == normalized_step:
            return None
        if normalized_step < self._next_due_step:
            return None

        self._next_due_step = _next_due_step(
            interval_steps=self._config.interval_steps,
            current_step=normalized_step,
        )

        milestone_path = find_latest_milestone_checkpoint(
            self._config.checkpoint_dir,
            max_step=normalized_step,
        )
        if milestone_path is None:
            return None

        milestone_step = parse_milestone_step(milestone_path)
        if milestone_step is None:
            raise RuntimeError(f"Invalid milestone checkpoint filename: {milestone_path}")

        self._console_writer(
            f"Elo eval: step {normalized_step} — playing "
            f"{self._config.num_games} games vs milestone {milestone_step} "
            f"({milestone_path.name})"
        )
        t0 = _time.monotonic()

        raw_outcome = self._match_runner(
            current_network=current_network,
            milestone_checkpoint=milestone_path,
            num_games=self._config.num_games,
            simulations_per_move=self._config.simulations_per_move,
        )
        outcome = _coerce_match_outcome(raw_outcome)
        if outcome.total_games != self._config.num_games:
            raise ValueError(
                "match_runner returned unexpected game count: "
                f"expected {self._config.num_games}, got {outcome.total_games}"
            )

        elapsed = _time.monotonic() - t0
        elo_difference = estimate_elo_difference(outcome)
        result = EloEvaluationResult(
            step=normalized_step,
            milestone_step=milestone_step,
            milestone_path=milestone_path,
            outcome=outcome,
            score=outcome.score,
            elo_difference=elo_difference,
        )
        self._last_evaluated_step = normalized_step
        self._emit_scalar("eval/elo_vs_milestone", result.elo_difference, normalized_step)
        self._emit_scalar("eval/score_vs_milestone", result.score, normalized_step)

        self._console_writer(
            f"Elo eval: step {normalized_step} — "
            f"W/D/L {outcome.wins}/{outcome.draws}/{outcome.losses}  "
            f"score {outcome.score:.2f}  Elo {elo_difference:+.0f}  "
            f"({elapsed:.1f}s)"
        )
        return result

    def _emit_scalar(self, tag: str, value: float, step: int) -> None:
        if self._scalar_logger is None:
            return

        if callable(self._scalar_logger):
            self._scalar_logger(tag, value, step)
            return
        self._scalar_logger.log_scalar(tag, value, step)


def _build_eval_state_evaluator(
    *,
    model: Any,
    game_config: Any,
    device: Any,
    use_mixed_precision: bool,
) -> Callable[[Any], dict[str, object]]:
    """Build a single-state evaluator closure for ``MctsSearch.run_simulations()``.

    Parameters
    ----------
    model:
        A PyTorch model (may be ``torch.compile``-d) in eval mode on *device*.
    game_config:
        A :class:`~alphazero.config.GameConfig` providing board dimensions,
        input channels, action-space size, and value-head type.
    device:
        ``torch.device`` the model lives on.
    use_mixed_precision:
        When *True* and *device* is CUDA, runs inference under
        ``torch.autocast(dtype=bfloat16)``.

    Returns
    -------
    Callable[[state], dict]
        ``{"policy": list[float], "value": float, "policy_is_logits": True}``
    """
    import torch as _torch

    rows, cols = game_config.board_shape
    encoded_state_size = game_config.input_channels * rows * cols

    def evaluator(state: Any) -> dict[str, object]:
        encoded = _torch.as_tensor(
            state.encode(), dtype=_torch.float32, device=device,
        ).flatten()
        if int(encoded.numel()) != encoded_state_size:
            raise ValueError(
                f"State encoding size mismatch: "
                f"expected {encoded_state_size}, got {int(encoded.numel())}"
            )

        model_input = encoded.reshape(1, game_config.input_channels, rows, cols)
        with _torch.no_grad():
            with _torch.autocast(
                device_type=device.type,
                dtype=_torch.bfloat16,
                enabled=use_mixed_precision and device.type == "cuda",
            ):
                model_output = model(model_input)
                if not isinstance(model_output, (tuple, list)):
                    raise TypeError(
                        "model(model_input) must return tuple/list with policy/value "
                        f"(and optional ownership), got {type(model_output).__name__}"
                    )
                if len(model_output) == 2:
                    policy_logits, value = model_output
                elif len(model_output) == 3:
                    policy_logits, value, _ownership = model_output
                else:
                    raise ValueError(
                        "model(model_input) must return a 2-tuple (policy, value) "
                        "or 3-tuple (policy, value, ownership)"
                    )

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
            raise ValueError(
                f"Unsupported value_head_type {game_config.value_head_type!r}"
            )

        policy = (
            policy_logits.reshape(1, -1)
            .detach()
            .to(device="cpu", dtype=_torch.float32)
        )
        if int(policy.shape[1]) != game_config.action_space_size:
            raise ValueError(
                f"Policy output size mismatch: "
                f"expected {game_config.action_space_size}, got {int(policy.shape[1])}"
            )

        return {
            "policy": policy[0].tolist(),
            "value": scalar_value,
            "policy_is_logits": True,
        }

    return evaluator


class _BatchInferenceServer:
    """Accumulates eval requests from concurrent game threads and runs batched inference.

    Game threads call the evaluator returned by :meth:`create_evaluator` which
    blocks until the main thread calls :meth:`process_pending` to run a batched
    forward pass and distribute results.
    """

    def __init__(
        self,
        *,
        model: Any,
        game_config: Any,
        device: Any,
        use_mixed_precision: bool,
    ) -> None:
        import torch as _torch

        self._model = model
        self._device = device
        self._use_mixed_precision = use_mixed_precision
        self._game_config = game_config
        self._torch = _torch

        rows, cols = game_config.board_shape
        self._rows = rows
        self._cols = cols
        self._channels = game_config.input_channels
        self._encoded_state_size = self._channels * rows * cols

        self._lock = threading.Lock()
        self._pending: list[tuple[Any, threading.Event, list[dict[str, object] | None]]] = []
        self._stopped = False

    def create_evaluator(self) -> Callable[[Any], dict[str, object]]:
        """Return a single-state evaluator for ``MctsSearch.run_simulations()``."""

        _torch = self._torch
        device = self._device
        encoded_state_size = self._encoded_state_size
        server = self

        def evaluator(state: Any) -> dict[str, object]:
            encoded = _torch.as_tensor(
                state.encode(), dtype=_torch.float32, device=device,
            ).flatten()
            if int(encoded.numel()) != encoded_state_size:
                raise ValueError(
                    f"State encoding size mismatch: "
                    f"expected {encoded_state_size}, got {int(encoded.numel())}"
                )

            event = threading.Event()
            result_holder: list[dict[str, object] | None] = [None]
            with server._lock:
                if server._stopped:
                    raise RuntimeError("BatchInferenceServer has been stopped")
                server._pending.append((encoded, event, result_holder))
            event.wait()
            if result_holder[0] is None:
                raise RuntimeError("BatchInferenceServer stopped before producing a result")
            return result_holder[0]

        return evaluator

    def process_pending(self) -> int:
        """Collect all pending requests, run batched inference, distribute results."""

        with self._lock:
            if not self._pending:
                return 0
            batch = self._pending[:]
            self._pending.clear()

        _torch = self._torch
        batch_size = len(batch)
        encoded_batch = _torch.stack([item[0] for item in batch])
        model_input = encoded_batch.reshape(
            batch_size, self._channels, self._rows, self._cols
        )

        with _torch.no_grad():
            with _torch.autocast(
                device_type=self._device.type,
                dtype=_torch.bfloat16,
                enabled=self._use_mixed_precision and self._device.type == "cuda",
            ):
                model_output = self._model(model_input)
                if not isinstance(model_output, (tuple, list)):
                    raise TypeError(
                        "model(model_input) must return tuple/list with policy/value "
                        f"(and optional ownership), got {type(model_output).__name__}"
                    )
                if len(model_output) == 2:
                    policy_logits, value = model_output
                elif len(model_output) == 3:
                    policy_logits, value, _ownership = model_output
                else:
                    raise ValueError(
                        "model(model_input) must return a 2-tuple (policy, value) "
                        "or 3-tuple (policy, value, ownership)"
                    )

        game_config = self._game_config
        if game_config.value_head_type == "scalar":
            value_scalars = value.reshape(batch_size, -1)[:, 0].detach()
        elif game_config.value_head_type == "wdl":
            reshaped = value.reshape(batch_size, -1)
            value_scalars = (reshaped[:, 0] - reshaped[:, 2]).detach()
        else:
            raise ValueError(
                f"Unsupported value_head_type {game_config.value_head_type!r}"
            )

        policy_cpu = (
            policy_logits.reshape(batch_size, -1)
            .detach()
            .to(device="cpu", dtype=_torch.float32)
        )

        for i, (_, event, holder) in enumerate(batch):
            holder[0] = {
                "policy": policy_cpu[i].tolist(),
                "value": float(value_scalars[i].item()),
                "policy_is_logits": True,
            }
            event.set()

        return batch_size

    def stop(self) -> None:
        """Stop the server and wake all waiting threads."""

        with self._lock:
            self._stopped = True
            for _, event, _ in self._pending:
                event.set()
            self._pending.clear()


def create_mcts_match_runner(
    *,
    game_config: Any,
    network_factory: Callable[[], Any],
    cpp_game_config: Any,
    device: Any | None = None,
    use_mixed_precision: bool = True,
) -> Callable[..., MatchOutcome]:
    """Create a :class:`MatchRunner`-compatible closure for MCTS self-play matches.

    Parameters
    ----------
    game_config:
        :class:`~alphazero.config.GameConfig` for the target game.
    network_factory:
        Zero-argument callable returning a fresh, untrained model instance
        (same architecture as the training model).
    cpp_game_config:
        C++ game config object (e.g. ``cpp.chess_game_config()``).
    device:
        ``torch.device`` for inference. ``None`` auto-selects CUDA if available.
    use_mixed_precision:
        Whether to use bfloat16 autocast during inference.

    Returns
    -------
    Callable
        A ``MatchRunner`` that plays ``num_games`` games between the current
        network and a milestone checkpoint, returning a :class:`MatchOutcome`.
    """

    def match_runner(
        *,
        current_network: object,
        milestone_checkpoint: Path,
        num_games: int,
        simulations_per_move: int,
    ) -> MatchOutcome:
        import copy

        import torch as _torch

        import alphazero_cpp as cpp  # type: ignore[import-not-found]
        from alphazero.utils.checkpoint import load_checkpoint

        resolved_device = device
        if resolved_device is None:
            resolved_device = _torch.device(
                "cuda" if _torch.cuda.is_available() else "cpu"
            )

        # --- snapshot current weights into a fresh model ---
        source_model = current_network
        if hasattr(source_model, "_orig_mod"):
            source_model = source_model._orig_mod
        current_model = network_factory()
        current_model.load_state_dict(copy.deepcopy(source_model.state_dict()))
        current_model.to(device=resolved_device)
        current_model.eval()

        # --- load milestone into a fresh model ---
        milestone_model = network_factory()
        load_checkpoint(
            milestone_checkpoint,
            milestone_model,
            optimizer=None,
            map_location="cpu",
        )
        milestone_model.to(device=resolved_device)
        milestone_model.eval()

        # --- batched inference servers (one per model) ---
        current_server = _BatchInferenceServer(
            model=current_model,
            game_config=game_config,
            device=resolved_device,
            use_mixed_precision=use_mixed_precision,
        )
        milestone_server = _BatchInferenceServer(
            model=milestone_model,
            game_config=game_config,
            device=resolved_device,
            use_mixed_precision=use_mixed_precision,
        )

        # --- configure search (deterministic, no noise) ---
        search_config = cpp.SearchConfig()
        search_config.simulations_per_move = simulations_per_move
        search_config.enable_dirichlet_noise = False
        search_config.dirichlet_epsilon = 0.0
        search_config.temperature = 0.0
        search_config.temperature_moves = 0
        search_config.enable_resignation = False

        game_name = game_config.name.strip().lower()
        node_arena_capacity = max(simulations_per_move * 2, 1000)
        results: list[float | None] = [None] * num_games
        errors: list[BaseException | None] = [None] * num_games

        def _play_game(game_index: int) -> None:
            try:
                current_player = game_index % 2

                if game_name == "chess":
                    state = cpp.ChessState()
                elif game_name == "go":
                    state = cpp.GoState()
                else:
                    raise ValueError(f"Unsupported game {game_config.name!r}")

                current_eval = current_server.create_evaluator()
                milestone_eval = milestone_server.create_evaluator()
                evaluators = [None, None]
                evaluators[current_player] = current_eval
                evaluators[1 - current_player] = milestone_eval

                searches = [
                    cpp.MctsSearch(cpp_game_config, search_config, node_arena_capacity),
                    cpp.MctsSearch(cpp_game_config, search_config, node_arena_capacity),
                ]

                while not bool(state.is_terminal()):
                    player = int(state.current_player())
                    search = searches[player]
                    search.set_root_state(state)
                    search.run_simulations(evaluators[player])
                    action = int(search.select_action(0))
                    state = state.apply_action(action)

                results[game_index] = float(state.outcome(current_player))
            except BaseException as exc:
                errors[game_index] = exc

        # --- run all games in parallel, batch-process inference on main thread ---
        threads = [
            threading.Thread(target=_play_game, args=(i,), daemon=True)
            for i in range(num_games)
        ]
        for t in threads:
            t.start()

        try:
            while any(t.is_alive() for t in threads):
                n1 = current_server.process_pending()
                n2 = milestone_server.process_pending()
                if n1 == 0 and n2 == 0:
                    _time.sleep(0.0001)
        except BaseException:
            current_server.stop()
            milestone_server.stop()
            raise
        finally:
            for t in threads:
                t.join(timeout=5.0)

        for i, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(f"Eval game {i} failed") from err

        wins = 0
        draws = 0
        losses = 0
        for r in results:
            if r is None:
                raise RuntimeError("Not all eval games completed")
            if r > 0:
                wins += 1
            elif r < 0:
                losses += 1
            else:
                draws += 1

        return MatchOutcome(wins=wins, draws=draws, losses=losses)

    return match_runner


__all__ = [
    "DEFAULT_EVAL_INTERVAL_STEPS",
    "DEFAULT_EVAL_NUM_GAMES",
    "DEFAULT_EVAL_SIMULATIONS_PER_MOVE",
    "DEFAULT_EVAL_CHECKPOINT_DIR",
    "EvaluationConfig",
    "MatchOutcome",
    "EloEvaluationResult",
    "MatchRunner",
    "ScalarLoggerLike",
    "ScalarLogCallback",
    "PeriodicEloEvaluator",
    "load_evaluation_config_from_config",
    "load_evaluation_config_from_yaml",
    "parse_milestone_step",
    "list_milestone_checkpoints",
    "find_latest_milestone_checkpoint",
    "estimate_elo_from_score",
    "estimate_elo_difference",
    "create_mcts_match_runner",
]
