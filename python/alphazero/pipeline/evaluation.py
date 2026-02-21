"""Periodic Elo estimation helpers for AlphaZero training monitoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
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
        self._emit_scalar(
            result.metric_tag,
            result.elo_difference,
            normalized_step,
        )
        return result

    def _emit_scalar(self, tag: str, value: float, step: int) -> None:
        if self._scalar_logger is None:
            return

        if callable(self._scalar_logger):
            self._scalar_logger(tag, value, step)
            return
        self._scalar_logger.log_scalar(tag, value, step)


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
]
