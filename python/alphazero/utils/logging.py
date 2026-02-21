"""TensorBoard logging helpers for AlphaZero training and self-play metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Callable, Mapping, Protocol, TextIO


DEFAULT_LOG_DIR = Path("logs")
DEFAULT_CONSOLE_SUMMARY_INTERVAL_STEPS = 100

REQUIRED_TRAINING_SCALARS: tuple[str, ...] = (
    "loss/total",
    "loss/policy",
    "loss/value",
    "loss/l2",
    "lr",
    "throughput/train_steps_per_sec",
    "buffer/size",
)

REQUIRED_SELFPLAY_SCALARS: tuple[str, ...] = (
    "selfplay/game_length",
    "selfplay/outcome",
    "selfplay/resigned",
    "selfplay/resign_false_positive",
    "selfplay/moves_per_second",
    "selfplay/games_per_hour",
    "selfplay/avg_simulations_per_second",
)


class ScalarWriter(Protocol):
    """Protocol for scalar-only TensorBoard writers."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        ...

    def flush(self) -> None:
        ...

    def close(self) -> None:
        ...


WriterFactory = Callable[[Path], ScalarWriter]


def _coerce_step(step: int) -> int:
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(f"step must be an integer, got {type(step).__name__}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    return step


def _coerce_scalar(name: str, value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _extract_field(source: Mapping[str, object] | object, name: str) -> object:
    if isinstance(source, Mapping):
        if name not in source:
            raise KeyError(f"Missing required field {name!r}")
        return source[name]
    if hasattr(source, name):
        return getattr(source, name)
    raise AttributeError(f"Missing required attribute {name!r}")


def _parse_required_scalars(
    metrics: Mapping[str, object],
    required: tuple[str, ...],
) -> dict[str, float]:
    missing: list[str] = []
    parsed: dict[str, float] = {}
    for key in required:
        if key not in metrics:
            missing.append(key)
            continue
        parsed[key] = _coerce_scalar(key, metrics[key])

    if missing:
        joined = ", ".join(sorted(missing))
        raise KeyError(f"Missing required metrics: {joined}")
    return parsed


def build_run_name(game_name: str | None = None, *, now: datetime | None = None) -> str:
    """Build a timestamped run directory name (for example ``chess_run_20260220_120000``)."""

    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    if game_name is None:
        return f"run_{timestamp}"

    normalized = game_name.strip().lower().replace(" ", "_")
    if not normalized:
        raise ValueError("game_name must not be empty")
    return f"{normalized}_run_{timestamp}"


def load_log_dir_from_config(config: Mapping[str, object]) -> Path:
    """Resolve ``system.log_dir`` from parsed configuration, defaulting to ``./logs``."""

    if not isinstance(config, Mapping):
        raise TypeError(f"config must be a mapping, got {type(config).__name__}")

    system = config.get("system", {})
    if system is None:
        system = {}
    if not isinstance(system, Mapping):
        raise ValueError("'system' section must be a mapping")

    raw_log_dir = system.get("log_dir", DEFAULT_LOG_DIR)
    if isinstance(raw_log_dir, Path):
        return raw_log_dir
    if isinstance(raw_log_dir, str):
        stripped = raw_log_dir.strip()
        if not stripped:
            raise ValueError("system.log_dir must not be empty")
        return Path(stripped)
    raise TypeError("system.log_dir must be a string or pathlib.Path")


def _default_writer_factory(run_dir: Path) -> ScalarWriter:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "tensorboard"}:
            raise
        raise ModuleNotFoundError(
            "TensorBoard logging requires torch + tensorboard. Install project dependencies "
            "or pass a custom writer_factory for tests."
        ) from exc

    return SummaryWriter(log_dir=str(run_dir))


@dataclass(slots=True)
class _SelfPlaySummary:
    games_completed: int = 0
    logged_games: int = 0
    total_logged_game_length: float = 0.0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    latest_moves_per_second: float = 0.0
    latest_games_per_hour: float = 0.0
    latest_avg_simulations_per_second: float = 0.0
    snapshot_average_game_length: float = 0.0

    def record_game(
        self,
        *,
        game_length: float,
        outcome: float,
        moves_per_second: float,
        games_per_hour: float,
        avg_simulations_per_second: float,
        known_games_completed: int | None = None,
    ) -> None:
        self.logged_games += 1
        self.total_logged_game_length += game_length

        if outcome > 0.0:
            self.wins += 1
        elif outcome < 0.0:
            self.losses += 1
        else:
            self.draws += 1

        self.latest_moves_per_second = moves_per_second
        self.latest_games_per_hour = games_per_hour
        self.latest_avg_simulations_per_second = avg_simulations_per_second

        if known_games_completed is None:
            self.games_completed += 1
        else:
            self.games_completed = max(self.games_completed, known_games_completed)

    def update_snapshot(
        self,
        *,
        games_completed: int,
        average_game_length: float,
        moves_per_second: float,
        games_per_hour: float,
        avg_simulations_per_second: float,
    ) -> None:
        self.games_completed = max(self.games_completed, games_completed)
        self.snapshot_average_game_length = average_game_length
        self.latest_moves_per_second = moves_per_second
        self.latest_games_per_hour = games_per_hour
        self.latest_avg_simulations_per_second = avg_simulations_per_second


class TensorBoardMetricsLogger:
    """Logs training and self-play scalars to TensorBoard plus periodic console summaries."""

    def __init__(
        self,
        run_name: str,
        *,
        log_dir: str | Path = DEFAULT_LOG_DIR,
        writer_factory: WriterFactory | None = None,
        console_stream: TextIO | None = sys.stdout,
        console_summary_interval_steps: int = DEFAULT_CONSOLE_SUMMARY_INTERVAL_STEPS,
    ) -> None:
        normalized_run_name = run_name.strip()
        if not normalized_run_name:
            raise ValueError("run_name must not be empty")
        if Path(normalized_run_name).name != normalized_run_name:
            raise ValueError("run_name must be a simple directory name")

        if (
            isinstance(console_summary_interval_steps, bool)
            or not isinstance(console_summary_interval_steps, int)
        ):
            raise TypeError(
                "console_summary_interval_steps must be an integer, got "
                f"{type(console_summary_interval_steps).__name__}"
            )
        if console_summary_interval_steps <= 0:
            raise ValueError(
                "console_summary_interval_steps must be positive, got "
                f"{console_summary_interval_steps}"
            )

        self.run_name = normalized_run_name
        self.log_root = Path(log_dir)
        self.run_dir = self.log_root / normalized_run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        active_factory = writer_factory or _default_writer_factory
        self._writer = active_factory(self.run_dir)
        self._console_stream = console_stream
        self._console_summary_interval_steps = console_summary_interval_steps

        self._selfplay = _SelfPlaySummary()
        self._last_logged_game_id: int | None = None
        self._latest_training_metrics: dict[str, float] = {}

    def _write_scalars(self, metrics: Mapping[str, float], *, step: int) -> None:
        for tag, value in metrics.items():
            self._writer.add_scalar(tag, value, step)

    def log_scalar(self, tag: str, value: object, step: int) -> None:
        """Log a single scalar metric, used for ad-hoc telemetry like Elo estimates."""

        if not isinstance(tag, str):
            raise TypeError(f"tag must be a string, got {type(tag).__name__}")
        normalized_tag = tag.strip()
        if not normalized_tag:
            raise ValueError("tag must not be empty")
        normalized_step = _coerce_step(step)
        normalized_value = _coerce_scalar(normalized_tag, value)
        self._writer.add_scalar(normalized_tag, normalized_value, normalized_step)

    def log_training_metrics(
        self,
        step: int,
        metrics: Mapping[str, object],
        *,
        games_total: int | float | None = None,
        emit_console: bool = True,
    ) -> None:
        """Log required training metrics and optionally emit a console summary."""

        normalized_step = _coerce_step(step)
        if not isinstance(metrics, Mapping):
            raise TypeError(f"metrics must be a mapping, got {type(metrics).__name__}")

        scalar_metrics = _parse_required_scalars(metrics, REQUIRED_TRAINING_SCALARS)
        if games_total is None:
            if "buffer/games_total" in metrics:
                games_total_value = _coerce_scalar(
                    "buffer/games_total",
                    metrics["buffer/games_total"],
                )
            else:
                games_total_value = float(self._selfplay.games_completed)
        else:
            games_total_value = _coerce_scalar("games_total", games_total)

        if games_total_value < 0.0:
            raise ValueError(f"buffer/games_total must be non-negative, got {games_total_value}")

        scalar_metrics["buffer/games_total"] = games_total_value
        self._write_scalars(scalar_metrics, step=normalized_step)
        self._latest_training_metrics = scalar_metrics

        if (
            emit_console
            and self._console_stream is not None
            and normalized_step % self._console_summary_interval_steps == 0
        ):
            self.emit_console_summary(normalized_step)

    def log_selfplay_metrics(self, step: int, metrics: Mapping[str, object]) -> None:
        """Log one completed self-play game's scalar metrics."""

        normalized_step = _coerce_step(step)
        if not isinstance(metrics, Mapping):
            raise TypeError(f"metrics must be a mapping, got {type(metrics).__name__}")

        scalar_metrics = _parse_required_scalars(metrics, REQUIRED_SELFPLAY_SCALARS)
        self._write_scalars(scalar_metrics, step=normalized_step)
        self._selfplay.record_game(
            game_length=scalar_metrics["selfplay/game_length"],
            outcome=scalar_metrics["selfplay/outcome"],
            moves_per_second=scalar_metrics["selfplay/moves_per_second"],
            games_per_hour=scalar_metrics["selfplay/games_per_hour"],
            avg_simulations_per_second=scalar_metrics["selfplay/avg_simulations_per_second"],
        )

    def log_selfplay_snapshot(self, step: int, snapshot: Mapping[str, object] | object) -> None:
        """Log latest self-play game from a ``SelfPlayMetricsSnapshot``-like object.

        The snapshot can come from pybind bindings or a plain mapping. Duplicate
        snapshots for the same ``latest_game_id`` are ignored.
        """

        normalized_step = _coerce_step(step)
        games_completed = int(
            _coerce_scalar(
                "games_completed",
                _extract_field(snapshot, "games_completed"),
            )
        )
        average_game_length = _coerce_scalar(
            "average_game_length",
            _extract_field(snapshot, "average_game_length"),
        )
        moves_per_second = _coerce_scalar(
            "moves_per_second",
            _extract_field(snapshot, "moves_per_second"),
        )
        games_per_hour = _coerce_scalar(
            "games_per_hour",
            _extract_field(snapshot, "games_per_hour"),
        )
        avg_sims_per_second = _coerce_scalar(
            "avg_simulations_per_second",
            _extract_field(snapshot, "avg_simulations_per_second"),
        )

        self._selfplay.update_snapshot(
            games_completed=games_completed,
            average_game_length=average_game_length,
            moves_per_second=moves_per_second,
            games_per_hour=games_per_hour,
            avg_simulations_per_second=avg_sims_per_second,
        )

        has_latest_game = bool(_extract_field(snapshot, "has_latest_game"))
        if not has_latest_game:
            return

        latest_game_id = int(
            _coerce_scalar(
                "latest_game_id",
                _extract_field(snapshot, "latest_game_id"),
            )
        )
        if self._last_logged_game_id is not None and latest_game_id <= self._last_logged_game_id:
            return

        scalar_metrics = {
            "selfplay/game_length": _coerce_scalar(
                "latest_game_length",
                _extract_field(snapshot, "latest_game_length"),
            ),
            "selfplay/outcome": _coerce_scalar(
                "latest_outcome_player0",
                _extract_field(snapshot, "latest_outcome_player0"),
            ),
            "selfplay/resigned": _coerce_scalar(
                "latest_game_resigned",
                _extract_field(snapshot, "latest_game_resigned"),
            ),
            "selfplay/resign_false_positive": _coerce_scalar(
                "latest_resignation_false_positive",
                _extract_field(snapshot, "latest_resignation_false_positive"),
            ),
            "selfplay/moves_per_second": moves_per_second,
            "selfplay/games_per_hour": games_per_hour,
            "selfplay/avg_simulations_per_second": avg_sims_per_second,
        }
        self._write_scalars(scalar_metrics, step=normalized_step)
        self._selfplay.record_game(
            game_length=scalar_metrics["selfplay/game_length"],
            outcome=scalar_metrics["selfplay/outcome"],
            moves_per_second=moves_per_second,
            games_per_hour=games_per_hour,
            avg_simulations_per_second=avg_sims_per_second,
            known_games_completed=games_completed,
        )
        self._last_logged_game_id = latest_game_id

    def make_training_step_logger(
        self,
        *,
        games_total_getter: Callable[[], int | float] | None = None,
        emit_console: bool = True,
    ) -> Callable[[int, Mapping[str, float]], None]:
        """Build a callback compatible with ``trainer.training_loop(step_logger=...)``."""

        def _logger(step: int, metrics: Mapping[str, float]) -> None:
            games_total: int | float | None
            if games_total_getter is None:
                games_total = None
            else:
                games_total = games_total_getter()
            self.log_training_metrics(
                step,
                metrics,
                games_total=games_total,
                emit_console=emit_console,
            )

        return _logger

    def render_console_summary(self, step: int) -> str:
        """Render the current multi-line console summary."""

        normalized_step = _coerce_step(step)
        if not self._latest_training_metrics:
            raise RuntimeError("No training metrics logged yet")

        loss_total = self._latest_training_metrics["loss/total"]
        loss_policy = self._latest_training_metrics["loss/policy"]
        loss_value = self._latest_training_metrics["loss/value"]
        loss_l2 = self._latest_training_metrics["loss/l2"]
        lr = self._latest_training_metrics["lr"]
        buffer_size = self._latest_training_metrics["buffer/size"]
        games_total = self._latest_training_metrics.get(
            "buffer/games_total",
            float(self._selfplay.games_completed),
        )

        if self._selfplay.logged_games > 0:
            avg_game_length = self._selfplay.total_logged_game_length / float(self._selfplay.logged_games)
            total_for_pct = float(self._selfplay.logged_games)
            win_pct = 100.0 * float(self._selfplay.wins) / total_for_pct
            draw_pct = 100.0 * float(self._selfplay.draws) / total_for_pct
            loss_pct = 100.0 * float(self._selfplay.losses) / total_for_pct
        else:
            avg_game_length = self._selfplay.snapshot_average_game_length
            win_pct = 0.0
            draw_pct = 0.0
            loss_pct = 0.0

        lines = [
            (
                f"Step {normalized_step} | Loss: {loss_total:.2f} "
                f"(policy: {loss_policy:.2f}, value: {loss_value:.2f}, l2: {loss_l2:.2f}) "
                f"| LR: {lr:.3f}"
            ),
            (
                "  Self-play: "
                f"{self._selfplay.latest_games_per_hour:.0f} games/hr, "
                f"avg length {avg_game_length:.0f} moves, "
                f"W/D/L: {win_pct:.0f}/{draw_pct:.0f}/{loss_pct:.0f}%"
            ),
            (
                "  Buffer: "
                f"{buffer_size:,.0f} positions, games {games_total:,.0f} "
                f"| Throughput: {self._latest_training_metrics['throughput/train_steps_per_sec']:.2f} train steps/s"
            ),
        ]
        return "\n".join(lines)

    def emit_console_summary(self, step: int) -> None:
        """Print the current console summary to the configured stream."""

        if self._console_stream is None:
            return
        print(self.render_console_summary(step), file=self._console_stream)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()

    def __enter__(self) -> "TensorBoardMetricsLogger":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def create_metrics_logger(
    *,
    run_name: str,
    config: Mapping[str, object] | None = None,
    log_dir: str | Path | None = None,
    writer_factory: WriterFactory | None = None,
    console_stream: TextIO | None = sys.stdout,
    console_summary_interval_steps: int = DEFAULT_CONSOLE_SUMMARY_INTERVAL_STEPS,
) -> TensorBoardMetricsLogger:
    """Construct a metrics logger using ``system.log_dir`` from config when available."""

    resolved_log_dir: str | Path
    if log_dir is not None:
        resolved_log_dir = log_dir
    elif config is not None:
        resolved_log_dir = load_log_dir_from_config(config)
    else:
        resolved_log_dir = DEFAULT_LOG_DIR

    return TensorBoardMetricsLogger(
        run_name,
        log_dir=resolved_log_dir,
        writer_factory=writer_factory,
        console_stream=console_stream,
        console_summary_interval_steps=console_summary_interval_steps,
    )


__all__ = [
    "DEFAULT_LOG_DIR",
    "DEFAULT_CONSOLE_SUMMARY_INTERVAL_STEPS",
    "REQUIRED_TRAINING_SCALARS",
    "REQUIRED_SELFPLAY_SCALARS",
    "ScalarWriter",
    "TensorBoardMetricsLogger",
    "WriterFactory",
    "build_run_name",
    "create_metrics_logger",
    "load_log_dir_from_config",
]
