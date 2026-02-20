"""Tests for TensorBoard metric logging utilities."""

from __future__ import annotations

from dataclasses import dataclass
import io
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from alphazero.utils.logging import (  # noqa: E402
    REQUIRED_SELFPLAY_SCALARS,
    REQUIRED_TRAINING_SCALARS,
    TensorBoardMetricsLogger,
    create_metrics_logger,
)


class _FakeWriter:
    def __init__(self, log_dir: pathlib.Path) -> None:
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.flush_calls = 0
        self.closed = False

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars.append((tag, float(scalar_value), int(global_step)))

    def flush(self) -> None:
        self.flush_calls += 1

    def close(self) -> None:
        self.closed = True


class _WriterFactory:
    def __init__(self) -> None:
        self.created: list[_FakeWriter] = []

    def __call__(self, run_dir: pathlib.Path) -> _FakeWriter:
        writer = _FakeWriter(run_dir)
        self.created.append(writer)
        return writer


@dataclass(slots=True)
class _Snapshot:
    games_completed: int
    average_game_length: float
    moves_per_second: float
    games_per_hour: float
    avg_simulations_per_second: float
    has_latest_game: bool
    latest_game_id: int
    latest_game_length: int
    latest_outcome_player0: float
    latest_game_resigned: bool
    latest_resignation_false_positive: bool


class TensorBoardMetricsLoggerTests(unittest.TestCase):
    def _base_training_metrics(self) -> dict[str, float]:
        return {
            "loss/total": 4.23,
            "loss/policy": 3.91,
            "loss/value": 0.30,
            "loss/l2": 0.02,
            "lr": 0.2,
            "throughput/train_steps_per_sec": 7.5,
            "buffer/size": 54_200,
        }

    def _base_selfplay_metrics(self) -> dict[str, float | bool]:
        return {
            "selfplay/game_length": 67,
            "selfplay/outcome": 1.0,
            "selfplay/resigned": False,
            "selfplay/resign_false_positive": False,
            "selfplay/moves_per_second": 150.0,
            "selfplay/games_per_hour": 142.0,
            "selfplay/avg_simulations_per_second": 9_500.0,
        }

    def _value_for_step(self, writer: _FakeWriter, *, tag: str, step: int) -> float:
        for event_tag, event_value, event_step in writer.scalars:
            if event_tag == tag and event_step == step:
                return event_value
        raise AssertionError(f"Missing scalar {tag!r} at step {step}")

    def test_training_metrics_logging_emits_all_required_scalars(self) -> None:
        """WHY: Missing any required scalar would silently break TensorBoard monitoring dashboards."""
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = _WriterFactory()
            logger = TensorBoardMetricsLogger(
                "chess_run_001",
                log_dir=pathlib.Path(temp_dir) / "logs",
                writer_factory=factory,
                console_stream=None,
            )

            logger.log_training_metrics(100, self._base_training_metrics(), emit_console=False)

            self.assertEqual(len(factory.created), 1)
            writer = factory.created[0]
            self.assertTrue(writer.log_dir.exists())
            self.assertTrue(writer.log_dir.name == "chess_run_001")

            expected_tags = set(REQUIRED_TRAINING_SCALARS) | {"buffer/games_total"}
            actual_tags = {tag for tag, _, step in writer.scalars if step == 100}
            self.assertEqual(actual_tags, expected_tags)
            self.assertEqual(self._value_for_step(writer, tag="buffer/games_total", step=100), 0.0)

            logger.close()

    def test_selfplay_metrics_logging_updates_games_total_for_training_logs(self) -> None:
        """WHY: Buffer game-count telemetry is needed to track data-generation pace alongside training loss."""
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = _WriterFactory()
            logger = TensorBoardMetricsLogger(
                "go_run_001",
                log_dir=pathlib.Path(temp_dir) / "logs",
                writer_factory=factory,
                console_stream=None,
            )

            logger.log_selfplay_metrics(25, self._base_selfplay_metrics())
            logger.log_training_metrics(100, self._base_training_metrics(), emit_console=False)

            writer = factory.created[0]
            selfplay_tags = {tag for tag, _, step in writer.scalars if step == 25}
            self.assertEqual(selfplay_tags, set(REQUIRED_SELFPLAY_SCALARS))
            self.assertEqual(self._value_for_step(writer, tag="buffer/games_total", step=100), 1.0)

            logger.close()

    def test_selfplay_snapshot_deduplicates_latest_game_logging_by_game_id(self) -> None:
        """WHY: Polling snapshots each cycle must not double-count per-game metrics when no new game finished."""
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = _WriterFactory()
            logger = TensorBoardMetricsLogger(
                "snapshot_run",
                log_dir=pathlib.Path(temp_dir) / "logs",
                writer_factory=factory,
                console_stream=None,
            )

            snapshot = _Snapshot(
                games_completed=7,
                average_game_length=63.0,
                moves_per_second=170.0,
                games_per_hour=130.0,
                avg_simulations_per_second=11_000.0,
                has_latest_game=True,
                latest_game_id=7,
                latest_game_length=55,
                latest_outcome_player0=-1.0,
                latest_game_resigned=True,
                latest_resignation_false_positive=False,
            )
            logger.log_selfplay_snapshot(40, snapshot)
            logger.log_selfplay_snapshot(41, snapshot)
            logger.log_selfplay_snapshot(
                42,
                _Snapshot(
                    games_completed=8,
                    average_game_length=64.0,
                    moves_per_second=171.0,
                    games_per_hour=131.0,
                    avg_simulations_per_second=11_100.0,
                    has_latest_game=True,
                    latest_game_id=8,
                    latest_game_length=61,
                    latest_outcome_player0=1.0,
                    latest_game_resigned=False,
                    latest_resignation_false_positive=True,
                ),
            )

            logger.log_training_metrics(100, self._base_training_metrics(), emit_console=False)

            writer = factory.created[0]
            game_length_events = [
                event
                for event in writer.scalars
                if event[0] == "selfplay/game_length"
            ]
            self.assertEqual(len(game_length_events), 2)
            self.assertEqual(self._value_for_step(writer, tag="buffer/games_total", step=100), 8.0)

            logger.close()

    def test_console_summary_emits_spec_style_sections(self) -> None:
        """WHY: Human-readable console summaries are the fast-path for spotting regressions during long runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            factory = _WriterFactory()
            console_stream = io.StringIO()
            logger = TensorBoardMetricsLogger(
                "console_run",
                log_dir=pathlib.Path(temp_dir) / "logs",
                writer_factory=factory,
                console_stream=console_stream,
                console_summary_interval_steps=5,
            )

            logger.log_selfplay_metrics(3, self._base_selfplay_metrics())
            logger.log_training_metrics(5, self._base_training_metrics())

            summary = console_stream.getvalue()
            self.assertIn("Step 5 | Loss:", summary)
            self.assertIn("Self-play:", summary)
            self.assertIn("Buffer:", summary)

            logger.close()

    def test_create_metrics_logger_uses_system_log_dir_from_config(self) -> None:
        """WHY: Logger creation must honor pipeline config so events land under the expected run directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_logs = pathlib.Path(temp_dir) / "custom_logs"
            factory = _WriterFactory()
            logger = create_metrics_logger(
                run_name="cfg_run",
                config={"system": {"log_dir": str(custom_logs)}},
                writer_factory=factory,
                console_stream=None,
            )

            self.assertEqual(logger.run_dir, custom_logs / "cfg_run")
            self.assertTrue(logger.run_dir.exists())
            logger.close()


if __name__ == "__main__":
    unittest.main()
