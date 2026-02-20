"""Tests for AlphaZero learning-rate schedule behavior."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

_YAML_AVAILABLE = importlib.util.find_spec("yaml") is not None

from alphazero.training.lr_schedule import (  # noqa: E402
    DEFAULT_STEP_DECAY_SCHEDULE,
    StepDecayLRSchedule,
    load_lr_schedule_from_config,
    load_lr_schedule_from_yaml,
    normalize_step_decay_schedule,
)


class LearningRateScheduleTests(unittest.TestCase):
    def test_default_schedule_matches_spec_milestones_and_boundaries(self) -> None:
        """Locks the canonical AlphaZero step-decay boundaries so later refactors cannot drift."""
        schedule = StepDecayLRSchedule()

        self.assertEqual(
            schedule.entries,
            DEFAULT_STEP_DECAY_SCHEDULE,
        )
        self.assertEqual(schedule.initial_lr, 0.2)
        self.assertEqual(schedule.milestones, (200000, 400000, 600000))

        self.assertEqual(schedule.lr_at_step(0), 0.2)
        self.assertEqual(schedule.lr_at_step(199999), 0.2)
        self.assertEqual(schedule.lr_at_step(200000), 0.02)
        self.assertEqual(schedule.lr_at_step(399999), 0.02)
        self.assertEqual(schedule.lr_at_step(400000), 0.002)
        self.assertEqual(schedule.lr_at_step(599999), 0.002)
        self.assertEqual(schedule.lr_at_step(600000), 0.0002)
        self.assertEqual(schedule.lr_at_step(900000), 0.0002)

    def test_schedule_loader_supports_custom_training_section(self) -> None:
        """Ensures YAML-derived config mappings can override decay boundaries for experiments."""
        config = {
            "game": "chess",
            "training": {
                "lr_schedule": [
                    [0, 0.3],
                    [5, 0.03],
                    [12, 0.003],
                ]
            },
        }

        schedule = load_lr_schedule_from_config(config)
        self.assertEqual(schedule.entries, ((0, 0.3), (5, 0.03), (12, 0.003)))
        self.assertEqual(schedule.milestones, (5, 12))
        self.assertEqual(schedule.lr_at_step(4), 0.3)
        self.assertEqual(schedule.lr_at_step(5), 0.03)
        self.assertEqual(schedule.lr_at_step(100), 0.003)

    def test_schedule_loader_falls_back_to_default_when_training_or_schedule_missing(self) -> None:
        """Keeps runtime startup resilient even when optional schedule config keys are omitted."""
        self.assertEqual(
            load_lr_schedule_from_config({"game": "go"}).entries,
            DEFAULT_STEP_DECAY_SCHEDULE,
        )
        self.assertEqual(
            load_lr_schedule_from_config({"training": {}}).entries,
            DEFAULT_STEP_DECAY_SCHEDULE,
        )

    def test_yaml_loader_uses_default_schedule_when_training_block_is_missing(self) -> None:
        """Ensures YAML config loading still returns the canonical schedule when no override is set."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = pathlib.Path(tmp_dir) / "no_training_schedule.yaml"
            config_path.write_text("game: go\n", encoding="utf-8")
            schedule = load_lr_schedule_from_yaml(config_path)
            self.assertEqual(schedule.entries, DEFAULT_STEP_DECAY_SCHEDULE)

    @unittest.skipUnless(_YAML_AVAILABLE, "PyYAML is required for nested YAML parsing")
    def test_yaml_loader_reads_nested_training_lr_schedule(self) -> None:
        """Guards the end-to-end YAML path used by training entry points."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = pathlib.Path(tmp_dir) / "custom.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    game: chess
                    training:
                      lr_schedule:
                        - [0, 0.4]
                        - [10, 0.04]
                        - [20, 0.004]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            schedule = load_lr_schedule_from_yaml(config_path)
            self.assertEqual(schedule.entries, ((0, 0.4), (10, 0.04), (20, 0.004)))
            self.assertEqual(schedule.lr_at_step(11), 0.04)

    def test_invalid_schedule_shapes_and_values_raise_clear_errors(self) -> None:
        """Fails fast on malformed schedules so training never silently uses bad LR settings."""
        with self.assertRaisesRegex(ValueError, "step 0"):
            normalize_step_decay_schedule([[1, 0.2]])

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            normalize_step_decay_schedule([[0, 0.2], [100, 0.1], [100, 0.05]])

        with self.assertRaisesRegex(ValueError, "exactly 2 values"):
            normalize_step_decay_schedule([[0, 0.2, 0.1]])

        with self.assertRaisesRegex(ValueError, "must be > 0"):
            normalize_step_decay_schedule([[0, 0.0]])

        with self.assertRaisesRegex(TypeError, "must be an integer"):
            normalize_step_decay_schedule([[0.5, 0.2]])

        with self.assertRaisesRegex(ValueError, "'training' section must be a mapping"):
            load_lr_schedule_from_config({"training": []})

        schedule = StepDecayLRSchedule()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            schedule.lr_at_step(-1)


if __name__ == "__main__":
    unittest.main()
