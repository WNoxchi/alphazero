"""Tests for periodic Elo evaluation utilities."""

from __future__ import annotations

import math
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from alphazero.pipeline.evaluation import (  # noqa: E402
    DEFAULT_EVAL_CHECKPOINT_DIR,
    DEFAULT_EVAL_INTERVAL_STEPS,
    DEFAULT_EVAL_NUM_GAMES,
    DEFAULT_EVAL_SIMULATIONS_PER_MOVE,
    EvaluationConfig,
    MatchOutcome,
    PeriodicEloEvaluator,
    _build_eval_state_evaluator,
    estimate_elo_difference,
    estimate_elo_from_score,
    find_latest_milestone_checkpoint,
    list_milestone_checkpoints,
    load_evaluation_config_from_config,
    parse_milestone_step,
)


class _FakeMatchRunner:
    def __init__(self, outcome: MatchOutcome) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        current_network: object,
        milestone_checkpoint: pathlib.Path,
        num_games: int,
        simulations_per_move: int,
    ) -> MatchOutcome:
        self.calls.append(
            {
                "current_network": current_network,
                "milestone_checkpoint": milestone_checkpoint,
                "num_games": num_games,
                "simulations_per_move": simulations_per_move,
            }
        )
        return self.outcome


class _FakeScalarLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, float, int]] = []

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.records.append((tag, float(value), int(step)))


class EvaluationConfigLoadingTests(unittest.TestCase):
    def test_load_config_uses_spec_defaults_when_sections_are_missing(self) -> None:
        """WHY: Missing YAML keys should still preserve the spec's evaluation cadence."""
        parsed = load_evaluation_config_from_config({"game": "chess"})

        self.assertEqual(parsed.interval_steps, DEFAULT_EVAL_INTERVAL_STEPS)
        self.assertEqual(parsed.num_games, DEFAULT_EVAL_NUM_GAMES)
        self.assertEqual(parsed.simulations_per_move, DEFAULT_EVAL_SIMULATIONS_PER_MOVE)
        self.assertEqual(parsed.checkpoint_dir, DEFAULT_EVAL_CHECKPOINT_DIR)

    def test_load_config_reads_nested_overrides_and_checkpoint_dir(self) -> None:
        """WHY: Runtime tuning requires evaluation and checkpoint settings to follow config values exactly."""
        parsed = load_evaluation_config_from_config(
            {
                "evaluation": {
                    "interval_steps": 5000,
                    "num_games": 64,
                    "simulations_per_move": 128,
                },
                "system": {"checkpoint_dir": "./custom_checkpoints"},
            }
        )

        self.assertEqual(
            parsed,
            EvaluationConfig(
                interval_steps=5000,
                num_games=64,
                simulations_per_move=128,
                checkpoint_dir=pathlib.Path("./custom_checkpoints"),
            ),
        )


class MilestoneDiscoveryTests(unittest.TestCase):
    def test_milestone_parsing_and_listing_are_step_sorted(self) -> None:
        """WHY: Evaluator must select chronological milestone baselines for stable progress tracking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = pathlib.Path(temp_dir)
            (checkpoint_dir / "milestone_00050000.pt").write_text("", encoding="utf-8")
            (checkpoint_dir / "milestone_00100000.pt").write_text("", encoding="utf-8")
            (checkpoint_dir / "checkpoint_00100000.pt").write_text("", encoding="utf-8")
            (checkpoint_dir / "milestone_abc.pt").write_text("", encoding="utf-8")

            milestones = list_milestone_checkpoints(checkpoint_dir)

        self.assertEqual(
            [path.name for path in milestones],
            ["milestone_00050000.pt", "milestone_00100000.pt"],
        )
        self.assertEqual(parse_milestone_step("milestone_00050000.pt"), 50000)
        self.assertIsNone(parse_milestone_step("checkpoint_00050000.pt"))

    def test_find_latest_milestone_supports_step_bounds(self) -> None:
        """WHY: Periodic evaluations should compare against the latest available prior milestone, not a future one."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = pathlib.Path(temp_dir)
            (checkpoint_dir / "milestone_00050000.pt").write_text("", encoding="utf-8")
            (checkpoint_dir / "milestone_00100000.pt").write_text("", encoding="utf-8")

            latest_any = find_latest_milestone_checkpoint(checkpoint_dir)
            latest_bounded = find_latest_milestone_checkpoint(checkpoint_dir, max_step=75000)

        self.assertIsNotNone(latest_any)
        assert latest_any is not None
        self.assertEqual(latest_any.name, "milestone_00100000.pt")
        self.assertIsNotNone(latest_bounded)
        assert latest_bounded is not None
        self.assertEqual(latest_bounded.name, "milestone_00050000.pt")


class EloMathTests(unittest.TestCase):
    def test_estimate_elo_from_score_matches_expected_symmetry(self) -> None:
        """WHY: Elo telemetry is meaningful only if score->Elo conversion follows the standard logistic formula."""
        self.assertAlmostEqual(estimate_elo_from_score(0.5), 0.0, places=6)
        self.assertAlmostEqual(estimate_elo_from_score(0.75), 190.8485, places=3)
        self.assertAlmostEqual(estimate_elo_from_score(0.25), -190.8485, places=3)

    def test_estimate_elo_from_match_outcome_handles_extremes(self) -> None:
        """WHY: Short evaluation matches can produce all-win/all-loss sweeps and must not crash telemetry."""
        self.assertTrue(math.isinf(estimate_elo_difference(MatchOutcome(50, 0, 0))))
        self.assertTrue(math.isinf(estimate_elo_difference(MatchOutcome(0, 0, 50))))


class PeriodicEloEvaluatorTests(unittest.TestCase):
    def test_evaluator_runs_periodically_logs_tensorboard_tag_and_avoids_duplicate_step(self) -> None:
        """WHY: Monitoring depends on running exactly-once evaluations per due step with the spec metric tag."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = pathlib.Path(temp_dir)
            (checkpoint_dir / "milestone_00050000.pt").write_text("", encoding="utf-8")
            runner = _FakeMatchRunner(MatchOutcome(wins=30, draws=10, losses=10))
            logged: list[tuple[str, float, int]] = []

            evaluator = PeriodicEloEvaluator(
                EvaluationConfig(
                    interval_steps=10_000,
                    num_games=50,
                    simulations_per_move=100,
                    checkpoint_dir=checkpoint_dir,
                ),
                match_runner=runner,
                scalar_logger=lambda tag, value, step: logged.append((tag, value, step)),
            )

            self.assertEqual(evaluator.next_due_step, 10_000)
            self.assertIsNone(evaluator.maybe_evaluate(step=10_000, current_network=object()))

            current_network = object()
            result = evaluator.maybe_evaluate(step=50_000, current_network=current_network)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.metric_tag, "eval/elo_vs_step_50000")
            self.assertEqual(result.milestone_path.name, "milestone_00050000.pt")
            self.assertAlmostEqual(result.score, 0.7, places=6)
            self.assertEqual(len(runner.calls), 1)
            self.assertIs(runner.calls[0]["current_network"], current_network)
            self.assertEqual(
                runner.calls[0]["milestone_checkpoint"],
                checkpoint_dir / "milestone_00050000.pt",
            )
            self.assertEqual(logged[0][0], "eval/elo_vs_milestone")
            self.assertEqual(logged[0][2], 50_000)
            self.assertEqual(logged[1][0], "eval/score_vs_milestone")
            self.assertEqual(logged[1][2], 50_000)

            self.assertIsNone(evaluator.maybe_evaluate(step=50_000, current_network=current_network))

    def test_evaluator_accepts_logger_object_and_start_step_alignment(self) -> None:
        """WHY: Warm resumes must schedule the next evaluation correctly and still emit scalars via logger objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = pathlib.Path(temp_dir)
            (checkpoint_dir / "milestone_00100000.pt").write_text("", encoding="utf-8")
            runner = _FakeMatchRunner(MatchOutcome(wins=25, draws=10, losses=15))
            logger = _FakeScalarLogger()

            evaluator = PeriodicEloEvaluator(
                EvaluationConfig(
                    interval_steps=10_000,
                    num_games=50,
                    simulations_per_move=100,
                    checkpoint_dir=checkpoint_dir,
                ),
                match_runner=runner,
                scalar_logger=logger,
                start_step=90_000,
            )

            self.assertEqual(evaluator.next_due_step, 100_000)
            result = evaluator.maybe_evaluate(step=100_000, current_network=object())
            self.assertIsNotNone(result)
            self.assertEqual(logger.records[0][0], "eval/elo_vs_milestone")
            self.assertEqual(logger.records[0][2], 100_000)
            self.assertEqual(logger.records[1][0], "eval/score_vs_milestone")
            self.assertEqual(logger.records[1][2], 100_000)

    def test_evaluator_rejects_match_result_with_wrong_game_count(self) -> None:
        """WHY: Elo estimates must be based on the configured match size to remain comparable across runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = pathlib.Path(temp_dir)
            (checkpoint_dir / "milestone_00050000.pt").write_text("", encoding="utf-8")
            runner = _FakeMatchRunner(MatchOutcome(wins=20, draws=10, losses=19))
            evaluator = PeriodicEloEvaluator(
                EvaluationConfig(
                    interval_steps=10_000,
                    num_games=50,
                    simulations_per_move=100,
                    checkpoint_dir=checkpoint_dir,
                ),
                match_runner=runner,
            )

            with self.assertRaisesRegex(ValueError, "unexpected game count"):
                evaluator.maybe_evaluate(step=50_000, current_network=object())


class BuildEvalStateEvaluatorTests(unittest.TestCase):
    """Tests for ``_build_eval_state_evaluator``."""

    def _require_torch(self) -> "Any":
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not available")
        return torch

    def test_scalar_value_head_returns_correct_policy_and_value(self) -> None:
        """WHY: The evaluator must faithfully forward the model's scalar output for MCTS consumption."""
        torch = self._require_torch()
        from alphazero.config import GameConfig

        game_config = GameConfig(
            name="test",
            board_shape=(3, 3),
            input_channels=2,
            action_space_size=9,
            value_head_type="scalar",
            supports_symmetry=False,
            num_symmetries=1,
        )

        expected_policy = torch.arange(9, dtype=torch.float32)
        expected_value = torch.tensor([[0.42]])

        class _FakeModel:
            def __call__(self, x: "Any") -> tuple["Any", "Any"]:
                return expected_policy.unsqueeze(0), expected_value

            def eval(self) -> None:
                pass

        class _FakeState:
            def encode(self) -> "Any":
                import numpy as np
                return np.ones((2, 3, 3), dtype=np.float32)

        evaluator = _build_eval_state_evaluator(
            model=_FakeModel(),
            game_config=game_config,
            device=torch.device("cpu"),
            use_mixed_precision=False,
        )
        result = evaluator(_FakeState())

        self.assertEqual(len(result["policy"]), 9)
        self.assertAlmostEqual(result["value"], 0.42, places=4)
        self.assertTrue(result["policy_is_logits"])
        for i in range(9):
            self.assertAlmostEqual(result["policy"][i], float(i), places=4)

    def test_wdl_value_head_returns_win_minus_loss(self) -> None:
        """WHY: WDL heads must be converted to scalar value as (win - loss) for MCTS."""
        torch = self._require_torch()
        from alphazero.config import GameConfig

        game_config = GameConfig(
            name="test",
            board_shape=(3, 3),
            input_channels=2,
            action_space_size=9,
            value_head_type="wdl",
            supports_symmetry=False,
            num_symmetries=1,
        )

        # win=0.7, draw=0.2, loss=0.1 → scalar = 0.7 - 0.1 = 0.6
        wdl_output = torch.tensor([[0.7, 0.2, 0.1]])
        policy = torch.zeros(1, 9)

        class _FakeModel:
            def __call__(self, x: "Any") -> tuple["Any", "Any"]:
                return policy, wdl_output

            def eval(self) -> None:
                pass

        class _FakeState:
            def encode(self) -> "Any":
                import numpy as np
                return np.ones((2, 3, 3), dtype=np.float32)

        evaluator = _build_eval_state_evaluator(
            model=_FakeModel(),
            game_config=game_config,
            device=torch.device("cpu"),
            use_mixed_precision=False,
        )
        result = evaluator(_FakeState())

        self.assertAlmostEqual(result["value"], 0.6, places=4)
        self.assertTrue(result["policy_is_logits"])


class CreateMctsMatchRunnerTests(unittest.TestCase):
    """Tests for ``create_mcts_match_runner``."""

    def _require_cpp(self) -> "Any":
        try:
            BUILD_SRC = ROOT / "build" / "src"
            if str(BUILD_SRC) not in sys.path:
                sys.path.insert(0, str(BUILD_SRC))
            import alphazero_cpp
        except (ModuleNotFoundError, ImportError):
            self.skipTest("alphazero_cpp not available")
        return alphazero_cpp

    def _require_torch(self) -> "Any":
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not available")
        return torch

    def test_match_runner_plays_correct_number_of_games_and_alternates_sides(self) -> None:
        """WHY: Side alternation ensures fair evaluation; game count must match config."""
        torch = self._require_torch()
        cpp = self._require_cpp()
        from alphazero.config import CHESS_CONFIG
        from alphazero.network import ResNetSE
        from alphazero.pipeline.evaluation import create_mcts_match_runner

        device = torch.device("cpu")

        model = ResNetSE(
            CHESS_CONFIG,
            num_blocks=1,
            num_filters=16,
            se_reduction=4,
        )
        model.to(device)
        model.eval()

        def network_factory() -> "Any":
            return ResNetSE(
                CHESS_CONFIG,
                num_blocks=1,
                num_filters=16,
                se_reduction=4,
            )

        # Save model weights as a fake milestone checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            milestone_path = pathlib.Path(temp_dir) / "milestone_00005000.pt"
            torch.save(
                {"model_state_dict": model.state_dict(), "step": 5000},
                milestone_path,
            )

            match_runner = create_mcts_match_runner(
                game_config=CHESS_CONFIG,
                network_factory=network_factory,
                cpp_game_config=cpp.chess_game_config(),
                device=device,
                use_mixed_precision=False,
            )

            num_games = 2
            outcome = match_runner(
                current_network=model,
                milestone_checkpoint=milestone_path,
                num_games=num_games,
                simulations_per_move=2,
            )

        self.assertIsInstance(outcome, MatchOutcome)
        self.assertEqual(outcome.total_games, num_games)

    def test_weight_snapshot_is_isolated_from_original_model(self) -> None:
        """WHY: Match runner must not modify the training model's weights."""
        torch = self._require_torch()
        cpp = self._require_cpp()
        from alphazero.config import CHESS_CONFIG
        from alphazero.network import ResNetSE
        from alphazero.pipeline.evaluation import create_mcts_match_runner

        device = torch.device("cpu")

        model = ResNetSE(
            CHESS_CONFIG,
            num_blocks=1,
            num_filters=16,
            se_reduction=4,
        )
        model.to(device)
        model.eval()

        # Capture original weights
        import copy
        original_state = copy.deepcopy(model.state_dict())

        def network_factory() -> "Any":
            return ResNetSE(
                CHESS_CONFIG,
                num_blocks=1,
                num_filters=16,
                se_reduction=4,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            milestone_path = pathlib.Path(temp_dir) / "milestone_00005000.pt"
            torch.save(
                {"model_state_dict": model.state_dict(), "step": 5000},
                milestone_path,
            )

            match_runner = create_mcts_match_runner(
                game_config=CHESS_CONFIG,
                network_factory=network_factory,
                cpp_game_config=cpp.chess_game_config(),
                device=device,
                use_mixed_precision=False,
            )

            match_runner(
                current_network=model,
                milestone_checkpoint=milestone_path,
                num_games=1,
                simulations_per_move=2,
            )

        # Verify original model weights are unchanged
        for key in original_state:
            self.assertTrue(
                torch.equal(original_state[key], model.state_dict()[key]),
                f"Model weight {key} was modified during match",
            )


if __name__ == "__main__":
    unittest.main()
