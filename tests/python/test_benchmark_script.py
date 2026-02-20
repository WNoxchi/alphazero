"""Tests for scripts/benchmark.py throughput benchmark orchestration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, cast
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "benchmark.py"

_SPEC = importlib.util.spec_from_file_location("alphazero_benchmark_script", SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import bootstrap guard.
    raise RuntimeError(f"Unable to load benchmark script module from {SCRIPT_PATH}")
benchmark_script = cast(Any, importlib.util.module_from_spec(_SPEC))
sys.modules[_SPEC.name] = benchmark_script
_SPEC.loader.exec_module(benchmark_script)


class BenchmarkScriptTests(unittest.TestCase):
    def _make_args(self, **overrides: object) -> Any:
        values: dict[str, object] = {
            "mode": "all",
            "game": "chess",
            "config": None,
            "batch_sizes": "8,16",
            "inference_batch_sizes": None,
            "training_batch_sizes": None,
            "games": "2,4",
            "threads": "1,3",
            "iterations": 3,
            "warmup_iterations": 2,
            "mcts_warmup_seconds": 0.0,
            "mcts_duration_seconds": 0.1,
            "mcts_simulations_per_move": 64,
            "num_blocks": 10,
            "num_filters": 32,
            "se_reduction": 2,
            "device": "cpu",
            "fp32": True,
            "random_seed": 123,
            "node_arena_capacity": 1024,
            "fail_fast": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_parse_positive_int_csv_enforces_numeric_positive_contract(self) -> None:
        """WHY: CLI grid parsing must reject malformed lists before any expensive benchmark starts."""
        self.assertEqual(
            benchmark_script._parse_positive_int_csv("batch_sizes", "32,64,128"),
            (32, 64, 128),
        )

        with self.assertRaisesRegex(ValueError, "non-integer"):
            benchmark_script._parse_positive_int_csv("batch_sizes", "64,abc")

        with self.assertRaisesRegex(ValueError, "positive"):
            benchmark_script._parse_positive_int_csv("batch_sizes", "64,0")

    def test_measure_loop_excludes_warmup_from_elapsed_throughput_window(self) -> None:
        """WHY: Throughput metrics are only valid if warmup work is excluded from timed measurements."""
        iteration_calls: list[str] = []
        clock_values = iter((10.0, 16.5))

        def run_iteration() -> None:
            iteration_calls.append("tick")

        elapsed = benchmark_script._measure_loop(
            warmup_iterations=2,
            timed_iterations=5,
            run_iteration=run_iteration,
            synchronize=None,
            perf_counter=lambda: next(clock_values),
        )

        self.assertEqual(elapsed, 6.5)
        self.assertEqual(len(iteration_calls), 7)

    def test_run_from_args_routes_batch_and_mcts_grids_to_mode_handlers(self) -> None:
        """WHY: Tuning workflows rely on exact propagation of user-provided batch/game/thread grids."""
        args = self._make_args(
            batch_sizes="9,27",
            inference_batch_sizes="4,12",
            training_batch_sizes="16,32",
            games="3,6",
            threads="2,5",
        )

        captured: dict[str, tuple[int, ...]] = {}

        def fake_benchmark_inference(**kwargs: Any) -> tuple[Any, ...]:
            captured["inference"] = tuple(int(value) for value in kwargs["batch_sizes"])
            return (
                benchmark_script.InferenceBenchmarkResult(
                    batch_size=4,
                    positions_per_second=100.0,
                    batch_latency_ms=10.0,
                ),
            )

        def fake_benchmark_training(**kwargs: Any) -> tuple[Any, ...]:
            captured["training"] = tuple(int(value) for value in kwargs["batch_sizes"])
            return (
                benchmark_script.TrainingBenchmarkResult(
                    batch_size=16,
                    steps_per_second=5.0,
                    positions_per_second=80.0,
                    step_latency_ms=20.0,
                ),
            )

        def fake_benchmark_mcts(**kwargs: Any) -> tuple[Any, ...]:
            captured["games"] = tuple(int(value) for value in kwargs["games"])
            captured["threads"] = tuple(int(value) for value in kwargs["threads"])
            return (
                benchmark_script.MctsBenchmarkResult(
                    concurrent_games=3,
                    threads_per_game=2,
                    simulations_per_move=64,
                    simulations_per_second=123.0,
                    moves_per_second=7.0,
                    games_per_hour=42.0,
                ),
            )

        original_inference = benchmark_script.benchmark_inference
        original_training = benchmark_script.benchmark_training
        original_mcts = benchmark_script.benchmark_mcts
        benchmark_script.benchmark_inference = fake_benchmark_inference
        benchmark_script.benchmark_training = fake_benchmark_training
        benchmark_script.benchmark_mcts = fake_benchmark_mcts
        try:
            report = benchmark_script.run_from_args(
                args,
                dependencies=benchmark_script.RuntimeDependencies(),
            )
        finally:
            benchmark_script.benchmark_inference = original_inference
            benchmark_script.benchmark_training = original_training
            benchmark_script.benchmark_mcts = original_mcts

        self.assertEqual(captured["inference"], (4, 12))
        self.assertEqual(captured["training"], (16, 32))
        self.assertEqual(captured["games"], (3, 6))
        self.assertEqual(captured["threads"], (2, 5))

        self.assertEqual(len(report.inference), 1)
        self.assertEqual(len(report.training), 1)
        self.assertEqual(len(report.mcts), 1)

    def test_format_report_renders_all_requested_sections(self) -> None:
        """WHY: Benchmark output must remain readable because operators tune runtime knobs from this text report."""
        report = benchmark_script.BenchmarkReport(
            mode="all",
            game="chess",
            device="cpu",
            inference=(
                benchmark_script.InferenceBenchmarkResult(
                    batch_size=32,
                    positions_per_second=1234.5,
                    batch_latency_ms=25.0,
                ),
            ),
            training=(
                benchmark_script.TrainingBenchmarkResult(
                    batch_size=256,
                    steps_per_second=7.5,
                    positions_per_second=1920.0,
                    step_latency_ms=133.3,
                    error="out of memory",
                ),
            ),
            mcts=(
                benchmark_script.MctsBenchmarkResult(
                    concurrent_games=16,
                    threads_per_game=8,
                    simulations_per_move=800,
                    simulations_per_second=45678.9,
                    moves_per_second=88.8,
                    games_per_hour=321.0,
                ),
            ),
        )

        rendered = benchmark_script.format_report(report)

        self.assertIn("Inference Throughput (positions/sec)", rendered)
        self.assertIn("Training Throughput (steps/sec)", rendered)
        self.assertIn("MCTS Throughput (simulations/sec)", rendered)
        self.assertIn("out of memory", rendered)


if __name__ == "__main__":
    unittest.main()
