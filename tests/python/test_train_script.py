"""Tests for scripts/train.py runtime bootstrap and shutdown behavior."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "train.py"

_SPEC = importlib.util.spec_from_file_location("alphazero_train_script", SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import bootstrap guard.
    raise RuntimeError(f"Unable to load training script module from {SCRIPT_PATH}")
train_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = train_script
_SPEC.loader.exec_module(train_script)


@dataclass(frozen=True, slots=True)
class _FakeTrainingConfig:
    batch_size: int = 1024
    momentum: float = 0.9
    wait_for_buffer_seconds: float = 1.0
    use_mixed_precision: bool = True
    export_folded_checkpoints: bool = True
    device: str | None = None
    checkpoint_dir: Path | None = Path("checkpoints")
    checkpoint_keep_last: int = 10
    max_steps: int = 100


class _FakeLogger:
    def __init__(self) -> None:
        self.training_steps: list[int] = []
        self.snapshots: list[int] = []
        self.scalars: list[tuple[str, float, int]] = []
        self.flush_calls = 0
        self.close_calls = 0
        self._console_summary_interval_steps = 100

    def make_training_step_logger(self, **_kwargs: Any) -> Any:
        def _logger(step: int, _metrics: Mapping[str, float]) -> None:
            self.training_steps.append(int(step))

        return _logger

    def render_console_summary(self, _step: int) -> str:
        return ""

    def log_selfplay_snapshot(self, step: int, _snapshot: Any) -> None:
        self.snapshots.append(int(step))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), int(step)))

    def flush(self) -> None:
        self.flush_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _FakeReplayBuffer:
    def __init__(self, *, capacity: int, random_seed: int) -> None:
        self.capacity = capacity
        self.random_seed = random_seed


class _FakeCompactReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        num_binary_planes: int,
        num_float_planes: int,
        float_plane_indices: list[int],
        full_policy_size: int,
        random_seed: int,
        sampling_strategy: object = "uniform",
        recency_weight_lambda: float = 1.0,
    ) -> None:
        self.capacity = capacity
        self.num_binary_planes = num_binary_planes
        self.num_float_planes = num_float_planes
        self.float_plane_indices = list(float_plane_indices)
        self.full_policy_size = full_policy_size
        self.random_seed = random_seed
        self.sampling_strategy = sampling_strategy
        self.recency_weight_lambda = float(recency_weight_lambda)


class _FakeReplaySamplingStrategy:
    UNIFORM = "uniform"
    RECENCY_WEIGHTED = "recency_weighted"


class _FakeEvalQueueConfig:
    def __init__(self) -> None:
        self.batch_size = 256
        self.flush_timeout_us = 100


class _FakeEvalQueue:
    def __init__(self, *, evaluator: Any, encoded_state_size: int, config: Any) -> None:
        self.evaluator = evaluator
        self.encoded_state_size = encoded_state_size
        self.config = config


class _FakeSelfPlayGameConfig:
    def __init__(self) -> None:
        self.simulations_per_move = 800
        self.mcts_threads = 8
        self.node_arena_capacity = 1024
        self.enable_playout_cap = False
        self.reduced_simulations = 50
        self.full_playout_probability = 0.25
        self.c_puct = 2.5
        self.c_fpu = 0.25
        self.enable_dirichlet_noise = True
        self.dirichlet_epsilon = 0.25
        self.randomize_dirichlet_epsilon = False
        self.dirichlet_epsilon_min = 0.15
        self.dirichlet_epsilon_max = 0.35
        self.dirichlet_alpha_override = 0.0
        self.temperature = 1.0
        self.temperature_moves = 30
        self.enable_resignation = True
        self.resign_threshold = -0.9
        self.resign_disable_fraction = 0.1
        self.random_seed = 123


class _FakeSelfPlayManagerConfig:
    def __init__(self) -> None:
        self.concurrent_games = 32
        self.max_games_per_slot = 0
        self.initial_game_id = 1
        self.random_seed = 123
        self.game_config = _FakeSelfPlayGameConfig()


class _FakeSelfPlayManager:
    def __init__(
        self,
        _game_config: Any,
        _replay_buffer: Any,
        _evaluator: Any,
        _config: Any,
    ) -> None:
        self.eval_source = _evaluator
        self._metrics = SimpleNamespace(games_completed=7)
        self.simulation_updates: list[int] = []

    def metrics(self) -> Any:
        return self._metrics

    def update_simulations_per_move(self, new_sims: int) -> None:
        self.simulation_updates.append(int(new_sims))


class _Harness:
    def __init__(self) -> None:
        self.logger = _FakeLogger()
        self.resume_calls: list[Path] = []
        self.saved_steps: list[int] = []
        self.run_mode = "return"
        self.run_final_step = 0
        self.pipeline_steps: list[int] = []
        self.resume_step = 0
        self.resume_schedule: Any = "resumed_lr_schedule"
        self.created_eval_queue: _FakeEvalQueue | None = None
        self.selfplay_adapter_calls = 0

    def build_dependencies(self) -> Any:
        cpp = SimpleNamespace(
            EvalQueueConfig=_FakeEvalQueueConfig,
            SelfPlayManagerConfig=_FakeSelfPlayManagerConfig,
            ReplayBuffer=_FakeReplayBuffer,
            CompactReplayBuffer=_FakeCompactReplayBuffer,
            ReplaySamplingStrategy=_FakeReplaySamplingStrategy,
            EvalQueue=self._make_eval_queue,
            SelfPlayManager=_FakeSelfPlayManager,
            chess_game_config=lambda: "chess_cpp_config",
            go_game_config=lambda: "go_cpp_config",
        )

        class _FakeResNet:
            def __init__(
                self,
                _game_config: Any,
                *,
                num_blocks: int,
                num_filters: int,
                se_reduction: int,
            ) -> None:
                self.num_blocks = num_blocks
                self.num_filters = num_filters
                self.se_reduction = se_reduction

        return train_script.RuntimeDependencies(
            cpp=cpp,
            ResNetSE=_FakeResNet,
            create_optimizer=lambda *_args, **_kwargs: "optimizer",
            load_lr_schedule_from_config=lambda _config: "initial_lr_schedule",
            load_pipeline_config_from_config=lambda _config: "pipeline_config",
            load_training_checkpoint=self._load_training_checkpoint,
            load_training_config_from_config=lambda _config: _FakeTrainingConfig(),
            make_eval_queue_batch_evaluator=lambda *_args, **_kwargs: "batch_evaluator",
            make_selfplay_evaluator_from_eval_queue=self._make_selfplay_evaluator_from_eval_queue,
            run_parallel_pipeline=self._run_parallel_pipeline,
            save_training_checkpoint=self._save_training_checkpoint,
            build_run_name=lambda game_name: f"{game_name}_unit_run",
            create_metrics_logger=lambda **_kwargs: self.logger,
        )

    def _make_eval_queue(self, **kwargs: Any) -> _FakeEvalQueue:
        queue = _FakeEvalQueue(**kwargs)
        self.created_eval_queue = queue
        return queue

    def _make_selfplay_evaluator_from_eval_queue(self, _queue: Any) -> Any:
        self.selfplay_adapter_calls += 1
        return "selfplay_evaluator"

    def _load_training_checkpoint(
        self,
        checkpoint_path: Path,
        _model: Any,
        _optimizer: Any,
        *,
        map_location: str,
    ) -> tuple[int, Any]:
        del map_location
        self.resume_calls.append(Path(checkpoint_path))
        return self.resume_step, self.resume_schedule

    def _run_parallel_pipeline(self, *_args: Any, **_kwargs: Any) -> Any:
        if self.run_mode == "interrupt":
            raise KeyboardInterrupt
        step_logger = _kwargs.get("step_logger")
        if callable(step_logger):
            for step in self.pipeline_steps:
                step_logger(int(step), {})
        return SimpleNamespace(final_step=self.run_final_step)

    def _save_training_checkpoint(self, *_args: Any, **kwargs: Any) -> Any:
        self.saved_steps.append(int(kwargs["step"]))
        step = int(kwargs["step"])
        return SimpleNamespace(checkpoint_path=Path(f"checkpoints/checkpoint_{step:08d}.pt"))


def _minimal_config() -> dict[str, object]:
    return {
        "game": "chess",
        "network": {
            "architecture": "resnet_se",
            "num_blocks": 10,
            "num_filters": 128,
            "se_reduction": 4,
        },
        "mcts": {
            "simulations_per_move": 32,
            "c_puct": 2.5,
            "c_fpu": 0.25,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "randomize_dirichlet_epsilon": False,
            "dirichlet_epsilon_min": 0.15,
            "dirichlet_epsilon_max": 0.35,
            "enable_playout_cap": False,
            "reduced_simulations": 16,
            "full_playout_probability": 0.25,
            "temperature": 1.0,
            "temperature_moves": 30,
            "concurrent_games": 2,
            "threads_per_game": 2,
            "batch_size": 16,
            "resign_threshold": -0.9,
            "resign_disable_fraction": 0.1,
            "enable_dirichlet_noise": True,
            "enable_resignation": True,
        },
        "training": {
            "wait_for_buffer_seconds": 0.25,
            "export_folded_checkpoints": True,
        },
        "pipeline": {
            "inference_batches_per_cycle": 2,
            "training_steps_per_cycle": 1,
        },
        "replay_buffer": {
            "capacity": 512,
            "random_seed": 42,
        },
        "system": {
            "precision": "bf16",
            "compile": False,
            "run_name": "explicit_run_name",
        },
    }


class TrainScriptRuntimeTests(unittest.TestCase):
    def test_build_runtime_uses_cold_start_with_no_resume_checkpoint(self) -> None:
        """WHY: cold start must not load stale optimizer/model state and should begin from step 0."""
        harness = _Harness()
        dependencies = harness.build_dependencies()

        runtime = train_script.build_training_runtime(
            config_path="configs/chess_default.yaml",
            resume_path=None,
            dependencies=dependencies,
            config_override=_minimal_config(),
        )

        self.assertEqual(runtime.start_step, 0)
        self.assertEqual(harness.resume_calls, [])
        self.assertEqual(runtime.lr_schedule, "initial_lr_schedule")
        self.assertEqual(runtime.optimizer, "optimizer")
        self.assertEqual(runtime.logger, harness.logger)
        self.assertIsNotNone(harness.created_eval_queue)
        assert harness.created_eval_queue is not None
        self.assertEqual(harness.created_eval_queue.encoded_state_size, 119 * 8 * 8)
        self.assertIs(runtime.self_play_manager.eval_source, harness.created_eval_queue)
        self.assertEqual(harness.selfplay_adapter_calls, 0)

    def test_build_runtime_loads_resume_checkpoint_and_uses_loaded_step(self) -> None:
        """WHY: warm resume must restore step/LR schedule so training continues from checkpoint state."""
        harness = _Harness()
        harness.resume_step = 1234
        harness.resume_schedule = "loaded_lr_schedule"
        dependencies = harness.build_dependencies()

        with tempfile.TemporaryDirectory() as temp_dir:
            resume_path = Path(temp_dir) / "checkpoint_00001234.pt"
            resume_path.write_text("placeholder", encoding="utf-8")

            runtime = train_script.build_training_runtime(
                config_path="configs/chess_default.yaml",
                resume_path=resume_path,
                dependencies=dependencies,
                config_override=_minimal_config(),
            )

        self.assertEqual(harness.resume_calls, [resume_path])
        self.assertEqual(runtime.start_step, 1234)
        self.assertEqual(runtime.lr_schedule, "loaded_lr_schedule")

    def test_build_runtime_compiles_model_by_default(self) -> None:
        """WHY: torch.compile should be enabled by default to deliver the expected GPU kernel-fusion speedup."""
        harness = _Harness()
        dependencies = harness.build_dependencies()
        config = _minimal_config()
        config["system"] = {"precision": "bf16", "run_name": "explicit_run_name"}

        compile_calls: list[tuple[Any, str]] = []
        compiled_model = SimpleNamespace(to=lambda **kw: None)

        def _compile(model: Any, *, mode: str) -> Any:
            compile_calls.append((model, mode))
            return compiled_model

        fake_device = SimpleNamespace(type="cpu")
        fake_torch = SimpleNamespace(
            compile=_compile,
            device=lambda x: fake_device,
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}), \
                patch.object(train_script, "_warmup_compiled_model"):
            runtime = train_script.build_training_runtime(
                config_path="configs/chess_default.yaml",
                resume_path=None,
                dependencies=dependencies,
                config_override=config,
            )

        self.assertEqual(len(compile_calls), 1)
        compiled_input, compile_mode = compile_calls[0]
        self.assertEqual(compile_mode, "default")
        self.assertIs(runtime.model, compiled_model)
        self.assertEqual(compiled_input.__class__.__name__, "_FakeResNet")

    def test_build_runtime_skips_compile_when_disabled(self) -> None:
        """WHY: debugging and CPU fallback workflows need a deterministic way to bypass graph compilation."""
        harness = _Harness()
        dependencies = harness.build_dependencies()

        def _compile(_model: Any, *, mode: str) -> Any:
            raise AssertionError(f"compile should not be called when disabled, received mode={mode!r}")

        fake_torch = SimpleNamespace(compile=_compile)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            runtime = train_script.build_training_runtime(
                config_path="configs/chess_default.yaml",
                resume_path=None,
                dependencies=dependencies,
                config_override=_minimal_config(),
            )

        self.assertEqual(runtime.model.__class__.__name__, "_FakeResNet")

    def test_build_runtime_rejects_non_boolean_compile_flag(self) -> None:
        """WHY: system.compile must be strictly typed so config mistakes fail fast before long training runs."""
        harness = _Harness()
        dependencies = harness.build_dependencies()
        config = _minimal_config()
        config["system"] = {"precision": "bf16", "compile": "true", "run_name": "explicit_run_name"}

        with self.assertRaisesRegex(TypeError, "system.compile must be a bool"):
            train_script.build_training_runtime(
                config_path="configs/chess_default.yaml",
                resume_path=None,
                dependencies=dependencies,
                config_override=config,
            )

    def test_build_replay_buffer_uses_compact_buffer_for_chess_metadata(self) -> None:
        """WHY: training must route chess replay storage through compact compression to unlock larger buffer capacity."""
        harness = _Harness()
        dependencies = harness.build_dependencies()
        config = _minimal_config()
        game_config = train_script.get_game_config("chess")

        replay_buffer = train_script._build_replay_buffer(
            dependencies.cpp,
            config,
            game_config,
        )

        self.assertIsInstance(replay_buffer, _FakeCompactReplayBuffer)
        self.assertEqual(replay_buffer.capacity, 512)
        self.assertEqual(replay_buffer.random_seed, 42)
        self.assertEqual(replay_buffer.num_binary_planes, 117)
        self.assertEqual(replay_buffer.num_float_planes, 2)
        self.assertEqual(replay_buffer.float_plane_indices, [113, 118])
        self.assertEqual(replay_buffer.full_policy_size, 4672)
        self.assertEqual(replay_buffer.sampling_strategy, _FakeReplaySamplingStrategy.UNIFORM)
        self.assertEqual(replay_buffer.recency_weight_lambda, 1.0)

    def test_build_replay_buffer_maps_recency_weighted_sampling_settings(self) -> None:
        """WHY: recency-weighted replay sampling must propagate YAML controls into compact buffer construction."""
        harness = _Harness()
        dependencies = harness.build_dependencies()
        config = _minimal_config()
        config["replay_buffer"] = {
            "capacity": 512,
            "random_seed": 42,
            "sampling_strategy": "recency_weighted",
            "recency_weight_lambda": 2.5,
        }
        game_config = train_script.get_game_config("chess")

        replay_buffer = train_script._build_replay_buffer(
            dependencies.cpp,
            config,
            game_config,
        )

        self.assertIsInstance(replay_buffer, _FakeCompactReplayBuffer)
        self.assertEqual(
            replay_buffer.sampling_strategy,
            _FakeReplaySamplingStrategy.RECENCY_WEIGHTED,
        )
        self.assertEqual(replay_buffer.recency_weight_lambda, 2.5)

    def test_build_replay_buffer_falls_back_to_dense_for_non_chess_board_shapes(self) -> None:
        """WHY: current compact encoding assumes 8x8 planes, so Go must stay on the dense buffer path."""
        harness = _Harness()
        dependencies = harness.build_dependencies()
        config = _minimal_config()
        go_config = train_script.get_game_config("go")

        replay_buffer = train_script._build_replay_buffer(
            dependencies.cpp,
            config,
            go_config,
        )

        self.assertIsInstance(replay_buffer, _FakeReplayBuffer)
        self.assertEqual(replay_buffer.capacity, 512)
        self.assertEqual(replay_buffer.random_seed, 42)

    def test_build_replay_buffer_rejects_unknown_sampling_strategy(self) -> None:
        """WHY: invalid replay sampling strategy values should fail fast before worker startup."""
        harness = _Harness()
        dependencies = harness.build_dependencies()
        config = _minimal_config()
        config["replay_buffer"] = {
            "capacity": 512,
            "random_seed": 42,
            "sampling_strategy": "newest_only",
        }
        game_config = train_script.get_game_config("chess")

        with self.assertRaisesRegex(
            ValueError,
            "replay_buffer.sampling_strategy must be 'uniform' or 'recency_weighted'",
        ):
            train_script._build_replay_buffer(dependencies.cpp, config, game_config)

    def test_build_replay_buffer_rejects_negative_recency_weight_lambda(self) -> None:
        """WHY: negative recency lambda would invert weighting intent and should be rejected explicitly."""
        harness = _Harness()
        dependencies = harness.build_dependencies()
        config = _minimal_config()
        config["replay_buffer"] = {
            "capacity": 512,
            "random_seed": 42,
            "sampling_strategy": "recency_weighted",
            "recency_weight_lambda": -0.1,
        }
        game_config = train_script.get_game_config("chess")

        with self.assertRaisesRegex(
            ValueError,
            "replay_buffer.recency_weight_lambda must be finite and non-negative",
        ):
            train_script._build_replay_buffer(dependencies.cpp, config, game_config)

    def test_build_selfplay_manager_config_maps_playout_cap_fields(self) -> None:
        """WHY: playout-cap throughput tuning only works when YAML values propagate into C++ game config."""
        harness = _Harness()
        dependencies = harness.build_dependencies()
        config = _minimal_config()
        mcts = dict(config["mcts"])
        mcts["enable_playout_cap"] = True
        mcts["reduced_simulations"] = 8
        mcts["full_playout_probability"] = 0.4
        config["mcts"] = mcts

        manager_config = train_script._build_selfplay_manager_config(dependencies.cpp, config)

        self.assertTrue(manager_config.game_config.enable_playout_cap)
        self.assertEqual(manager_config.game_config.reduced_simulations, 8)
        self.assertAlmostEqual(manager_config.game_config.full_playout_probability, 0.4)

    def test_build_selfplay_manager_config_rejects_reduced_simulations_above_full_budget(self) -> None:
        """WHY: invalid reduced playout budget should fail fast instead of crashing inside long-running workers."""
        harness = _Harness()
        dependencies = harness.build_dependencies()
        config = _minimal_config()
        mcts = dict(config["mcts"])
        mcts["enable_playout_cap"] = True
        mcts["simulations_per_move"] = 16
        mcts["reduced_simulations"] = 17
        config["mcts"] = mcts

        with self.assertRaisesRegex(
            ValueError,
            "mcts.reduced_simulations must not exceed mcts.simulations_per_move",
        ):
            train_script._build_selfplay_manager_config(dependencies.cpp, config)

    def test_build_selfplay_manager_config_rejects_playout_probability_outside_unit_interval(self) -> None:
        """WHY: playout-cap probability must remain a true probability to avoid undefined sampling behavior."""
        harness = _Harness()
        dependencies = harness.build_dependencies()
        config = _minimal_config()
        mcts = dict(config["mcts"])
        mcts["enable_playout_cap"] = True
        mcts["full_playout_probability"] = 1.5
        config["mcts"] = mcts

        with self.assertRaisesRegex(
            ValueError,
            "mcts.full_playout_probability must be finite and in \\[0, 1\\]",
        ):
            train_script._build_selfplay_manager_config(dependencies.cpp, config)

    def test_build_selfplay_manager_config_maps_randomized_dirichlet_fields(self) -> None:
        """WHY: per-game root-noise randomization only works when YAML epsilon bounds reach the C++ game config."""
        harness = _Harness()
        dependencies = harness.build_dependencies()
        config = _minimal_config()
        mcts = dict(config["mcts"])
        mcts["randomize_dirichlet_epsilon"] = True
        mcts["dirichlet_epsilon_min"] = 0.2
        mcts["dirichlet_epsilon_max"] = 0.4
        config["mcts"] = mcts

        manager_config = train_script._build_selfplay_manager_config(dependencies.cpp, config)

        self.assertTrue(manager_config.game_config.randomize_dirichlet_epsilon)
        self.assertAlmostEqual(manager_config.game_config.dirichlet_epsilon_min, 0.2)
        self.assertAlmostEqual(manager_config.game_config.dirichlet_epsilon_max, 0.4)

    def test_build_selfplay_manager_config_rejects_invalid_randomized_dirichlet_bounds(self) -> None:
        """WHY: invalid epsilon bounds should fail fast so workers do not start with undefined noise schedules."""
        harness = _Harness()
        dependencies = harness.build_dependencies()
        config = _minimal_config()
        mcts = dict(config["mcts"])
        mcts["randomize_dirichlet_epsilon"] = True
        mcts["dirichlet_epsilon_min"] = 0.4
        mcts["dirichlet_epsilon_max"] = 0.3
        config["mcts"] = mcts

        with self.assertRaisesRegex(
            ValueError,
            "mcts.dirichlet_epsilon_min must be <= mcts.dirichlet_epsilon_max",
        ):
            train_script._build_selfplay_manager_config(dependencies.cpp, config)

    def test_run_session_interrupt_saves_final_checkpoint_and_flushes_logger(self) -> None:
        """WHY: graceful shutdown must preserve progress and flush metrics when interrupted by a signal."""
        harness = _Harness()
        harness.resume_step = 250
        harness.run_mode = "interrupt"
        dependencies = harness.build_dependencies()

        with tempfile.TemporaryDirectory() as temp_dir:
            resume_path = Path(temp_dir) / "checkpoint_00000250.pt"
            resume_path.write_text("placeholder", encoding="utf-8")
            runtime = train_script.build_training_runtime(
                config_path="configs/chess_default.yaml",
                resume_path=resume_path,
                dependencies=dependencies,
                config_override=_minimal_config(),
            )

        summary = train_script.run_training_session(runtime, dependencies=dependencies)

        self.assertTrue(summary.interrupted)
        self.assertEqual(summary.final_step, 250)
        self.assertEqual(harness.saved_steps, [250])
        self.assertEqual(summary.final_checkpoint_path, Path("checkpoints/checkpoint_00000250.pt"))
        self.assertEqual(harness.logger.flush_calls, 1)
        self.assertEqual(harness.logger.close_calls, 1)

    def test_run_session_uses_pipeline_result_step_for_final_checkpoint(self) -> None:
        """WHY: non-interrupted exits should save final state at the true terminal training step."""
        harness = _Harness()
        harness.run_mode = "return"
        harness.run_final_step = 777
        dependencies = harness.build_dependencies()

        runtime = train_script.build_training_runtime(
            config_path="configs/chess_default.yaml",
            resume_path=None,
            dependencies=dependencies,
            config_override=_minimal_config(),
        )

        summary = train_script.run_training_session(runtime, dependencies=dependencies)

        self.assertFalse(summary.interrupted)
        self.assertEqual(summary.final_step, 777)
        self.assertEqual(harness.saved_steps, [777])
        self.assertEqual(summary.final_checkpoint_path, Path("checkpoints/checkpoint_00000777.pt"))

    def test_run_session_updates_simulation_schedule_on_step_progress(self) -> None:
        """WHY: dynamic self-play throughput tuning must switch from 100 to 200 simulations at step 10k."""
        harness = _Harness()
        harness.run_mode = "return"
        harness.run_final_step = 12_000
        harness.pipeline_steps = [1, 9_999, 10_000, 12_000]
        dependencies = harness.build_dependencies()

        runtime = train_script.build_training_runtime(
            config_path="configs/chess_default.yaml",
            resume_path=None,
            dependencies=dependencies,
            config_override=_minimal_config(),
        )

        summary = train_script.run_training_session(runtime, dependencies=dependencies)

        self.assertFalse(summary.interrupted)
        self.assertEqual(summary.final_step, 12_000)
        self.assertEqual(runtime.self_play_manager.simulation_updates, [100, 100, 100, 200, 200])

    def test_run_session_applies_resume_step_schedule_before_pipeline(self) -> None:
        """WHY: resumed runs past the switch step should immediately restore the 200-simulation budget."""
        harness = _Harness()
        harness.run_mode = "return"
        harness.run_final_step = 15_000
        harness.resume_step = 15_000
        dependencies = harness.build_dependencies()

        with tempfile.TemporaryDirectory() as temp_dir:
            resume_path = Path(temp_dir) / "checkpoint_00015000.pt"
            resume_path.write_text("placeholder", encoding="utf-8")
            runtime = train_script.build_training_runtime(
                config_path="configs/chess_default.yaml",
                resume_path=resume_path,
                dependencies=dependencies,
                config_override=_minimal_config(),
            )

        summary = train_script.run_training_session(runtime, dependencies=dependencies)

        self.assertFalse(summary.interrupted)
        self.assertEqual(summary.final_step, 15_000)
        self.assertEqual(runtime.self_play_manager.simulation_updates, [200])


if __name__ == "__main__":
    unittest.main()
