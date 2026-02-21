# Test Pyramid Implementation Plan

## Context

The AlphaZero codebase has 36 test files (18 Python, 18 C++) with solid unit and
component coverage. However, the chess/go end-to-end testing story has gaps:

1. **No pytest-driven chess/go pipeline smoke tests.** The test configs
   (`configs/chess_test.yaml`, `configs/go_test.yaml`) exist and work with
   `scripts/train.py`, but no pytest test exercises them. "Testing chess" means
   running the training script manually.

2. **`test_integration_smoke.py` uses synthetic components.** It constructs a
   `_SyntheticEvalQueue` and `_SyntheticSelfPlayManager` — it never exercises the
   real C++/Python boundary (EvalQueue batching, GIL handoff, SelfPlayManager worker
   threads, game state encoding/decoding).

3. **`test_bindings.py` has 2 pre-existing failures:**
   - `test_game_state_interface_is_callable_from_python`: `len(state.encode())`
     returns 119 (first dimension of 3D numpy array), not 7616. Should use
     `state.encode().size` or `np.prod(state.encode().shape)`.
   - `test_replay_buffer_round_trips_positions`: float32 precision loss when
     comparing `sample.encoded_state.tolist()` to Python float list. Should use
     `assertAlmostEqual` or `numpy.testing.assert_allclose`.

## Bugs Fixed (prerequisite for pipeline tests)

These bugs in the pipeline code were fixed in commit `d213fd8` and are required for
any real chess/go pipeline test to work:

| Bug | File | Fix |
|-----|------|-----|
| 3D array not flattened for EvalQueue | `orchestrator.py:~388` | Added `.ravel().tolist()` before `submit_and_wait()` |
| GIL livelock in main-thread `process_batch()` | `orchestrator.py:~460` | Run `process_batch()` in `threading.Thread`, `join()` releases GIL |
| Shutdown deadlock | `orchestrator.py:~567` | Stop `eval_queue` before `self_play_manager` |
| `process_batch()` hangs indefinitely | `eval_queue.h/cpp` | Added `wait_timeout` (100ms) to `EvalQueueConfig` |
| Static lib not position-independent | `src/CMakeLists.txt` | Added `POSITION_INDEPENDENT_CODE ON` |

## Current Test Inventory

### Unit Tests (Layer 1) - Fast, Isolated

| File | What It Tests | Status |
|------|---------------|--------|
| `test_loss.py` | Policy CE, scalar/WDL value loss, L2 reg | Complete |
| `test_config.py` | GameConfig shapes, YAML loading | Complete |
| `test_network.py` | Network forward/backward, shape validation | Complete |
| `test_lr_schedule.py` | Learning rate schedule | Complete |
| `test_bn_fold.py` | Batch norm folding | Complete |
| `test_checkpoint_utils.py` | Checkpoint save/load | Complete |
| `test_logging.py` | TensorBoard logging | Complete |
| `test_scaffold.py` | Python import scaffold | Complete |
| C++ chess tests (6 files) | Bitboard, movegen, state, encoding, serialization | Complete |
| C++ go tests (4 files) | Rules, state, encoding, serialization | Complete |
| C++ mcts/infra (6 files) | MCTS, eval queue, replay buffer, arena, self-play | Complete |

### Component Tests (Layer 2) - Moderate Speed, Pairs of Components

| File | What It Tests | Status |
|------|---------------|--------|
| `test_training.py` | `train_one_step`, `prepare_replay_batch`, checkpoint round-trip | Complete |
| `test_orchestrator.py` | Pipeline config, interleaved schedule logic | Complete |
| `test_train_script.py` | Script bootstrap, cold start, resume, interrupt handling | Complete |
| `test_bindings.py` | C++/Python pybind11 bindings | 2 pre-existing failures |
| `test_evaluation.py` | Model evaluation | Complete |
| `test_play_script.py` | Play script | Complete |
| `test_benchmark_script.py` | Benchmark script | Complete |
| `test_export_model_script.py` | Model export script | Complete |

### Integration Tests (Layer 3) - Slow, Full Pipeline

| File | What It Tests | Status |
|------|---------------|--------|
| `test_integration_smoke.py` | 100-step pipeline with **synthetic** eval queue + self-play | Complete |
| `test_connect_four_learning.py` | Full AlphaZero loop learns Connect Four (>75% vs random) | Complete |
| Chess pipeline smoke test | Real chess game through real C++ pipeline | **MISSING** |
| Go pipeline smoke test | Real go game through real C++ pipeline | **MISSING** |

## What Needs to Change

### Change 1: Fix `test_bindings.py` Pre-existing Failures

**File to modify:** `tests/python/test_bindings.py`

**Fix 1 — `test_game_state_interface_is_callable_from_python`:**
```python
# BEFORE
encoded = state.encode()
self.assertEqual(len(encoded), 119 * 8 * 8)

# AFTER
encoded = state.encode()
self.assertEqual(encoded.size, 119 * 8 * 8)
```

**Fix 2 — `test_replay_buffer_round_trips_positions`:**
```python
# BEFORE
self.assertEqual(sample.encoded_state.tolist(), encoded_state)

# AFTER
import numpy as np
np.testing.assert_allclose(sample.encoded_state, encoded_state, rtol=1e-6)
```

### Change 2: Create Chess Pipeline Smoke Test

**File to create:** `tests/python/test_chess_pipeline_smoke.py`

**Purpose:** Verify that chess-specific shapes (119-channel encoding, 4672 action
space, WDL value head) flow through the real C++ SelfPlayManager, EvalQueue, and
Python training pipeline without errors.

**This test should:**
1. Build a real `TrainingRuntime` via `train.py:build_training_runtime()` using
   `configs/chess_test.yaml`
2. Call `run_training_session()` from `train.py`
3. Assert: pipeline completes, replay buffer has positions, training steps executed
4. Mark with `@pytest.mark.slow`

**Implementation using the real `train.py` bootstrap (verified working):**

```python
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    import alphazero_cpp  # noqa: F401
    _CPP = True
except ImportError:
    _CPP = False


@unittest.skipUnless(_TORCH and _CPP, "Requires torch and alphazero_cpp")
class ChessPipelineSmokeTest(unittest.TestCase):

    @pytest.mark.slow
    def test_chess_pipeline_completes_3_training_steps(self):
        """
        Smoke test: run the real AlphaZero pipeline with chess for 3 training
        steps using configs/chess_test.yaml. Verifies shapes, types, and
        C++/Python component integration. Does NOT verify learning.
        """
        from scripts.train import build_training_runtime, run_training_session

        with tempfile.TemporaryDirectory() as tmpdir:
            # Override checkpoint/log dirs to use temp directory
            config_path = ROOT / "configs" / "chess_test.yaml"
            from alphazero.config import load_yaml_config
            config = dict(load_yaml_config(config_path))
            config["system"] = dict(config.get("system", {}))
            config["system"]["checkpoint_dir"] = tmpdir
            config["system"]["log_dir"] = tmpdir

            runtime = build_training_runtime(
                config_path=config_path,
                resume_path=None,
                config_override=config,
            )

            summary = run_training_session(runtime)

            self.assertEqual(summary.final_step, 3)
            self.assertFalse(summary.interrupted)
            self.assertGreater(summary.games_completed, 0)
            self.assertGreater(runtime.replay_buffer.size(), 0)
```

**Key points:**
- Uses `build_training_runtime()` which constructs the full C++ pipeline exactly
  as production does (verified working in manual testing).
- The C++ module is `alphazero_cpp`, NOT `alphazero_engine`.
- `configs/chess_test.yaml` uses: 2 concurrent games, batch_size=2,
  simulations_per_move=4, inference_batches_per_cycle=500, max_steps=3.
- With the `wait_timeout` fix in EvalQueue, `inference_batches_per_cycle: 500` is
  fine — `process_batch()` returns promptly (100ms timeout) when no eval requests
  are pending, so unused batches don't cause blocking.
- Chess completes in ~5-15 seconds (verified: 2 games in ~10s on aarch64 CPU).

### Change 3: Create Go Pipeline Smoke Test

**File to create:** `tests/python/test_go_pipeline_smoke.py`

**Identical structure** to the chess test, using `configs/go_test.yaml`. Go uses
17 input channels, 362 action space, scalar value head, dirichlet_alpha=0.03.

Go games on a 19x19 board with 4 sims/move are slower than chess. Go training
with max_steps=3 completes in ~10-30 seconds (verified on aarch64 CPU). Go is
hardcoded to 19x19 in `src/games/go/go_state.h` (`kBoardSize = 19`), so no
option to reduce board size.

### Change 4: Add `slow` pytest marker

**File to modify:** `pyproject.toml`

Add a pytest marker registration so `@pytest.mark.slow` doesn't produce warnings:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

### Change 5 (Optional): Tag Existing Slow Tests

Consider adding `@pytest.mark.slow` to:
- `test_connect_four_learning.py::test_short_alphazero_training_beats_random_connect_four_above_ninety_percent`
  (runs 20 self-play games + 40 evaluation games)

## Architecture Decision: Why NOT Full End-to-End Learning Tests for Chess/Go

A full "train chess AlphaZero for N steps and verify the model improves" test is the
wrong approach for several reasons:

1. **Irreducible cost.** Chess has a 119-channel encoding, 4672 action space, and
   complex move generation. Even with a 2-block/32-filter network and 4 sims/move,
   each self-play game takes seconds. You'd need dozens of games to see any measurable
   learning signal. This is 5-10 minutes minimum.

2. **Flaky by nature.** Learning tests depend on training dynamics. A "does loss
   decrease?" assertion can fail due to random seed variance, especially with tiny
   networks and few steps. Connect Four works because it's simple enough that learning
   is nearly guaranteed in 20 games.

3. **Redundant with component tests.** If the unit tests verify loss computation is
   correct, training steps work, MCTS produces valid policies, and chess encoding
   produces correct tensors — then the only remaining failure mode is a shape/type
   mismatch at integration boundaries. The smoke test catches exactly that.

The test pyramid for this codebase should be:

```
              /\
             /  \   Chess/Go pipeline smoke tests (~10-30s each)
            /    \  Connect Four learning test (~30-60s)
           /------\
          /        \  Component tests: trainer, orchestrator, bindings (~1-5s each)
         /          \
        /------------\
       /              \  Unit tests: loss, config, network, C++ game logic (<1s each)
      /________________\
```

## File Checklist

| Action | File | Priority |
|--------|------|----------|
| Modify | `pyproject.toml` (add slow marker) | P0 |
| Modify | `tests/python/test_bindings.py` (fix 2 pre-existing failures) | P1 |
| Create | `tests/python/test_chess_pipeline_smoke.py` | P1 |
| Create | `tests/python/test_go_pipeline_smoke.py` | P1 |
| Modify (optional) | `tests/python/test_connect_four_learning.py` (add @slow) | P2 |

## Verification

After implementation, verify:

```bash
# Fast tests only (should complete in <30 seconds)
pytest tests/python/ -m "not slow" -v

# Full suite including pipeline smoke tests (should complete in <2 minutes)
pytest tests/python/ -v

# Just the pipeline smoke tests
pytest tests/python/test_chess_pipeline_smoke.py tests/python/test_go_pipeline_smoke.py -v
```

The chess and go pipeline smoke tests should each complete in under 30 seconds.
