# MCTS Evaluation Pipeline: GIL Bottleneck Analysis

## Problem Statement

Self-play training achieves only ~30% GPU utilization on the DGX Spark GB10 despite
having 64 concurrent games with 4 MCTS threads each (256 threads). Inference batches
process at ~1.2/second instead of the expected hundreds per second. No games complete
after 10+ minutes of running.

## Root Cause: Python GIL Serialization in the Hot Path

Every MCTS leaf evaluation goes through Python, forcing all 256 C++ worker threads
to serialize on the GIL.

### Current Evaluation Flow (per MCTS simulation)

```
C++ MCTS thread (one of 256)
  │
  ├─ Tree traversal to leaf node (pure C++, no GIL needed)
  │
  ├─ Call evaluator(GameState&)
  │     │
  │     ├─ py::gil_scoped_acquire         ← ACQUIRE GIL (contention with 255 other threads)
  │     ├─ py::cast GameState → Python object
  │     ├─ Call Python selfplay_evaluator(state):
  │     │     ├─ state.encode()           → C++ encode → numpy array (119,8,8) → back to Python
  │     │     ├─ .ravel().tolist()        → create 7,616 Python float objects
  │     │     ├─ eval_queue.submit_and_wait(python_list):
  │     │     │     ├─ cast_float_sequence → copy 7,616 floats into std::vector
  │     │     │     ├─ py::gil_scoped_release  ← RELEASE GIL
  │     │     │     ├─ C++ submit_and_wait (blocks on semaphore)
  │     │     │     ├─ ... process_batch() eventually runs ...
  │     │     │     ├─ py::gil_scoped_acquire  ← RE-ACQUIRE GIL
  │     │     │     └─ return EvalResult to Python
  │     │     ├─ list(eval_result.policy_logits)  → create 4,672 Python float objects
  │     │     └─ return dict
  │     ├─ parse_search_evaluation_result → copy back to C++ EvaluationResult
  │     └─ py::gil_scoped_release         ← RELEASE GIL
  │
  └─ Backpropagate value (pure C++, no GIL needed)
```

### Why This Is Catastrophically Slow

1. **GIL contention**: 256 threads compete for one lock. Each thread holds the GIL
   for the entire encode→submit→wait→return cycle. Only one thread can submit at
   a time. With Python's GIL scheduling (default 5ms switch interval), most threads
   are starved.

2. **Redundant data conversion**: Each evaluation round-trips data through Python:
   - C++ `encode()` → numpy array → Python `.tolist()` → 7,616 Python objects →
     C++ `cast_float_sequence` → `std::vector<float>` → `const float*`
   - C++ `EvalResult` → Python `eval_result` → Python `list()` → Python `dict` →
     C++ `parse_search_evaluation_result` → `EvaluationResult`

   The data starts in C++ and ends in C++. The Python detour creates ~12,000
   temporary Python objects per evaluation.

3. **process_batch() also needs GIL**: The inference thread's evaluator callback
   acquires the GIL to run PyTorch inference. This competes with MCTS threads
   trying to submit requests, creating a circular dependency where inference can't
   run because threads are holding the GIL trying to submit, and threads can't
   submit because inference is holding the GIL.

### Measured Impact

- Expected throughput: hundreds of inference batches per second
- Actual throughput: ~1.2 batches per 5 seconds (~0.24/sec)
- Slowdown factor: ~1000x
- Buffer fill time: >10 minutes with zero games completing (expected: <1 minute)

## The Key Insight: C++ Already Has Everything Needed

The entire Python roundtrip is unnecessary. Both ends of the connection already
exist in pure C++:

**Encoding (already in C++):**
```cpp
// src/games/game_state.h
class GameState {
    virtual void encode(float* buffer) const = 0;  // Fills pre-allocated buffer
};

// Chess: fills 119 * 8 * 8 = 7,616 floats
// Go: fills 17 * 19 * 19 = 6,137 floats
```

**Eval queue submission (already in C++):**
```cpp
// src/mcts/eval_queue.h
class EvalQueue {
    EvalResult submit_and_wait(const float* encoded_state);  // Takes raw float pointer
};
```

**MCTS evaluator interface:**
```cpp
// src/mcts/mcts_search.h
using EvaluateFn = std::function<EvaluationResult(const GameState&)>;
```

A pure C++ adapter can connect these directly:
```cpp
EvaluationResult eval_via_queue(const GameState& state) {
    thread_local std::vector<float> buffer(encoded_state_size);
    state.encode(buffer.data());
    EvalResult result = eval_queue.submit_and_wait(buffer.data());
    // Convert EvalResult → EvaluationResult (trivial field mapping)
}
```

No GIL. No Python objects. No data conversion. Each thread has its own buffer
(thread_local), so no contention beyond the eval queue's internal mutex.

## Architecture After Refactor

```
C++ MCTS thread (one of 256)
  │
  ├─ Tree traversal to leaf (pure C++)
  │
  ├─ Call evaluator(GameState&):
  │     ├─ state.encode(thread_local_buffer)        ← pure C++
  │     ├─ eval_queue.submit_and_wait(buffer)        ← pure C++, blocks on semaphore
  │     └─ Convert EvalResult → EvaluationResult     ← pure C++
  │
  └─ Backpropagate (pure C++)
```

The GIL is only needed by:
- The inference worker thread (for the PyTorch evaluator callback in process_batch)
- The training worker thread (for PyTorch training steps)
- The main thread (for progress monitoring)

MCTS threads never touch Python. No GIL contention from self-play.

## Files Involved

| File | Role |
|------|------|
| `src/mcts/eval_queue.h` | EvalQueue with `submit_and_wait(const float*)` |
| `src/mcts/eval_queue.cpp` | Queue implementation, batching, process_batch |
| `src/mcts/mcts_search.h` | `EvaluateFn` typedef, MctsSearch |
| `src/mcts/mcts_search.cpp` | `run_simulation()` calls `evaluator(*state)` at line 176 |
| `src/games/game_state.h` | `virtual void encode(float*) const = 0` |
| `src/games/chess/chess_state.cpp` | Chess encoding (lines 603-634) |
| `src/games/go/go_state.cpp` | Go encoding (lines 817-832) |
| `src/selfplay/self_play_manager.h` | SelfPlayManager, stores `EvaluateFn` |
| `src/selfplay/self_play_manager.cpp` | Worker threads, creates SelfPlayGame |
| `src/bindings/python_bindings.cpp` | PyEvalQueue, make_selfplay_evaluator, SelfPlayManager binding |
| `python/alphazero/pipeline/orchestrator.py` | `make_selfplay_evaluator_from_eval_queue` (the Python wrapper to eliminate) |
| `scripts/train.py` | Pipeline setup, passes evaluator to SelfPlayManager |

## Expected Impact

- MCTS threads run without GIL → full CPU parallelism across 20 cores
- Eval queue fills at native speed → GPU batch utilization jumps from ~30% to target ~85%
- Games complete in seconds instead of never → training can actually start
- Data conversion overhead eliminated → ~12,000 fewer Python objects per evaluation
