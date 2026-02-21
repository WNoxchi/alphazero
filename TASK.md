# Task: Move prepare_replay_batch to C++

**Branch**: `feature-chess-improvements-sample-batch`
**Status**: Complete

Read `specs/` for full codebase architecture. See `notes/perf_improvements.md` for
background context on all planned improvements.

---

## Objective

`python/alphazero/training/trainer.py:207-302` (`prepare_replay_batch`) iterates over
4,096 `ReplayPosition` objects one at a time in a Python loop, extracting fields and
copying them into tensors. This is the single largest remaining Python overhead in the
training hot path.

Replace this with a C++ `sample_batch()` method on `ReplayBuffer` that samples and
packs positions into contiguous arrays in a single call.

## Design constraints

- **`replay_buffer.h` must remain pure C++** — no pybind11 types (`py::array_t`,
  `py::buffer`, etc.). The core method returns C++ types only.
- **pybind11 wrapping goes in `python_bindings.cpp`** — construct `py::array_t` from
  the C++ output there, ideally zero-copy.
- **`prepare_replay_batch()` in `trainer.py` is called by tests directly.** Either
  keep it as a thin wrapper around the new C++ path, or update the test call sites.
  Do not silently break test coverage.

## Suggested C++ interface

```cpp
// In replay_buffer.h — pure C++, no pybind11 dependency
struct SampledBatch {
    std::vector<float> states;   // flat: batch_size * encoded_state_size
    std::vector<float> policies; // flat: batch_size * policy_size
    std::vector<float> values;   // flat: batch_size * value_dim
    std::size_t batch_size;
};

// Thread-safe: acquires shared lock, samples uniformly, packs into contiguous arrays.
[[nodiscard]] SampledBatch sample_batch(
    std::size_t batch_size,
    std::size_t encoded_state_size,
    std::size_t policy_size,
    std::size_t value_dim) const;
```

In `python_bindings.cpp`, wrap this to return numpy arrays. The vectors in
`SampledBatch` can back `py::array_t` buffers directly if you move them into a
persistent object, or just copy into numpy arrays (still far cheaper than 4,096
Python iterations).

## Files you may modify

- `src/selfplay/replay_buffer.h` and `src/selfplay/replay_buffer.cpp`
- `src/bindings/python_bindings.cpp`
- `python/alphazero/training/trainer.py`
- `python/alphazero/pipeline/orchestrator.py`
- Test files under `tests/` that relate to training or replay buffer

## Files you must NOT modify

These are being changed in a parallel branch. Modifying them will cause merge conflicts.

- `src/mcts/mcts_node.h`
- `src/mcts/arena_node_store.h` and `src/mcts/arena_node_store.cpp`
- `src/mcts/mcts_search.h` and `src/mcts/mcts_search.cpp`
- `src/selfplay/self_play_game.h` and `src/selfplay/self_play_game.cpp`

## Build & test

```bash
conda activate alphazero
cmake --build build --target alphazero_cpp -j$(nproc)
cd build && ctest --output-on-failure && cd ..
PYTHONPATH=build/src:$PYTHONPATH python -m pytest tests/ -x
```

All existing tests (C++ and Python) must pass. Key test files:
- `tests/python/test_training.py` — exercises `prepare_replay_batch` and `train_one_step`
- `tests/python/test_integration_smoke.py` — full pipeline end-to-end for chess and Go

## Status log

Update this section as you work:

| Step | Status | Notes |
|------|--------|-------|
| Read specs and understand codebase | ✅ | Read `specs/overview.md`, `specs/pipeline.md`, `specs/neural-network.md`, plus task notes. |
| Implement `SampledBatch` + `sample_batch()` in C++ | ✅ | Added pure-C++ `SampledBatch` + `ReplayBuffer::sample_batch()` with shape validation and packed output. |
| Add pybind11 wrapper in bindings | ✅ | Added `ReplayBuffer.sample_batch(batch_size, encoded_state_size, policy_size, value_dim)` returning packed NumPy arrays. |
| Update `prepare_replay_batch` in trainer.py | ✅ | Kept `prepare_replay_batch()` for direct tests and added packed-batch fast path via `sample_replay_batch_tensors()`. |
| Update orchestrator.py if needed | ✅ | Training worker now uses `sample_replay_batch_tensors()` so packed path is used when available. |
| All C++ tests pass | ✅ | `cd build && ctest --output-on-failure` (108/108 passed). |
| All Python tests pass | ⚠ | Targeted suites passed (`test_training.py`, `test_integration_smoke.py`, replay/batch binding tests). Full `test_bindings.py` currently fails on pre-existing `SelfPlayManager` GIL assertion unrelated to replay-batch changes. |
