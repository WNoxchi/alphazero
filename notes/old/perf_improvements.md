# Performance Improvements

Read `specs/` to understand the full codebase architecture before implementing.

**Build & test**:
```bash
cmake --build build --target alphazero_cpp -j$(nproc)
cd build && ctest --output-on-failure
PYTHONPATH=build/src:$PYTHONPATH python -m pytest tests/ -x
```

---

## 1. Game-specific MCTSNode sizing

**Problem**: `src/mcts/mcts_node.h:67` defines `using MCTSNode = GoMCTSNode`, so every
node is sized for Go's 362 max actions even when playing chess (218 max actions). Each
node has 7 arrays of `MaxActions` elements, so Go-sized nodes are ~7.5 KB vs ~4.5 KB
for chess-sized nodes.

With 256 concurrent games x 8,192-node arenas, this wastes ~6 GB of memory and hurts
CPU cache efficiency during PUCT selection (fewer nodes fit per cache line).

**Fix**: Make the node type game-specific. The templated `MCTSNodeT<MaxActions>` already
exists — chess code should use `ChessMCTSNode` (218) and Go code should use
`GoMCTSNode` (362).

**Recommended approach**: Do NOT template the entire class hierarchy (`ArenaNodeStore`,
`MctsSearch`, `SelfPlayGame`, etc.) — that would propagate template parameters through
every layer up to the Python bindings. Instead, branch at the entry point:

- Keep `ArenaNodeStore`, `MctsSearch`, and `SelfPlayGame` working with a runtime-known
  max-actions size. One way: make `ArenaNodeStore` allocate nodes of a given byte size
  (determined at construction from `GameConfig::action_space_size`), and have `MctsSearch`
  use the actual `num_actions` stored in each node rather than the compile-time max.
- Alternatively, instantiate separate template specializations for chess and Go in the
  bindings layer (`python_bindings.cpp`) and expose them under the same Python class
  name, selected by game config. This keeps the C++ core templated but limits the
  template explosion to the bindings file.

The key constraint is that `python_bindings.cpp` must expose a single `MctsSearch` /
`ArenaNodeStore` interface to Python — the game-specific dispatch should be invisible
to the Python side.

**Files**:
- `src/mcts/mcts_node.h` — the `using MCTSNode = GoMCTSNode` alias
- `src/mcts/arena_node_store.h/.cpp` — allocates nodes, templated on node type
- `src/mcts/mcts_search.h/.cpp` — uses `MCTSNode` throughout
- `src/selfplay/self_play_game.h/.cpp` — creates arenas and searches
- `src/bindings/python_bindings.cpp` — exposes MctsSearch and ArenaNodeStore

**Do not modify**: `src/selfplay/replay_buffer.h/.cpp` or
`python/alphazero/training/trainer.py` (those are being changed in a parallel branch).

**Impact**: ~6 GB memory savings for chess, better cache utilization in MCTS hot loop.

**Verification**: All existing tests must pass. Run both the C++ tests (`ctest`) and
Python tests (`pytest tests/`). The chess_test and go_test integration smoke tests
exercise the full pipeline end-to-end and will catch regressions.

---

## 2. Config tuning for chess_1hr.yaml

**Current hardware observations** (DGX Spark, 20-block/256-filter model):
- GPU utilization: 80-95% (fluctuating)
- Memory: 72.5 / 128 GB
- CPU: 20 cores, 0-90% fluctuating

**Current config** (`configs/chess_1hr.yaml`):
```yaml
mcts:
  concurrent_games: 256
  threads_per_game: 1
  batch_size: 256

training:
  batch_size: 4096

replay_buffer:
  capacity: 500000
```

**Recommended changes**:
```yaml
mcts:
  concurrent_games: 384       # was 256 — more games keeps eval queue full, smooths GPU dips
  batch_size: 384             # was 256 — match concurrent_games for larger, more efficient batches

replay_buffer:
  capacity: 750000            # was 500000 — more training diversity (~12 GB additional memory)
```

**Rationale**:
- More concurrent games means the eval queue fills faster, reducing GPU idle gaps between
  batches. Larger batches have better GPU arithmetic intensity.
- Going from 256 to 384 concurrent games costs ~8 GB in arena memory (with GoMCTSNode)
  or ~5 GB (with ChessMCTSNode). With 55 GB free this is affordable.
- Larger replay buffer gives training more diverse samples across game
  histories/network versions, reducing overfitting to recent self-play.
- Training batch_size stays at 4096 — already well-sized. Doubling would halve weight
  updates per unit time and require LR scaling.
- inference_batches_per_cycle (100) and training_steps_per_cycle (1) stay unchanged —
  the 100:1 ratio allocates ~90%+ GPU time to inference, appropriate for early training
  where game generation is the bottleneck.

This config change can be included in either branch.

---

## 3. Move `prepare_replay_batch` to C++

**Problem**: `python/alphazero/training/trainer.py:207-302` iterates over 4,096
`ReplayPosition` objects one at a time in a Python loop, extracting fields and copying
them into tensors. This is the single largest remaining Python overhead in the hot path.

**Fix**: Add a `sample_batch()` method to `ReplayBuffer` that samples and packs
positions into contiguous arrays in a single C++ call, then expose it to Python.

**Constraints**:
- `replay_buffer.h` must remain pure C++ — no pybind11 types (`py::array_t`, etc.).
  The core C++ method should return something like a struct of `std::vector<float>`
  or write into caller-provided buffers. The pybind11 wrapping (returning numpy arrays)
  belongs in `python_bindings.cpp`.
- The Python-side `prepare_replay_batch()` function in `trainer.py` is also called by
  tests directly. Either keep it as a thin wrapper around the new C++ path, or update
  the tests that call it.

**Suggested C++ interface**:
```cpp
// In replay_buffer.h — pure C++, no pybind11 dependency
struct SampledBatch {
    std::vector<float> states;   // flat: batch_size * encoded_state_size
    std::vector<float> policies; // flat: batch_size * policy_size
    std::vector<float> values;   // flat: batch_size * value_dim
    std::size_t batch_size;
};

SampledBatch sample_batch(std::size_t batch_size,
                          std::size_t encoded_state_size,
                          std::size_t policy_size,
                          std::size_t value_dim) const;
```

Then in `python_bindings.cpp`, wrap this to return numpy arrays with zero-copy
(`py::array_t` constructed from the vector data).

**Files**:
- `src/selfplay/replay_buffer.h/.cpp` — add `SampledBatch` struct and `sample_batch()`
- `src/bindings/python_bindings.cpp` — expose as Python method returning numpy arrays
- `python/alphazero/training/trainer.py` — replace `prepare_replay_batch` internals
- `python/alphazero/pipeline/orchestrator.py` — update training worker if needed

**Do not modify**: `src/mcts/` files (those are being changed in a parallel branch).

**Verification**: All existing tests must pass. The training tests in
`tests/python/test_training.py` exercise `prepare_replay_batch` and `train_one_step`
end-to-end. The integration smoke tests in `tests/python/test_integration_smoke.py`
run the full pipeline.
