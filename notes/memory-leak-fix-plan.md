# Memory Leak & GIL Safety Fix Plan

## Investigation Summary

A thorough investigation of the C++ core, pybind11 bindings, MCTS, self-play, and training
pipeline code identified several memory-safety and GIL-management issues. The most impactful
are GIL-related deadlock risks in the Python bindings layer, followed by MCTS node-mutex
lifecycle inefficiency and minor exception-safety gaps in capsule-based numpy views.

### What was ruled out

- **`emitted_checkpoints` list growth** (orchestrator.py:528): Accumulates ~256 bytes per
  checkpoint entry. Even at 10,000 checkpoints this is ~2.5 MB — negligible.
- **Milestone checkpoint disk accumulation**: Out of scope per user request.
- **`py::function` captured in C++ lambdas** (evaluator, completion callback): This is by
  design — the py::function prevents Python GC from collecting the callable, which is the
  correct behavior while SelfPlayManager/EvalQueue are alive. RAII ensures cleanup on
  destruction.
- **Thread-local `encoded_state_buffer`** (eval_queue.cpp:36-39): By design for buffer
  reuse. ~12 MB across 32 workers — acceptable.
- **ArenaNodeStore vectors never shrink**: Bounded by `node_arena_capacity` (default 8192).
  Memory is reused across allocations within a game and freed when the game ends. Correct.
- **Go `position_history` deep copies**: Not a leak — each MCTS clone needs its own history
  for superko detection. Memory is freed when nodes are released.
- **`node_mutexes_` unbounded growth**: Initially flagged as CRITICAL but on closer analysis
  this is bounded by arena capacity (default 8192) and cleared every move via
  `advance_root()` → `clear_node_mutexes()`. Each `SelfPlayGame` creates its own
  `RuntimeMctsSearch`, so the map is also destroyed between games. See TASK-003 for the
  remaining efficiency improvement.

---

## Prioritized Tasks

### TASK-001: Add GIL release to `PyEvalQueue::stop()` binding

- **File**: `src/bindings/python_bindings.cpp`
- **Current state**: COMPLETE (2026-02-27) — `stop()` now releases the GIL before stopping the queue.
- **Priority**: HIGH — deadlock risk under specific shutdown sequences.
- **Rationale**: `PyEvalQueue::process_batch()` (line 744) and `submit_and_wait()` (line 734)
  both correctly use `py::gil_scoped_release` before calling into C++. However `stop()`
  (line 749) does not:
  ```cpp
  void stop() { queue_.stop(); }
  ```
  `queue_.stop()` may block while failing pending requests and waiting for internal state to
  settle. If the evaluator lambda (captured `py::function` at line 706) is simultaneously
  trying to acquire the GIL in `process_batch()` on the same or another thread, and the
  caller of `stop()` holds the GIL, this creates a deadlock:
  - Thread A: holds GIL → calls `stop()` → blocks waiting for pending work
  - Thread B: in `process_batch()` → evaluator lambda tries to acquire GIL → blocks

  This matches the existing pattern where `SelfPlayManager.stop()` already has
  `py::call_guard<py::gil_scoped_release>()` (line 1444).

- **Fix**: Release the GIL inside `PyEvalQueue::stop()`:
  ```cpp
  void stop() {
      py::gil_scoped_release release_gil;
      queue_.stop();
  }
  ```
- **Acceptance criteria**:
  1. `stop()` releases the GIL before calling `queue_.stop()`
  2. Existing tests pass (build with `cmake --build build --target alphazero_cpp -j$(nproc)`)
  3. Manual verification: the training pipeline (`scripts/train.py`) shuts down cleanly
     without hanging
- **Implementation notes (2026-02-27)**:
  - Updated `PyEvalQueue::stop()` to use `py::gil_scoped_release` before `queue_.stop()`.
  - Added Python regression coverage: `test_eval_queue_stop_unblocks_waiting_submitters_without_consumer`.
- **Validation (2026-02-27)**:
  - `cmake --build build --target alphazero_cpp -j$(nproc)` ✅
  - `PYTHONPATH=build/src:$PYTHONPATH /home/hakan/miniconda3/envs/alphazero/bin/python -m pytest tests/python/test_bindings.py` ✅ (15 passed)
  - `python3 -m compileall -q python tests/python/test_bindings.py` ✅
  - `mypy` and `ruff` are not installed in this sandbox session; used compile-time/build checks plus `compileall` fallback.
  - `scripts/train.py --config configs/chess_test.yaml` startup/shutdown smoke run exited cleanly when interrupted (`Training interrupted at step 0`), with no shutdown hang observed.

---

### TASK-002: Add GIL release to `SelfPlayManager.start()` and other bindings

- **File**: `src/bindings/python_bindings.cpp`
- **Current state**: COMPLETE (2026-02-27) — `start()`, `update_simulations_per_move()`, and
  `metrics()` now release the GIL via `py::call_guard<py::gil_scoped_release>()`.
- **Priority**: HIGH — GIL contention during thread spawning.
- **Rationale**: The SelfPlayManager Python bindings (around lines 1443-1450) have
  inconsistent GIL handling:
  ```cpp
  .def("start", &SelfPlayManager::start)                                        // NO GIL release
  .def("stop", &SelfPlayManager::stop, py::call_guard<py::gil_scoped_release>()) // GIL released
  .def("is_running", &SelfPlayManager::is_running)                               // NO GIL release
  .def("update_simulations_per_move", ...)                                       // NO GIL release
  .def("metrics", &SelfPlayManager::metrics)                                     // NO GIL release
  ```

  `start()` (self_play_manager.cpp lines 83-128) acquires `lifecycle_mutex_`, resets state,
  then spawns `config_.concurrent_games` worker threads in a loop. While the thread spawning
  itself is fast, it holds the GIL the entire time. If worker threads immediately try to
  acquire the GIL (e.g., via a `py::function` evaluator), they'll block until `start()`
  returns. More importantly, holding the GIL during `start()` blocks all other Python threads
  for the duration.

  `is_running()`, `update_simulations_per_move()`, and `metrics()` are lightweight reads of
  atomic variables or small structs and are unlikely to cause issues, but releasing the GIL
  for them is consistent and free (no Python objects accessed in C++).

- **Fix**: Add `py::call_guard<py::gil_scoped_release>()` to `start()`. Optionally add it to
  `is_running()`, `update_simulations_per_move()`, and `metrics()` for consistency:
  ```cpp
  .def("start", &SelfPlayManager::start, py::call_guard<py::gil_scoped_release>())
  .def("stop", &SelfPlayManager::stop, py::call_guard<py::gil_scoped_release>())
  .def("is_running", &SelfPlayManager::is_running)
  .def("update_simulations_per_move", &SelfPlayManager::update_simulations_per_move,
       py::arg("new_sims"), py::call_guard<py::gil_scoped_release>())
  .def("metrics", &SelfPlayManager::metrics, py::call_guard<py::gil_scoped_release>())
  ```
  Note: `is_running()` returns a `bool` from an atomic and is trivially fast — GIL release
  is optional here. The others involve mutex acquisition or non-trivial work.

- **Acceptance criteria**:
  1. `start()` releases GIL before entering C++
  2. `update_simulations_per_move()` and `metrics()` release GIL
  3. Existing tests pass
  4. The training pipeline starts and runs self-play without hanging
- **Implementation notes (2026-02-27)**:
  - Added `py::call_guard<py::gil_scoped_release>()` to `SelfPlayManager.start()`.
  - Added `py::call_guard<py::gil_scoped_release>()` to
    `SelfPlayManager.update_simulations_per_move(...)`.
  - Added `py::call_guard<py::gil_scoped_release>()` to `SelfPlayManager.metrics()`.
  - Added regression coverage:
    `test_self_play_manager_bindings_release_gil_for_lifecycle_and_metrics_calls`.
- **Validation (2026-02-27)**:
  - `cmake --build build --target alphazero_cpp -j$(nproc)` ✅
  - `PYTHONPATH=build/src:$PYTHONPATH /home/hakan/miniconda3/envs/alphazero/bin/python -m pytest tests/python/test_bindings.py` ✅ (16 passed)
  - `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix --ignore-installed` ✅
  - `ruff` and `mypy` are not installed in this sandbox session; used `python3 -m compileall -q tests/python/test_bindings.py` fallback ✅
  - `PYTHONPATH=python:build/src:$PYTHONPATH /home/hakan/miniconda3/envs/alphazero/bin/python scripts/train.py --config configs/chess_test.yaml` startup/shutdown smoke run exited cleanly when interrupted (`Training interrupted at step 0`), with no shutdown hang observed.

---

### TASK-003: Remove eager mutex creation in `allocate_node()`

- **File**: `src/mcts/mcts_search.cpp`, `src/mcts/mcts_search.h`
- **Current state**: COMPLETE (2026-02-27) — `allocate_node()` now only allocates
  node storage, and mutexes remain lazily materialized by `node_mutex()`.
- **Priority**: MEDIUM — reduces memory overhead and allocation pressure during MCTS search.
- **Rationale**: `allocate_node()` (mcts_search.cpp lines 587-594) eagerly creates a mutex
  for every node:
  ```cpp
  NodeId MctsSearchT<NodeType>::allocate_node() {
      std::scoped_lock store_lock(store_mutex_);
      const NodeId id = node_store_.allocate();
      std::scoped_lock node_mutexes_lock(node_mutex_map_mutex_);
      node_mutexes_.emplace(id, std::make_shared<std::mutex>());
      return id;
  }
  ```

  However, `node_mutex()` (lines 579-584) already handles lazy creation:
  ```cpp
  std::shared_ptr<std::mutex> MctsSearchT<NodeType>::node_mutex(const NodeId node_id) const {
      std::scoped_lock node_mutexes_lock(node_mutex_map_mutex_);
      auto [it, inserted] = node_mutexes_.emplace(node_id, std::make_shared<std::mutex>());
      (void)inserted;
      return it->second;
  }
  ```

  Since `node_mutex()` creates-on-demand, the eager creation in `allocate_node()` is
  redundant. During a single MCTS search with 800 simulations, only a subset of nodes are
  concurrently accessed and need mutexes. Removing the eager creation avoids
  `make_shared<mutex>` + unordered_map insertion for every allocation.

  The `node_mutexes_` map is properly cleared every move (via `advance_root()` →
  `clear_node_mutexes()`) and destroyed with the `MctsSearchT` instance when the game ends,
  so this is an efficiency fix, not a leak fix.

- **Fix**: Remove the mutex creation from `allocate_node()`:
  ```cpp
  NodeId MctsSearchT<NodeType>::allocate_node() {
      std::scoped_lock store_lock(store_mutex_);
      return node_store_.allocate();
  }
  ```

- **Acceptance criteria**:
  1. `allocate_node()` no longer creates mutex entries
  2. `node_mutex()` still lazily creates mutexes on demand (unchanged)
  3. Existing tests pass (C++ tests: `cmake --build build --target test_mcts -j$(nproc)`,
     then run the test binary)
  4. Self-play still functions correctly (MCTS search produces valid moves, no deadlocks)
- **Implementation notes (2026-02-27)**:
  - Removed eager `node_mutexes_.emplace(...)` from `MctsSearchT<NodeType>::allocate_node()`.
  - Added `MctsSearchT<NodeType>::cached_node_mutex_count()` to expose current mutex-cache
    size for diagnostics/regression coverage.
  - Added regression test
    `MctsSearchTest.RootAllocationKeepsNodeMutexMapEmptyUntilLockIsNeeded` to prove root
    allocation/reset do not pre-populate mutex entries and that first lock access creates them.
- **Validation (2026-02-27)**:
  - `cmake --build build --target alphazero_cpp_tests -j$(nproc)` ✅
  - `ctest --test-dir build --output-on-failure -R "MctsSearchTest\\."` ✅ (9 passed)
  - `ctest --test-dir build --output-on-failure -R "SelfPlay"` ✅ (18 passed)
  - `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix --ignore-installed` ✅
    (rerun with `--ignore-installed` required in sandbox due non-writable existing install)
  - `ruff` and `mypy` are not installed in this sandbox session; ran
    `python3 -m compileall -q python tests/python/test_bindings.py` fallback ✅

---

### TASK-004: Improve exception safety in capsule-based numpy views

- **File**: `src/bindings/python_bindings.cpp`
- **Current state**: COMPLETE (2026-02-27) — capsule owner handoff now uses exception-safe
  `std::unique_ptr` transfer in sampled-batch NumPy view helpers.
- **Priority**: LOW — extremely unlikely to trigger in practice, but easy to fix.
- **Rationale**: `sampled_batch_array_view()` (lines 130-143) and
  `sampled_batch_vector_view()` (lines 145-156) use raw `new` to heap-allocate a
  `shared_ptr<SampledBatch>` for the capsule owner:
  ```cpp
  py::capsule owner(
      new std::shared_ptr<SampledBatch>(batch),
      [](void* ptr) { delete static_cast<std::shared_ptr<SampledBatch>*>(ptr); });
  ```
  If `py::capsule`'s constructor throws after the `new` but before taking ownership, the
  allocation leaks. In practice this is near-impossible (pybind11 capsule construction is
  trivial) but the pattern is fragile.

- **Fix**: Use a local `unique_ptr` to hold the allocation, then release it into the capsule:
  ```cpp
  auto prevent_leak = std::make_unique<std::shared_ptr<SampledBatch>>(batch);
  py::capsule owner(
      prevent_leak.get(),
      [](void* ptr) { delete static_cast<std::shared_ptr<SampledBatch>*>(ptr); });
  prevent_leak.release();  // capsule now owns the pointer
  ```
  Apply this pattern to both `sampled_batch_array_view()` and `sampled_batch_vector_view()`.

- **Acceptance criteria**:
  1. Both functions use exception-safe allocation pattern
  2. Numpy array views still work correctly (test via `ReplayBuffer.sample()` in Python)
  3. Existing tests pass
- **Implementation notes (2026-02-27)**:
  - Added `sampled_batch_owner_capsule(...)` helper that allocates
    `std::shared_ptr<SampledBatch>` ownership via `std::make_unique` and transfers ownership
    to `py::capsule` only after successful construction.
  - Updated both `sampled_batch_array_view(...)` and `sampled_batch_vector_view(...)` to use
    the helper, removing direct `new std::shared_ptr<SampledBatch>(batch)` allocations.
  - Added regression test
    `test_sample_batch_numpy_capsule_owner_transfer_is_exception_safe` to lock in the
    exception-safe ownership transfer pattern.
- **Validation (2026-02-27)**:
  - `cmake --build build --target alphazero_cpp -j$(nproc)` ✅
  - `PYTHONPATH=build/src:$PYTHONPATH /home/hakan/miniconda3/envs/alphazero/bin/python -m pytest tests/python/test_bindings.py` ✅ (17 passed)
  - `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix` ⚠️ failed due non-writable existing install; reran with `--ignore-installed`
  - `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix --ignore-installed` ✅
- `python3 -m ruff check tests/python/test_bindings.py` ❌ (`No module named ruff`)
- `python3 -m mypy tests/python/test_bindings.py` ❌ (`No module named mypy`)
- `python3 -m compileall -q python tests/python/test_bindings.py` ✅ (fallback static check)

---

### TASK-005: Release GIL for replay-buffer binding hot paths

- **File**: `src/bindings/python_bindings.cpp`
- **Current state**: COMPLETE (2026-02-27) — direct replay-buffer hot-path bindings now
  release the GIL, while NumPy-wrapper bindings keep scoped internal release windows.
- **Priority**: HIGH — long replay-buffer calls can hold the GIL and stall Python threads.
- **Rationale**: `ReplayBuffer` and `CompactReplayBuffer` bindings currently expose hot
  paths without `py::call_guard<py::gil_scoped_release>()`, including direct methods like
  `add_game`, `sample`, and compact checkpoint I/O (`save_to_file`, `load_from_file`).
  Additionally, NumPy-wrapper bindings (`sample_batch`, `export_buffer`, `import_buffer`)
  already perform internal `py::gil_scoped_release` around native buffer work and must keep
  the GIL while constructing Python objects.
- **Fix**:
  - Add `py::call_guard<py::gil_scoped_release>()` to direct C++ methods that do not touch
    Python objects: dense/compact `add_game`, dense/compact `sample`, and compact
    `save_to_file` / `load_from_file`.
  - Keep `sample_batch` / `export_buffer` / `import_buffer` on internal scoped-release
    helpers to avoid releasing the GIL across Python array creation/conversion code.
- **Acceptance criteria**:
  1. Direct dense/compact replay-buffer hot-path bindings release the GIL
  2. NumPy-wrapper replay bindings retain safe internal scoped GIL release behavior
  3. Existing replay-buffer binding behavior remains unchanged
  4. Binding tests pass
- **Implementation notes (2026-02-27)**:
  - Added `py::call_guard<py::gil_scoped_release>()` to:
    - `ReplayBuffer.add_game(...)`
    - `ReplayBuffer.sample(...)`
    - `CompactReplayBuffer.add_game(...)`
    - `CompactReplayBuffer.sample(...)`
    - `CompactReplayBuffer.save_to_file(...)`
    - `CompactReplayBuffer.load_from_file(...)`
  - Preserved existing internal `py::gil_scoped_release` windows in:
    `replay_buffer_sample_batch_numpy_impl(...)`,
    `replay_buffer_export_numpy_impl(...)`, and
    `replay_buffer_import_numpy_impl(...)`.
  - Added regression test:
    `test_replay_buffer_bindings_release_gil_for_hot_paths`.
- **Validation (2026-02-27)**:
  - `cmake --build build --target alphazero_cpp -j$(nproc)` ✅
  - `PYTHONPATH=build/src:$PYTHONPATH /home/hakan/miniconda3/envs/alphazero/bin/python -m pytest tests/python/test_bindings.py` ✅ (18 passed)
  - `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix` ⚠️ failed due non-writable existing install; reran with `--ignore-installed`
  - `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix --ignore-installed` ✅
  - `python3 -m ruff check tests/python/test_bindings.py` ❌ (`No module named ruff`)
  - `python3 -m mypy tests/python/test_bindings.py` ❌ (`No module named mypy`)
  - `python3 -m compileall -q python tests/python/test_bindings.py` ✅ (fallback static check)
