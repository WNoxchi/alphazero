# Performance Improvement Plan

## How to Use This Document

This plan is designed for **incremental, autonomous implementation**. Each item is self-contained. A future AI should:

1. Read `specs/*` for system context and `configs/chess_1hr.yaml` for current parameters.
2. Pick the highest-priority unchecked item below.
3. Implement it, run the existing test suite (`ctest --test-dir build` and `pytest`), and validate with a short training run.
4. Mark the item as done (check the box) and commit.

Items are ordered by ROI (impact ÷ effort). Dependencies between items are noted explicitly — skip any item whose dependencies are not yet complete.

---

## Current Bottleneck Summary

**Observed**: GPU 1-5%, CPU ~25%/20 cores, 41/128 GB memory during training with `configs/chess_1hr.yaml`.

**Root cause**: The pipeline is fully serial. The main thread loops: call `process_batch()` (one small GPU inference) → wait → repeat 50-100 times → do one training step → repeat. Inference and training never overlap. The GPU does a batch-256 forward pass in microseconds, then idles while waiting for the next batch to accumulate. Additionally, every eval batch crosses the C++→Python boundary with per-sample data copies and Python object allocation.

---

## Plan Items

### P1: Contiguous Batch Transfer in EvalQueue Callback
- [x] **Completed (2026-02-21)** — Switched EvalQueue C++↔Python bridge to contiguous batch arrays in both directions.
  `PyEvalQueue` now submits a single `(batch, encoded_state_size)` numpy array and parses batched
  `(policy_logits, value)` arrays directly; `make_eval_queue_batch_evaluator()` now consumes contiguous
  batch input and returns contiguous policy/value arrays without per-sample dict packing.

**Priority**: Highest — eliminates the single largest per-batch overhead with minimal architectural risk.

**Problem**: The C++ evaluator callback in `PyEvalQueue` (src/bindings/python_bindings.cpp:386-417) creates a separate numpy array per sample, appends them to a Python list, then the Python evaluator (python/alphazero/pipeline/orchestrator.py:298-381) converts that list back into a single tensor via `np.array(encoded_states)`. On the return path, results are packed into per-sample Python dicts and parsed back one-by-one. This is 3 full copies of the batch data per inference call, with N Python object allocations.

**Solution**:
1. In `PyEvalQueue`'s lambda (python_bindings.cpp:386-417), replace the per-sample loop with a single contiguous `py::array_t<float>` of shape `(batch_size, encoded_state_size)`:
   ```cpp
   py::array_t<float> batch_array({(py::ssize_t)inputs.size(), (py::ssize_t)encoded_state_size});
   float* dest = batch_array.mutable_data();
   for (const float* input : inputs) {
       std::copy_n(input, encoded_state_size, dest);
       dest += encoded_state_size;
   }
   py::object py_outputs = evaluator(batch_array);
   ```
2. Update `make_eval_queue_batch_evaluator()` (orchestrator.py:298-381) to accept a single numpy array instead of a list of arrays. Replace `np.array(encoded_states)` with direct `torch.from_numpy()` or `torch.as_tensor()` on the contiguous array (zero-copy if the array is already contiguous float32).
3. For the return path, change the Python evaluator to return a tuple `(policy_logits_tensor, values_tensor)` instead of a list of dicts. Update the C++ callback to parse these two contiguous arrays directly into `std::vector<EvalResult>`.

**Files to modify**:
- `src/bindings/python_bindings.cpp` — `PyEvalQueue` constructor lambda (~line 386-417), `parse_eval_queue_result` may need a batch variant
- `python/alphazero/pipeline/orchestrator.py` — `make_eval_queue_batch_evaluator()` (~line 298-381)

**Validation**: Existing tests must pass. Run `ctest --test-dir build` and `pytest`. Then do a short training run (3-5 cycles) to verify inference still produces valid policy/value outputs. Compare batch processing time before/after using the pipeline metrics logged to TensorBoard (`pipeline/inference_seconds`).

**Dependencies**: None.

---

### P2: Persistent Inference Thread
- [x] **Completed (2026-02-21)** — `run_interleaved_pipeline()` now uses a single
  persistent `eval-queue-inference-worker` thread with semaphore-based request/completion
  signaling (one `process_batch()` per scheduled inference batch), propagates worker
  failures back to the scheduler, and performs explicit worker shutdown/join before
  pipeline teardown. Integration smoke coverage now asserts all eval batches run on the
  single persistent worker thread.

**Priority**: High — eliminates thread creation overhead (50-100 thread spawns per cycle) and enables continuous inference.

**Problem**: `inference_batch_fn()` in orchestrator.py:461-467 spawns a new `threading.Thread` for every single `process_batch()` call, then joins it immediately. This is called 100 times per cycle (configurable via `inference_batches_per_cycle`). Each thread spawn involves OS thread creation, GIL negotiation, and teardown.

**Solution**:
1. Replace the per-call thread spawn with a persistent daemon thread that continuously calls `process_batch()` in a loop, controlled by an `Event` or `Condition`:
   ```python
   # In run_interleaved_pipeline(), before the schedule loop:
   inference_stop_event = threading.Event()
   inference_batches_completed = threading.Semaphore(0)

   def inference_worker():
       while not inference_stop_event.is_set():
           model.eval()
           eval_queue.process_batch()  # already releases GIL internally
           inference_batches_completed.release()

   inference_thread = threading.Thread(target=inference_worker, daemon=True)
   inference_thread.start()
   ```
2. Update `inference_batch_fn()` to just wait on the semaphore:
   ```python
   def inference_batch_fn() -> None:
       inference_batches_completed.acquire()
   ```
3. Signal `inference_stop_event` during shutdown, before `eval_queue.stop()`.

**Files to modify**:
- `python/alphazero/pipeline/orchestrator.py` — `run_interleaved_pipeline()` (~line 404-591), specifically `inference_batch_fn()` at line 461

**Validation**: All existing tests must pass. Verify that shutdown is still clean (no hanging threads). Run a short training session and confirm `pipeline/inference_seconds` decreases.

**Dependencies**: None. Compatible with all other items.

---

### P3: Decouple Inference and Training (Pipeline Parallelism)
- [ ] **Not started**

**Priority**: High — the single largest architectural improvement. Allows GPU to do inference and training simultaneously.

**Problem**: `run_interleaved_schedule()` (orchestrator.py:206-278) is strictly serial: it runs N inference batches, then M training steps, then repeats. During inference, no training happens. During training, no inference happens (self-play workers block). The GPU alternates between tiny forward passes and one large forward+backward pass with idle gaps between each.

**Solution**:
1. Run training in a persistent background thread that pulls from the replay buffer independently of the inference loop:
   ```python
   training_thread = threading.Thread(target=training_worker, daemon=True)
   ```
   The training worker continuously: checks buffer size → samples batch → calls `train_one_step()` → repeats. It uses its own copy of the model for forward/backward, or acquires a lock for the shared model.
2. The main thread runs a continuous inference loop (builds on P2): keeps calling `process_batch()` to serve self-play workers.
3. **Model synchronization**: Training updates the model weights. Inference needs the latest weights. Two approaches (pick one):
   - **Option A (simpler)**: Use the same model object. PyTorch operations are thread-safe at the tensor level. Training calls `model.train()` and inference calls `model.eval()` — protect mode switches with a lock, or use `torch.no_grad()` context without mode switching (BN running stats are not critical for self-play eval quality).
   - **Option B (cleaner)**: Maintain a separate `inference_model` that periodically copies weights from the training model (e.g., every N training steps). This eliminates all contention.
4. Replace `run_interleaved_schedule()` with a new `run_parallel_pipeline()` that starts both threads and waits for `max_steps` to be reached.

**Files to modify**:
- `python/alphazero/pipeline/orchestrator.py` — new function `run_parallel_pipeline()`, modify `run_interleaved_pipeline()`
- `scripts/train.py` — update to use the new pipeline function

**Validation**: All existing tests. Run a training session and verify that both `pipeline/inference_seconds` and `throughput/train_steps_per_sec` improve. GPU utilization should increase to 20-40%+.

**Dependencies**: Recommended to do P2 first (persistent inference thread), as P3 builds on the same pattern. P1 is independent but compounds well.

---

### P4: Vectorize Training Batch Preparation
- [ ] **Not started**

**Priority**: Medium — removes a Python-level bottleneck that blocks every training step.

**Problem**: `prepare_replay_batch()` (trainer.py:207-302) iterates over each of the 4096 samples individually in Python, doing per-sample attribute access, type checking, `_as_float_tensor()` conversion, and shape validation. At batch size 4096, this is thousands of Python function calls per training step.

**Solution**:
1. Add a C++ method to `ReplayBuffer` that returns batch data as contiguous numpy arrays directly:
   ```cpp
   // In replay_buffer.h/.cpp:
   struct BatchSample {
       std::vector<float> states;      // flat (batch * encoded_state_size)
       std::vector<float> policies;    // flat (batch * action_space_size)
       std::vector<float> values;      // flat (batch) or (batch * 3) for WDL
   };
   BatchSample sample_batch(size_t batch_size, size_t encoded_state_size, size_t policy_size);
   ```
2. Expose this in Python bindings as returning a tuple of numpy arrays with the correct shapes.
3. Simplify `prepare_replay_batch()` to just reshape and move the arrays to the target device:
   ```python
   states_np, policy_np, value_np = replay_buffer.sample_batch(batch_size, encoded_state_size, policy_size)
   states = torch.from_numpy(states_np).reshape(batch_size, C, H, W).to(device)
   target_policy = torch.from_numpy(policy_np).to(device)
   target_value = torch.from_numpy(value_np).to(device)
   ```

**Files to modify**:
- `src/selfplay/replay_buffer.h` and `.cpp` — add `sample_batch()` method
- `src/bindings/python_bindings.cpp` — expose `sample_batch()` with numpy return
- `python/alphazero/training/trainer.py` — simplify `prepare_replay_batch()` (~line 207-302)

**Validation**: Existing tests plus a new unit test for `sample_batch()`. Measure `throughput/train_steps_per_sec` before/after.

**Dependencies**: None. Independent of P1-P3.

---

### P5: `torch.compile()` for Inference
- [ ] **Not started**

**Priority**: Medium — free 2-5x inference speedup with minimal code change.

**Problem**: Every inference forward pass goes through Python's eager execution, with per-op dispatch overhead. For the small batch sizes and repeated identical graph structure, this overhead is significant relative to actual compute.

**Solution**:
1. In `make_eval_queue_batch_evaluator()` (orchestrator.py:298-381), after `model.to(device)`, compile the model:
   ```python
   compiled_model = torch.compile(model, mode="reduce-overhead")
   ```
2. Use `compiled_model` for inference inside the evaluator closure. Keep the uncompiled `model` for training (compilation with changing batch sizes and backward pass can cause recompilation).
3. Guard with a try/except for environments where `torch.compile` is unavailable (older PyTorch).
4. Note: The first few inference calls will be slower due to compilation. This is acceptable for a training run.

**Files to modify**:
- `python/alphazero/pipeline/orchestrator.py` — `make_eval_queue_batch_evaluator()` (~line 298-312)

**Validation**: All existing tests. Verify compiled model produces identical outputs to uncompiled (within floating point tolerance). Measure `pipeline/inference_seconds`.

**Dependencies**: None. But compounds best with P1 (less Python overhead to compile away) and P3 (inference runs continuously).

---

### P6: Thread Pool for MCTS Simulation Workers
- [ ] **Not started**

**Priority**: Low-Medium — reduces per-move overhead in self-play games.

**Problem**: `SelfPlayGame::run_simulation_batch()` (src/selfplay/self_play_game.cpp:188-226) creates `mcts_threads` new `std::thread` objects per move, then joins them all. With `threads_per_game: 1` (current config), this is a no-op. But if `threads_per_game` is increased (e.g., for Go with longer simulations), this creates and destroys threads hundreds of times per game.

**Solution**:
1. Add a simple thread pool class (or use a pre-allocated vector of threads with a work queue) to `SelfPlayGame`.
2. Initialize the pool in the constructor with `mcts_threads` workers.
3. In `run_simulation_batch()`, submit simulation work to the pool instead of creating threads.

**Files to modify**:
- `src/selfplay/self_play_game.h` and `.cpp` — add thread pool member, modify `run_simulation_batch()`

**Validation**: Existing MCTS and self-play tests. Benchmark with `threads_per_game > 1`.

**Dependencies**: None. Only matters when `threads_per_game > 1`.

---

### P7: Pinned Memory for GPU Transfers
- [ ] **Not started**

**Priority**: Low — the DGX Spark has unified memory, so CPU↔GPU copies may already be optimized by the hardware. Measure before implementing.

**Problem**: Tensor transfers from CPU to GPU (`torch.as_tensor(..., device=device)`) go through pageable memory. Pinned memory enables faster DMA transfers.

**Solution**:
1. In `make_eval_queue_batch_evaluator()`, use `torch.from_numpy(array).pin_memory().to(device)` for the input tensor.
2. In `prepare_replay_batch()`, similarly pin memory before device transfer.
3. **Important**: Measure first on the actual DGX Spark hardware. Unified memory architectures may see no benefit or even regression from explicit pinning.

**Files to modify**:
- `python/alphazero/pipeline/orchestrator.py` — evaluator closure
- `python/alphazero/training/trainer.py` — `prepare_replay_batch()`

**Validation**: Measure transfer times with and without pinning on the target hardware.

**Dependencies**: None.

---

## Config Changes Already Applied

The following changes were made to `configs/chess_1hr.yaml` (no code changes required):

| Parameter | Old | New | Rationale |
|---|---|---|---|
| `simulations_per_move` | 800 | 400 | 2x faster game generation; acceptable quality for early/mid training |
| `concurrent_games` | 64 | 256 | 4x more games keeps eval queue full; workers mostly block on eval so CPU impact is modest |
| `threads_per_game` | 4 | 1 | With eval latency as bottleneck, extra threads per game add little value; saves CPU for more games |
| `training.batch_size` | 2048 | 4096 | Better GPU utilization per training step |
| `min_buffer_size` | 4096 | 8192 | 2x training batch ensures sample diversity |
| `inference_batches_per_cycle` | 50 | 100 | More inference per cycle = faster buffer fill |

**Expected impact of config-only changes**: ~2-4x throughput improvement (more concurrent games generating data, faster games, better GPU utilization per training step). CPU usage should rise from ~25% to ~50-70%. GPU will still be low (5-15%) until the code-level serialization bottlenecks (P1-P3) are addressed.

---

## Measurement & Validation

After each item, verify improvement using:

1. **GPU utilization**: `nvidia-smi dmon -s u -d 1` during a training run. Target: >30% after P1-P3.
2. **Training throughput**: TensorBoard metric `throughput/train_steps_per_sec`. Target: >5 steps/sec.
3. **Self-play throughput**: TensorBoard metric `games_per_hour`. Target: >100 games/hr after config changes, >500 after P1-P3.
4. **Pipeline timing**: `pipeline/inference_seconds` and `pipeline/training_seconds` per cycle.
5. **Correctness**: Loss curves should look similar (not worse) after optimization. Run the test suite after every change.
