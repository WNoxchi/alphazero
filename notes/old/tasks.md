# GPU Training Pipeline Optimization — Task List

**Date**: 2026-02-21
**Design doc**: `notes/gpu_optimization.md`
**Baseline**: 0.52 train steps/sec, 83% GPU util, ~1072 games/hr

All five tasks are independent and can be executed in parallel.

---

## Task 01: `torch.compile` — GPU Kernel Fusion

**Priority**: Highest | **Difficulty**: Low | **Expected impact**: 20-40% faster forward/backward

The ResNetSE model (20 blocks, 256 filters, ~35M params) launches hundreds of small CUDA kernels per forward pass. `torch.compile(model, mode='reduce-overhead')` fuses them into fewer, larger kernels.

### File: `scripts/train.py`

**Function**: `build_training_runtime()` (line 397)

After `model = _build_model(...)` (line 413) and before checkpoint loading (line 430), add:

```python
system = _section(config, "system")
compile_model = system.get("compile", True)
if compile_model:
    import torch
    model = torch.compile(model, mode="reduce-overhead")
```

### Config: `configs/chess_1hr.yaml`

Optionally add `compile: true` under `system:`.

### Constraints

- Do not use `fullgraph=True` or `dynamic=True`
- Only compile the model itself, not evaluator closures or loss functions
- First 2 steps will be slow (~5-30s each) due to compilation; subsequent steps are faster

### Verification

- Run 100+ steps, confirm no `torch._dynamo` errors
- Throughput: baseline ~0.52 → expected ~0.65-0.75 steps/sec
- Loss values should be comparable to non-compiled runs (minor FP noise acceptable)
- Disable with `system.compile: false` in YAML

---

## Task 02: Batch Gradient Statistics Computation

**Priority**: High | **Difficulty**: Low-Medium | **Expected impact**: Eliminates ~300 GPU sync points per step

`_gradient_statistics()` iterates over ~100 parameters, calling `.item()` 3 times each (finite check, squared norm, nonzero count) = ~300 GPU→CPU sync stalls per step.

### File: `python/alphazero/training/trainer.py`

**Function**: `_gradient_statistics()` (lines 530-545)

Replace with batched GPU-side computation:

```python
def _gradient_statistics(model: nn.Module) -> tuple[float, int]:
    grads = []
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is not None:
            grads.append(gradient.detach())

    if not grads:
        return 0.0, 0

    # Check all-finite in one GPU operation
    finite_checks = torch.stack([torch.isfinite(g).all() for g in grads])
    if not bool(finite_checks.all()):  # Single sync
        raise FloatingPointError("Encountered non-finite gradients during training")

    # Compute per-parameter L2 norms on GPU, then global norm — single .item()
    per_param_norms = torch.stack([
        torch.linalg.vector_norm(g, 2.0, dtype=torch.float32) for g in grads
    ])
    global_norm = float(torch.linalg.vector_norm(per_param_norms, 2.0).item())

    # Count non-zero gradient parameters — single .item()
    nonzero_flags = torch.stack([(torch.count_nonzero(g) > 0) for g in grads])
    nonzero_count = int(nonzero_flags.sum().item())

    return global_norm, nonzero_count
```

Reduces sync points from ~300 to 3. Pattern matches `torch.nn.utils.clip_grad_norm_`.

### Verification

- Gradient norm values must match old implementation (< 0.01% relative error)
- Run `grep -r "_gradient_statistics" tests/` for existing tests
- Run 100+ training steps, confirm no `FloatingPointError` regressions

---

## Task 03: Inference CUDA Stream + Non-Blocking Transfers

**Priority**: High | **Difficulty**: Medium | **Expected impact**: 10-30% throughput improvement

Inference and training Python threads share the GIL. The inference evaluator's `.to("cpu")` synchronizes the GPU while holding the GIL, blocking training. Fix: use a dedicated CUDA stream with `non_blocking=True` transfers and `stream.synchronize()` (which releases the GIL).

### File: `python/alphazero/pipeline/orchestrator.py`

**Function**: `make_eval_queue_batch_evaluator()` (lines 298-377)

#### Change 1: Create inference stream (after line 311)

```python
resolved_device = _resolve_torch_device(device)
model = model.to(device=resolved_device)

inference_stream = (
    torch.cuda.Stream(device=resolved_device)
    if resolved_device.type == "cuda"
    else None
)
```

#### Change 2: Wrap evaluator in stream context

```python
stream_ctx = torch.cuda.stream(inference_stream) if inference_stream else nullcontext()
with stream_ctx:
    model.eval()
    flat_states = flat_states.to(device=resolved_device, non_blocking=True)
    # ... forward pass unchanged ...
    policy_cpu = policy_logits.detach().to(device="cpu", dtype=torch.float32, non_blocking=True)
    value_cpu = value_scalars.detach().to(device="cpu", dtype=torch.float32, non_blocking=True)

if inference_stream:
    inference_stream.synchronize()  # Releases GIL while waiting

return policy_cpu.contiguous().numpy(), value_cpu.contiguous().numpy()
```

### BatchNorm Safety

Inference sets `model.eval()` (read-only BN stats), training sets `model.train()` (updates stats). Mode switches are GIL-serialized. Slight staleness in BN running stats during concurrent execution is acceptable — same design as intentional weight staleness.

### Constraints

- Do not modify `train_one_step()` or the training worker
- Do not remove `torch.no_grad()` context
- Keep value extraction logic (WDL vs scalar) unchanged
- CPU-only fallback: `inference_stream=None`, code path unchanged

### Verification

- Run 500+ steps, confirm no deadlocks
- GPU util: baseline ~83% → expected 90%+
- Throughput: baseline ~0.52 → expected ~0.6-0.7 steps/sec

---

## Task 04: Batch Loss Metric Extraction

**Priority**: Medium | **Difficulty**: Low | **Expected impact**: Minor (4 GPU syncs → 1)

`train_one_step()` extracts four loss values with four `.detach().item()` calls = four sequential GPU sync points.

### File: `python/alphazero/training/trainer.py`

**Function**: `train_one_step()` (lines 609-621)

Replace:

```python
loss_values = torch.stack([
    loss_components.total_loss.detach(),
    loss_components.policy_loss.detach(),
    loss_components.value_loss.detach(),
    loss_components.l2_loss.detach(),
]).cpu().tolist()  # Single sync point

return TrainingStepMetrics(
    step=global_step + 1,
    loss_total=loss_values[0],
    loss_policy=loss_values[1],
    loss_value=loss_values[2],
    loss_l2=loss_values[3],
    # ... remaining fields unchanged ...
)
```

### Constraints

- Do not change the finite-check on lines 596-601 (unavoidable sync for control flow)
- `.tolist()` returns Python floats, satisfying `TrainingStepMetrics` type requirements

### Verification

- Loss values must be numerically identical to old implementation
- Run `grep -r "train_one_step" tests/` for existing tests

---

## Task 05: Config Tuning for Maximum Throughput

**Priority**: Medium | **Difficulty**: Low (config only, no code) | **Expected impact**: ~2x self-play throughput

### File: `configs/chess_1hr.yaml`

| Parameter | Current | New | Rationale |
|-----------|---------|-----|-----------|
| `mcts.simulations_per_move` | 400 | 200 | 2x faster self-play; network too weak at step ~10K for deep search to help |
| `training.batch_size` | 4096 | 8192 | Better GPU arithmetic intensity; 2x data per step |
| `training.min_buffer_size` | 8192 | 16384 | 2x training batch for sample diversity |

### Constraints

- Do not change `concurrent_games` (384), `threads_per_game` (1), or `mcts.batch_size` (384)
- Do not change `replay_buffer.capacity` (750000)

### Staged approach for long runs (100K+ steps)

- **Early (0-50K)**: 200 sims, batch 8192
- **Mid (50K-200K)**: 400 sims, batch 4096
- **Late (200K+)**: 800 sims, batch 4096

### Verification

- Self-play throughput: baseline ~1072 → expected ~2000+ games/hr
- Training steps/sec will decrease (~0.35-0.45) but net positions/sec increases
- Loss curve should match baseline when plotted against total positions trained (not steps)
- Buffer fills to `min_buffer_size: 16384` in ~30 seconds

---

## Projected Results (All Optimizations Combined)

| Metric | Baseline | Target |
|--------|----------|--------|
| Training steps/sec | 0.52 | 0.8-1.2 |
| GPU utilization | 83% | 90-95% |
| Self-play games/hr | 1072 | 2000+ |
| Positions generated/hr | ~19K | ~40K+ |
