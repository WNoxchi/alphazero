# Task 02: Batch Gradient Statistics Computation

**Priority**: High — eliminates ~100 GPU sync points per training step
**Expected impact**: Several milliseconds of GPU stall eliminated per step
**Dependencies**: None
**Difficulty**: Low-Medium

## Background

Read `specs/*` for the full AlphaZero specification, and `notes/gpu_optimization.md` for the bottleneck analysis motivating this change.

The function `_gradient_statistics()` in `trainer.py` computes the global gradient norm and counts non-zero gradient parameters after each training step. It does this by iterating over every model parameter (~100 for a 20-block ResNetSE), calling `.item()` on each parameter's squared norm. Each `.item()` forces a GPU→CPU synchronization — the CPU stalls until the GPU completes all pending work and transfers the scalar value.

This creates **~100 GPU pipeline stalls per training step**, which adds up to significant overhead when training steps take ~1.9 seconds total.

## File to Modify

### `python/alphazero/training/trainer.py`

**Function**: `_gradient_statistics()` (lines 530-545)

**Current code**:
```python
def _gradient_statistics(model: nn.Module) -> tuple[float, int]:
    squared_norm_sum = 0.0
    nonzero_grad_parameters = 0
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Encountered non-finite gradients during training")
        gradient_float = gradient.detach().to(dtype=torch.float32)
        squared_norm_sum += float((gradient_float * gradient_float).sum().item())  # GPU SYNC
        if bool(torch.count_nonzero(gradient_float)):  # GPU SYNC
            nonzero_grad_parameters += 1
    return math.sqrt(squared_norm_sum), nonzero_grad_parameters
```

**Problems**:
1. `.item()` on line 541 — GPU→CPU sync for each parameter's squared norm
2. `torch.count_nonzero(...).item()` implicit in `bool()` on line 542 — another sync per parameter
3. `torch.isfinite(gradient).all()` on line 537 — yet another sync per parameter (bool conversion)

Total: ~300 GPU sync points for ~100 parameters (3 per parameter).

## Required Change

Replace with a batched computation that collects all gradients first, then computes statistics in bulk with minimal sync points.

**New implementation**:
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
    if not bool(finite_checks.all()):  # Single sync for finite check
        raise FloatingPointError("Encountered non-finite gradients during training")

    # Compute per-parameter L2 norms on GPU (no sync yet)
    per_param_norms = torch.stack([
        torch.linalg.vector_norm(g, 2.0, dtype=torch.float32) for g in grads
    ])

    # Compute global norm — single .item() sync
    global_norm = float(torch.linalg.vector_norm(per_param_norms, 2.0).item())

    # Count non-zero gradient parameters — single .item() sync
    nonzero_flags = torch.stack([
        (torch.count_nonzero(g) > 0) for g in grads
    ])
    nonzero_count = int(nonzero_flags.sum().item())

    return global_norm, nonzero_count
```

This reduces sync points from ~300 to **3** (one for finite check, one for norm, one for nonzero count). The `torch.stack` + `torch.linalg.vector_norm` pattern is the same used internally by `torch.nn.utils.clip_grad_norm_`.

**Note**: The `for g in grads` loops still execute on CPU to build the list, but the actual computation (norm, count) happens in bulk on GPU.

## Caller Context

`_gradient_statistics` is called from `train_one_step()` at line 605:
```python
scaler.unscale_(optimizer)
grad_global_norm, nonzero_grad_parameters = _gradient_statistics(model)  # line 605
scaler.step(optimizer)
```

The return values feed into `TrainingStepMetrics`:
```python
grad_global_norm=grad_global_norm,           # line 618
grad_nonzero_param_count=nonzero_grad_parameters,  # line 619
```

These are logged to TensorBoard for monitoring. The values must be numerically equivalent (or very close) to the original implementation.

## Verification

1. **Numerical correctness**: On the same training checkpoint and batch, compare the gradient norm values between old and new implementations. They should be identical or differ only by floating-point noise (< 0.01% relative error). The `torch.linalg.vector_norm` computation is mathematically equivalent to `sqrt(sum(g^2))`.

2. **Performance**: Time the `_gradient_statistics` function before and after. Expected improvement: from several milliseconds (with ~100 sync points) to sub-millisecond (with 3 sync points).

3. **Existing tests**: Check if there are tests for `_gradient_statistics` in the test suite:
   ```bash
   grep -r "_gradient_statistics\|gradient_statistics" tests/
   ```
   If tests exist, they should still pass. If not, consider adding a simple test that compares old vs new on a small model.

4. **Training run**: Run 100+ training steps and confirm:
   - No `FloatingPointError` exceptions (finite check still works)
   - Gradient norm values in logs look reasonable
   - Training loss continues to decrease normally
