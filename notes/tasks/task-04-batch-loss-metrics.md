# Task 04: Batch Loss Metric Extraction

**Priority**: Medium — small but easy improvement
**Expected impact**: Minor per-step reduction (4 GPU syncs → 1)
**Dependencies**: None
**Difficulty**: Low

## Background

Read `specs/*` for the full AlphaZero specification, and `notes/gpu_optimization.md` for the bottleneck analysis motivating this change.

At the end of each training step, `train_one_step()` extracts four loss component values (total, policy, value, L2) from GPU tensors. Each extraction uses `.detach().item()`, which synchronizes the GPU — the CPU blocks until the GPU finishes all pending work and transfers the scalar. Four separate `.item()` calls create four sequential GPU→CPU sync points.

This is easy to fix: stack the four tensors and transfer them to CPU in a single operation.

## File to Modify

### `python/alphazero/training/trainer.py`

**Function**: `train_one_step()` (lines 548-621)

**Current code** (lines 609-621):
```python
    step_duration = max(time.perf_counter() - step_start_time, 1e-8)
    return TrainingStepMetrics(
        step=global_step + 1,
        loss_total=float(loss_components.total_loss.detach().item()),            # GPU SYNC 1
        loss_policy=float(loss_components.policy_loss.detach().item()),          # GPU SYNC 2
        loss_value=float(loss_components.value_loss.detach().item()),            # GPU SYNC 3
        loss_l2=float(loss_components.l2_loss.detach().item()),                  # GPU SYNC 4
        lr=lr,
        grad_global_norm=grad_global_norm,
        grad_nonzero_param_count=nonzero_grad_parameters,
        buffer_size=0,
        train_steps_per_second=1.0 / step_duration,
    )
```

## Required Change

Replace the four `.item()` calls with a single batched GPU→CPU transfer:

```python
    step_duration = max(time.perf_counter() - step_start_time, 1e-8)

    # Transfer all loss components in a single GPU→CPU sync
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
        lr=lr,
        grad_global_norm=grad_global_norm,
        grad_nonzero_param_count=nonzero_grad_parameters,
        buffer_size=0,
        train_steps_per_second=1.0 / step_duration,
    )
```

**Note**: The `loss_components` values are already scalar tensors (0-dim) on GPU. `torch.stack()` creates a 1-D tensor of 4 elements, `.cpu()` transfers all 4 in one operation, and `.tolist()` converts to Python floats.

## What NOT to Change

- The finite-check on lines 596-601 should remain as-is — those use `torch.isfinite()` which returns a boolean tensor, and the implicit `.item()` via `not` is necessary for the control flow. These are unavoidable sync points (we need the value to decide whether to raise an error).
- The `step_duration` computation (line 609) uses `time.perf_counter()` on CPU — no change needed.
- The `grad_global_norm` and `nonzero_grad_parameters` values come from `_gradient_statistics()` (Task 02) — they're already Python floats by the time they reach this code.

## Context: TrainingStepMetrics

The `TrainingStepMetrics` dataclass (defined earlier in the same file) expects float values:
```python
@dataclass(frozen=True, slots=True)
class TrainingStepMetrics:
    step: int
    loss_total: float
    loss_policy: float
    loss_value: float
    loss_l2: float
    lr: float
    grad_global_norm: float
    grad_nonzero_param_count: int
    buffer_size: int
    train_steps_per_second: float
```

The `.tolist()` method returns Python floats, which satisfies these type requirements.

## Verification

1. Run training for 100+ steps and confirm loss values in the log output look normal (same magnitude, decreasing over time).

2. Spot-check: The loss values should be numerically identical to the old implementation — `torch.stack().cpu().tolist()` is mathematically equivalent to calling `.item()` on each individually.

3. If there are existing tests for `train_one_step`, ensure they still pass:
   ```bash
   grep -r "train_one_step" tests/
   ```
