# Training Performance Bottleneck Analysis — 2026-02-24

## Observed Performance

| Metric | Value | Source |
|---|---|---|
| `train_one_step()` time | **2.38s** | `1/0.42 = 2.38s` (reported throughput) |
| Wall clock per step | **5.16s** | tqdm measured |
| **Overhead per step** | **~2.78s (54%)** | Difference |

The GPU is idle for over half the wall-clock time between training steps.
The nvtop GPU utilization graph shows a clear sawtooth pattern (alternating ~96% → ~0%),
confirming significant GPU idle periods between bursts of work.

Model: 20-block ResNet SE, 256 filters. ~215 parameter tensors with gradients.
Config: batch_size=8192 training, 384 concurrent games, 200 sims/move.

## Bottleneck #1 (HIGH): `_gradient_statistics()` runs every step needlessly

`trainer.py:530-559` — Called inside `train_one_step()` on **every** step, but results
are only logged every 50 steps (`log_interval: 50`). It:

1. Iterates ~215 parameter tensors in a Python for-loop
2. Launches **~645 small GPU kernels** (3 per tensor: `isfinite`, `vector_norm`, `count_nonzero`)
3. Calls `.item()` twice — each forces a full CUDA pipeline flush/stall

The `count_nonzero` per parameter is particularly wasteful — purely diagnostic with no
training value.

**Impact estimate**: The 215x Python-loop overhead + 2 CUDA syncs likely accounts for
500ms-1s of the 2.38s per-step time. Removing it from non-log steps could speed up
`train_one_step()` by ~25-40%.

## Bottleneck #2 (HIGH): Replay buffer sampling is synchronous and unoverlapped

`orchestrator.py:583-588` — Each training step calls `sample_replay_batch_tensors()`
synchronously before `train_one_step()`. For chess with batch_size=8192:

- Encoded state: 119 x 8 x 8 = 7,616 floats per position
- Policy: 4,672 floats per position
- Total data: 8,192 x (7,616 + 4,672 + 3) x 4 bytes ~ **400 MB** per sample

This sampling + tensor conversion happens while the GPU sits completely idle (the previous
step's GPU work is done, and the next step hasn't started yet). Combined with GIL contention
from the inference thread, this likely accounts for a large portion of the 2.78s overhead.

**Fix**: Double-buffer replay batch sampling — prefetch the next training batch in a
background thread while the current step trains on GPU (which releases the GIL during
CUDA kernels).

## Bottleneck #3 (MODERATE): GIL contention between inference and training threads

Both threads are Python threads sharing the GIL:

- **Inference thread**: Calls Python evaluator callback ~34x/sec. Each call holds GIL for
  numpy->tensor conversion, model forward pass setup, and result extraction.
- **Training thread**: Needs GIL for replay sampling, tensor ops, gradient stats Python loops.

The inference thread interrupts the training thread's Python-heavy operations (sampling,
gradient stats), serializing work that could overlap.

## Bottleneck #4 (MODERATE): Buffer fill delay on resume

When resuming from a checkpoint, the replay buffer starts empty. With
`min_buffer_size: 16384` and ~26 positions/sec self-play generation rate, training is
blocked for roughly ~10 minutes at startup. This inflates the tqdm average.

**Fix**: Persist replay buffer contents alongside checkpoints so training can resume
immediately.

## Bottleneck #5 (LOW-MODERATE): CUDA synchronization in loss transfer

`trainer.py:626-633`:
```python
loss_values = torch.stack([...]).cpu().tolist()
```
Forces GPU->CPU synchronization every step, flushing the CUDA pipeline. Combined with the
`.item()` syncs in gradient stats, there are 3-4 GPU stall points per step.

## Recommended Fixes (by priority)

### Fix 1: Make `_gradient_statistics()` conditional on log interval
Only compute gradient norms when the results will actually be logged. Eliminates ~645 GPU
kernel launches + 2 CUDA syncs on 49 out of every 50 steps. Low risk, high impact.

### Fix 2: Double-buffer replay batch sampling
Prefetch the next training batch in a background thread while the current step trains on
GPU. Overlaps the ~2s sampling overhead with GPU compute.

### Fix 3: Save replay buffer snapshot on resume
Persist replay buffer contents alongside checkpoints so training can resume immediately
instead of waiting 10+ minutes for the buffer to refill.

### Fix 4: Batch loss transfer to avoid repeated CUDA syncs
Move all GPU->CPU transfers to a single synchronization point at the end of
`train_one_step()`.

### Fix 5: Simplify gradient statistics when computed
Replace the Python-loop implementation with `torch.nn.utils.clip_grad_norm_()` or a single
fused operation. Drop the `count_nonzero` diagnostic entirely.
