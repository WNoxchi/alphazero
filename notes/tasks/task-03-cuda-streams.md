# Task 03: Inference CUDA Stream + Non-Blocking Transfers

**Priority**: High — enables GPU overlap between inference and training
**Expected impact**: 10-30% throughput improvement
**Dependencies**: None (pairs well with Task 01 but independent)
**Difficulty**: Medium

## Background

Read `specs/*` for the full AlphaZero specification, and `notes/gpu_optimization.md` for the bottleneck analysis motivating this change.

The training pipeline has two Python threads sharing one model:
- **Inference thread**: continuously calls `eval_queue.process_batch()`, which invokes a Python evaluator callback
- **Training thread**: continuously samples from replay buffer and calls `train_one_step()`

Both threads compete for the Python GIL. Currently, the inference evaluator uses **blocking GPU transfers** that hold the GIL while waiting for the GPU to complete:

1. `flat_states.to(device=resolved_device)` — CPU→GPU copy, blocks with GIL held
2. `model(model_inputs)` — GPU forward pass, GIL held during kernel launch
3. `policy_logits.detach().to(device="cpu", ...).contiguous()` — **GPU→CPU sync, GIL held until GPU completes entire forward pass and copies data back**

Step 3 is the worst offender: it synchronizes the GPU (waiting for the forward pass to finish) while holding the GIL, preventing the training thread from doing any work.

**The fix**: Use a dedicated CUDA stream for inference with `non_blocking=True` transfers, and call `stream.synchronize()` which **releases the GIL while waiting**. This allows the training thread to acquire the GIL and launch its own GPU work while the inference thread waits for its stream to complete.

## File to Modify

### `python/alphazero/pipeline/orchestrator.py`

**Function**: `make_eval_queue_batch_evaluator()` (lines 298-377)

**Current evaluator closure** (lines 314-375):
```python
def evaluator(encoded_states: Any) -> tuple[Any, Any]:
    import numpy as np

    encoded_states_array = np.asarray(encoded_states, dtype=np.float32)
    # ... shape validation ...
    if not encoded_states_array.flags.c_contiguous:
        encoded_states_array = np.ascontiguousarray(encoded_states_array)

    batch_size = int(encoded_states_array.shape[0])
    if batch_size == 0:
        return (np.empty((0, game_config.action_space_size), ...), np.empty((0,), ...))

    flat_states = torch.from_numpy(encoded_states_array)
    if resolved_device.type != "cpu":
        flat_states = flat_states.to(device=resolved_device)          # ← BLOCKING
    # Keep forward pass mode-agnostic so shared-model parallel workers avoid
    # train/eval mode thrashing while still using no-grad inference.
    model_inputs = flat_states.reshape(batch_size, game_config.input_channels, rows, cols)

    with torch.no_grad():
        with torch.autocast(
            device_type=resolved_device.type,
            dtype=torch.bfloat16,
            enabled=use_mixed_precision,
        ):
            policy_logits, value = model(model_inputs)                # ← GPU FORWARD

    # ... value extraction (WDL or scalar) ...
    value_scalars = value[:, 0] - value[:, 2]  # WDL case

    policy_cpu = policy_logits.detach().to(device="cpu", dtype=torch.float32).contiguous()  # ← SYNC!
    value_cpu = value_scalars.detach().to(device="cpu", dtype=torch.float32).contiguous()    # ← SYNC!
    return policy_cpu.numpy(), value_cpu.numpy()
```

## Required Changes

### 1. Create a CUDA stream in the outer function

At the top of `make_eval_queue_batch_evaluator()`, after `resolved_device` is set (line 311), create the inference stream:

```python
resolved_device = _resolve_torch_device(device)
model = model.to(device=resolved_device)

# Dedicated CUDA stream for inference — allows overlap with training on default stream
inference_stream = (
    torch.cuda.Stream(device=resolved_device)
    if resolved_device.type == "cuda"
    else None
)
```

### 2. Wrap evaluator GPU ops in stream context with non-blocking transfers

Replace the evaluator body with:

```python
def evaluator(encoded_states: Any) -> tuple[Any, Any]:
    import numpy as np
    from contextlib import nullcontext

    encoded_states_array = np.asarray(encoded_states, dtype=np.float32)
    # ... existing shape validation (unchanged) ...
    if not encoded_states_array.flags.c_contiguous:
        encoded_states_array = np.ascontiguousarray(encoded_states_array)

    batch_size = int(encoded_states_array.shape[0])
    if batch_size == 0:
        return (
            np.empty((0, game_config.action_space_size), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    stream_ctx = torch.cuda.stream(inference_stream) if inference_stream else nullcontext()
    with stream_ctx:
        model.eval()  # Use eval mode for inference (no BN stat updates)

        flat_states = torch.from_numpy(encoded_states_array)
        if resolved_device.type != "cpu":
            flat_states = flat_states.to(device=resolved_device, non_blocking=True)
        model_inputs = flat_states.reshape(
            batch_size, game_config.input_channels, rows, cols
        )

        with torch.no_grad():
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.bfloat16,
                enabled=use_mixed_precision,
            ):
                policy_logits, value = model(model_inputs)

        # ... value extraction (same as current, unchanged) ...

        policy_cpu = policy_logits.detach().to(
            device="cpu", dtype=torch.float32, non_blocking=True
        )
        value_cpu = value_scalars.detach().to(
            device="cpu", dtype=torch.float32, non_blocking=True
        )

    # Synchronize outside the stream context — releases GIL while waiting!
    if inference_stream:
        inference_stream.synchronize()

    return policy_cpu.contiguous().numpy(), value_cpu.contiguous().numpy()
```

### 3. Ensure training worker sets model.train()

The training worker already calls `model.train()` at orchestrator.py line 545 before `train_one_step()`. This is correct and needs no change. The mode switching is safe because:
- `model.eval()` (inference) and `model.train()` (training) are Python calls serialized by the GIL
- The CUDA stream context ensures GPU kernels launched by inference use inference-mode BN behavior
- The default stream (training) uses train-mode BN behavior

### Key: Why this helps

Before (serialized):
```
[Inference: GIL held, GPU forward + sync] → [Training: GIL held, GPU fwd+bwd] → repeat
```

After (overlapped):
```
[Inference: GIL held, launch async ops] → [release GIL, wait on stream] ←──┐
                                            [Training: GIL held, GPU fwd+bwd] │
                                            [GPU: both streams may overlap]    │
```

During `inference_stream.synchronize()`, the GIL is released. The training thread can acquire the GIL and start launching work on the default CUDA stream. If the GPU supports concurrent kernel execution (Blackwell does), kernels from both streams can execute simultaneously.

## BatchNorm Safety

With concurrent CUDA streams, there's a potential data race on BatchNorm `running_mean`/`running_var` tensors:
- Training mode: BN updates these tensors during forward pass
- Eval mode: BN reads these tensors (no writes)

By setting `model.eval()` in the inference path, inference BN uses fixed running stats (read-only). Training sets `model.train()` which updates running stats. Since the mode switches are GIL-serialized, and CUDA kernels on each stream see the mode that was set before their launch, there is no GPU-level race on the mode flag. However, training's BN updates to `running_mean`/`running_var` could overlap with inference's reads of the same tensors across streams.

**This is acceptable**: BN running stats are exponential moving averages used for normalization. A slightly stale read during concurrent execution produces a negligible difference. The AlphaZero pipeline already intentionally uses slightly stale weights (training updates are immediately visible to inference by design).

## What NOT to Change

- Do not modify `train_one_step()` or the training worker — they should continue using the default CUDA stream
- Do not add stream synchronization between training and inference beyond what the GIL provides — the slight staleness is intentional
- Do not remove the existing `torch.no_grad()` context — it's still needed for memory efficiency
- Keep the value extraction logic (WDL vs scalar, lines 359-371) unchanged

## Verification

1. **Correctness**: Run training for 100+ steps. Inference results should be numerically close to baseline (minor differences from BN eval mode vs train mode are expected and acceptable).

2. **GPU utilization**: Monitor with `nvidia-smi` (there's a `watch nvidia-smi` in tmux pane 1). Expect utilization to increase from ~83% toward 90%+.

3. **Throughput**: Check training logs for `Throughput: X train steps/s`. Expect improvement from ~0.52 to ~0.6-0.7 steps/sec (combined with other optimizations, higher).

4. **No deadlocks**: The stream synchronization must not deadlock with the eval queue's condition variable. Since `stream.synchronize()` releases the GIL, and the C++ eval queue uses its own mutex (not the GIL), there should be no interaction. Verify by running for 500+ steps without hangs.

5. **CPU-only fallback**: If `resolved_device.type == "cpu"`, the `inference_stream` is `None` and the code path is unchanged (no CUDA stream, no non_blocking). Verify CPU mode still works if you have CPU tests.

## Testing the Change in Isolation

To test just this change without other optimizations:
```bash
# Run with the current config
python scripts/train.py --config configs/chess_1hr.yaml

# Monitor in another terminal
watch -n 0.5 nvidia-smi
```

Compare throughput numbers in the training log output against the baseline of 0.52 steps/sec.
