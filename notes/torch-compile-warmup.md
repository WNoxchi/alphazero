# torch.compile Warmup Pre-compilation

**Date**: 2026-02-22
**Status**: Not implemented — design note for future work

## Problem

`torch.compile` compiles lazily on the first forward pass for each distinct input shape. During the AlphaZero pipeline, inference (batch 384) and training (batch 8192, forward + backward) trigger compilation at different times. The training compilation holds the GIL for minutes, blocking the inference thread and stalling the entire pipeline.

With `mode='reduce-overhead'` (CUDAGraphs), each shape also requires graph capture, making this worse. We switched to `mode='default'` to avoid CUDAGraph overhead, but the lazy compilation still causes a latency spike on the first training step.

## Proposed Solution

Add a warmup phase in `build_training_runtime()` (or at the start of `run_pipeline()`) that triggers compilation for both shapes before the pipeline threads start:

```python
# After torch.compile(model, mode="default"):
def _warmup_compiled_model(model, game_config, training_batch_size, eval_batch_size, device):
    """Pre-compile forward/backward for both inference and training shapes."""
    import torch
    rows, cols = game_config.board_shape
    channels = game_config.input_channels

    # 1. Inference shape (forward only, eval mode)
    model.eval()
    dummy_inf = torch.zeros(eval_batch_size, channels, rows, cols, device=device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            model(dummy_inf)

    # 2. Training shape (forward + backward, train mode)
    model.train()
    dummy_train = torch.zeros(training_batch_size, channels, rows, cols, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        policy, value = model(dummy_train)
        loss = policy.sum() + value.sum()  # dummy loss to trigger backward compilation
        loss.backward()
    model.zero_grad()
```

This runs sequentially during startup, so there's no GIL contention. Both graphs are compiled before any threads start.

## Key details

- Place after `torch.compile()` and before `make_eval_queue_batch_evaluator()` in `scripts/train.py`
- The dummy backward pass compiles the autograd graph; the actual loss function doesn't matter
- Call `model.zero_grad()` after to clear dummy gradients before real training
- With `mode='reduce-overhead'`, this also captures CUDAGraphs upfront (if we ever switch back)
- Eval batch size = `eval_queue_config.batch_size` (384), training batch size = `training_config.batch_size` (8192)

## Relevant files

- `scripts/train.py` — `build_training_runtime()`, lines ~440-460
- `python/alphazero/pipeline/orchestrator.py` — `make_eval_queue_batch_evaluator()`, `run_pipeline()`
- `notes/gpu_optimization.md` — overall optimization context
