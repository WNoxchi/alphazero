# GPU Training Pipeline Optimization — Design Doc

**Date**: 2026-02-21
**Target**: DGX Spark (GB10 Blackwell, 128 GB unified memory, 20-core ARM Cortex-X925/A725)
**Codebase**: AlphaZero single-machine chess/Go training (see `specs/*` for full specification)

## Problem Statement

AlphaZero chess training on DGX Spark achieves 0.52 training steps/sec at 83% GPU utilization. Analysis shows the primary bottleneck is **GPU serialization between inference and training** caused by Python GIL contention and unnecessary GPU-CPU synchronization points, not raw GPU compute.

The system has headroom: 83% GPU utilization with 30 GB free RAM and 20 CPU cores mostly idle (self-play threads block on eval queue). This document describes five optimizations to close those gaps.

## System Profile (Baseline)

| Resource | Value |
|----------|-------|
| GPU | NVIDIA GB10 Blackwell, 83% utilization |
| GPU Memory | ~15 GB used (unified memory, shared with CPU) |
| System RAM | 91 GB / 128 GB used (67 GB by training process) |
| CPU | 20 cores, 448 threads in training process (384 self-play + workers) |
| Training throughput | 0.52 steps/sec, batch_size=4096 |
| Self-play throughput | ~1072 games/hr, avg 18-21 moves |
| Replay buffer | 60K-78K positions (750K capacity pre-allocated = 36.9 GB) |
| Model | ResNetSE 20 blocks × 256 filters, ~35M params, bf16 mixed precision |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  384 C++ self-play threads (MCTS, 1 thread per game)        │
│    ├─ Run MCTS simulations (tree traversal, state encoding) │
│    └─ submit_and_wait(encoded_state) → blocks on semaphore  │
├─────────────────────────────────────────────────────────────┤
│  EvalQueue (C++, thread-safe MPSC queue)                    │
│    ├─ Accumulates requests until batch_size=384 or timeout  │
│    └─ process_batch() → calls Python evaluator callback     │
├─────────────────────────────────────────────────────────────┤
│  Python Inference Thread         Python Training Thread      │
│    eval_queue.process_batch()     sample_replay_batch_tensors│
│    ├─ numpy → tensor (.to GPU)    ├─ C++ sample_batch()     │
│    ├─ model(inputs) [forward]     ├─ tensor.to(GPU)         │
│    ├─ .to("cpu") [SYNC, GIL]     ├─ model.train()          │
│    └─ return numpy arrays         ├─ forward + backward     │
│         ↕                         ├─ optimizer.step()        │
│    Shares GIL with training ──→   └─ .item() × 4 [SYNC]    │
├─────────────────────────────────────────────────────────────┤
│  GPU: Single default CUDA stream (all ops serialize)        │
│  Model: Shared reference, used by both threads              │
└─────────────────────────────────────────────────────────────┘
```

Key files:
- `python/alphazero/pipeline/orchestrator.py` — evaluator closure, parallel pipeline workers
- `python/alphazero/training/trainer.py` — training step, gradient stats, replay sampling
- `python/alphazero/network/resnet_se.py` — ResNetSE model architecture
- `scripts/train.py` — entry point, runtime construction
- `configs/chess_1hr.yaml` — training configuration

## Bottleneck Analysis

### 1. GIL-Mediated GPU Serialization

The inference and training workers are Python threads competing for the GIL. Critical GPU operations hold the GIL while synchronizing:

**Inference evaluator** (`orchestrator.py:338-374`):
```python
flat_states = flat_states.to(device=resolved_device)           # CPU→GPU, blocks with GIL held
policy_logits, value = model(model_inputs)                     # GPU forward, GIL held for launch
policy_cpu = policy_logits.detach().to("cpu", dtype=...).contiguous()  # GPU sync, GIL held!
```

The `.to("cpu")` call at line 373 **synchronizes the GPU and holds the GIL until data is copied back**. During this entire time — GPU forward pass execution plus GPU→CPU copy — the training thread cannot acquire the GIL.

**Training step** (`trainer.py:603-615`):
```python
scaler.scale(loss).backward()                                  # GPU backward, GIL held for launch
grad_norm, _ = _gradient_statistics(model)                     # ~100 .item() calls! (see below)
scaler.step(optimizer)                                         # GPU optimizer step
loss_total = float(loss_components.total_loss.detach().item()) # GPU sync × 4
```

**Net effect**: Inference and training can never overlap on the GPU. The GPU alternates between them with idle gaps during GIL handoffs.

### 2. Per-Parameter Gradient Synchronization

`_gradient_statistics()` (`trainer.py:530-545`) computes gradient norm by iterating over every model parameter (~100 for a 20-block ResNet), calling `.item()` on each. Each `.item()` is a GPU→CPU synchronization point. This creates **~100 GPU pipeline stalls per training step**.

### 3. Redundant Loss Metric Syncs

`train_one_step()` extracts four loss values (`trainer.py:612-615`) with four separate `.detach().item()` calls, creating four sequential GPU sync points where one would suffice.

### 4. No GPU Kernel Fusion

The model launches hundreds of small CUDA kernels per forward pass (conv, bn, relu, SE attention × 20 blocks). Without `torch.compile`, each kernel has launch overhead and inter-kernel data round-trips through GPU global memory.

### 5. Self-Play Throughput vs Quality Tradeoff

At 400 simulations/move, each game takes ~400 inference batches worth of evaluations. For early training where the network plays randomly (18-move games), this is more computation per position than needed. Halving simulations doubles data generation rate.

## Optimizations

### Optimization 1: `torch.compile` — GPU Kernel Fusion

Apply `torch.compile(model, mode='reduce-overhead')` to the shared model. The inductor backend fuses conv+bn+relu sequences, SE block operations, and head computations into fewer, larger CUDA kernels. The ResNetSE model is an ideal candidate: purely static control flow, no custom autograd, standard nn.Module ops only.

**Expected impact**: 20-40% faster forward/backward passes.

### Optimization 2: Batched Gradient Statistics

Replace the per-parameter `.item()` loop in `_gradient_statistics()` with a single GPU-side norm computation using `torch.linalg.vector_norm` on stacked per-parameter norms. Reduces ~100 GPU sync points to 1.

**Expected impact**: Eliminates several milliseconds of GPU stall per training step.

### Optimization 3: Inference CUDA Stream + Non-Blocking Transfers

Create a dedicated `torch.cuda.Stream()` for inference. Use `non_blocking=True` for CPU↔GPU transfers. Call `stream.synchronize()` (which releases the GIL while waiting) instead of implicit synchronization via `.to("cpu")`.

This allows the training thread to acquire the GIL and launch GPU work on the default stream while the inference thread waits for its stream to complete. The GPU can potentially execute kernels from both streams concurrently.

**Expected impact**: 10-30% throughput improvement from reduced GIL contention and potential GPU kernel overlap.

### Optimization 4: Batched Loss Metric Extraction

Stack the four loss component tensors and transfer to CPU in a single operation instead of four separate `.item()` calls.

**Expected impact**: Minor (4 syncs → 1), but contributes to overall sync reduction.

### Optimization 5: Config Tuning

- `simulations_per_move: 400 → 200` — doubles self-play data generation rate
- `training.batch_size: 4096 → 8192` — better GPU arithmetic intensity per training step

**Expected impact**: ~2x self-play throughput, ~15-25% better GPU utilization during training steps.

## Projected Results

| Metric | Baseline | Target |
|--------|----------|--------|
| Training steps/sec | 0.52 | 0.8-1.2 |
| GPU utilization | 83% | 90-95% |
| Self-play games/hr | 1072 | 2000+ |
| Positions generated/hr | ~19K | ~40K+ |

## Implementation

See `notes/tasks/task-{01..05}-*.md` for self-contained implementation tasks. All five tasks are independent and can be executed in parallel. Each task file contains the full context needed for implementation: file paths, line numbers, current code, expected changes, and verification criteria.
