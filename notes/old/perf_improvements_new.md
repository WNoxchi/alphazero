# Performance Improvements — Next Steps

**Date:** 2026-02-23
**Baseline:** 0.42 train steps/s, 96% GPU utilization, 104/121 GiB RAM used

## Current Bottleneck Profile

| Metric | Value |
|---|---|
| Train steps/s | 0.42 |
| GPU utilization | 96% |
| RAM used | 104 / 121 GiB |
| Model | 20-block ResNet-SE, 256 filters (~25M params) |
| Training batch | 8192 |
| Inference batch | 384 |
| Precision | BF16 |
| torch.compile | `mode="default"` (reduce-overhead hangs on GB10) |

GPU is saturated at 96%. Memory is tight at 104/121 GiB. Gains must come from doing less work per step, doing it more efficiently, or architectural changes.

---

## Tier 1: High Impact, Moderate Effort

### 1. Shrink the Network (10 blocks instead of 20)

Single biggest lever. The 20-block, 256-filter model is ~25M params. A 10-block model is ~12.5M params and halves forward+backward FLOPs.

**Why this is viable:**
- LC0 research shows that for early/mid training, smaller networks train faster and reach the same Elo earlier. LC0's strongest nets use 20+ blocks, but they trained for months on hundreds of GPUs.
- On a single GB10, diminishing returns on network capacity hit long before 20 blocks × 256 filters are exhausted. A 10b×256f model is functionally equivalent for the first ~100k+ training steps of game quality.
- AlphaZero's 2017 paper used 20 blocks with 256 filters — on 5,000 TPUs. One GPU requires a different efficiency tradeoff.

**Recommended approach — progressive widening:**
1. Start with **10 blocks, 256 filters** (~12.5M params) for rapid iteration
2. When Elo plateaus (visible in TensorBoard), distill into **20 blocks, 256 filters**
3. Gets ~2x train steps/s immediately; smaller model generates higher-quality self-play data sooner

**Expected impact:** 0.42 → ~0.8 train steps/s

### 2. FP8 Inference (Blackwell Native)

GB10 has native FP8 Tensor Cores. Inference is ~85% of GPU time. FP8 inference would nearly double inference throughput.

**Implementation paths:**
- **`torch.float8_e4m3fn`** via PyTorch's native FP8 support (torch 2.10 has this)
- **TensorRT with FP8 quantization** — export ONNX, build TRT engine
- TensorRT is more work but gives kernel fusion + FP8 combined

**Expected impact on inference:** ~1.5–2x throughput → more self-play data per unit time → can increase S:T ratio or reduce batch padding waste

### 3. torch.compile Warmup (Already Designed, Not Implemented)

`torch-compile-warmup.md` describes the solution. The first compilation stalls the pipeline for minutes. Pre-compiling both shapes (inference batch=384, training batch=8192) during startup eliminates this.

**Impact:** Eliminates startup stall + prevents mid-run recompilation jitter. Not a steady-state throughput gain, but reduces wasted wall-clock time significantly on shorter runs (like the 40k-step chess config).

---

## Tier 2: Medium Impact, Lower Effort

### 4. Move `prepare_replay_batch` to C++

Already identified in `perf_improvements.md` but not implemented. The Python loop over 8192 `ReplayPosition` objects is the largest remaining Python overhead in the training hot path. A C++ `sample_batch()` returning packed tensors eliminates this.

**Expected impact:** ~5-15% train step speedup (depends on current Python overhead fraction)

### 5. Game-Specific MCTSNode Sizing

Also from `perf_improvements.md`: nodes are sized for Go's 362 max actions even when playing chess (218). With 384 concurrent games × 8192-node arenas, this wastes ~2-3 GB and hurts CPU cache efficiency.

**Expected impact:** Better MCTS throughput → more positions/hr → training maintains pace without starving

### 6. Reduce Simulations Further in Early Training

Currently at 200 sims/move (down from 800 in the spec). In early training when the model is weak, even 200 sims produces noisy policies. Consider:
- **100 sims/move for steps 0–5000** (model is near-random anyway)
- **200 sims/move for steps 5000+**
- Doubles self-play throughput in the critical buffer-filling phase

### 7. Dedicated Inference CUDA Stream

From `gpu_optimization.md`: inference and training currently compete on the default stream. A dedicated inference stream with `non_blocking=True` transfers and explicit `stream.synchronize()` would overlap inference compute with training's CPU-side work.

**Expected impact:** 10-30% less GIL contention between inference and training workers

---

## Tier 3: Architectural Exploration

### 8. Transformer Architecture?

**Recommendation: Not yet.** Rationale:

| Factor | ResNet-SE | Transformer |
|---|---|---|
| FLOPs per position | Lower (convolutions) | Higher (attention is O(n²)) |
| Memory bandwidth | Conv is compute-bound | Attention is memory-bound |
| GB10 bandwidth | 273 GB/s (our bottleneck) | Would be worse |
| Implementation maturity | Battle-tested (LC0, AZ) | Experimental for chess |
| Training stability | Very stable with BN | Needs careful LR warmup |

On memory-bandwidth-constrained hardware, transformers would likely be **slower** per training step. Attention's memory access pattern is unfavorable on GB10's 273 GB/s (4x less than RTX 4090). ResNet-SE with convolutions has better arithmetic intensity.

**When to reconsider:** If scaling to multiple GPUs or moving to hardware with higher memory bandwidth. Google's recent work shows transformers can match ResNets at higher compute budgets.

### 9. Hybrid Architecture: Fewer Blocks + Wider Filters

Alternative to pure shrinking: **10 blocks, 384 filters** (~18M params):
- Shallower (fewer sequential ops → lower latency)
- Wider (more parallelism per layer → better GPU utilization)
- Fewer total FLOPs than 20×256 but more expressive per layer

Could be a sweet spot for GB10 where wider convolutions map well to 6,144 CUDA cores.

---

## Tier 4: Longer-Term / Experimental

### 10. Knowledge Distillation Pipeline

Train a small model (10b×128f, ~2.5M params) very fast, then use it as a teacher for a larger model:
- Small model generates self-play data 4-5x faster
- Large model trains on the small model's improved policies
- Common in LC0 community for bootstrapping

### 11. ONNX Runtime or TensorRT for Inference

Export the inference model to TensorRT. Combined with FP8, this could give 3-4x inference throughput via kernel fusion + quantization + graph optimization. The inference path is read-only so this is safe.

### 12. Replay Buffer Compression

At 49KB/position and 750K capacity, the buffer uses ~37 GB. Compressing to ~500 bytes/position (as noted in the spec) would free ~35 GB, allowing either a larger buffer (more training diversity) or freeing RAM for larger training batches.

---

## Recommended Action Plan (Priority Order)

| # | Action | Expected Impact | Effort |
|---|---|---|---|
| 1 | **Shrink to 10 blocks × 256 filters** | ~2x steps/s | Config change + retrain |
| 2 | **torch.compile warmup** | Eliminate startup stall | ~50 lines Python |
| 3 | **Move prepare_replay_batch to C++** | 5-15% step speedup | Medium (C++ + pybind) |
| 4 | **Dedicated inference CUDA stream** | 10-30% overlap improvement | ~100 lines Python |
| 5 | **FP8 inference** | ~1.5-2x inference throughput | Medium (PyTorch FP8 or TRT) |
| 6 | **Game-specific MCTSNode sizing** | Better cache, ~3 GB freed | Template specialization |
| 7 | **Reduce sims to 100 for early training** | 2x self-play in warmup phase | Config change |

Items 1-2 alone could push from 0.42 → ~0.8+ train steps/s. Items 3-5 could push toward 1.0+. The NN shrink is the biggest single win because we're compute-bound at 96% GPU — making each step do half the FLOPs is the most direct path to faster training.
