# AlphaZero Implementation: System Overview

## 1. Project Goals

Build a fast, single-machine implementation of DeepMind's AlphaZero algorithm capable of:

- Training from scratch (tabula rasa) to strong play in both **Chess** and **Go**
- Running entirely on a single **NVIDIA DGX Spark** (GB10 Grace Blackwell)
- Maximizing GPU utilization through an async self-play/training pipeline
- Supporting future research through clean abstractions and swappable components

The implementation follows the **AlphaZero** variant (Silver et al., 2017 — "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"), not the earlier AlphaGo Zero, wherever the two differ.

### Key AlphaZero-over-AlphaGo-Zero Decisions

| Aspect | AlphaGo Zero | AlphaZero (our choice) |
|---|---|---|
| Network updates | Best-player selection with evaluator gate | Continuous training from latest network |
| Evaluator | 400-game matches, 55% win threshold | No evaluator gate |
| Symmetry in MCTS | Random rotation/reflection during eval | No symmetry during MCTS |
| Game outcomes | Binary win/loss | Expected outcome (handles draws) |
| Hyperparameters | Bayesian optimization per game | Same hyperparameters across games (except Dirichlet noise) |
| Data augmentation | 8 symmetries during training (Go) | Training data augmentation for Go only; none for chess |

## 2. Target Hardware: NVIDIA DGX Spark

| Spec | Value |
|---|---|
| SoC | NVIDIA GB10 Grace Blackwell |
| CPU | 20-core ARM (10x Cortex-X925 + 10x Cortex-A725) |
| GPU | Blackwell: 6,144 CUDA cores, 192 5th-gen Tensor Cores |
| Memory | 128 GB unified LPDDR5x (shared CPU/GPU, cache-coherent) |
| Memory bandwidth | 273 GB/s |
| AI compute | ~1 PFLOP sparse FP4; supports FP8, BF16, FP16 |
| Storage | 4 TB NVMe PCIe Gen5 |
| OS | Ubuntu 24.04 |

### Hardware Implications for Design

- **Unified memory**: CPU and GPU share the same physical memory. No `cudaMemcpy` needed. The replay buffer, model weights, and MCTS trees are all zero-copy accessible. Grace Blackwell provides hardware cache coherence between CPU and GPU.
- **Memory bandwidth is the bottleneck**: At 273 GB/s, bandwidth is ~4x less than an RTX 4090 and ~12x less than an H100. Maximizing arithmetic intensity (FLOPS per byte moved) is critical.
- **Single GPU**: Self-play inference and training compete for the same GPU. An async pipeline with interleaved scheduling is required.
- **ARM CPU**: 20 cores available for MCTS tree traversal, game logic, and pipeline orchestration. The CPU is underutilized relative to the GPU — we exploit this with many MCTS threads.

## 3. Algorithm Overview

AlphaZero learns to play board games through self-play reinforcement learning, with no human data or domain knowledge beyond the game rules.

### Core Loop

```
1. Self-Play: Generate games using MCTS guided by neural network f_θ
2. Store: Save (state, search_policy π, outcome z) tuples to replay buffer
3. Train: Update f_θ to predict π and z from state
4. Repeat: Self-play uses the updated network immediately
```

### Neural Network

A single dual-headed network `(p, v) = f_θ(s)`:
- **Input** `s`: Board position encoded as spatial planes (history of T=8 positions)
- **Policy output** `p`: Probability distribution over all legal actions
- **Value output** `v`: Expected game outcome from the current player's perspective
  - Scalar (tanh, range [-1, +1]) for Go
  - Win/Draw/Loss (3-way softmax) for Chess

### MCTS

Each move selection runs 800 simulations of Monte-Carlo Tree Search:
1. **Select**: Traverse tree choosing actions by PUCT (Q + exploration bonus)
2. **Expand**: Add leaf node to tree
3. **Evaluate**: Neural network inference on leaf position
4. **Backup**: Propagate value up the tree, updating visit counts and Q-values

### Training Objective

```
L = L_value + L_policy + c * ||θ||²

L_policy = -π^T log(p)                     (cross-entropy)
L_value  = (z - v)²                         (MSE, for scalar/Go)
L_value  = -z_wdl^T log(v_wdl)             (cross-entropy, for WDL/Chess)
c        = 10^-4                             (L2 regularization)
```

## 4. Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| MCTS engine, self-play, game logic | C++20 | Performance-critical; CUDA ecosystem; reference implementations |
| Neural network, training loop | Python 3.11+ / PyTorch | Mature; DGX software stack; fast iteration |
| C++ ↔ Python bridge | pybind11 / torch::extension | Standard in PyTorch ecosystem |
| Build system | CMake | Standard for C++ CUDA projects |
| GPU compute | CUDA, cuDNN | Native Blackwell support |
| Monitoring | TensorBoard | Standard; local logging to NVMe |
| Future: custom CUDA kernels | CUDA C++ | Natural extension of engine code |

## 5. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DGX Spark (GB10)                         │
│                                                                 │
│  ┌─────────────────────────────────────┐   ┌─────────────────┐  │
│  │        Self-Play Engine (C++)       │   │ Training (Python │  │
│  │                                     │   │   + PyTorch)     │  │
│  │  ┌──────┐ ┌──────┐     ┌──────┐    │   │                 │  │
│  │  │Game 0│ │Game 1│ ... │Game M│    │   │  Forward pass   │  │
│  │  │K thds│ │K thds│     │K thds│    │   │  Loss + backward│  │
│  │  └──┬───┘ └──┬───┘     └──┬───┘    │   │  Optimizer step │  │
│  │     │        │            │         │   │  LR scheduling  │  │
│  │     └────────┼────────────┘         │   └────────┬────────┘  │
│  │              ▼                      │            │           │
│  │      ┌──────────────┐               │            │           │
│  │      │  Eval Queue  │               │            │           │
│  │      └──────┬───────┘               │            │           │
│  └─────────────┼───────────────────────┘            │           │
│                │                                     │           │
│  ══════════════╪═════════════════════════════════════╪═══════    │
│          Unified Memory (128 GB)                                │
│                │                                     │           │
│       ┌────────▼────────┐    ┌───────────────────────▼───────┐  │
│       │   NN Weights    │◄──►│         GPU (Blackwell)       │  │
│       │  (single copy)  │    │  • Self-play inference (BF16) │  │
│       └─────────────────┘    │  • Training fwd+bwd (BF16)    │  │
│                              │  • Interleaved scheduling     │  │
│       ┌─────────────────┐    └───────────────────────────────┘  │
│       │  Replay Buffer  │                                       │
│       │  (ring buffer,  │◄── written by self-play               │
│       │   1-2M pos)     │──► read by training                   │
│       └─────────────────┘                                       │
│                                                                 │
│       ┌─────────────────┐                                       │
│       │   Checkpoints   │──► NVMe SSD (4 TB)                    │
│       │   Logs/Metrics  │                                       │
│       └─────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

## 6. Component Map

| Spec Document | Contents |
|---|---|
| [`game-interface.md`](game-interface.md) | Abstract game interface; Chess (bitboard) and Go implementations; input/output encoding |
| [`neural-network.md`](neural-network.md) | ResNet + SE architecture; policy/value heads; precision; batch norm folding |
| [`mcts.md`](mcts.md) | PUCT search; hybrid parallelism; eval queue; tree memory; virtual loss |
| [`pipeline.md`](pipeline.md) | Async self-play/training pipeline; replay buffer; GPU scheduling; checkpointing |
| [`infrastructure.md`](infrastructure.md) | Build system; project layout; testing; monitoring; DGX Spark deployment |

## 7. Reference Papers

1. Silver et al. (2016) — "Mastering the game of Go with deep neural networks and tree search" (AlphaGo)
2. Silver et al. (2017) — "Mastering the Game of Go without Human Knowledge" (AlphaGo Zero)
3. Silver et al. (2017) — "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero)
4. Schrittwieser et al. (2020) — "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero)

## 8. Reference Implementations

- **Leela Chess Zero** (lczero.org) — Most mature open-source AlphaZero for chess. C++ engine, Python training. Migrated from ResNet+SE to Transformers.
- **Minigo** (github.com/tensorflow/minigo) — Google's reference Go implementation. Demonstrated root parallelism superiority.
- **AlphaZero.jl** (github.com/jonathan-laurent/AlphaZero.jl) — Clean Julia implementation with generic game interface.
