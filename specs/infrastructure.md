# Infrastructure

## 1. Project Directory Structure

```
alphazero/
├── CMakeLists.txt                    # Top-level CMake build
├── pyproject.toml                    # Python package configuration
├── configs/
│   ├── chess_default.yaml            # Default chess training config
│   └── go_default.yaml              # Default Go training config
│
├── src/                              # C++ source code
│   ├── CMakeLists.txt
│   ├── games/
│   │   ├── game_state.h             # Abstract GameState interface
│   │   ├── game_config.h            # GameConfig struct
│   │   ├── chess/
│   │   │   ├── chess_state.h
│   │   │   ├── chess_state.cpp
│   │   │   ├── bitboard.h           # Bitboard utilities
│   │   │   ├── bitboard.cpp
│   │   │   ├── movegen.h            # Move generation
│   │   │   ├── movegen.cpp
│   │   │   └── chess_config.cpp     # Chess GameConfig
│   │   └── go/
│   │       ├── go_state.h
│   │       ├── go_state.cpp
│   │       ├── go_rules.h           # Liberty tracking, capture, ko
│   │       ├── go_rules.cpp
│   │       ├── scoring.h            # Tromp-Taylor scoring
│   │       ├── scoring.cpp
│   │       └── go_config.cpp        # Go GameConfig
│   │
│   ├── mcts/
│   │   ├── mcts_node.h              # MCTSNode struct (SoA layout)
│   │   ├── node_store.h             # NodeStore interface
│   │   ├── arena_node_store.h       # Arena allocator implementation
│   │   ├── arena_node_store.cpp
│   │   ├── mcts_search.h            # MCTS simulation logic
│   │   ├── mcts_search.cpp
│   │   ├── eval_queue.h             # Async evaluation queue
│   │   └── eval_queue.cpp
│   │
│   ├── selfplay/
│   │   ├── self_play_manager.h      # Orchestrates concurrent games
│   │   ├── self_play_manager.cpp
│   │   ├── self_play_game.h         # Single game lifecycle
│   │   ├── self_play_game.cpp
│   │   ├── replay_buffer.h          # Ring buffer in unified memory
│   │   └── replay_buffer.cpp
│   │
│   ├── nn/
│   │   ├── nn_inference.h           # NeuralNetInference interface
│   │   ├── libtorch_inference.h     # LibTorch implementation
│   │   └── libtorch_inference.cpp
│   │
│   └── bindings/
│       └── python_bindings.cpp      # pybind11 bindings
│
├── python/                           # Python source code
│   └── alphazero/
│       ├── __init__.py
│       ├── config.py                # Configuration loading (YAML)
│       ├── network/
│       │   ├── __init__.py
│       │   ├── base.py              # AlphaZeroNetwork base class
│       │   ├── resnet_se.py         # ResNet + SE implementation
│       │   ├── heads.py             # Policy and value head implementations
│       │   └── bn_fold.py           # Batch norm folding utility
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py           # Training loop
│       │   ├── loss.py              # Loss function implementations
│       │   └── lr_schedule.py       # Learning rate schedule
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── orchestrator.py      # GPU scheduling (S:T interleaving)
│       │   └── evaluation.py        # Periodic Elo estimation
│       └── utils/
│           ├── __init__.py
│           ├── logging.py           # TensorBoard logging
│           └── checkpoint.py        # Checkpoint save/load
│
├── tests/
│   ├── cpp/
│   │   ├── CMakeLists.txt
│   │   ├── test_chess_movegen.cpp   # Chess move generation perft tests
│   │   ├── test_chess_encoding.cpp  # Chess input/output encoding
│   │   ├── test_go_rules.cpp        # Go rules (capture, ko, scoring)
│   │   ├── test_go_encoding.cpp     # Go input/output encoding
│   │   ├── test_mcts.cpp            # MCTS correctness
│   │   ├── test_eval_queue.cpp      # Eval queue threading
│   │   ├── test_replay_buffer.cpp   # Replay buffer concurrency
│   │   └── test_arena.cpp           # Arena allocator
│   └── python/
│       ├── test_network.py          # Network forward pass shapes
│       ├── test_loss.py             # Loss function computation
│       ├── test_bn_fold.py          # BN folding correctness
│       └── test_training.py         # End-to-end training step
│
├── scripts/
│   ├── train.py                     # Main training entry point
│   ├── play.py                      # Play against the trained model
│   ├── benchmark.py                 # GPU throughput benchmarking
│   └── export_model.py             # Export model for deployment
│
├── specs/                            # Specification documents (this directory)
│   ├── overview.md
│   ├── game-interface.md
│   ├── neural-network.md
│   ├── mcts.md
│   ├── pipeline.md
│   └── infrastructure.md
│
└── .local/                           # Local development files (not committed)
    └── refs/                         # Reference papers and links
```

## 2. Build System

### CMake (C++)

The C++ code is built with CMake (minimum version 3.24 for CUDA support).

```cmake
cmake_minimum_required(VERSION 3.24)
project(alphazero LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Engine library
add_library(alphazero_engine
    src/games/chess/chess_state.cpp
    src/games/chess/bitboard.cpp
    src/games/chess/movegen.cpp
    src/games/go/go_state.cpp
    src/games/go/go_rules.cpp
    src/games/go/scoring.cpp
    src/mcts/arena_node_store.cpp
    src/mcts/mcts_search.cpp
    src/mcts/eval_queue.cpp
    src/selfplay/self_play_manager.cpp
    src/selfplay/self_play_game.cpp
    src/selfplay/replay_buffer.cpp
    src/nn/libtorch_inference.cpp
)
target_link_libraries(alphazero_engine
    ${TORCH_LIBRARIES}
    CUDA::cudart
    pthread
)
target_compile_options(alphazero_engine PRIVATE
    -O3 -march=armv9-a     # ARM optimization for DGX Spark
    -Wall -Wextra
    -fsanitize=thread      # Enable during development only
)

# Python bindings
pybind11_add_module(alphazero_cpp src/bindings/python_bindings.cpp)
target_link_libraries(alphazero_cpp PRIVATE alphazero_engine)

# Tests
enable_testing()
add_subdirectory(tests/cpp)
```

### Python Package

Python code is managed as a standard package. Dependencies in `pyproject.toml`:

```toml
[project]
name = "alphazero"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "tensorboard",
    "pyyaml",
    "numpy",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-benchmark",
]
```

### Build Targets

| Target | Command | Description |
|---|---|---|
| Build C++ engine | `cmake --build build` | Compiles engine library and python bindings |
| Run C++ tests | `ctest --test-dir build` | Chess perft, Go rules, MCTS, concurrency tests |
| Run Python tests | `pytest tests/python/` | Network shapes, loss, training step |
| Install Python package | `pip install -e .` | Editable install for development |
| Full build + test | `cmake --build build && ctest --test-dir build && pytest` | CI pipeline |

## 3. Dependencies

### System Dependencies (DGX Spark / Ubuntu 24.04)

| Package | Version | Purpose |
|---|---|---|
| GCC | 13+ | C++20 compiler (ARM) |
| CMake | 3.24+ | Build system |
| CUDA Toolkit | 12.x | GPU compute |
| cuDNN | 9.x | Optimized NN primitives |
| Python | 3.11+ | Training code |

### C++ Dependencies

| Library | Purpose | Integration |
|---|---|---|
| LibTorch | NN inference from C++ | CMake `find_package(Torch)` |
| pybind11 | Python bindings | CMake `find_package(pybind11)` |
| Google Test | C++ unit testing | CMake FetchContent |

### Python Dependencies

| Package | Purpose |
|---|---|
| PyTorch (torch) | Neural network training |
| TensorBoard | Metrics visualization |
| PyYAML | Configuration file parsing |
| NumPy | Data manipulation |
| pytest | Testing |

## 4. Testing Strategy

### Test Pyramid

```
┌──────────────────────────┐
│    Integration Tests     │  End-to-end: self-play → train → improved play
│    (few, slow)           │  Run: nightly or manual
├──────────────────────────┤
│    Component Tests       │  MCTS correctness, eval queue threading,
│    (moderate)            │  replay buffer concurrency, training step
│                          │  Run: every build
├──────────────────────────┤
│    Unit Tests            │  Move generation, encoding, rules, loss functions,
│    (many, fast)          │  network shapes, BN folding
│                          │  Run: every build
└──────────────────────────┘
```

### C++ Tests

**Chess move generation (perft)**:
- Compare move counts at depths 1-6 against known-correct perft results for multiple standard positions (initial position, kiwipete, various endgames).
- This is the gold-standard correctness test for chess engines.
- If perft counts match at depth 6, move generation is almost certainly correct.

**Chess encoding**:
- Verify input tensor for known positions (initial position, specific mid-game positions).
- Verify action index ↔ move mapping round-trips correctly.
- Verify board flipping for black-to-move positions.

**Go rules**:
- Capture scenarios: simple capture, snapback, large group capture.
- Ko detection and prohibition.
- Superko detection.
- Liberty counting correctness.
- Tromp-Taylor scoring against known game results.
- Self-capture prohibition.

**Go encoding**:
- Verify input tensor for known positions.
- Verify symmetry transforms produce equivalent positions.
- Verify policy vector transforms are consistent with board transforms.

**MCTS correctness**:
- Run MCTS with a mock NN (fixed policy and value) and verify visit counts converge to expected distributions.
- Verify backup correctly negates values at alternating levels.
- Verify FPU computation.
- Verify Dirichlet noise is applied only at root.
- Verify tree reuse preserves statistics correctly.

**Eval queue threading**:
- Multiple producer threads, single consumer thread.
- Verify all requests are processed and results dispatched correctly.
- Verify flush timeout triggers on partial batches.
- Stress test under high contention.

**Replay buffer concurrency**:
- Concurrent writes from multiple game threads.
- Concurrent reads from training thread.
- Verify no data corruption or lost positions.
- Verify ring buffer wrapping.

### Python Tests

**Network shapes**:
- Instantiate ResNet+SE with various configs and verify output shapes for both chess and Go.
- Verify policy output size matches `action_space_size`.
- Verify value head output size (1 for scalar, 3 for WDL).

**Loss functions**:
- Verify policy cross-entropy against hand-computed values.
- Verify value MSE and cross-entropy against hand-computed values.
- Verify L2 regularization includes all parameters.

**BN folding**:
- Run inference before and after BN folding.
- Verify outputs match within floating-point tolerance (1e-5).

**Training step**:
- Run a single training step on synthetic data.
- Verify loss decreases.
- Verify gradients are non-zero.
- Verify mixed precision doesn't produce NaN.

### Integration Tests

**Smoke test**:
- Run the full pipeline (self-play + training) for 100 training steps.
- Verify the replay buffer fills.
- Verify checkpoints are saved.
- Verify metrics are logged.
- Does not verify playing strength — just that the pipeline runs without crashing.

**Learning test (Connect Four)**:
- Implement Connect Four as a simple game (small board, fast games).
- Run AlphaZero training for a short duration.
- Verify the trained model beats a random player >90% of the time.
- This tests the entire algorithm end-to-end on a tractable problem.

## 5. Monitoring

### TensorBoard

All metrics are logged to TensorBoard format on the local NVMe:

```
logs/
├── chess_run_001/
│   ├── events.out.tfevents.*     # TensorBoard event files
│   └── config.yaml               # Copy of training config
└── go_run_001/
    ├── events.out.tfevents.*
    └── config.yaml
```

Access via: `tensorboard --logdir logs/`

### Logged Scalars

See `pipeline.md` Section 8 for the full list of training and self-play metrics.

### Console Output

The main training script prints periodic summaries:

```
Step 1000 | Loss: 4.23 (policy: 3.91, value: 0.30, l2: 0.02) | LR: 0.200
  Self-play: 142 games/hr, avg length 67 moves, W/D/L: 48/4/48%
  Buffer: 54,200 / 1,000,000 positions
  GPU: 83% inference, 17% training
```

## 6. DGX Spark Deployment

### Environment Setup

```bash
# Create conda environment
conda create -n alphazero python=3.11
conda activate alphazero

# Install PyTorch (ARM + CUDA for Grace Blackwell)
# Use the appropriate PyTorch build for the DGX Spark platform.
# As of 2025, NVIDIA provides optimized PyTorch wheels via the
# DGX software stack or NGC containers.
pip install torch  # or use NVIDIA's provided wheel

# Install project
pip install -e ".[dev]"

# Build C++ engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
cd ..

# Run tests
ctest --test-dir build
pytest tests/python/
```

### Running Training

```bash
# Chess training
python scripts/train.py --config configs/chess_default.yaml

# Go training
python scripts/train.py --config configs/go_default.yaml

# Resume from checkpoint
python scripts/train.py --config configs/chess_default.yaml --resume checkpoints/step_100000.pt
```

### Performance Benchmarking

Before starting a training run, benchmark GPU throughput:

```bash
# Benchmark inference throughput (positions/second at various batch sizes)
python scripts/benchmark.py --mode inference --game chess --batch-sizes 32,64,128,256,512

# Benchmark training throughput (steps/second at various batch sizes)
python scripts/benchmark.py --mode training --game chess --batch-sizes 256,512,1024,2048,4096

# Benchmark MCTS throughput (simulations/second with various thread configs)
python scripts/benchmark.py --mode mcts --game chess --games 16,32,64 --threads 4,8,16
```

Use benchmark results to tune `concurrent_games`, `threads_per_game`, `batch_size`, and the S:T ratio.

### Resource Monitoring

Monitor system resources during training:

```bash
# GPU utilization, memory, temperature
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 5

# CPU utilization across all cores
htop

# Memory usage
free -h
```

Target: GPU utilization >80%, CPU utilization 30-50% (MCTS threads), memory usage <64 GB (leaving headroom).

## 7. Playing Against the Trained Model

```bash
# Interactive play (human vs AI)
python scripts/play.py --game chess --model checkpoints/step_500000.pt --simulations 800

# AI vs external engine (e.g., Stockfish via UCI protocol)
python scripts/play.py --game chess --model checkpoints/step_500000.pt --opponent stockfish --games 100
```

The play script uses the same MCTS engine as self-play but with:
- Deterministic move selection (τ → 0)
- No Dirichlet noise
- Pondering disabled (future feature)
- Resignation enabled with calibrated threshold

## 8. Future Optimizations (Not in v1)

The following optimizations are identified but deferred to post-v1:

| Optimization | Expected Impact | Complexity |
|---|---|---|
| FP8 inference via TensorRT | ~2× inference throughput | Medium |
| Custom CUDA kernels for MCTS batching | Reduced CPU→GPU latency | High |
| Replay buffer compression | ~100× memory reduction | Low |
| Transposition table (chess) | ~10-20% fewer NN evals | Medium |
| Pondering (background search) | Better time utilization during play | Low |
| ONNX Runtime backend | Potentially faster inference | Low |
| Multi-DGX Spark training (2-unit cluster) | 2× throughput | Medium |
| Transformer network architecture | Better Elo/FLOP | Medium |
