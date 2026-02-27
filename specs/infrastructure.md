# Infrastructure

## 1. Project Directory Structure

```
alphazero/
├── CMakeLists.txt                    # Top-level CMake build
├── pyproject.toml                    # Python package configuration
├── configs/
│   ├── chess.yaml                   # Chess training config (production)
│   ├── chess_default.yaml           # Chess training config (alias)
│   ├── chess_test.yaml              # Chess test config (tiny, fast)
│   ├── go.yaml                      # Go training config (production)
│   ├── go_default.yaml              # Go training config (alias)
│   └── go_test.yaml                 # Go test config (tiny, fast)
│
├── src/                              # C++ source code
│   ├── CMakeLists.txt
│   ├── games/
│   │   ├── game_state.h             # Abstract GameState interface
│   │   ├── game_config.h            # GameConfig base class
│   │   ├── chess/
│   │   │   ├── chess_state.h
│   │   │   ├── chess_state.cpp
│   │   │   ├── chess_config.h       # ChessGameConfig
│   │   │   ├── chess_config.cpp
│   │   │   ├── bitboard.h           # Bitboard utilities
│   │   │   ├── bitboard.cpp
│   │   │   ├── movegen.h            # Move generation
│   │   │   └── movegen.cpp
│   │   └── go/
│   │       ├── go_state.h
│   │       ├── go_state.cpp
│   │       ├── go_config.h          # GoGameConfig
│   │       ├── go_config.cpp
│   │       ├── go_rules.h           # Liberty tracking, capture, ko
│   │       ├── go_rules.cpp
│   │       ├── scoring.h            # Tromp-Taylor scoring
│   │       └── scoring.cpp
│   │
│   ├── mcts/
│   │   ├── mcts_node.h              # MCTSNodeT<MaxActions> template (SoA layout)
│   │   ├── node_store.h             # NodeStoreT<NodeType> interface
│   │   ├── arena_node_store.h       # ArenaNodeStoreT<NodeType> implementation
│   │   ├── arena_node_store.cpp
│   │   ├── mcts_search.h            # MctsSearchT<NodeType> + RuntimeMctsSearch
│   │   ├── mcts_search.cpp
│   │   ├── eval_queue.h             # Async evaluation queue
│   │   └── eval_queue.cpp
│   │
│   ├── selfplay/
│   │   ├── self_play_manager.h      # Orchestrates concurrent games
│   │   ├── self_play_manager.cpp
│   │   ├── self_play_game.h         # Single game lifecycle + AddGameFn type erasure
│   │   ├── self_play_game.cpp
│   │   ├── replay_buffer.h          # Dense ReplayBuffer + ReplayPosition
│   │   ├── replay_buffer.cpp
│   │   ├── compact_replay_buffer.h  # CompactReplayBuffer (bitpacked + sparse policy)
│   │   ├── compact_replay_buffer.cpp
│   │   ├── replay_compression.h     # State compression/decompression utilities
│   │   └── replay_compression.cpp
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
│       ├── config.py                # GameConfig + YAML loading
│       ├── network/
│       │   ├── __init__.py
│       │   ├── base.py              # AlphaZeroNetwork base class
│       │   ├── resnet_se.py         # ResNet + SE implementation
│       │   ├── heads.py             # Policy and value head implementations
│       │   └── bn_fold.py           # Batch norm folding utility
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py           # Training step + batch preparation
│       │   ├── loss.py              # Loss function implementations
│       │   └── lr_schedule.py       # Learning rate schedule
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── orchestrator.py      # Parallel pipeline (inference + training threads)
│       │   └── evaluation.py        # Periodic Elo estimation
│       └── utils/
│           ├── __init__.py
│           ├── logging.py           # TensorBoard + console logging
│           └── checkpoint.py        # Checkpoint save/load + replay buffer persistence
│
├── web/                              # Web UI for browser play
│   ├── server.py                    # FastAPI application + WebSocket endpoints
│   ├── engine.py                    # ChessEngine wrapper for human play
│   ├── watch_engine.py              # WatchEngine for auto-play viewing
│   ├── model_manager.py             # Checkpoint discovery and model loading
│   ├── requirements.txt             # FastAPI, uvicorn, websockets
│   └── static/                      # HTML/CSS/JS frontend
│       ├── index.html
│       └── watch.html
│
├── tests/
│   ├── cpp/
│   │   ├── CMakeLists.txt
│   │   ├── test_scaffold_smoke.cpp  # Minimal smoke test (always built)
│   │   ├── test_game_interfaces.cpp # Game abstraction interface tests
│   │   ├── test_chess_bitboard.cpp  # Bitboard operations
│   │   ├── test_chess_movegen.cpp   # Chess move generation perft tests
│   │   ├── test_chess_state.cpp     # Chess state logic
│   │   ├── test_chess_encoding.cpp  # Chess input/output encoding
│   │   ├── test_chess_serialization.cpp  # FEN/PGN round-trip
│   │   ├── test_go_rules.cpp        # Go rules (capture, ko, scoring)
│   │   ├── test_go_state.cpp        # Go state logic
│   │   ├── test_go_encoding.cpp     # Go input/output encoding
│   │   ├── test_go_serialization.cpp # SGF round-trip
│   │   ├── test_mcts.cpp            # MCTS correctness
│   │   ├── test_eval_queue.cpp      # Eval queue threading
│   │   ├── test_replay_buffer.cpp   # Dense replay buffer concurrency
│   │   ├── test_compact_replay_buffer.cpp  # Compact replay buffer
│   │   ├── test_replay_compression.cpp     # Compression round-trip
│   │   ├── test_arena.cpp           # Arena allocator
│   │   ├── test_self_play_game.cpp  # Single game lifecycle
│   │   ├── test_self_play_manager.cpp  # Manager orchestration
│   │   └── test_libtorch_inference.cpp # LibTorch integration
│   └── python/
│       ├── test_network.py          # Network forward pass shapes
│       ├── test_loss.py             # Loss function computation
│       ├── test_bn_fold.py          # BN folding correctness
│       ├── test_training.py         # End-to-end training step
│       ├── test_bindings.py         # C++ bindings integration tests
│       └── test_train_script.py     # Training script logic tests
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
└── .claude/                          # Claude Code project settings
    └── settings.json
```

## 2. Build System

### CMake (C++)

The C++ code is built with CMake (minimum version 3.24).

```cmake
cmake_minimum_required(VERSION 3.24)
project(alphazero VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional build features (all default ON)
option(ALPHAZERO_ENABLE_CUDA "Enable CUDA support" ON)
option(ALPHAZERO_ENABLE_TORCH "Enable LibTorch" ON)
option(ALPHAZERO_ENABLE_PYBIND "Build Python extension" ON)
option(ALPHAZERO_ENABLE_CPP_TESTS "Build C++ tests" ON)

# Dependencies (optional — build degrades gracefully)
find_package(CUDAToolkit)    # warns if not found
find_package(Torch)          # warns if not found
find_package(pybind11)       # required for Python extension
find_package(GTest)          # required for full test suite

# Engine library
add_library(alphazero_engine STATIC
    src/games/chess/chess_state.cpp
    src/games/chess/bitboard.cpp
    src/games/chess/movegen.cpp
    src/games/chess/chess_config.cpp
    src/games/go/go_state.cpp
    src/games/go/go_rules.cpp
    src/games/go/scoring.cpp
    src/games/go/go_config.cpp
    src/mcts/arena_node_store.cpp
    src/mcts/mcts_search.cpp
    src/mcts/eval_queue.cpp
    src/selfplay/self_play_manager.cpp
    src/selfplay/self_play_game.cpp
    src/selfplay/replay_buffer.cpp
    src/selfplay/compact_replay_buffer.cpp
    src/selfplay/replay_compression.cpp
    src/nn/libtorch_inference.cpp
)
set_target_properties(alphazero_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(alphazero_engine Threads::Threads)
# Link optional deps if found: Torch, CUDA::cudart

# Python bindings
pybind11_add_module(alphazero_cpp src/bindings/python_bindings.cpp)
target_link_libraries(alphazero_cpp PRIVATE alphazero_engine)

# Tests
enable_testing()
add_subdirectory(tests/cpp)
```

Note: `POSITION_INDEPENDENT_CODE ON` is required on the engine library for the pybind11 `.so` extension to link correctly.

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
| CUDA Toolkit | 13.0 | GPU compute |
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
# Default PyPI only ships CPU-only for aarch64; use the CUDA index:
pip install torch --index-url https://download.pytorch.org/whl/cu130

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

## 8. Web UI

A browser-based interface for playing against trained models and watching self-play games.

### Dependencies

```
fastapi>=0.110
uvicorn[standard]>=0.27
websockets>=12.0
```

### Running

```bash
pip install -r web/requirements.txt
python -m web.server --model checkpoints/checkpoint_00009000.pt --simulations 800 --port 8000
```

Then open `http://127.0.0.1:8000`.

### Components

- **`server.py`**: FastAPI application with WebSocket endpoints for real-time game communication
- **`engine.py`**: ChessEngine wrapper for human vs. AI play
- **`watch_engine.py`**: WatchEngine for automated AI vs. AI games with live viewing
- **`model_manager.py`**: Checkpoint discovery, architecture inference from state_dict, and lazy model loading with caching

## 9. Future Optimizations

| Optimization | Status | Expected Impact | Complexity |
|---|---|---|---|
| Replay buffer compression | **Implemented** (`CompactReplayBuffer`) | ~100× memory reduction | Low |
| FP8 inference via TensorRT | Deferred | ~2× inference throughput | Medium |
| Custom CUDA kernels for MCTS batching | Deferred | Reduced CPU→GPU latency | High |
| Transposition table (chess) | Deferred | ~10-20% fewer NN evals | Medium |
| Pondering (background search) | Deferred | Better time utilization during play | Low |
| ONNX Runtime backend | Deferred | Potentially faster inference | Low |
| Multi-DGX Spark training (2-unit cluster) | Deferred | 2× throughput | Medium |
| Transformer network architecture | Deferred | Better Elo/FLOP | Medium |
