# AlphaZero How-To Guide

## How It Works

The system runs an interleaved pipeline on a single GPU:

1. **Self-play** — 32 concurrent games, each with 8 MCTS threads doing 800 simulations/move. Leaf positions are batched and sent to the GPU for neural network evaluation.
2. **Replay buffer** — Completed games write (board state, MCTS policy, game outcome) tuples to a 1M-position ring buffer.
3. **Training** — SGD updates the network to predict the MCTS policy and game outcome from board states.
4. **Repeat** — The updated network immediately improves self-play quality, creating a virtuous cycle.

The GPU alternates between ~50 inference batches and 1 training step each cycle (~85% inference, ~15% training). No human data is used — it learns entirely from self-play.

## Setup

```bash
# Create conda environment
conda create -n alphazero python=3.11
conda activate alphazero

# Install Python package (editable, with dev deps)
pip install -e ".[dev]"

# Build the C++ engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
cd ..
```

## Run Tests

```bash
ctest --test-dir build          # C++ tests (movegen, rules, MCTS, etc.)
pytest tests/python/            # Python tests (network, loss, training)
```

## Train

```bash
# Chess (full training run)
PYTHONPATH=build/src:$PYTHONPATH python scripts/train.py --config configs/chess_default.yaml

# Go
PYTHONPATH=build/src:$PYTHONPATH python scripts/train.py --config configs/go_default.yaml

# Resume from checkpoint
PYTHONPATH=build/src:$PYTHONPATH python scripts/train.py --config configs/chess_default.yaml --resume checkpoints/step_100000.pt
```

### Configuration

Pre-made configs in `configs/`:

| Config | Use case |
|---|---|
| `configs/chess_default.yaml` | Full chess training (20-block net, 800 sims, 700K steps) |
| `configs/go_default.yaml` | Full Go training (same scale) |
| `configs/chess_test.yaml` | Tiny smoke test (2-block net, 3 steps) |
| `configs/go_test.yaml` | Tiny smoke test for Go |

Copy and customize any of these. Key parameters:

| Parameter | Default | What it does |
|---|---|---|
| `network.num_blocks` | 20 | Residual blocks (10 for dev, 20 for production, 40 for large) |
| `network.num_filters` | 256 | Channel width (128 for dev, 256 for production) |
| `mcts.simulations_per_move` | 800 | MCTS rollouts per move — lower is faster but weaker |
| `mcts.concurrent_games` | 32 | Parallel self-play games |
| `mcts.threads_per_game` | 8 | MCTS threads per game (total threads = games x threads) |
| `training.max_steps` | 700000 | Total training steps |
| `training.batch_size` | 1024 | Training mini-batch size |
| `training.lr_schedule` | step decay | Piecewise-constant LR: `[[0, 0.2], [200000, 0.02], ...]` |
| `replay_buffer.capacity` | 1000000 | Ring buffer size in positions |
| `pipeline.inference_batches_per_cycle` | 50 | Inference batches before each training step |

### Monitoring

```bash
tensorboard --logdir logs/
```

Metrics: loss (policy/value/total), learning rate, replay buffer fill, games completed, self-play throughput, and periodic Elo estimates vs saved milestones.

Checkpoints are saved every 1,000 steps (last 5 kept), with permanent milestones every 50,000 steps.

## Play Against a Trained Model

### Interactive (human vs AI)

```bash
PYTHONPATH=build/src:$PYTHONPATH python scripts/play.py \
  --game chess \
  --model checkpoints/step_500000.pt \
  --opponent human \
  --simulations 800
```

Enter moves in UCI notation (e.g. `e2e4`). For Go, use coordinates.

### Engine Match (AI vs Stockfish)

```bash
PYTHONPATH=build/src:$PYTHONPATH python scripts/play.py \
  --game chess \
  --model checkpoints/step_500000.pt \
  --opponent stockfish \
  --games 100 \
  --simulations 800
```

## Export a Trained Model

```bash
# ONNX (with batch norm folding for zero-overhead inference)
PYTHONPATH=build/src:$PYTHONPATH python scripts/export_model.py \
  --checkpoint checkpoints/step_500000.pt \
  --format onnx \
  --fold-bn \
  --config configs/chess_default.yaml

# TorchScript
PYTHONPATH=build/src:$PYTHONPATH python scripts/export_model.py \
  --checkpoint checkpoints/step_500000.pt \
  --format torchscript \
  --fold-bn
```

## Benchmark

```bash
PYTHONPATH=build/src:$PYTHONPATH python scripts/benchmark.py --mode inference --game chess --batch-sizes 32,64,128,256,512
PYTHONPATH=build/src:$PYTHONPATH python scripts/benchmark.py --mode training --game chess --batch-sizes 256,512,1024,2048,4096
PYTHONPATH=build/src:$PYTHONPATH python scripts/benchmark.py --mode all --game chess
```

## Architecture

```
src/                     C++ engine
  games/chess/           Bitboard chess engine with full move generation
  games/go/              Go engine with Tromp-Taylor scoring
  mcts/                  MCTS with PUCT, virtual loss, tree reuse, arena allocator
  nn/                    LibTorch inference (BF16)
  bindings/              pybind11 bindings -> alphazero_cpp module

python/alphazero/        Python layer
  network/               ResNet + Squeeze-and-Excitation architecture
  training/              Trainer, loss functions, LR scheduling
  pipeline/              Orchestrator (interleaved self-play + training)
  utils/                 Checkpointing, logging

scripts/                 Entry points
  train.py               Training pipeline
  play.py                Interactive play and engine matches
  export_model.py        Model export (TorchScript, ONNX)
  benchmark.py           Performance benchmarking

configs/                 YAML training configurations
specs/                   Design specifications
```

### Neural Network

ResNet with Squeeze-and-Excitation blocks and dual heads:

- **Input**: Board state history (chess: 8x8x119, Go: 19x19x17)
- **Tower**: N residual SE blocks (conv -> BN -> ReLU -> conv -> BN -> SE -> skip)
- **Policy head**: Predicts move probabilities (chess: 4672 actions, Go: 362 actions)
- **Value head**: Predicts game outcome (chess: win/draw/loss softmax, Go: scalar tanh)

### MCTS

800 simulations per move, each following select -> expand -> evaluate -> backup:

- **PUCT selection**: Balances exploitation (Q-value) vs exploration (prior + visit count)
- **Virtual loss**: Prevents parallel threads from all exploring the same path
- **Tree reuse**: After each move, the chosen subtree becomes the new root
- **Dirichlet noise**: Injected at root for exploration during self-play

## Tips

- **Start small**: Use a 10-block/128-filter network with 100-200 simulations to iterate quickly. Scale up once things work.
- **Hardware**: The default config targets NVIDIA DGX Spark with unified memory. On other hardware, reduce `concurrent_games`/`threads_per_game`, or use `fp32` precision if BF16 isn't supported.
- **Chess vs Go**: Chess uses a WDL value head (handles draws); Go uses a scalar value head. Chess encoding is 119 channels; Go is 17.
