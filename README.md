# alphazero
C++ Implementation of AlphaZero deeplearning-augmented MCTS

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
# Chess
python scripts/train.py --config configs/chess_default.yaml

# Go
python scripts/train.py --config configs/go_default.yaml

# Resume from checkpoint
python scripts/train.py --config configs/chess_default.yaml --resume checkpoints/step_100000.pt
```

## Play Against a Trained Model

```bash
# Interactive play
python scripts/play.py --game chess --model checkpoints/step_500000.pt --simulations 800

# AI vs Stockfish (100 games)
python scripts/play.py --game chess --model checkpoints/step_500000.pt --opponent stockfish --games 100
```

## Play in the Browser

```bash
# Install web dependencies (once)
pip install -r web/requirements.txt

# Launch the web UI
python -m web.server --model checkpoints/checkpoint_00009000.pt

# Options
python -m web.server \
  --model checkpoints/checkpoint_00009000.pt \
  --simulations 800 \    # MCTS simulations per move (default: 800)
  --device cuda \         # cpu or cuda (default: auto)
  --port 8000             # server port (default: 8000)
```

Then open http://127.0.0.1:8000 in a browser.

## Benchmark

```bash
python scripts/benchmark.py --mode inference --game chess --batch-sizes 32,64,128,256,512
python scripts/benchmark.py --mode training --game chess --batch-sizes 256,512,1024,2048,4096
```

## Monitor

```bash
tensorboard --logdir logs/
```
