# Task 05: Config Tuning for Maximum Throughput

**Priority**: Medium — easy wins from config changes alone
**Expected impact**: ~2x self-play throughput, ~15-25% better GPU utilization during training
**Dependencies**: None
**Difficulty**: Low (config change only, no code)

## Background

Read `specs/*` for the full AlphaZero specification, and `notes/gpu_optimization.md` for the bottleneck analysis motivating this change.

The current `configs/chess_1hr.yaml` uses `simulations_per_move: 400` and `training.batch_size: 4096`. At the current training stage (step ~10K, average game length 18 moves, effectively random play), the network is too weak for high-quality MCTS search to matter. More positions at lower quality per position beats fewer high-quality positions for early training.

## File to Modify

### `configs/chess_1hr.yaml`

**Current config** (relevant sections):
```yaml
mcts:
  simulations_per_move: 400
  concurrent_games: 384
  threads_per_game: 1
  batch_size: 384

training:
  batch_size: 4096
  max_steps: 10500
  min_buffer_size: 8192
```

## Required Changes

```yaml
mcts:
  simulations_per_move: 200   # was 400 — 2x faster self-play for early training
  concurrent_games: 384       # unchanged — well-matched with batch_size
  threads_per_game: 1         # unchanged
  batch_size: 384             # unchanged — matches concurrent_games

training:
  batch_size: 8192            # was 4096 — better GPU arithmetic intensity
  max_steps: 10500            # unchanged for 1hr config
  min_buffer_size: 16384      # was 8192 — 2x training batch for sample diversity
```

## Rationale

### `simulations_per_move: 400 → 200`

Each MCTS move requires `simulations_per_move` neural network evaluations. Halving this:
- Halves the GPU inference work per move → games complete ~2x faster
- Doubles self-play throughput (games/hr and positions/hr)
- Reduces MCTS search quality per position, but at step 10K with random-level play (18-move games, W/D/L: 53/12/35%), the network isn't strong enough for deep search to help

The AlphaZero paper used 800 sims/move for the final strong network. During early training when the network is weak, fewer simulations is standard practice. The search quality matters more in later training when the network has learned meaningful patterns.

For long runs (100K+ steps), consider increasing back to 400-800 sims/move after ~50K steps when game lengths increase and play quality improves.

### `training.batch_size: 4096 → 8192`

Larger training batches have better **arithmetic intensity** — more useful computation per kernel launch and memory access. On the GB10 GPU:
- 4096 samples: moderate GPU utilization during forward/backward
- 8192 samples: better utilization, each training step processes more data

The tradeoff is that each training step takes longer wall-clock time (roughly 1.5-1.8x, not 2x, because GPU kernel efficiency improves with larger batches). But it processes 2x more data per step, so net positions-processed-per-second improves.

The replay buffer (750K capacity) currently has ~78K positions and is growing at ~19K positions/hr. At 8192 batch size, each training batch samples ~10% of the buffer — this is sufficient diversity (standard AlphaZero uses 1-5% sampling ratios).

### `min_buffer_size: 8192 → 16384`

Set to 2x the training batch size. This ensures that when training begins, each sampled position has at most a 50% chance of appearing in the same training batch. Without this, the first few training steps would oversample the small initial buffer.

## What NOT to Change

- `concurrent_games: 384` — already well-matched with `batch_size: 384`. After each inference batch, all 384 threads wake up and quickly resubmit, keeping the queue full.
- `threads_per_game: 1` — with 1 thread per game and 384 games, eval latency dominates. Multi-threading per game adds MCTS tree parallelism overhead without GPU benefit.
- `batch_size` (mcts): 384 — matches concurrent_games for optimal queue filling.
- `replay_buffer.capacity: 750000` — appropriate for long training runs where the buffer should hold a diverse window of recent play.

## For Long Training Runs

If extending `max_steps` beyond 10.5K to 100K+ steps, consider a staged approach:

**Early training (steps 0-50K)**: Use aggressive throughput settings
```yaml
mcts:
  simulations_per_move: 200
training:
  batch_size: 8192
```

**Mid training (steps 50K-200K)**: Increase search quality as network improves
```yaml
mcts:
  simulations_per_move: 400
training:
  batch_size: 4096
```

**Late training (steps 200K+)**: Full search quality for final polish
```yaml
mcts:
  simulations_per_move: 800
training:
  batch_size: 4096
```

This can be implemented by creating separate config files (`chess_early.yaml`, `chess_mid.yaml`, `chess_late.yaml`) and resuming from checkpoints with the appropriate config.

## Verification

1. Run training with the updated config:
   ```bash
   python scripts/train.py --config configs/chess_1hr.yaml
   ```

2. Check self-play throughput in the training log output:
   - Baseline: ~1072 games/hr
   - Expected: ~2000+ games/hr (from halved simulations)

3. Check training throughput:
   - Baseline: ~0.52 steps/sec
   - Expected: ~0.35-0.45 steps/sec (steps are slower with 8192 batch, but each step processes 2x data)
   - Net positions/sec should increase

4. Monitor loss convergence — the loss curve should look similar when plotted against **total positions trained on** (not steps). Since each step now processes 2x positions, the loss should decrease at roughly the same rate per-position.

5. Check that buffer fills appropriately:
   - `min_buffer_size: 16384` means training starts after ~16K positions accumulate
   - With 2000+ games/hr at ~20 moves each, this takes ~30 seconds
