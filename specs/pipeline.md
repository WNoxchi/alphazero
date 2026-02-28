# Self-Play and Training Pipeline

## 1. Overview

The pipeline is the orchestration layer that coordinates self-play game generation, neural network inference, training, and data management. It implements an **asynchronous architecture** where self-play and training interleave on a single GPU, connected by a replay buffer in unified memory.

This follows the AlphaZero approach: continuous training from the latest network, no evaluator gate, no best-player selection.

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Self-Play Manager (C++)                                       │
│   ┌──────────┐ ┌──────────┐         ┌──────────┐               │
│   │  Game 0   │ │  Game 1   │  ...    │  Game M   │              │
│   │  1 thread │ │  1 thread │         │  1 thread │              │
│   └─────┬─────┘ └─────┬─────┘        └─────┬─────┘              │
│         │              │                     │                   │
│         └──────────────┼─────────────────────┘                   │
│                        ▼                                         │
│              ┌─────────────────┐                                 │
│              │   Eval Queue    │                                 │
│              └────────┬────────┘                                 │
│                       │                                          │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│           Parallel Pipeline Threads                              │
│                       │                                          │
│         ┌─────────────┼─────────────┐                            │
│         ▼                           ▼                            │
│  ┌──────────────┐          ┌──────────────┐                      │
│  │  Inference    │          │   Training   │                      │
│  │  Thread       │          │   Thread     │                      │
│  │  (BF16)       │          │   (BF16 AMP) │                      │
│  └──────┬───────┘          └──────┬───────┘                      │
│         │                          │                              │
│         ▼                          │                              │
│  ┌──────────────┐                  │                              │
│  │  Results      │                  │                              │
│  │  dispatched   │                  │                              │
│  │  to threads   │                  │                              │
│  └──────────────┘                  │                              │
│                                    │                              │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│               Unified Memory       │                              │
│                                    ▼                              │
│  ┌────────────────────────────────────────┐                       │
│  │          Replay Buffer                 │                       │
│  │    (ring buffer, up to 5M positions)   │                       │
│  │                                        │                       │
│  │   ◄── self-play writes positions       │                       │
│  │   ──► training reads mini-batches      │                       │
│  └────────────────────────────────────────┘                       │
│                                                                   │
│  ┌────────────────────────────────────────┐                       │
│  │          Model Weights                 │                       │
│  │    (single copy, unified memory)       │                       │
│  │                                        │                       │
│  │   ◄── training writes updated weights  │                       │
│  │   ──► inference reads current weights  │                       │
│  └────────────────────────────────────────┘                       │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## 3. GPU Time-Sharing: Parallel Pipeline

Since self-play inference and training share one GPU, they run in **parallel threads** that naturally time-share the GPU via CUDA stream scheduling:

```python
# Inference thread (daemon) — continuously services eval queue
def inference_worker():
    while not stopped:
        eval_queue.process_batch()   # GPU: forward only, dedicated CUDA stream

# Training thread (daemon) — continuously trains from replay buffer
def training_worker():
    while step < max_steps:
        if replay_buffer.size() < min_buffer_size:
            time.sleep(wait_for_buffer_seconds)
            continue
        batch = replay_buffer.sample(train_batch_size)
        loss = train_step(model, batch)   # GPU: fwd + bwd + optim
        log_metrics(loss, step)
        step += 1
```

The main thread coordinates startup, shutdown, checkpointing, and metrics collection. Synchronization between threads uses `threading.Condition`.

### Scheduling Parameters

| Parameter | Chess | Go | Notes |
|---|---|---|---|
| Inference batch size | 384 | 384 | = M × K (concurrent games × threads) |
| Training batch size | 8,192 | 4,096 | Tuned for GPU arithmetic intensity |
| Min buffer size | 16,384 | 8,192 | 2× training batch for sample diversity |

### Shutdown Order

1. Stop `eval_queue` first — this unblocks any MCTS workers waiting in `submit_and_wait()`
2. Join inference thread
3. Stop `self_play_manager`
4. Join training thread
5. Save final checkpoint

### Weight Visibility

Training updates model weights in-place (unified memory). The inference thread automatically uses the updated weights on its next batch. This is the "slightly stale network" behavior that AlphaZero uses by design — self-play games generated with a marginally older network are still valid training data.

No explicit weight synchronization mechanism is needed on the DGX Spark's cache-coherent unified memory architecture.

## 4. Self-Play Manager

The self-play manager owns and orchestrates all concurrent games.

### Responsibilities

1. Maintain M concurrent game slots (M=384), each with its own `GameState` and MCTS tree.
2. Spawn K MCTS worker threads per game (K=1 by default).
3. Track simulation counts per game and trigger move selection when the budget is reached (configurable, with playout cap randomization and dynamic schedule support).
4. When a game ends, write the game record to the replay buffer (via type-erased `AddGameFn` callback) and start a new game in that slot.
5. Collect and report self-play metrics via `SelfPlayMetricsSnapshot`.
6. Support runtime simulation budget updates via `update_simulations_per_move()`.

### Game Lifecycle

```
NEW GAME
  │
  ├─ Initialize GameState (starting position)
  ├─ Reset MCTS arena
  ├─ Add Dirichlet noise to root priors
  │
  ▼
MOVE LOOP
  │
  ├─ K threads run simulations concurrently
  ├─ When simulation_count >= budget (playout cap may reduce this):
  │     ├─ Compute move policy: π(a) ∝ N(root, a)^(1/τ)
  │     ├─ Select move (sample from π if τ=1, argmax if τ→0)
  │     ├─ Store training sample: (state, π, _)  [outcome filled at game end]
  │     ├─ Apply move to game state
  │     ├─ Reuse MCTS subtree (child becomes new root)
  │     ├─ Add Dirichlet noise to new root
  │     └─ Reset simulation count
  │
  ├─ If game is terminal:
  │     ├─ Compute outcome z for each stored position
  │     ├─ Write all (state, π, z) tuples to replay buffer
  │     ├─ Log game metrics (length, outcome, resignation)
  │     └─ Reset slot → NEW GAME
  │
  └─ If move_number > max_game_length:
        ├─ Adjudicate (draw for chess; score by Tromp-Taylor for Go)
        └─ Same as terminal handling
```

### Resignation Logic

```
should_resign = (root_value < resign_threshold) AND
                (best_child_value < resign_threshold)
```

Resignation is **disabled** in a configurable fraction of games (default: 10%) for false positive calibration. These games play to natural termination regardless of evaluation.

## 5. Replay Buffer

### Design

Two implementations are provided:

1. **`ReplayBuffer`** — Dense ring buffer of `ReplayPosition` structs. Simple, used as fallback for non-64-square boards.
2. **`CompactReplayBuffer`** (default for chess and Go) — Compressed ring buffer using bitpacking and sparse policy storage. ~100× memory reduction.

Both are circular buffers in unified memory. Oldest positions are overwritten when the buffer is full.

### ReplayPosition (Dense Format)

```cpp
struct ReplayPosition {
    float encoded_state[MAX_INPUT_SIZE];   // NN input (e.g., 119*8*8 for chess)
    float policy[MAX_ACTION_SIZE];          // π(a) for each action
    float value;                            // z ∈ {-1, 0, +1} for scalar
    float value_wdl[3];                     // [win, draw, loss] for WDL
    float training_weight;                  // sample weight (default 1.0; reduced for playout cap)
    uint32_t game_id;
    uint16_t move_number;
    uint32_t encoded_state_size;            // actual used size
    uint32_t policy_size;                   // actual used size
};
```

### CompactReplayPosition (Compressed Format)

```cpp
struct CompactReplayPosition {
    uint64_t bitpacked_planes[117];        // binary planes as uint64 (1 per plane)
    uint8_t  quantized_float_planes[2];    // constant float planes as uint8
    int16_t  policy_actions[64];           // sparse policy: action indices
    uint16_t policy_probs_fp16[64];        // sparse policy: FP16 probabilities
    uint8_t  num_policy_entries;           // sparse policy size
    float    value, value_wdl[3], training_weight;
    uint32_t game_id;
    uint16_t move_number;
    // shape metadata: num_binary_planes, num_float_planes, policy_size
};
```

The `CompactReplayBuffer` uses `GameConfig::float_plane_indices` to determine which planes are bitpacked (binary) vs. quantized (float). Decompression to dense format happens on-the-fly during sampling.

### Sampling Strategies

| Strategy | Description |
|---|---|
| `kUniform` (default) | Standard uniform random sampling |
| `kRecencyWeighted` | Exponential decay by insertion order: weight ∝ (1 - exp(-λ × age)) |

Recency-weighted sampling biases toward more recent (higher-quality) data. Configurable via `sampling_strategy` and `recency_weight_lambda` in config.

### Configuration

| Parameter | Chess | Go | Notes |
|---|---|---|---|
| Capacity | 5,000,000 | 800,000 | Tuned per game for memory budget |
| Sampling | Uniform | Uniform | Recency-weighted also available |
| Min fill before training | 16,384 | 8,192 | 2× training batch size |

### Memory Budget

**Dense format** (ReplayBuffer): ~49 KB/position for chess. 5M × 49 KB = ~245 GB — too large.

**Compact format** (CompactReplayBuffer): ~300-500 bytes/position. 5M × 500 bytes = ~2.5 GB. Fits comfortably in 128 GB unified memory.

The `CompactReplayBuffer` is automatically selected for boards with 64 squares (chess). It supports binary serialization to/from files via `save_to_file()` / `load_from_file()`.

### Type Erasure: AddGameFn

The `SelfPlayManager` and `SelfPlayGame` accept buffer writes via a type-erased callback:

```cpp
using AddGameFn = std::function<void(const std::vector<ReplayPosition>&)>;
```

This allows writing to either `ReplayBuffer` or `CompactReplayBuffer` (or any custom buffer) without template parameterization.

### Symmetry Augmentation (Go only)

When sampling a mini-batch for Go training, each sampled position is randomly transformed by one of 8 symmetries. This effectively multiplies the dataset by 8× without additional storage. The transform is applied to both the state encoding and the policy vector.

For chess, no symmetry augmentation is applied.

## 6. Training Loop

The training loop runs in Python (PyTorch), reading from the replay buffer and updating the model.

```python
def training_loop(model, replay_buffer, game_config, train_config):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_config.initial_lr,
        momentum=0.9,
        weight_decay=0,  # L2 applied in loss, not optimizer
    )
    scheduler = StepLR(optimizer, milestones=train_config.lr_milestones)
    scaler = torch.amp.GradScaler()
    step = 0

    while step < train_config.max_steps:
        # Wait for sufficient data
        if replay_buffer.size() < train_config.min_buffer_size:
            time.sleep(1)
            continue

        # Sample mini-batch
        batch = replay_buffer.sample(train_config.batch_size)
        states, target_pi, target_z = prepare_batch(batch, game_config)

        # Apply symmetry augmentation (Go only)
        if game_config.supports_symmetry:
            states, target_pi = apply_random_symmetry(states, target_pi)

        # Forward + backward with mixed precision
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            policy_logits, value = model(states)
            loss = compute_loss(
                policy_logits, value,
                target_pi, target_z,
                game_config.value_head_type,
                l2_weight=train_config.l2_reg,
                model=model,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Logging
        if step % train_config.log_interval == 0:
            log_training_metrics(step, loss, optimizer)

        # Checkpointing
        if step % train_config.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step)
            export_folded_weights(model, step)  # BN-folded for inference

        step += 1
```

### Loss Computation

```python
def compute_loss(policy_logits, value, target_pi, target_z, value_type, l2_weight, model):
    # Policy loss: cross-entropy
    # target_pi is a probability distribution (sums to 1 over legal moves, 0 for illegal)
    # policy_logits are raw logits
    policy_loss = -torch.sum(target_pi * F.log_softmax(policy_logits, dim=-1), dim=-1).mean()

    # Value loss
    if value_type == "scalar":
        # MSE between predicted value (tanh output) and target outcome
        value_loss = F.mse_loss(value.squeeze(-1), target_z)
    elif value_type == "wdl":
        # Cross-entropy between predicted WDL and target WDL
        value_loss = -torch.sum(target_z * torch.log(value + 1e-8), dim=-1).mean()

    # L2 regularization (explicit, not via optimizer weight_decay)
    l2_loss = sum(torch.sum(p ** 2) for p in model.parameters())

    total_loss = policy_loss + value_loss + l2_weight * l2_loss
    return total_loss
```

## 7. Checkpointing

### What is Saved

Each checkpoint contains:
- Model weights (PyTorch `state_dict`)
- Optimizer state (for training resumption)
- Training step number
- Learning rate schedule state
- Replay buffer metadata (write head position, count)

### Checkpoint Schedule

| Action | Chess | Go | Notes |
|---|---|---|---|
| Save checkpoint | Every 1,000 steps | Every 1,000 steps | Full model + optimizer state |
| Export folded weights | Every 1,000 steps | Every 1,000 steps | BN-folded model for inference |
| Keep last K checkpoints | K = 10 | K = 10 | Delete older checkpoints to save disk |
| Save "milestone" checkpoint | Every 5,000 steps | Every 25,000 steps | Permanent, for Elo tracking |

### Storage

Checkpoints are saved to the local NVMe SSD (4 TB). A 25M parameter model in FP32 is ~100 MB per checkpoint. With K=10 rolling checkpoints plus milestones, storage is negligible.

## 8. Monitoring and Metrics

### Training Metrics (logged every N steps)

| Metric | Description |
|---|---|
| `loss/total` | Total loss |
| `loss/policy` | Policy cross-entropy loss |
| `loss/value` | Value MSE or cross-entropy loss |
| `loss/l2` | L2 regularization loss |
| `lr` | Current learning rate |
| `throughput/train_steps_per_sec` | Training steps per second |
| `buffer/size` | Current replay buffer fill level |
| `buffer/games_total` | Total games completed |

### Self-Play Metrics (logged per game completion)

| Metric | Description |
|---|---|
| `selfplay/game_length` | Number of moves in the game |
| `selfplay/outcome` | Win/draw/loss from first player's perspective |
| `selfplay/resigned` | Whether the game ended by resignation |
| `selfplay/resign_false_positive` | Whether a disabled-resignation game was recoverable |
| `selfplay/moves_per_second` | Self-play throughput |
| `selfplay/games_per_hour` | Aggregate throughput |
| `selfplay/avg_simulations_per_second` | MCTS simulation throughput |

### Elo Estimation (periodic, non-gating)

A `PeriodicEloEvaluator` runs in a background thread, matching the current network against historical milestone checkpoints:

| Parameter | Chess | Go | Notes |
|---|---|---|---|
| Evaluation interval | 5,000 steps | 10,000 steps | Match against milestones |
| Games per match | 20 | 20 | Reduced from paper for speed |
| Simulations per move | 100 | 100 | Fast time control |

- Discovers milestones by pattern `milestone_XXXXXXXX.pt` in checkpoint directory
- Logs Elo difference via TensorBoard
- Purely for monitoring — does not gate training or select best players

## 9. Configuration

All pipeline parameters are specified in a single configuration file (YAML or similar):

```yaml
# Game selection
game: "chess"                    # or "go"

# Network
network:
  architecture: "resnet_se"
  num_blocks: 20
  num_filters: 256
  se_reduction: 4

# MCTS
mcts:
  simulations_per_move: 200        # chess default; 400 for go
  enable_playout_cap: true         # randomize between full and reduced sims
  reduced_simulations: 50
  full_playout_probability: 0.25
  c_puct: 2.5
  c_fpu: 0.25
  dirichlet_alpha: 0.3            # 0.03 for go
  dirichlet_epsilon: 0.25
  randomize_dirichlet_epsilon: true
  dirichlet_epsilon_min: 0.15
  dirichlet_epsilon_max: 0.35
  temperature_moves: 40            # 30 for go
  concurrent_games: 384
  threads_per_game: 1
  batch_size: 384
  resign_threshold: -0.9
  resign_disable_fraction: 0.1

# Training
training:
  batch_size: 8192                 # 4096 for go
  max_steps: 125000                # 350000 for go
  lr_schedule:
    - [0, 0.2]
    - [7000, 0.02]                 # go: [100000, 0.02]
    - [9000, 0.002]                # go: [200000, 0.002], [300000, 0.0002]
  momentum: 0.9
  l2_reg: 0.0001
  checkpoint_interval: 1000
  milestone_interval: 5000         # 25000 for go
  log_interval: 50
  min_buffer_size: 16384           # 8192 for go
  wait_for_buffer_seconds: 0.01

# Pipeline
pipeline:
  inference_batches_per_cycle: 100
  training_steps_per_cycle: 1

# Replay buffer
replay_buffer:
  capacity: 5000000                # 800000 for go
  sampling_strategy: uniform       # or "recency_weighted"

# Evaluation
evaluation:
  interval_steps: 5000             # 10000 for go
  num_games: 20
  simulations_per_move: 100

# System
system:
  precision: "bf16"
  compile: true                    # enable torch.compile
  num_gpu: 1
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"
```

## 10. Startup and Resumption

### Cold Start

1. Initialize a new neural network with random weights.
2. Create an empty replay buffer.
3. Start self-play games (all positions are effectively random initially).
4. Begin training once `min_buffer_size` positions are accumulated.

### Warm Resume

1. Load the latest checkpoint (model weights, optimizer state, step count).
2. Replay buffer state:
   - **Option A (simple)**: Start with an empty buffer. Self-play quickly fills it with positions from the loaded model's quality level. Some initial training steps may use lower-quality data, but this is transient.
   - **Option B (full resume)**: Persist the replay buffer to disk periodically. Reload on resume. More complex but avoids the cold-start transient.
3. Resume training from the saved step count with the saved learning rate.

Recommendation: **Option A for v1** (simplicity). The replay buffer fills within minutes of self-play. Option B can be added later if training run interruptions are frequent.

## 11. Shutdown

Graceful shutdown procedure:
1. Signal all self-play threads to stop after their current simulation.
2. Wait for all threads to complete.
3. Save a final checkpoint.
4. Flush all metrics to disk.
5. Report final statistics (total games played, training steps completed, estimated Elo).
