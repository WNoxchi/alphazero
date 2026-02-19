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
│   │  K threads│ │  K threads│         │  K threads│              │
│   └─────┬─────┘ └─────┬─────┘        └─────┬─────┘              │
│         │              │                     │                   │
│         └──────────────┼─────────────────────┘                   │
│                        ▼                                         │
│              ┌─────────────────┐                                 │
│              │   Eval Queue    │                                 │
│              └────────┬────────┘                                 │
│                       │                                          │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│           GPU Scheduler (interleaved)                            │
│                       │                                          │
│         ┌─────────────┼─────────────┐                            │
│         ▼                           ▼                            │
│  ┌──────────────┐          ┌──────────────┐                      │
│  │  Inference    │          │   Training   │                      │
│  │  (BF16)       │          │   (BF16 AMP) │                      │
│  │  S batches    │          │   T steps    │                      │
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
│  │    (ring buffer, 1-2M positions)       │                       │
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

## 3. GPU Time-Sharing: Interleaved Scheduling

Since self-play inference and training share one GPU, they are interleaved:

```python
while not done:
    # --- Self-play inference phase ---
    for i in range(S):                              # S inference batches
        batch = eval_queue.collect(batch_size)       # collect from MCTS threads
        policies, values = model.infer(batch)        # GPU: forward only
        dispatch_results(policies, values)           # wake MCTS threads

    # --- Training phase ---
    for j in range(T):                              # T training steps
        states, target_pi, target_z = replay_buffer.sample(train_batch_size)
        loss = train_step(model, states, target_pi, target_z)  # GPU: fwd + bwd + optim
        log_metrics(loss, step)
```

### Scheduling Parameters

| Parameter | Default | Notes |
|---|---|---|
| S (inference batches per cycle) | 50 | ~50ms of inference at 1ms/batch |
| T (training steps per cycle) | 1 | ~5-10ms per training step |
| Inference batch size | 256 | = M * K (concurrent games * threads) |
| Training batch size | 1024 | Scale to 4096 as replay buffer fills |

The **S:T ratio** controls the balance between data generation and learning:
- **Early training** (replay buffer underfull): Increase S to generate more data before training.
- **Steady state**: S=50, T=1 provides a good balance. The GPU spends ~85% of time on inference (feeding self-play) and ~15% on training.
- This ratio is tunable and can be adjusted dynamically.

### Weight Visibility

Training updates model weights in-place (unified memory). The next inference batch automatically uses the updated weights. This is the "slightly stale network" behavior that AlphaZero uses by design — self-play games generated with a marginally older network are still valid training data.

No explicit weight synchronization mechanism is needed on the DGX Spark's cache-coherent unified memory architecture.

## 4. Self-Play Manager

The self-play manager owns and orchestrates all concurrent games.

### Responsibilities

1. Maintain M concurrent game slots, each with its own `GameState` and MCTS tree.
2. Spawn K MCTS worker threads per game.
3. Track simulation counts per game and trigger move selection when the budget (800) is reached.
4. When a game ends, write the game record to the replay buffer and start a new game in that slot.
5. Collect and report self-play metrics.

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
  ├─ When simulation_count >= 800:
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

A **ring buffer** (circular buffer) of fixed capacity in unified memory. Oldest positions are overwritten when the buffer is full.

```cpp
struct ReplayPosition {
    // Board encoding: stored as compressed/raw representation.
    // NOT the full NN input tensor (which includes history) — the tensor is
    // reconstructed on-the-fly during training from sequential positions.
    // For simplicity in v1: store the full encoded NN input tensor.
    float encoded_state[MAX_INPUT_SIZE];   // NN input (e.g., 119*8*8 for chess)

    // Policy target: MCTS visit count distribution.
    float policy[MAX_ACTION_SIZE];          // π(a) for each action

    // Value target: game outcome from current player's perspective.
    float value;                            // z ∈ {-1, 0, +1} for scalar
    float value_wdl[3];                     // [win, draw, loss] for WDL

    // Metadata
    uint32_t game_id;                       // which game this came from
    uint16_t move_number;                   // move within the game
};

class ReplayBuffer {
public:
    // Capacity: number of positions the buffer can hold.
    ReplayBuffer(size_t capacity);

    // Write: called by self-play when a game completes.
    // Thread-safe (multiple games may complete concurrently).
    void add_game(const std::vector<ReplayPosition>& positions);

    // Read: called by training to sample a mini-batch.
    // Returns batch_size positions sampled uniformly at random.
    // Thread-safe (training reads while self-play writes).
    std::vector<ReplayPosition> sample(size_t batch_size);

    // Current fill level.
    size_t size() const;

private:
    std::vector<ReplayPosition> buffer;     // contiguous, in unified memory
    std::atomic<size_t> write_head;
    std::atomic<size_t> count;
    mutable std::shared_mutex mutex;        // readers-writer lock
};
```

### Configuration

| Parameter | Value | Notes |
|---|---|---|
| Capacity | 1,000,000 positions | ~500K games × ~60 moves for chess |
| Sampling | Uniform random | Per AlphaZero paper |
| Minimum fill before training | 10,000 positions | Avoid training on too-small dataset |

### Memory Budget

Per-position memory for chess: `119*8*8*4 (state) + 4672*4 (policy) + 4 (value) = ~49 KB`

1M positions × 49 KB = ~49 GB. This is significant.

**Optimization**: Compress the stored representation:
- Store board state as raw piece positions (not full NN tensor): ~120 bytes for chess, ~400 bytes for Go.
- Store policy as sparse (only legal moves, ~30-40 for chess): ~200 bytes.
- Reconstruct full NN input tensor and dense policy on-the-fly during training.
- Compressed: ~500 bytes per position × 1M = ~500 MB. Fits comfortably.

The v1 implementation can use the uncompressed format for simplicity (49 GB fits in 128 GB), with compression as a clear optimization path.

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

| Action | Frequency | Notes |
|---|---|---|
| Save checkpoint | Every 1,000 training steps | Full model + optimizer state |
| Export folded weights | Every 1,000 training steps | BN-folded model for inference |
| Keep last K checkpoints | K = 10 | Delete older checkpoints to save disk |
| Save "milestone" checkpoint | Every 50,000 steps | Permanent, for Elo tracking |

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

Every N training steps (e.g., 10,000), run a small evaluation match:
- Current network vs. a saved milestone checkpoint
- Fast time control: 100 simulations per move (not 800, for speed)
- 50-100 games
- Estimate Elo difference
- Log as `eval/elo_vs_step_N`

This is purely for monitoring — it does not gate training or select best players. It provides a human-readable measure of training progress.

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
  simulations_per_move: 800
  c_puct: 2.5
  c_fpu: 0.25
  dirichlet_alpha: 0.3           # auto-set from game if not specified
  dirichlet_epsilon: 0.25
  temperature_moves: 30
  concurrent_games: 32
  threads_per_game: 8
  batch_size: 256
  resign_threshold: -0.9
  resign_disable_fraction: 0.1

# Training
training:
  batch_size: 1024
  max_steps: 700000
  lr_schedule:
    - [0, 0.2]
    - [200000, 0.02]
    - [400000, 0.002]
    - [600000, 0.0002]
  momentum: 0.9
  l2_reg: 0.0001
  checkpoint_interval: 1000
  milestone_interval: 50000
  log_interval: 100
  min_buffer_size: 10000

# Pipeline
pipeline:
  inference_batches_per_cycle: 50    # S
  training_steps_per_cycle: 1        # T

# Replay buffer
replay_buffer:
  capacity: 1000000

# Evaluation
evaluation:
  interval_steps: 10000
  num_games: 50
  simulations_per_move: 100

# System
system:
  precision: "bf16"
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
