# Monte-Carlo Tree Search (MCTS)

## 1. Overview

MCTS is the search algorithm that converts the neural network's raw policy and value estimates into much stronger move selections. Each move in self-play is selected by running MCTS simulations (configurable; defaults: 200 for chess, 400 for Go), where each simulation traverses the game tree, evaluates a leaf position with the neural network, and backpropagates the result.

The implementation uses **hybrid parallelism**: root parallelism across concurrent games, tree parallelism within each game, and an asynchronous evaluation queue for batched GPU inference.

## 2. MCTS Algorithm

### Per-Simulation Steps

Each simulation consists of four phases:

#### 2.1 Select

Starting from the root node, traverse the tree by selecting at each node the action that maximizes the PUCT score:

```
a* = argmax_a [ Q(s, a) + U(s, a) ]

U(s, a) = c_puct * P(s, a) * sqrt(N_total(s)) / (1 + N(s, a))
```

Where:
- `Q(s, a)`: Mean action-value (average value of all simulations that passed through this edge)
- `P(s, a)`: Prior probability from the neural network
- `N(s, a)`: Visit count for this edge
- `N_total(s) = Σ_b N(s, b)`: Total visit count for the parent node
- `c_puct`: Exploration constant (default: 2.5, tunable)

Selection continues until reaching a **leaf node** (a node with unexpanded children or a terminal game state).

#### 2.2 Expand

When a leaf node is reached (a position not yet evaluated by the NN):
1. The leaf position is submitted to the **evaluation queue** for neural network inference.
2. The thread blocks until the result is available.
3. Upon receiving `(policy, value)` from the NN:
   - Create child edges for all legal actions.
   - Initialize each edge: `N(s,a)=0, W(s,a)=0, Q(s,a)=0, P(s,a)=p_a` where `p_a` is the masked and renormalized policy prior.

If the leaf is a **terminal state**, no NN evaluation is needed. The terminal value is used directly for backup.

#### 2.3 Evaluate

Neural network inference is batched across all concurrent threads and games (see Section 4: Evaluation Queue). The NN returns:
- `policy`: Logits over all actions (illegal actions masked, then softmax applied)
- `value`: Scalar in [-1, 1] (for scalar head) or WDL probabilities (for WDL head)

For WDL, convert to scalar for tree backup: `v = win - loss`.

#### 2.4 Backup

Propagate the value `v` back up the path from leaf to root. For each edge `(s, a)` traversed during selection:

```
N(s, a) += 1
W(s, a) += v
Q(s, a) = W(s, a) / N(s, a)
```

The value `v` is negated at each level (since players alternate): the current player's win is the opponent's loss.

```
v_backup = v                    (at leaf)
v_backup = -v_backup            (at each parent level)
```

## 3. Edge Data Structure

### Struct of Arrays (SoA) Layout

Each MCTS node uses SoA layout for SIMD-friendly PUCT computation:

```cpp
template<int MaxActions>
struct MCTSNodeT {
    // --- Edge statistics (SoA for vectorized PUCT) ---
    int32_t   visit_count[MaxActions];     // N(s, a)
    float     total_value[MaxActions];     // W(s, a)
    float     mean_value[MaxActions];      // Q(s, a) = W/N
    float     prior[MaxActions];           // P(s, a) from NN

    // --- Node metadata ---
    int16_t   actions[MaxActions];         // Legal action indices
    int16_t   num_actions;                 // Number of legal actions
    int32_t   total_visits;                // Σ N(s, a) (cached)
    float     node_value;                  // V(s): NN value of this node

    // --- Tree structure ---
    NodeId    children[MaxActions];        // Child node handles (NULL_NODE if unexpanded)
    NodeId    parent;                      // Parent node handle
    int16_t   parent_action;              // Action that led to this node from parent

    // --- Virtual loss tracking ---
    int32_t   virtual_loss[MaxActions];    // In-flight virtual losses per edge
};

// Game-specific specializations:
using ChessMCTSNode = MCTSNodeT<218>;   // max legal chess moves
using GoMCTSNode    = MCTSNodeT<362>;   // 19*19 + pass
using MCTSNode      = GoMCTSNode;       // universal alias
```

`MaxActions` is a compile-time template parameter per game type:
- Chess: 218 (theoretical maximum legal moves)
- Go: 362 (19*19 + pass)

The `NodeStore`, `ArenaNodeStore`, and `MctsSearch` classes are similarly templatized (`NodeStoreT<NodeType>`, `ArenaNodeStoreT<NodeType>`, `MctsSearchT<NodeType>`). A `RuntimeMctsSearch` wrapper uses `std::variant` for runtime game-type dispatch.

### NodeId Abstraction

Nodes are referenced by `NodeId` (a `uint32_t` index) rather than raw pointers, enabling future swapping of the backing store:

```cpp
using NodeId = uint32_t;
constexpr NodeId NULL_NODE = UINT32_MAX;
```

## 4. Node Store

### Interface

```cpp
class NodeStore {
public:
    virtual ~NodeStore() = default;

    // Allocate a new node, returns its ID.
    virtual NodeId allocate() = 0;

    // Access a node by ID.
    virtual MCTSNode& get(NodeId id) = 0;
    virtual const MCTSNode& get(NodeId id) const = 0;

    // Release a subtree rooted at the given node (mark memory as reusable).
    virtual void release_subtree(NodeId root) = 0;

    // Reset the entire store (e.g., when a game ends).
    virtual void reset() = 0;

    // Statistics (for monitoring).
    virtual size_t nodes_allocated() const = 0;
    virtual size_t memory_used_bytes() const = 0;
};
```

### Arena Implementation (Default)

```cpp
class ArenaNodeStore : public NodeStore {
    // Pre-allocated contiguous array of MCTSNode.
    std::vector<MCTSNode> arena;
    uint32_t next_free;

    // Arena capacity is set at construction.
    // For 800 sims/move with tree reuse, ~2000-5000 nodes per game is sufficient.
    // With 32 concurrent games: ~160K nodes total.
    static constexpr size_t DEFAULT_CAPACITY = 8192;  // per game
};
```

**Allocation**: Bump pointer (`arena[next_free++]`). O(1).

**Tree reuse**: After playing a move, the chosen child's subtree is preserved. Sibling subtrees are released. The arena is compacted or a simple free-list is maintained for released nodes.

**Reset**: Set `next_free = 0`. O(1). Used when a game ends.

### Future: Transposition Table Implementation

A `TranspositionNodeStore` would use a hash map keyed by position hash (`GameState::hash()`). Multiple parent edges can point to the same node, forming a DAG. This requires:
- Reference counting to determine when a node is unreachable.
- Careful handling during tree reuse (subtree release must not free shared nodes).
- The `NodeId` abstraction enables this swap without changing MCTS logic.

## 5. Hybrid Parallelism

### Architecture

```
M concurrent games (root parallelism)  ×  K threads per game (tree parallelism)
= M * K total threads feeding one evaluation queue

Current defaults:
  M = 384 games
  K = 1 thread per game
  Total = 384 threads
  GPU batch size = 384

All values are configurable.
```

In practice, K=1 (pure root parallelism) performs best on the DGX Spark: NN eval latency dominates over MCTS tree traversal, so additional tree parallelism threads add CPU overhead without improving GPU utilization. High M keeps the eval queue full and smooths GPU utilization.

### Root Parallelism (Game Level)

- M self-play games run concurrently, each with its own MCTS tree and game state.
- Games are independent — no shared state between them.
- When a game terminates, its data is written to the replay buffer and a new game begins immediately in that slot.
- Root parallelism provides natural batch diversity and pipeline stability.

### Tree Parallelism (Per-Game Level)

- Within each game, K threads explore the same MCTS tree concurrently.
- Each thread performs: select → submit leaf to eval queue → wait → backup.
- **Virtual loss** prevents threads from exploring the same path (see Section 6).
- K > 1 is supported but currently not used (K=1 is preferred, see above).

### Synchronization

- Each game's MCTS tree is accessed only by its K threads. No cross-game synchronization needed.
- Within a game, edge statistics are protected by fine-grained locks:
  - Per-node `shared_ptr<std::mutex>` for node-level locking during select/backup.
  - `virtual_loss`: `int32_t` (protected by node mutex).
- The evaluation queue is thread-safe (see Section 7).

## 6. Virtual Loss

Virtual loss is applied during the **select** phase to discourage multiple threads from traversing the same path. When a thread selects an edge `(s, a)`:

### Apply (during select, before submitting leaf)

```cpp
virtual_loss[a] += 1;
visit_count[a] += 1;        // inflate visit count
total_visits += 1;
total_value[a] -= 1.0f;     // assume loss
// Recompute Q for PUCT:
mean_value[a] = total_value[a] / visit_count[a];
```

### Revert (during backup, after receiving NN value)

```cpp
virtual_loss[a] -= 1;
visit_count[a] -= 1;        // remove inflated count
total_visits -= 1;
total_value[a] += 1.0f;     // remove assumed loss
// Then apply real backup as usual
```

The virtual loss value of -1 (assume loss) biases Q(s,a) downward, pushing other threads toward unexplored branches. With K=8 threads, the maximum virtual loss on any edge is 8, which is small relative to typical visit counts after a few simulations.

## 7. Evaluation Queue

The evaluation queue decouples MCTS tree traversal (CPU) from neural network inference (GPU), enabling fully asynchronous operation.

### Design

```cpp
class EvalQueue {
public:
    // Called by MCTS threads. Submits a leaf position for NN evaluation.
    // Blocks until the result is available (the batch containing this
    // position has been processed by the GPU).
    //
    // Returns: (policy_logits, value) for this position.
    EvalResult submit_and_wait(const float* encoded_state);

    // Called by the GPU inference thread. Collects pending requests
    // into a batch and runs inference.
    void process_batch();

private:
    struct PendingRequest {
        const float* encoded_state;        // input (in unified memory)
        EvalResult* result;                // output slot
        std::counting_semaphore<1>* done;  // signal when result is ready
    };

    // Thread-safe queue of pending requests.
    // Implementation: lock-free MPSC (multi-producer, single-consumer) queue,
    // or a simple mutex-protected deque.
    ConcurrentQueue<PendingRequest> pending;

    // Flush triggers:
    size_t batch_size = 256;               // flush when this many requests are pending
    std::chrono::microseconds timeout{100}; // flush partial batch after timeout
};
```

### Flow

1. MCTS thread encodes the leaf position into a buffer (in unified memory).
2. Thread calls `submit_and_wait()`, which enqueues the request and blocks on a semaphore.
3. The GPU inference thread monitors the queue:
   - When `pending.size() >= batch_size`, flush immediately.
   - When `timeout` has elapsed since the first pending request, flush a partial batch.
4. The GPU thread collects the batch, stacks the inputs into a contiguous tensor, and runs `NeuralNetInference::infer()`.
5. Results are written to each request's output slot, and semaphores are signaled.
6. MCTS threads wake up and proceed with expansion and backup.

### Flush Timeout

The timeout prevents the last few threads from waiting indefinitely for a full batch (e.g., when most games have finished a move and few simulations remain). The partial batch may underutilize the GPU but avoids tail latency.

### Unified Memory Advantage

The encoded state buffers and result buffers are in unified memory. The GPU reads input and writes output directly — no explicit `cudaMemcpy`. This simplifies the queue implementation and eliminates transfer latency.

## 8. First Play Urgency (FPU)

When evaluating the PUCT score for an unvisited action (N(s,a) = 0), the Q-value is undefined (no visits). FPU provides a default Q-value.

### Leela-Style FPU Reduction (Chosen Approach)

```
Q_fpu(s) = V(parent) - c_fpu * sqrt(Σ_{visited a} P(s, a))

Where:
  V(parent)          = the NN value of this node (stored in node_value)
  c_fpu              = 0.2 to 0.4 (default: 0.25, tunable)
  Σ visited P(s, a)  = sum of priors for already-visited actions
```

Intuition: unvisited actions are assumed to be somewhat worse than the current evaluation, proportional to how much of the policy has already been explored. This encourages the search to first try high-prior actions, while still exploring low-prior actions when the high-prior ones prove disappointing.

For unvisited actions, the PUCT formula uses `Q_fpu(s)` in place of `Q(s, a)`.

## 9. Dirichlet Noise

To ensure exploration, Dirichlet noise is added to the prior probabilities at the **root node only**:

```
P(root, a) = (1 - ε) * p_a + ε * η_a

Where:
  p_a       = neural network policy prior for action a
  η ~ Dir(α) = Dirichlet noise vector
  ε         = 0.25 (or randomized per-game from a range, see below)
  α         = 0.3 (chess) or 0.03 (Go)
```

The Dirichlet α is scaled inversely with the typical number of legal moves: `α ≈ 10 / avg_legal_moves`. This ensures a similar level of exploration regardless of action space size.

Noise is sampled **once per move** (when the root changes), not per simulation.

### Randomized Dirichlet Epsilon (Optional)

When `randomize_dirichlet_epsilon` is enabled, each game samples ε uniformly from `[dirichlet_epsilon_min, dirichlet_epsilon_max]` instead of using a fixed value. This increases opening diversity. Chess defaults: `[0.15, 0.35]`.

## 10. Temperature and Move Selection

After completing all simulations, a move is selected from the root based on visit counts:

```
π(a) ∝ N(root, a)^(1/τ)
```

### Temperature Schedule

| Move number | Temperature τ | Selection behavior |
|---|---|---|
| 1 – N | 1.0 | Proportional to visit count (stochastic, ensures diverse openings) |
| N+1 and beyond | → 0 (effectively: argmax) | Select most-visited move (deterministic, strongest play) |

N = `temperature_moves` (chess: 40, Go: 30).

When τ → 0, this becomes greedy selection of the most-visited action. In practice, implement this as `argmax(N(root, a))` rather than computing the power.

The search probability vector `π` (with temperature applied) is stored as the **policy training target** for this position.

## 11. Tree Reuse

After selecting and playing a move `a*`:

1. The child node `children[a*]` becomes the new root.
2. All sibling subtrees (children of the old root except `a*`) are released via `NodeStore::release_subtree()`.
3. The new root retains its edge statistics from the previous search.
4. The next search starts with pre-existing knowledge, potentially saving many simulations.

The `parent` pointer of the new root is set to `NULL_NODE`.

### Benefits

- Simulations from the previous move that explored lines after `a*` carry forward.
- Opponent's likely responses already have visit counts and Q-values.
- Effectively increases the simulation budget per move without additional NN evaluations.

## 12. Resignation

To save computation, a player resigns when the position is clearly lost:

```
Resign if:
  V(root) < v_resign  AND  max_child_V < v_resign

Where:
  V(root)       = neural network value of the root position
  max_child_V   = maximum Q-value among root's children
  v_resign      = resignation threshold (configurable, e.g., -0.9)
```

Both conditions must be met: the position must be evaluated as lost, and no explored continuation is promising.

### Calibration

To prevent premature resignations (false positives), disable resignation in a fraction of self-play games (e.g., 10%) and play until natural termination. Track how often a resigned game would have been won. Adjust `v_resign` to keep the false positive rate below 5%.

## 13. MCTS Configuration Summary

| Parameter | Chess | Go | Notes |
|---|---|---|---|
| Simulations per move | 200 | 400 | Tunable; reduced from AlphaZero paper's 800 for faster data generation |
| Playout cap | enabled | — | 25% full (200), 75% reduced (50 sims); see below |
| c_puct | 2.5 | 2.5 | Exploration constant; tunable |
| c_fpu | 0.25 | 0.25 | FPU reduction constant; tunable |
| Dirichlet α | 0.3 | 0.03 | ~10 / avg_legal_moves |
| Dirichlet ε | 0.25 (randomized 0.15–0.35) | 0.25 | Root noise weight; chess uses per-game randomization |
| Temperature moves | 40 | 30 | Stochastic selection for first N moves |
| Concurrent games (M) | 384 | 384 | High M keeps eval queue full; tunable |
| Threads per game (K) | 1 | 1 | Eval latency dominates; K=1 saves CPU for more games |
| GPU batch size | 384 | 384 | = M × K; tunable |
| Arena capacity (per game) | 8,192 nodes | 2,048 nodes | Go reduced to save memory (~22 GB savings) |
| Resign threshold | -0.9 | -0.9 | Tunable; calibrate via false positive rate |
| Resign disable fraction | 10% | 10% | For calibration |
| Max game length | 512 | 722 (19*19*2) | Draw/score after this |

### Playout Cap Randomization

When `enable_playout_cap` is true, each move randomly selects between the full simulation budget and a reduced budget:
- With probability `full_playout_probability` (default 0.25): use full `simulations_per_move` (200)
- Otherwise: use `reduced_simulations` (default 50)

Positions generated with reduced simulations receive a proportionally lower `training_weight` in the replay buffer. This accelerates self-play data generation while maintaining training signal quality.

### Dynamic Simulation Schedule

The simulation budget can be adjusted at runtime via `SelfPlayManager::update_simulations_per_move()`. The current training script uses a step-based schedule:
- Steps 0–10,000: 100 simulations/move (fast early exploration)
- Steps 10,000+: 200 simulations/move (higher quality data)

## 14. Pseudocode: Complete MCTS Simulation

```
function mcts_simulate(root, game_state, eval_queue):
    node = root
    path = []               // list of (node, action) pairs traversed
    state = game_state

    // --- SELECT ---
    while node is expanded and not state.is_terminal():
        action = select_action_puct(node)
        apply_virtual_loss(node, action)
        path.append((node, action))
        state = state.apply_action(action)
        child = node.children[action]
        if child == NULL_NODE:
            break            // reached unexpanded child
        node = child

    // --- EVALUATE ---
    if state.is_terminal():
        value = state.outcome(state.current_player())
    else:
        encoded = state.encode()
        policy, value = eval_queue.submit_and_wait(encoded)

        // --- EXPAND ---
        child_id = node_store.allocate()
        child = node_store.get(child_id)
        legal = state.legal_actions()
        child.num_actions = legal.size()
        for i, a in enumerate(legal):
            child.actions[i] = a
            child.prior[i] = masked_softmax(policy, legal)[i]
            child.visit_count[i] = 0
            child.total_value[i] = 0
            child.mean_value[i] = 0
            child.children[i] = NULL_NODE
        child.node_value = value
        child.parent = node_id
        child.parent_action = action
        node.children[action_idx] = child_id

    // --- BACKUP ---
    v = value
    for (node, action) in reversed(path):
        revert_virtual_loss(node, action)
        v = -v                           // negate for alternating players
        node.visit_count[action] += 1
        node.total_visits += 1
        node.total_value[action] += v
        node.mean_value[action] = node.total_value[action] / node.visit_count[action]
```
