# AlphaZero Codebase Review

**Date:** 2026-02-20
**Scope:** Full codebase review against specs/* for critical/functional bugs that would prevent the program from working.

---

## Summary

The codebase is well-structured and largely correct. Six subsystems were reviewed in parallel: chess game logic, Go game logic, MCTS, self-play/replay buffer, Python neural network/training, and build system/configs. Constants (board sizes, input channels, action spaces, value head types) are consistent across C++, Python, YAML configs, and spec documents.

**3 bugs found that could prevent training from working.** Several additional moderate issues documented below.

---

## Critical Bugs

### BUG-1 (HIGH): WDL value loss validation crashes training under BF16 mixed precision

**Files:**
- `python/alphazero/training/loss.py:127` (`_validate_probability_distribution` call)
- `python/alphazero/network/heads.py:85` (softmax computed in BF16)
- `python/alphazero/training/trainer.py:438-442` (autocast context)

**Problem:** The `wdl_value_loss` function calls `_validate_probability_distribution("value", value)` which checks that all WDL probability rows sum to 1.0 within `atol=1e-4`. However, the WDL head computes `torch.softmax(...)` inside the `torch.autocast` BF16 context. BF16 has a 7-bit mantissa (~2.4 decimal digits of precision). For values near 1.0, the representable precision gap is ~0.008 — far larger than the 1e-4 tolerance. A 3-element BF16 softmax output can easily have a sum of 0.998 or 1.002.

**Impact:** Deterministic training crash for chess (WDL mode) with BF16 mixed precision. Raises `ValueError: value rows must sum to 1` and halts training. This will fire on essentially every batch.

**Fix:** Either relax the tolerance (e.g., `atol=1e-2`), cast to FP32 before validation, or remove the runtime validation on model output (softmax output is a valid distribution by construction).

---

### BUG-2 (MEDIUM): L2 regularization applied to ALL parameters including biases

**File:** `python/alphazero/training/loss.py:138-153`

**Problem:** The `l2_regularization_loss` function iterates over `model.parameters()` which includes both weights and biases:

```python
for parameter in model.parameters():
    l2_loss = l2_loss + parameter.to(dtype=torch.float32).pow(2).sum()
```

The spec (`specs/neural-network.md:218`) explicitly states: "L2 regularization applies to all network parameters (weights, not biases)."

**Impact:** Adds unnecessary regularization pressure on bias terms, pushing them toward zero. This subtly degrades training quality — biases should be free to take any value the loss landscape requires.

**Fix:** Filter to weight parameters only:
```python
for name, parameter in model.named_parameters():
    if 'bias' not in name:
        l2_loss = l2_loss + parameter.to(dtype=torch.float32).pow(2).sum()
```

---

### BUG-3 (MEDIUM): `torch.load` missing `weights_only` parameter

**File:** `python/alphazero/utils/checkpoint.py:317`

**Problem:**
```python
payload = torch.load(path, map_location=map_location)
```

Starting with PyTorch 2.6 (Feb 2025), `weights_only` defaults to `True`, which rejects anything that isn't a pure tensor/dict/list/primitive. The checkpoint payload likely contains objects (e.g., optimizer state with custom types) that fail this check. On PyTorch 2.2–2.5, this emits a `FutureWarning`.

**Impact:** On PyTorch >= 2.6 (which `pyproject.toml` permits via `torch>=2.2`), checkpoint **resume crashes** with `UnpicklingError`. Cold-start training works fine — only resume-from-checkpoint is broken.

**Fix:** Add explicit parameter:
```python
payload = torch.load(path, map_location=map_location, weights_only=False)
```

---

## Moderate Issues

### MOD-1: Go liberty count double-counting in `analyze_board`

**File:** `src/games/go/go_rules.cpp:142-166`

**Problem:** The `liberty_seen_for_root` array uses a single `int` per intersection to track which group root has counted a given liberty. When stones from two different groups (different roots) are adjacent to the same empty intersection, and iteration order interleaves them, a liberty can be counted twice for one group.

**Concrete example:**
```
Row 0: B . W    (B=black group A, W=white group B)
Row 1: B B .
```
Empty at (0,1) is adjacent to black at (0,0), white at (0,2), and black at (1,1). Processing order: intersection 0 sets `liberty_seen[(0,1)] = root_A`, intersection 2 overwrites to `root_B`, intersection 20 sees `root_B != root_A` and counts the liberty again for group A.

**Impact:** Liberty counts are overcounted (never undercounted). Game play correctness is **not affected** for two reasons: (1) capture/self-capture checks only test `liberties == 0`, and overcounting can only increase counts above the true value; (2) a group with truly 0 liberties has no adjacent empty intersections at all, so the counting loop body is never entered for that group — the count stays at exactly 0 regardless of this bug. The bug only inflates counts for groups that already have 1+ liberties. No impact on training.

**Fix:** Process liberties per-group rather than per-stone, or use a per-intersection set of roots.

---

### MOD-2: `advance_root` leaks old root node in non-arena NodeStore path

**File:** `src/mcts/mcts_search.cpp:425-437`

**Problem:** In the `else` branch of `advance_root` (when the node store is not `ArenaNodeStore`), sibling subtrees are released but the old root node itself is never released. The arena path handles this correctly via `release_single_node(old_root)`.

**Impact:** Currently unreachable — the code always uses `ArenaNodeStore` (the `dynamic_cast` succeeds). Would leak one node per move if a `TranspositionNodeStore` or other `NodeStore` implementation is added.

**Fix:** Add `node_store_.release_subtree(old_root)` after releasing siblings in the else branch.

---

### MOD-3: Go SGF serialization produces invalid output for interleaved setup stones

**File:** `src/games/go/go_state.cpp:683-702`

**Problem:** When the initial position has interleaved black and white stones (by intersection order), the SGF writer can append a stone's coordinate to the wrong property. Example: black at (0,0), white at (0,1), black at (0,2) produces `AB[as]AW[bs][cs]` where `[cs]` is incorrectly attributed to white.

**Impact:** Only affects SGF import/export. Normal self-play starts from an empty board. No impact on training.

**Fix:** Collect all black and white setup coordinates into separate lists, then write each property at once.

---

## Performance Issues (non-blocking)

### PERF-1: Thread creation/destruction per move in self-play

**File:** `src/selfplay/self_play_game.cpp:198-222`

`run_simulation_batch()` creates K threads (default 8) for every move, then joins them. With 800 sims/move, 60 moves/game, 32 games → ~15,360 thread creations per batch. A thread pool would eliminate this overhead.

### PERF-2: `release_subtree` iterates over all `kMaxActions` children

**File:** `src/mcts/arena_node_store.cpp:70`

The loop iterates over all 362 (Go) or 218 (Chess) children entries even when only ~30 are legal. Should iterate up to `num_actions` only.

### PERF-3: `advance_root` resets `root_expanded_` unnecessarily for reused subtrees

**File:** `src/mcts/mcts_search.cpp:440`

After tree reuse, `root_expanded_` is set to false even though the reused child is already expanded. This causes one unnecessary `GameState::clone()` per move.

---

## Verified Correct

The following critical aspects were verified against the spec and found to be **correct**:

**Chess:**
- Board representation: `pieces[2][6]` (color x piece_type) bitboards
- Input encoding: 8x8x119 tensor (14 planes x 8 history + 7 constant planes)
- Board flipped for black perspective via `orient_square_for_side`
- Action encoding: `[0, 4672)`, formula `from_square * 73 + move_type_index`, mirrored for black
- Move generation: sliding pieces (ray tables), knights/kings (precomputed), pawns (push/capture/EP/promotion), castling (rights + empty + not-attacked checks), pseudo-legal filtered for legality
- Terminal conditions: checkmate, stalemate, 50-move rule, threefold repetition, insufficient material, max game length (512)
- History: T=8 positions, copy-on-apply, zero-filled for missing steps
- Config: WDL value head, Dirichlet alpha 0.3, action space 4672

**Go:**
- Board representation: `uint8_t board[19][19]`, union-find for groups
- Capture logic: correct order (remove opponent groups first, then check self-capture)
- Ko detection and prohibition
- Superko: positional superko using board-only Zobrist hash history
- Scoring: Tromp-Taylor rules correctly implemented
- Input encoding: 19x19x17 tensor (2 planes x 8 history + 1 constant plane)
- Action encoding: `[0, 362)`, `row*19+col` for placement, 361 for pass
- Terminal: two consecutive passes OR move_number >= 722
- Symmetry: 8 dihedral transforms for both board tensor and policy vector
- Config: scalar value head, Dirichlet alpha 0.03

**MCTS:**
- PUCT formula: `Q + c_puct * P * sqrt(N_total) / (1 + N)` — correct
- Virtual loss: apply during select (+1 visit, -1 value), revert during backup — correct
- Backup: value negated at each level for alternating players — correct
- FPU: `Q_fpu = V(parent) - c_fpu * sqrt(sum_visited_priors)` — correct
- Dirichlet noise: applied at root only, once per move — correct
- Temperature: tau=1 for moves 1-30, argmax for 31+ — correct
- Tree reuse: child becomes root, siblings released — correct
- Eval queue: MPSC with batch collection and flush timeout — correct

**Self-play / Replay Buffer:**
- Game lifecycle: 800 sims → compute pi → select move → store sample → terminal → write to buffer
- Training samples: (state, pi, z) with z filled at game end from current player's perspective
- Replay buffer: ring buffer, shared_mutex, uniform sampling
- Resignation: both root_value and best_child_value < threshold, disabled in 10% of games

**Neural Network:**
- ResNet + SE architecture matches spec (SE with scale+bias variant)
- Policy head: Conv1x1(F,32) → BN → ReLU → Flatten → Linear(32*H*W, action_space_size)
- Value head scalar: Conv1x1(F,1) → BN → ReLU → Flatten → Linear(H*W, 256) → ReLU → Linear(256,1) → Tanh
- Value head WDL: same but Linear(256,3) → Softmax
- Weight init: Kaiming for conv/linear, Xavier for SE FC, zero for final heads
- BN folding: correct formula `W_folded = W * gamma / sqrt(var + eps)`

**Build System:**
- All source files in CMakeLists.txt exist on disk
- Optional dependencies handled gracefully (Torch, pybind11, CUDA, GTest)
- Constants consistent across C++, Python, YAML configs, and specs
- pyproject.toml dependencies correct
- Test targets properly defined (17 C++ test files, all present)
