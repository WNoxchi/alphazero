# Go Training Improvements Plan

## Context

Config-only changes have already been applied to `configs/go.yaml`:
- `c_puct`: 2.5 → 1.5 (ELF OpenGo / Leela Zero convention)
- Playout cap randomization enabled (400 full / 100 reduced, p=0.25)
- Replay buffer: 800K → 5M positions (~6 GB compact storage)
- LR schedule milestones scaled to 700K steps (DeepMind AZ default)
- `c_fpu`: 0.25 → 0.2 (KataGo in-tree value; root FPU needs code change below)

This plan covers the remaining improvements that require code changes.

---

## Prioritized Tasks

### TASK-001: Add separate root vs. in-tree FPU reduction

- **Files**: `src/mcts/mcts_search.h`, `src/mcts/mcts_search.cpp`, `src/bindings/python_bindings.cpp`
- **Current state**: COMPLETE (2026-02-28)
- **Priority**: HIGH — directly affects search quality; KataGo showed root FPU=0.0 significantly
  improves exploration at the root when Dirichlet noise is applied.
- **Rationale**: The MCTS implementation uses a single `c_fpu` value for all nodes. KataGo uses
  `c_fpu = 0.2` in-tree but `c_fpu = 0.0` at root (when Dirichlet noise is enabled). The reasoning:
  at the root node, Dirichlet noise already provides exploration, so FPU reduction on unvisited
  children is counterproductive — it penalizes unexplored moves that the noise is trying to
  encourage visiting. In-tree, FPU reduction remains valuable to bias search toward the parent's
  value estimate for unvisited children.

- **Current code** (`src/mcts/mcts_search.cpp`, `select_action_slot`):
  ```cpp
  const float fpu = compute_fpu_value(node, config_.c_fpu);
  ```
  This is called for every node (root and in-tree) with the same `config_.c_fpu`.

- **Fix**:
  1. Add `c_fpu_root` field to `SearchConfig` in `src/mcts/mcts_search.h` (default -1.0 meaning
     "use c_fpu", so existing behavior is preserved):
     ```cpp
     struct SearchConfig {
         // ... existing fields ...
         float c_fpu = 0.25F;
         float c_fpu_root = -1.0F;  // if >= 0, used at root instead of c_fpu
         // ...
     };
     ```
  2. The `select_action_slot` function needs to know whether it's operating on the root. The
     simplest approach: add a `bool is_root` parameter to `select_action_slot`. The caller
     (`run_simulation` or equivalent) already knows if it's at the root — it's the first step of
     the selection phase. Pass `is_root = true` for the root call and `false` for all subsequent
     calls in the tree traversal.
  3. In `select_action_slot`, choose the FPU value:
     ```cpp
     const float effective_fpu = (is_root && config_.c_fpu_root >= 0.0F)
         ? config_.c_fpu_root : config_.c_fpu;
     const float fpu = compute_fpu_value(node, effective_fpu);
     ```
  4. Add `c_fpu_root` to `SelfPlayGameConfig` in `src/selfplay/self_play_game.h` (mirrors
     `SearchConfig`).
  5. Propagate `c_fpu_root` through config loading in Python (`python/alphazero/config.py` or
     wherever `SearchConfig` is populated from YAML).
  6. Expose `c_fpu_root` in pybind11 bindings (`src/bindings/python_bindings.cpp`).
  7. Add to `configs/go.yaml`:
     ```yaml
     c_fpu_root: 0.0    # KataGo: no FPU reduction at root (Dirichlet noise handles exploration)
     ```
     Leave `configs/chess.yaml` unchanged (will use default -1.0, preserving current behavior).

- **Acceptance criteria**:
  1. `SearchConfig` has `c_fpu_root` field with default -1.0 (backward compatible)
  2. When `c_fpu_root >= 0`, it is used at the root node; otherwise `c_fpu` is used everywhere
  3. Config loading from YAML correctly populates `c_fpu_root`
  4. Existing MCTS tests pass unchanged (default -1.0 preserves old behavior)
  5. New test: a search with `c_fpu_root = 0.0` should produce FPU = node_value (no reduction)
     at the root, while in-tree nodes still get reduction
  6. `cmake --build build --target alphazero_cpp -j$(nproc)` succeeds
  7. `ctest --test-dir build --output-on-failure -R "MctsSearchTest\\."` passes
  8. Python binding tests pass

- **Testing guidance**:
  - Add a C++ unit test in `tests/cpp/test_mcts.cpp` that:
    a. Creates a search with `c_fpu = 0.2` and `c_fpu_root = 0.0`
    b. Runs a small search (e.g., 10 sims)
    c. Verifies root node's FPU calculation uses 0.0 (no reduction from parent value)
    d. Verifies a child node's FPU calculation uses 0.2
  - The test should document WHY root FPU=0 matters (interaction with Dirichlet noise)

- **Implementation notes (2026-02-28)**:
  - Added `c_fpu_root` to `SearchConfig` and `SelfPlayGameConfig` with default `-1.0`.
  - `select_action_slot` now accepts `is_root` and applies `c_fpu_root` only at root when set.
  - Wired `c_fpu_root` through pybind (`SearchConfig`, `SelfPlayGameConfig`) and `scripts/train.py`
    YAML loading (`mcts.c_fpu_root`).
  - Set `mcts.c_fpu_root: 0.0` in `configs/go.yaml`; chess configs unchanged.
  - Added regression coverage in `tests/cpp/test_mcts.cpp` for root-vs-in-tree FPU behavior and
    Python wiring assertions in `tests/python/test_bindings.py` and `tests/python/test_train_script.py`.
  - Validation run: `cmake --build build --target alphazero_cpp -j$(nproc)`,
    `cmake --build build --target alphazero_cpp_tests -j$(nproc)`,
    `ctest --test-dir build --output-on-failure -R "MctsSearchTest\\."`,
    `ctest --test-dir build --output-on-failure -R "SelfPlayGameTest\\."`,
    `PYTHONPATH=build/src:$PYTHONPATH /home/hakan/miniconda3/envs/alphazero/bin/python -m pytest tests/python/test_bindings.py tests/python/test_train_script.py`.
  - No additional follow-up tasks discovered while implementing TASK-001.

---

### TASK-002: Add dynamic Dirichlet alpha scaling

- **Files**: `src/mcts/mcts_search.cpp`, `src/mcts/mcts_search.h`, `src/bindings/python_bindings.cpp`
- **Current state**: PENDING
- **Priority**: MEDIUM — improves exploration calibration across positions with varying numbers of
  legal moves. Currently the Dirichlet noise concentration is the same whether there are 10 or
  350 legal moves, which means early-game positions (many legal moves) get nearly uniform noise
  while late-game positions (few legal moves) get highly concentrated noise.
- **Rationale**: KataGo scales Dirichlet alpha as `base_alpha * board_area / num_legal_moves`
  (for Go: `0.03 * 361 / N_legal`). This keeps the expected noise per-move roughly constant
  regardless of how many legal moves exist. DeepMind used a fixed alpha (0.03 for Go, 0.3 for
  chess) which implicitly assumes `~10/alpha` legal moves on average.

  The scaling formula `alpha = base_alpha * reference_moves / actual_moves` where
  `reference_moves` is the game's average legal move count (361 for Go, ~30 for chess) ensures:
  - When `actual_moves ≈ reference_moves`: alpha ≈ base_alpha (matches DeepMind)
  - When `actual_moves < reference_moves`: alpha > base_alpha (more noise per move to maintain
    exploration pressure)
  - When `actual_moves > reference_moves`: alpha < base_alpha (less noise per move to avoid
    washing out the policy signal)

- **Current code** (`src/mcts/mcts_search.cpp`, `dirichlet_alpha()`):
  ```cpp
  float MctsSearchT<NodeType>::dirichlet_alpha() const {
      if (config_.dirichlet_alpha_override > 0.0F) {
          return config_.dirichlet_alpha_override;
      }
      return game_config_.dirichlet_alpha;
  }
  ```
  This returns a fixed value regardless of the position.

- **Fix**:
  1. Add a `dynamic_dirichlet_alpha` bool to `SearchConfig` (default `false` for backward
     compatibility):
     ```cpp
     struct SearchConfig {
         // ... existing fields ...
         bool dynamic_dirichlet_alpha = false;
         // ...
     };
     ```
  2. Add a `dirichlet_alpha_reference_moves` field to the game config (or compute it from action
     space size). For Go this is 361 (board area). For chess, 30 is typical. This can be set as a
     constant in the game config structs (`go_config.cpp`, `chess_config.cpp`):
     ```cpp
     // go_config.cpp
     dirichlet_alpha_reference_moves = 361;
     // chess_config.cpp
     dirichlet_alpha_reference_moves = 30;
     ```
     Add the field to the `GameConfig` base (likely in `game_config.h` or wherever the game
     config struct is defined — search for `dirichlet_alpha` definition).
  3. Modify `dirichlet_alpha()` to accept the number of legal moves and scale accordingly:
     ```cpp
     float MctsSearchT<NodeType>::dirichlet_alpha(int num_legal_moves) const {
         float base = config_.dirichlet_alpha_override > 0.0F
             ? config_.dirichlet_alpha_override
             : game_config_.dirichlet_alpha;
         if (config_.dynamic_dirichlet_alpha && num_legal_moves > 0) {
             base = base * static_cast<float>(game_config_.dirichlet_alpha_reference_moves)
                  / static_cast<float>(num_legal_moves);
         }
         return base;
     }
     ```
  4. Update the caller of `dirichlet_alpha()` to pass the root node's `num_actions` (which equals
     the number of legal moves at the root). Search for where Dirichlet noise is applied — likely
     in a function like `apply_dirichlet_noise` or at the start of search after root expansion.
  5. Add `dynamic_dirichlet_alpha` to config loading and pybind11 bindings.
  6. Add to `configs/go.yaml`:
     ```yaml
     dynamic_dirichlet_alpha: true
     ```
     Leave `configs/chess.yaml` unchanged (default `false`, preserving current behavior).

- **Acceptance criteria**:
  1. `SearchConfig` has `dynamic_dirichlet_alpha` bool (default false)
  2. Game configs have `dirichlet_alpha_reference_moves` (361 for Go, 30 for chess)
  3. When disabled: behavior is identical to current (fixed alpha)
  4. When enabled: alpha = `base_alpha * reference_moves / num_legal_moves`
  5. Existing tests pass unchanged (feature is off by default)
  6. New test: with dynamic alpha enabled and Go config (base=0.03, ref=361):
     - 361 legal moves → alpha ≈ 0.03
     - 100 legal moves → alpha ≈ 0.108
     - 10 legal moves → alpha ≈ 1.083
  7. `cmake --build build --target alphazero_cpp -j$(nproc)` succeeds
  8. MCTS tests pass

- **Testing guidance**:
  - Add a C++ unit test in `tests/cpp/test_mcts.cpp` that verifies the scaling formula
  - Test edge cases: 1 legal move, very large number of legal moves
  - Document WHY the scaling matters (noise concentration vs. number of moves)

---

## Reference: DeepMind / KataGo / Leela Zero Comparison

For agents implementing these tasks, here are the key hyperparameter references:

| Parameter | DeepMind AZ (Go) | KataGo | Leela Zero | This Project |
|---|---|---|---|---|
| c_puct | ~1.25 (log formula) | 1.1 (dynamic) | ~1.5 | 1.5 |
| c_fpu (in-tree) | unpublished | 0.2 | — | 0.2 |
| c_fpu (root) | unpublished | 0.0 | — | 0.0 (TASK-001) |
| Dirichlet alpha | 0.03 (fixed) | 0.03*361/N (dynamic) | 0.03 (fixed) | dynamic (TASK-002) |
| Dirichlet epsilon | 0.25 | 0.25 | 0.25 | 0.25 |
| Simulations | 800 | 600-2000 | 1,600 | 400 (playout cap) |
| Playout cap | no | yes (p=0.25) | no | yes (p=0.25) |
| Replay buffer | 1M games | 250K-22M samples | 500K games | 5M positions |
| Training batch | 4,096 | 256 | 96 | 4,096 |
| LR schedule | 0.2→0.02→0.002→0.0002 | per-sample 6e-5 | varies | 0.2→0.02→0.002→0.0002 |
| Value head | scalar | scalar | scalar | scalar |
| Symmetry aug | yes (8x D4) | yes | yes | yes (already implemented) |
