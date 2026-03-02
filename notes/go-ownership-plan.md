# Go Training: Pass Restriction & Auxiliary Ownership Head

## Why This Work Is Needed

Go training has collapsed into a **degenerate equilibrium**: the network learned to pass
immediately on move 1, both players pass, the game ends, and white wins by komi on an empty
board. Self-play shows avg game length of 1 move and W/D/L of 8/0/92%. The replay buffer is
flooded with trivial 1-move games that reinforce this behavior.

This is a known bootstrap problem for single-machine AlphaZero Go training. DeepMind avoided it
through sheer compute (5,000 TPUs generating enough diverse games that the random policy never
converged on passing). KataGo solved it structurally by adding **auxiliary training targets**
(ownership prediction) that give the network rich spatial gradients from the start — the network
learns "what territory looks like" before it needs to play well, which naturally prevents the
degenerate equilibrium.

This plan addresses the problem in two phases:
1. **Immediate fix**: Forbid pass before move 30 (unblocks training now)
2. **Structural fix**: Add an ownership prediction head (KataGo's most impactful innovation)

---

## Prioritized Tasks

### TASK-001: Forbid pass before move 30 in Go

- **Files**: `src/games/go/go_state.h`, `src/games/go/go_rules.cpp`, `tests/cpp/test_go_state.cpp`
- **Current state**: COMPLETE (2026-03-02)
- **Priority**: CRITICAL — training is stuck in a degenerate equilibrium and cannot make progress
  until this is fixed. The training run must be restarted after this change.
- **Rationale**: In real Go, passing before move 30 is never correct play — there are always
  productive moves available in the opening and early midgame. DeepMind's temperature schedule
  (τ=1 for the first 30 moves) implicitly recognizes moves 1-30 as the "opening phase." Removing
  pass from the legal action space during this phase eliminates the degenerate double-pass
  equilibrium without affecting the quality of learned play. This is a permanent rule that aligns
  with Go's structure, not a temporary hack.

- **Current code** — `src/games/go/go_rules.cpp`, `play_pass()` (line ~200):
  ```cpp
  MoveResult play_pass(const GoPosition& position) {
      // ... validation ...
      GoPosition next = position;
      next.move_number = position.move_number + 1;
      next.consecutive_passes = position.consecutive_passes + 1;
      // ... return legal result ...
  }
  ```
  Pass is always legal regardless of move number. All legality checks flow through
  `play_action()` → `play_pass()`, so this is the single point of control.

- **Fix**:
  1. Add a constant to `src/games/go/go_state.h`:
     ```cpp
     constexpr int kMinPassMove = 30;
     ```
  2. In `play_pass()` (`src/games/go/go_rules.cpp`), add an early rejection:
     ```cpp
     MoveResult play_pass(const GoPosition& position) {
         if (position.move_number < kMinPassMove) {
             return illegal_result(position, kPassAction, MoveStatus::kPassTooEarly);
         }
         // ... rest unchanged ...
     }
     ```
  3. Add `kPassTooEarly` to the `MoveStatus` enum (`src/games/go/go_rules.h`). Search for the
     existing enum definition and add the new variant.
  4. Update `move_status_to_string()` to handle the new variant.

- **Acceptance criteria**:
  1. `legal_actions()` does not include `kPassAction` (index 361) for positions with
     `move_number < 30`
  2. `is_legal_action(position, kPassAction)` returns false when `move_number < 30`
  3. `apply_action(kPassAction)` throws when `move_number < 30`
  4. Pass is legal at `move_number >= 30` (unchanged behavior)
  5. Two consecutive passes at move 30+ still terminate the game normally
  6. All existing Go tests pass (some may need adjustment if they assume pass is legal early)
  7. `cmake --build build --target alphazero_cpp -j$(nproc)` succeeds
  8. `ctest --test-dir build --output-on-failure -R "GoStateTest\\."` passes

- **Testing guidance**:
  - Add tests in `tests/cpp/test_go_state.cpp`:
    a. Verify pass is illegal at move 0, 15, 29
    b. Verify pass is legal at move 30, 31, 100
    c. Verify that `legal_actions()` at move 0 contains exactly 361 actions (all board
       intersections, no pass) instead of 362
    d. Verify a game can still terminate via double pass after move 30
  - Check existing tests — any test that applies pass before move 30 needs to be updated
    (e.g., by playing 30 dummy moves first, or by constructing a position with
    `move_number >= 30`)
  - Document WHY: "Prevents degenerate double-pass equilibrium in early self-play training"

- **Completion notes (2026-03-02)**:
  - Implemented `kMinPassMove = 30`, added `MoveStatus::kPassTooEarly`, and enforced the new
    legality check in `play_pass()`.
  - Updated `move_status_to_string()` and Go tests to reflect the opening-phase pass restriction.
  - Updated SGF serialization tests to build actual 30+ move histories before pass moves because
    SGF import/export does not encode a non-zero starting `move_number`.
  - Validation run:
    - `cmake --build build --target alphazero_cpp -j$(nproc)`
    - `cmake --build build --target alphazero_cpp_tests -j$(nproc)`
    - `ctest --test-dir build --output-on-failure -R "(GoStateTest|GoRulesEngineTest|GoSerializationTest)\\."`
    - `ctest --test-dir build --output-on-failure -R "GoStateTest\\."`
    - `python3 -m compileall python` (`ruff` and `mypy` unavailable in this environment)

---

### TASK-002: Compute and store ownership targets in self-play

- **Files**: `src/games/go/scoring.h`, `src/games/go/scoring.cpp`,
  `src/selfplay/replay_buffer.h`, `src/selfplay/compact_replay_buffer.h`,
  `src/selfplay/compact_replay_buffer.cpp`, `src/selfplay/replay_compression.h`,
  `src/selfplay/replay_compression.cpp`, `src/selfplay/self_play_game.cpp`,
  `src/bindings/python_bindings.cpp`, `tests/cpp/test_go_state.cpp`,
  `tests/cpp/test_replay_buffer.cpp`
- **Current state**: COMPLETE (2026-03-02)
- **Priority**: HIGH — prerequisite for TASK-003; without ownership data in the replay buffer,
  the ownership head cannot be trained.
- **Rationale**: KataGo's ownership prediction head requires a per-intersection training target
  for every position in the replay buffer. At game end, Tromp-Taylor scoring already determines
  which player owns each intersection. This target is the same for all positions in a game — the
  network learns to predict the eventual territorial outcome from any intermediate board state.

  The existing `compute_tromp_taylor_score()` function (`src/games/go/scoring.cpp`) computes
  aggregate point totals but does not return a per-intersection ownership map. This task extends
  the scoring code to produce that map, adds storage for it in both replay buffer formats, and
  propagates it through self-play game completion.

- **Fix — Part A: Ownership map computation**:

  1. Add a new function to `src/games/go/scoring.h`:
     ```cpp
     // Per-intersection ownership for training targets.
     // Values: +1.0 = black territory, -1.0 = white territory, 0.0 = neutral/dame.
     void compute_tromp_taylor_ownership(
         const GoPosition& position,
         float* out_ownership);  // Must point to kBoardArea (361) floats
     ```
  2. Implement in `src/games/go/scoring.cpp`. The logic mirrors `compute_tromp_taylor_score()`
     but writes per-intersection values instead of accumulating totals:
     - Black stone → +1.0
     - White stone → -1.0
     - Empty region reachable only by black → +1.0
     - Empty region reachable only by white → -1.0
     - Empty region reachable by both (dame) → 0.0
     The `analyze_empty_region()` helper already exists and returns `EmptyRegionInfo` with
     `reaches_black`, `reaches_white`, and the set of intersections. Reuse it.

- **Fix — Part B: Replay buffer storage**:

  3. Add ownership field to `ReplayPosition` (`src/selfplay/replay_buffer.h`):
     ```cpp
     static constexpr std::size_t kMaxBoardArea = 361U;  // 19*19
     std::array<float, kMaxBoardArea> ownership{};
     std::uint16_t ownership_size = 0U;  // 0 = no ownership data (chess), 361 = Go
     ```
  4. Add compact ownership storage to `CompactReplayPosition`:
     ```cpp
     // Ownership bitpacked as two planes: black_owns and white_owns.
     // Each plane uses ceil(board_area / 64) words, same as state bitpacking.
     static constexpr std::size_t kMaxOwnershipWords = 12U;  // 2 planes * 6 words for 19x19
     std::array<std::uint64_t, kMaxOwnershipWords> bitpacked_ownership{};
     std::uint16_t num_ownership_words = 0U;  // 0 = no ownership data
     ```
     The ownership encoding: plane 0 = "black owns" (bit set if ownership > 0),
     plane 1 = "white owns" (bit set if ownership < 0). Neutral intersections have both
     bits unset. This is lossless for the 3-state ownership values.
  5. Add compression/decompression helpers to `src/selfplay/replay_compression.h`:
     ```cpp
     void compress_ownership(
         std::span<const float> ownership,      // kBoardArea floats (+1/-1/0)
         std::size_t board_area,
         std::span<std::uint64_t> out_bitpacked);  // 2 * words_per_plane words

     void decompress_ownership(
         std::span<const std::uint64_t> bitpacked,
         std::size_t board_area,
         std::span<float> out_ownership);          // kBoardArea floats
     ```
  6. Update `CompactReplayBuffer` to compress/decompress ownership in `add_game()`, `sample()`,
     `sample_batch()`, `export_positions()`, and `import_positions()`.
  7. Bump compact replay file version (2 → 3) to include ownership words in the header/data.
     Add backward compatibility for version 2 files (assume `num_ownership_words = 0`).

- **Fix — Part C: Self-play propagation**:

  8. In `src/selfplay/self_play_game.cpp`, after game completion (lines 225-244), compute
     ownership from the terminal position and attach it to every `ReplayPosition`:
     ```cpp
     // After determining game outcome:
     std::array<float, kMaxBoardArea> ownership{};
     std::uint16_t ownership_size = 0U;
     if (game_config.compute_ownership) {
         compute_tromp_taylor_ownership(final_go_state.position(), ownership.data());
         ownership_size = kBoardArea;  // 361
     }
     // In the loop creating replay_positions:
     position.ownership = ownership;
     position.ownership_size = ownership_size;
     ```
     The `compute_ownership` flag should be part of `SelfPlayGameConfig` (default false,
     backward compatible). Set it to true for Go in the config loading path.
  9. Add `compute_ownership` to `SelfPlayGameConfig` (`src/selfplay/self_play_game.h`).
  10. Expose `compute_ownership` in pybind11 bindings and thread it through
      `scripts/train.py` YAML loading.

- **Fix — Part D: Python bindings**:

  11. Update `sample_batch` bindings to return ownership data as an additional numpy array
      when ownership is present. The return type changes from a 4-tuple to a 5-tuple:
      `(states, policies, values, weights, ownership)`. When no ownership data exists
      (chess), the ownership array should be empty (size 0).
  12. Update `export_buffer` / `import_buffer` similarly.

- **Acceptance criteria**:
  1. `compute_tromp_taylor_ownership()` produces correct per-intersection maps
  2. Ownership round-trips through compact replay buffer (compress → store → decompress)
  3. Chess replay buffer is unaffected (ownership_size = 0, no storage cost)
  4. Self-play games for Go produce ownership targets for all positions
  5. Python `sample_batch` returns ownership data
  6. Existing tests pass, compact replay version 2 files still load
  7. `cmake --build build --target alphazero_cpp -j$(nproc)` succeeds
  8. All replay buffer and Go state tests pass

- **Testing guidance**:
  - Test `compute_tromp_taylor_ownership()` on known board positions:
    a. Empty board → all 0.0 (dame, reaches both colors)
    b. Board with enclosed black territory → enclosed intersections = +1.0
    c. Board with enclosed white territory → enclosed intersections = -1.0
    d. Board with stones but no territory → stones are +1/-1, empty is 0.0
  - Test compact round-trip: create Go game, add to compact buffer, sample, verify ownership
  - Test chess round-trip: verify ownership_size = 0 and no extra storage
  - Document WHY: "Ownership targets teach the network spatial territory understanding,
    which is KataGo's most impactful training innovation"

- **Completion notes (2026-03-02)**:
  - Added `compute_tromp_taylor_ownership()` and validated ownership maps for stones,
    enclosed territory, and neutral regions.
  - Extended replay payloads (`ReplayPosition`, `CompactReplayPosition`, `SampledBatch`) to
    carry ownership with compact two-plane bitpacking plus decompress/compress helpers.
  - Threaded ownership through dense + compact replay sampling/export/import paths and added
    compact replay file format v3 with backward compatibility for v1/v2 files.
  - Added `SelfPlayGameConfig.compute_ownership`, propagated terminal Go ownership into replay
    rows when enabled, and enabled it by default for Go in `scripts/train.py`.
  - Updated pybind replay APIs:
    - `sample_batch` now returns `(states, policies, values, weights, ownership)`
    - `export_buffer` now appends ownership as a sixth return array
    - `import_buffer` accepts optional ownership input
  - Validation run:
    - `cmake --build build --target alphazero_cpp -j$(nproc)`
    - `cmake --build build --target alphazero_cpp_tests -j$(nproc)`
    - `ctest --test-dir build --output-on-failure -R "(GoScoringTest|GoRulesEngineTest|ReplayBufferTest|CompactReplayBufferTest|ReplayCompressionTest|SelfPlayGameTest)\\."`
    - `python3 -m pytest tests/python/test_bindings.py tests/python/test_training.py tests/python/test_train_script.py tests/python/test_checkpoint_utils.py`
    - `python3 -m compileall python scripts/train.py tests/python/test_bindings.py tests/python/test_training.py tests/python/test_train_script.py tests/python/test_checkpoint_utils.py`
    - `ruff` and `mypy` were unavailable in this environment.

---

### TASK-003: Add ownership prediction head and training loss

- **Files**: `python/alphazero/network/heads.py`, `python/alphazero/network/resnet_se.py`,
  `python/alphazero/network/base.py`, `python/alphazero/training/trainer.py`,
  `python/alphazero/training/loss.py`, `python/alphazero/config.py`,
  `python/alphazero/pipeline/orchestrator.py`, `python/alphazero/utils/checkpoint.py`,
  `configs/go.yaml`, `scripts/train.py`,
  `tests/python/test_network.py`, `tests/python/test_training.py`,
  `python/alphazero/pipeline/evaluation.py`, `scripts/play.py`,
  `tests/python/test_loss.py`, `tests/python/test_orchestrator.py`,
  `tests/python/test_checkpoint_utils.py`, `tests/python/test_config.py`
- **Current state**: COMPLETE (2026-03-02)
- **Priority**: HIGH — the structural fix for Go training quality. Ownership prediction gives
  the network 361 spatial gradient signals per position instead of a single scalar, which
  dramatically accelerates learning and prevents degenerate equilibria.
- **Rationale**: KataGo showed that auxiliary ownership prediction is one of the single largest
  efficiency gains for Go self-play training — removing it caused a **1.65x training slowdown**
  (190 Elo loss) in ablation studies at 2.5B training queries. The network backbone already
  computes spatial features; the ownership head just reads them off to predict per-intersection
  territory. This costs minimal extra compute (one 1x1 conv vs. 20 residual blocks) but provides
  vastly richer gradients during the critical early training phase.

  The loss function uses **binary cross-entropy with logits** (not MSE), following KataGo's
  proven approach. BCE has better gradient properties than MSE+tanh: at extreme target values
  (+1/-1), tanh saturates and gradients vanish, while BCE with logits provides consistent
  gradient signal regardless of prediction confidence. The ownership targets {-1, 0, +1} are
  mapped to BCE probabilities {0, 0.5, 1} via `(1 + target) / 2`, and the relationship
  `sigmoid(2x) = (1 + tanh(x)) / 2` bridges the logit and ownership domains.

- **Fix — Part A: Network architecture**:

  1. Add `OwnershipHead` class to `python/alphazero/network/heads.py`:
     ```python
     class OwnershipHead(nn.Module):
         """Predicts per-intersection ownership logits for Go training.

         Outputs raw logits (pre-sigmoid). During training, these are passed to
         BCE loss which handles the sigmoid internally. To convert to ownership
         predictions: ownership = tanh(logits), mapping to [-1, +1].
         """
         def __init__(self, num_filters: int) -> None:
             super().__init__()
             self.conv = nn.Conv2d(num_filters, 1, kernel_size=1)
             # Initialize small so ownership loss doesn't dominate early training.
             # Follows KataGo's approach of initializing aux spatial convs at small scale.
             nn.init.normal_(self.conv.weight, std=0.01)
             nn.init.zeros_(self.conv.bias)

         def forward(self, x: torch.Tensor) -> torch.Tensor:
             return self.conv(x).flatten(1)  # (batch, H*W) raw logits
     ```
     This is deliberately minimal — a single 1x1 convolution reads spatial features from the
     backbone and projects to one ownership logit per intersection. No BN, no activation
     function. The backbone's final ReLU already provides nonlinearity; the loss function
     (BCE with logits) handles the output nonlinearity implicitly. No FC layers are needed
     because ownership is inherently spatial (unlike value, which is a global scalar).

     **Why no tanh output**: The loss uses `F.binary_cross_entropy_with_logits`, which is
     numerically stable because it never computes `sigmoid` explicitly. If we applied `tanh`
     in the model, we'd need to invert it for BCE, losing numerical stability. KataGo follows
     the same pattern — their ownership conv outputs raw logits with no activation.

  2. Add `ownership_head` to `ResNetSE` (`python/alphazero/network/resnet_se.py`):
     - Constructor: create `OwnershipHead` only when `game_config.supports_ownership` is true
       (Go = true, chess = false). Store as `self.ownership_head: OwnershipHead | None`.
     - Forward: return 3-tuple when ownership head exists, 2-tuple otherwise:
       ```python
       def forward(self, x):
           features = self._backbone(x)
           policy_logits = self.policy_head(features)
           value = self.value_head(features)
           if self.ownership_head is not None:
               ownership = self.ownership_head(features)
               return policy_logits, value, ownership
           return policy_logits, value
       ```

  3. Update `AlphaZeroNetwork` base class (`python/alphazero/network/base.py`) to document
     the variable return type. The base class's shape validation should handle both 2-tuple
     and 3-tuple returns.

- **Fix — Part B: Training integration**:

  4. Add `ownership_loss()` to `python/alphazero/training/loss.py`:
     ```python
     def ownership_loss(
         predicted_logits: torch.Tensor,  # (batch, board_area) raw logits
         target: torch.Tensor,            # (batch, board_area) in {-1, 0, +1}
         weights: torch.Tensor,           # (batch,)
     ) -> torch.Tensor:
         """BCE ownership loss following KataGo's approach.

         Maps targets from {-1, 0, +1} to {0, 0.5, 1} for BCE.
         Scales logits by 2.0 so that tanh(logit) <-> sigmoid(2*logit).
         """
         target_probs = (1.0 + target) / 2.0           # {-1,0,+1} -> {0, 0.5, 1}
         scaled_logits = predicted_logits * 2.0         # tanh <-> sigmoid scaling
         per_point = F.binary_cross_entropy_with_logits(
             scaled_logits, target_probs, reduction='none'
         )                                              # (batch, board_area)
         per_sample = per_point.mean(dim=-1)            # average over intersections
         return (per_sample * weights).mean()
     ```
     **Why BCE over MSE**: MSE with tanh output suffers from vanishing gradients when the
     model is confident (tanh saturates as `x → ±∞`, so `d/dx tanh(x) → 0`). BCE with
     logits provides consistent gradient signal regardless of confidence level. The 2x logit
     scaling maps between domains: `sigmoid(2x) = (1 + tanh(x)) / 2`. This is KataGo's
     proven approach — they use exactly this formulation with a loss weight of 1.5.

  5. Update `train_one_step()` in `python/alphazero/training/trainer.py`:
     - The model forward pass at **line 712** currently reads:
       ```python
       policy_logits, predicted_value = model(states)
       ```
       Change to handle the optional 3rd output:
       ```python
       model_output = model(states)
       if len(model_output) == 3:
           policy_logits, predicted_value, ownership_pred = model_output
       else:
           policy_logits, predicted_value = model_output
           ownership_pred = None
       ```
     - After the existing loss computation, add ownership loss:
       ```python
       if ownership_pred is not None and ownership_target is not None:
           loss_own = ownership_loss(ownership_pred, ownership_target, weights)
           total_loss = policy_loss + value_loss + ownership_weight * loss_own + l2_loss
       ```
     - The `ownership_weight` hyperparameter controls relative importance. KataGo uses
       **1.5** — ownership is weighted *more* than value (1.2) because 361 per-intersection
       gradients are individually noisier but collectively far more informative than a single
       scalar. The per-sample ownership loss is averaged over intersections, so its magnitude
       is comparable to a single-scalar loss (both ~0.7 at initialization), and the 1.5
       weight makes it the dominant learning signal — by design. Use **1.5** as default.
     - Log `loss_ownership` in the training metrics dict alongside existing losses.

  6. Update `sample_replay_batch_tensors()` in `python/alphazero/training/trainer.py`
     (lines 480-510) to include ownership targets from the replay buffer sample. Currently
     returns a 4-tuple `(states, target_policy, target_value, sample_weights)`. When ownership
     data is present in the replay buffer (TASK-002 adds this as a 5th array in `sample_batch`),
     return it as a 5th tensor `(batch, board_area)` on the training device. When `ownership_size
     = 0` (chess), return `None` for the ownership tensor — the loss computation will be skipped
     via the `ownership_target is None` check.

- **Fix — Part C: Configuration**:

  7. Add `supports_ownership: bool` to the Python `GameConfig` dataclass
     (`python/alphazero/config.py`). Set `True` for Go, `False` for chess.

  8. Add `ownership_loss_weight: float` to `TrainingConfig` (default 0.0 = disabled).
     A value > 0 enables ownership training.

  9. Add to `configs/go.yaml`:
     ```yaml
     training:
       ownership_loss_weight: 1.5   # KataGo-style auxiliary ownership loss (BCE)
     ```
     Leave chess config unchanged (default 0.0, ownership head not created).

- **Fix — Part D: Inference path compatibility**:

  10. Update the inference callback in `python/alphazero/pipeline/orchestrator.py`. The batch
      evaluator closure built by `make_eval_queue_batch_evaluator()` unpacks the model output
      at **line 378**:
      ```python
      policy_logits, value = model(model_inputs)   # BREAKS with 3-tuple
      ```
      Change to handle the optional ownership output:
      ```python
      model_output = model(model_inputs)
      if len(model_output) == 3:
          policy_logits, value, _ = model_output  # Discard ownership during inference
      else:
          policy_logits, value = model_output
      ```
      **Why this is critical**: Without this change, the inference thread will crash with
      `ValueError: too many values to unpack` the moment a Go model with an ownership head
      is used. The ownership output is purely a training signal — MCTS only needs policy
      logits and value. The discarded ownership tensor is immediately freed by PyTorch.

- **Fix — Part E: Checkpoint backward compatibility**:

  11. Update `load_checkpoint()` in `python/alphazero/utils/checkpoint.py`. The current code
      at **line 440** uses:
      ```python
      model.load_state_dict(payload["model_state_dict"])  # strict=True by default
      ```
      This will fail when loading an old checkpoint (without ownership head weights) into a
      new model (with ownership head). Change to:
      ```python
      missing, unexpected = model.load_state_dict(
          payload["model_state_dict"], strict=False
      )
      # Log missing keys (expected: ownership_head.conv.weight, ownership_head.conv.bias)
      if missing:
          logger.info(f"Checkpoint missing keys (new head initialized randomly): {missing}")
      if unexpected:
          logger.warning(f"Checkpoint has unexpected keys: {unexpected}")
      ```
      This allows:
      - Loading old checkpoints → ownership head uses its small random initialization
      - Loading new checkpoints → all weights loaded normally
      - Loading new checkpoints into old model → unexpected keys logged as warning

      Add a `strict` parameter to `load_checkpoint()` (default `False` for backward
      compatibility) so callers can opt into strict loading if desired.

- **Acceptance criteria**:
  1. `OwnershipHead` produces `(batch, board_area)` raw logits (unbounded, not [-1, +1])
  2. `OwnershipHead` weights initialized small (std=0.01, bias=0)
  3. `ResNetSE` creates ownership head only for Go (when `supports_ownership=True`)
  4. Forward pass returns 3-tuple for Go, 2-tuple for chess
  5. Ownership loss uses BCE with logit scaling (2x), not MSE
  6. Ownership loss weight defaults to 1.5 for Go
  7. Training metrics include `loss_ownership`
  8. Chess training is completely unaffected (no ownership head, no ownership loss)
  9. Ownership weight of 0.0 disables the loss even if the head exists
  10. Inference callback in orchestrator handles 3-tuple without crashing
  11. Old checkpoints (without ownership head) load successfully into new model
  12. All existing tests pass
  13. `PYTHONPATH=build/src:$PYTHONPATH python -m pytest tests/python/` passes

- **Testing guidance**:
  - Test `OwnershipHead` output shape: input `(2, 256, 19, 19)` → output `(2, 361)`
  - Test `OwnershipHead` output range: values are unbounded raw logits (NOT in [-1, +1])
  - Test `OwnershipHead` init: `conv.weight.std() ≈ 0.01`, `conv.bias == 0`
  - Test `ResNetSE` with Go config returns 3-tuple, chess config returns 2-tuple
  - Test ownership loss computation with known values:
    - All targets = +1 (probs=1.0), logits = 0 → loss ≈ 0.693
    - All targets = 0 (probs=0.5), logits = 0 → loss ≈ 0.693
    - All targets = +1, logits = +5 → loss ≈ 0.00005 (near zero, correct prediction)
    - All targets = -1, logits = -5 → loss ≈ 0.00005 (near zero, correct prediction)
  - Test that `ownership_loss_weight=0.0` means ownership loss is not added
  - Test backward pass: gradients flow through ownership head to backbone
  - Test inference path: mock model returning 3-tuple, verify callback extracts policy/value
- Test checkpoint loading: load 2-head checkpoint into 3-head model without crash
- Document WHY: "Ownership prediction gives the network 361 spatial gradient signals per
  position instead of a single scalar, dramatically accelerating Go training. KataGo's
  ablation showed 1.65x slowdown without ownership/score targets."

- **Completion notes (2026-03-02)**:
  - Added `GameConfig.supports_ownership` (`go=True`, `chess=False`) and implemented
    `OwnershipHead` with small initialization (`std=0.01`, zero bias), wired into `ResNetSE`
    so Go returns `(policy, value, ownership)` while chess remains `(policy, value)`.
  - Added `ownership_loss()` (BCE-with-logits with 2x-logit scaling), threaded optional
    ownership targets through replay batch parsing, training-step loss composition, and
    `TrainingStepMetrics` (`loss/ownership`).
  - Added `TrainingConfig.ownership_loss_weight` (default `0.0`) and set
    `training.ownership_loss_weight: 1.5` in `configs/go.yaml`.
  - Updated inference compatibility for optional third model output in:
    `pipeline/orchestrator.py`, `pipeline/evaluation.py`, `scripts/train.py` warmup, and
    `scripts/play.py`.
  - Updated checkpoint loading to support backward-compatible head additions:
    `load_checkpoint(..., strict=False)` now logs missing/unexpected keys and preserves
    `strict=True` opt-in behavior.
  - Added/updated tests for ownership head shape/init, Go 3-tuple forward path, ownership BCE
    behavior, replay ownership tensor parsing, ownership-loss enable/disable behavior,
    3-output inference adapter compatibility, checkpoint strict/non-strict loading, and config
    ownership flags.
  - Validation run:
    - `python3 -m pytest tests/python/test_config.py tests/python/test_network.py tests/python/test_loss.py tests/python/test_training.py tests/python/test_orchestrator.py tests/python/test_checkpoint_utils.py tests/python/test_evaluation.py tests/python/test_play_script.py`
    - `python3 -m pytest tests/python/test_integration_smoke.py`
    - `python3 -m compileall python/alphazero/config.py python/alphazero/network python/alphazero/training python/alphazero/pipeline/orchestrator.py python/alphazero/pipeline/evaluation.py python/alphazero/utils/checkpoint.py scripts/train.py scripts/play.py tests/python/test_config.py tests/python/test_network.py tests/python/test_loss.py tests/python/test_training.py tests/python/test_orchestrator.py tests/python/test_checkpoint_utils.py`
    - `ruff` and `mypy` were unavailable in this environment.

---

## Reference

### KataGo Auxiliary Targets

KataGo (arXiv:1902.10565) used several auxiliary prediction heads beyond the standard
policy + value. In ablation studies at 2.5B training queries:

| Removed Component          | Elo Loss | Slowdown |
|----------------------------|----------|----------|
| Ownership + score targets  | -190     | 1.65x    |
| Opponent policy target     | -74      | 1.30x    |
| All other auxiliary targets | smaller  | <1.15x   |

Ownership was the most impactful single auxiliary target. KataGo's full auxiliary head set:
- **Ownership**: Per-intersection territory prediction (+1/-1/0) — *the one we're implementing*
- **Score prediction**: Per-intersection expected score contribution
- **Opponent policy**: Predict the opponent's next move (weight 0.15)
- **Score belief distribution**: Full distribution over final scores (PDF + CDF losses)
- **Future position**: Where stones will be placed in the near future
- **Seki detection**: Identify seki (mutual life) situations
- **TD value heads**: Temporal-difference value predictions at 3 time horizons

### Ownership Head Technical Details

**Architecture**: Single 1x1 convolution from trunk features to 1 channel. No activation —
outputs raw logits (called `ownership_pretanh` in KataGo's code). KataGo initializes the
conv weights at scale 0.2 so predictions start near zero (uncertain).

**Loss function**: Binary cross-entropy with logits.
- Targets: {-1, 0, +1} mapped to {0, 0.5, 1} via `(1 + target) / 2`
- Logits scaled by 2.0 before BCE: `sigmoid(2x) = (1 + tanh(x)) / 2`
- Per-sample loss averaged over valid board intersections (`/ board_area`)
- **Loss weight: 1.5** (weighted MORE than value at 1.2 in KataGo, because per-intersection
  gradients are more informative than a single scalar)

**Why BCE over MSE**: MSE with tanh output suffers from vanishing gradients at extreme
confidence levels (`d/dx tanh(x) → 0` as `x → ±∞`). BCE with logits provides consistent
gradient signal regardless of prediction confidence. At the boundaries (target +1 or -1),
BCE keeps pushing the logit further in the correct direction, while MSE+tanh plateaus.

**The ownership output is NOT used during MCTS** — it is purely an auxiliary training signal.
It teaches the backbone to represent territory spatially, which indirectly improves both
the policy and value heads.

### KataGo Loss Weight Summary (for context)

| Loss Component     | Weight | Notes                            |
|--------------------|--------|----------------------------------|
| Policy (player)    | 1.0    | Main policy                      |
| Value (outcome)    | 1.20   | 3-class CE (win/loss/no-result)  |
| **Ownership**      | **1.5**| **BCE, averaged per board**      |
| Policy (opponent)  | 0.15   | Auxiliary                        |
| Scoring            | 0.25   | MSE with sqrt transform          |
| Score belief       | 0.02   | PDF + CDF losses                 |
| L2 regularization  | 3e-5   | On all weights                   |

### Impact on Training

With the ownership head, the network receives 361 spatial learning signals per position
instead of just 1 scalar value. This means:
- The backbone learns spatial features much faster (territory, influence, groups)
- The value head improves indirectly (better backbone features → better value predictions)
- The policy head improves indirectly (understanding territory helps choose moves)
- Degenerate equilibria (like double-pass) are eliminated because predicting ownership on
  an empty board provides no useful gradient — the network is pushed to create positions
  with meaningful territorial structure

As the KataGo paper states: *"Whenever a desired target can be expressed as a sum,
conjunction, or disjunction of separate subevents, predicting those subevents is likely to
help."* The game outcome (win/loss) is a function of the total score, which is the sum of
per-intersection ownership — so predicting ownership decomposes the sparse win/loss signal
into spatially localized components.

### Memory Impact

- Ownership head parameters: ~257 extra parameters (1×1 conv: 256 weights + 1 bias).
  Negligible compared to the 25M parameter backbone.
- Per-position compact storage: +96 bytes (12 uint64 words for 2 bitpacked planes).
  At 20M positions this adds ~1.8 GB. Well within the memory budget.
- Training activations: One additional (batch, 1, 19, 19) tensor per forward pass.
  Negligible compared to the 20-block backbone activations (~31 GB).
