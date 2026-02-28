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
- **Current state**: COMPLETE (2026-02-28)
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

- **Implementation notes (2026-02-28)**:
  - Added `dynamic_dirichlet_alpha` to `SearchConfig` and `SelfPlayGameConfig` with default `false` to preserve
    backward compatibility.
  - Added `dirichlet_alpha_reference_moves` to `GameConfig`, set to `361` in Go and `30` in chess.
  - Updated `MctsSearchT::dirichlet_alpha` to accept `num_legal_moves` and apply
    `base_alpha * reference_moves / num_legal_moves` when dynamic scaling is enabled.
  - Root Dirichlet noise now calls `dirichlet_alpha(root_node.num_actions)` so scaling depends on current root
    legal-move count.
  - Exposed new config fields through pybind (`GameConfig`, `SearchConfig`, `SelfPlayGameConfig`) and threaded
    `mcts.dynamic_dirichlet_alpha` through `scripts/train.py` config loading.
  - Enabled `mcts.dynamic_dirichlet_alpha: true` in `configs/go.yaml`; chess configs remain unchanged.
  - Added C++ regression coverage for dynamic scaling in `tests/cpp/test_mcts.cpp` including acceptance examples
    (`361`, `100`, `10`) and edge cases (`1`, `362` legal moves); added config/binding assertions in
    `tests/cpp/test_chess_state.cpp`, `tests/cpp/test_go_state.cpp`, `tests/python/test_bindings.py`, and
    `tests/python/test_train_script.py`.
  - Validation run:
    - `cmake --build build --target alphazero_cpp -j$(nproc)`
    - `cmake --build build --target alphazero_cpp_tests -j$(nproc)`
    - `ctest --test-dir build --output-on-failure -R "(MctsSearchTest\\.|ChessStateTest\\.|GoStateTest\\.)"`
    - `PYTHONPATH=build/src:$PYTHONPATH /home/hakan/miniconda3/envs/alphazero/bin/python -m pytest tests/python/test_bindings.py tests/python/test_train_script.py`
    - `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`
    - `python3 -m compileall scripts/train.py tests/python/test_bindings.py tests/python/test_train_script.py`
  - Tooling notes:
    - `ruff` unavailable in this sandbox (`ruff: command not found`).
    - `mypy` is installed but reports existing environment/type issues unrelated to this task (notably missing `torch`
      stubs/imports and existing typed-API issues in untouched modules).
  - No additional follow-up tasks discovered while implementing TASK-002.

---

### TASK-003: Extend CompactReplayBuffer to support 19×19 Go boards

- **Files**: `src/selfplay/replay_buffer.h`, `src/selfplay/replay_compression.h`,
  `src/selfplay/replay_compression.cpp`, `src/selfplay/compact_replay_buffer.h`,
  `src/selfplay/compact_replay_buffer.cpp`, `src/bindings/python_bindings.cpp`,
  `scripts/train.py`, `configs/go.yaml`, `tests/cpp/test_replay_buffer.cpp`,
  `tests/cpp/test_replay_compression.cpp`, `tests/cpp/test_compact_replay_buffer.cpp`,
  `tests/python/test_train_script.py`, `tests/python/test_bindings.py`
- **Current state**: COMPLETE (2026-02-28)
- **Priority**: HIGH — the uncompressed `ReplayBuffer` uses ~48 KB per position (fixed-size arrays
  at chess maximums), so Go's 5M-position target requires ~229 GB and crashes with `std::bad_alloc`.
  The compact buffer stores ~1.2 KB per position but currently only supports boards ≤ 64 squares
  (8×8). Extending it to 19×19 reduces Go memory from 229 GB to ~6 GB.
- **Rationale**: `CompactReplayPosition` bitpacks each binary plane into a single `uint64_t`,
  which can hold 64 bits — one per square on an 8×8 board. A 19×19 board has 361 squares and
  needs `ceil(361/64) = 6` uint64 words per plane. The existing `bitpacked_planes` array has
  `kMaxBinaryPlanes = 117` slots. Go needs `16 binary planes × 6 words = 96 words`, which fits
  within the existing 117-word array. The core change is reinterpreting the array as "words" rather
  than "planes" and making the compression/decompression routines board-size-aware.

  **Stopgap (resolved 2026-02-28)**: `configs/go.yaml` previously used
  `replay_buffer.capacity: 1000000` with dense replay fallback (~46 GB). This task restored compact
  replay support for 19×19 and reset Go capacity to 5,000,000.

- **Problem analysis** — three blocking issues:

  1. **`replay_compression.cpp` line 16**: `constexpr kSquaresPerPlane = 64U` is hardcoded.
     Used in validation (`dense_state.size() % 64`), the binary packing loop (one `uint64_t` per
     plane), and decompression (`fill_n` with stride 64). Must become a runtime parameter.

  2. **`compact_replay_buffer.cpp` line 97**: `encoded_state_size_ = total_planes * 64U` assumes
     64 squares per plane. Must use the actual board area.

  3. **`scripts/train.py` line 411**: `if rows * cols != 64` guard explicitly rejects non-chess
     boards from using the compact buffer.

- **Fix**:

  1. **Rename `kMaxBinaryPlanes` → `kMaxBinaryWords`** in `CompactReplayPosition`
     (`src/selfplay/replay_buffer.h` line 43). Keep the value at 117. This is a semantic rename:
     the array stores `words_per_plane × num_binary_planes` total words. For chess (117 planes ×
     1 word) and Go (16 planes × 6 words = 96 words), both fit within 117.
     ```cpp
     static constexpr std::size_t kMaxBinaryWords = 117U;
     // ...
     std::array<std::uint64_t, kMaxBinaryWords> bitpacked_planes{};
     ```

  2. **Add `squares_per_plane` parameter** to `CompactReplayBuffer` constructor
     (`src/selfplay/compact_replay_buffer.cpp`). Store it as `squares_per_plane_` member.
     Compute `words_per_plane_ = (squares_per_plane_ + 63) / 64` (ceiling division).
     Update constructor validation:
     ```cpp
     const std::size_t total_binary_words = num_binary_planes_ * words_per_plane_;
     if (total_binary_words > CompactReplayPosition::kMaxBinaryWords) {
         throw std::invalid_argument("binary planes exceed storage capacity");
     }
     encoded_state_size_ = (num_binary_planes_ + num_float_planes_) * squares_per_plane_;
     ```

  3. **Update `compress_state` and `decompress_state`** signatures in
     `src/selfplay/replay_compression.h` to accept `std::size_t squares_per_plane`:
     ```cpp
     [[nodiscard]] StateCompressionLayout compress_state(
         std::span<const float> dense_state,
         std::span<const std::size_t> float_plane_indices,
         std::size_t squares_per_plane,
         std::span<std::uint64_t> out_bitpacked_planes,
         std::span<std::uint8_t> out_quantized_float_planes);

     void decompress_state(
         std::span<const std::uint64_t> bitpacked_planes,
         std::span<const std::uint8_t> quantized_float_planes,
         std::span<const std::size_t> float_plane_indices,
         std::size_t squares_per_plane,
         std::span<float> out_dense_state);
     ```

  4. **Update `compress_state` implementation** (`replay_compression.cpp`):
     - Remove `constexpr kSquaresPerPlane = 64U`.
     - Use the `squares_per_plane` parameter throughout.
     - Change validation to `dense_state.size() % squares_per_plane != 0`.
     - Pack binary planes into `words_per_plane = (squares_per_plane + 63) / 64` words each:
       ```cpp
       const std::size_t words_per_plane = (squares_per_plane + 63U) / 64U;
       // For each binary plane:
       for (std::size_t w = 0; w < words_per_plane; ++w) {
           std::uint64_t bits = 0U;
           const std::size_t bit_start = w * 64U;
           const std::size_t bit_end = std::min(bit_start + 64U, squares_per_plane);
           for (std::size_t sq = bit_start; sq < bit_end; ++sq) {
               if (dense_state[plane_offset + sq] >= 0.5F) {
                   bits |= (std::uint64_t{1} << (sq - bit_start));
               }
           }
           out_bitpacked_planes[binary_word_index++] = bits;
       }
       ```
     - Float plane quantization: sample the first cell only (unchanged logic), but use
       `squares_per_plane` for stride instead of 64.

  5. **Update `decompress_state` implementation** (`replay_compression.cpp`):
     - Mirror the multi-word unpacking:
       ```cpp
       for (std::size_t w = 0; w < words_per_plane; ++w) {
           const std::uint64_t bits = bitpacked_planes[binary_word_index++];
           const std::size_t bit_start = w * 64U;
           const std::size_t bit_end = std::min(bit_start + 64U, squares_per_plane);
           for (std::size_t sq = bit_start; sq < bit_end; ++sq) {
               out_dense_state[plane_offset + sq] =
                   ((bits >> (sq - bit_start)) & std::uint64_t{1}) != 0U ? 1.0F : 0.0F;
           }
       }
       ```
     - Float plane fill: use `fill_n(..., squares_per_plane, value)`.

  6. **Update all callers** of `compress_state` / `decompress_state` in
     `compact_replay_buffer.cpp` to pass `squares_per_plane_`:
     - `add_game()` — compression call
     - `sample()` — decompression call
     - `sample_batch()` — decompression call
     - `export_positions()` — decompression call
     - `import_positions()` — compression call

  7. **Update `CompactReplayPosition` metadata** (`src/selfplay/replay_buffer.h`):
     - Add `uint16_t num_binary_words = 0U` field (the total words used, for serialization).
       This replaces the role of `num_binary_planes` for indexing into `bitpacked_planes`.
       Keep `num_binary_planes` as well (it records the logical plane count).
     - This changes `sizeof(CompactReplayPosition)`. Bump the test assertion
       (`sizeof <= 1300`) to `sizeof <= 1304` to account for the new field.

  8. **Update file serialization** (`compact_replay_buffer.cpp`):
     - Bump `kFileVersion` from 1 to 2.
     - Add `squares_per_plane` to the file header (after the existing fields):
       ```
       [magic: 4 bytes "AZRB"]
       [version: uint32 = 2]
       [count: uint64]
       [sizeof_position: uint64]
       [squares_per_plane: uint32]    // NEW
       [positions: count × CompactReplayPosition]
       ```
     - On load, verify `squares_per_plane` matches the buffer's configured value.
     - Version 1 files can still be loaded by assuming `squares_per_plane = 64`.

  9. **Update pybind11 bindings** (`src/bindings/python_bindings.cpp`):
     - Add `squares_per_plane` parameter to `CompactReplayBuffer.__init__` (with default 64 for
       backward compatibility):
       ```cpp
       py::arg("squares_per_plane") = 64U,
       ```

  10. **Remove the board-size guard in `scripts/train.py`** (line 411):
      - Replace `if rows * cols != 64 or not hasattr(cpp, "CompactReplayBuffer"):` with
        `if not hasattr(cpp, "CompactReplayBuffer"):`.
      - Pass `squares_per_plane=rows * cols` to the `CompactReplayBuffer` constructor.
      - The Go config in `python/alphazero/config.py` already has `float_plane_indices=()`
        (0 float planes) and `input_channels=17` (16 binary + 1 constant plane). Verify whether
        Go's constant plane (side-to-move, a uniform 0/1) should be treated as binary or float.
        If it's binary-valued (0.0 or 1.0), it should be a binary plane. If it can be fractional
        (e.g., komi encoding), it should be a float plane. Check `go_state.cpp`'s `encode()`
        implementation for the last plane's semantics and update `GO_CONFIG.float_plane_indices`
        if needed.

  11. **Restore Go buffer capacity** in `configs/go.yaml`:
      ```yaml
      replay_buffer:
        capacity: 5000000   # ~6 GB compact storage (1.2 KB/pos)
      ```

- **Acceptance criteria**:
  1. `CompactReplayBuffer` accepts `squares_per_plane` parameter (default 64)
  2. Compression/decompression correctly handles 19×19 boards (361 squares, 6 words/plane)
  3. Chess (8×8, 1 word/plane) continues to work identically — full backward compatibility
  4. Go training starts without `std::bad_alloc` at 5M capacity (~6 GB memory)
  5. `save_to_file` / `load_from_file` work for Go-sized buffers (version 2 format)
  6. Version 1 files (chess) can still be loaded
  7. The Python `train.py` uses `CompactReplayBuffer` for Go (no board-size guard)
  8. All existing replay buffer tests pass
  9. `cmake --build build --target alphazero_cpp -j$(nproc)` succeeds
  10. `ctest --test-dir build --output-on-failure` passes

- **Testing guidance**:
  - Add C++ unit tests in `tests/cpp/test_replay_buffer.cpp`:
    a. **Round-trip test (19×19)**: Create a `CompactReplayBuffer` with `squares_per_plane=361`,
       `num_binary_planes=16`, `num_float_planes=1`. Add a game with known board states. Sample
       back and verify decompressed states match originals within quantization tolerance.
    b. **Round-trip test (8×8)**: Same as above with `squares_per_plane=64` to verify chess still
       works after the refactor.
    c. **Compression unit test**: Directly test `compress_state` / `decompress_state` with a
       361-element plane. Verify specific bit patterns: all zeros, all ones, checkerboard, single
       bit in the 6th word (square index > 320).
    d. **Serialization round-trip**: `save_to_file` then `load_from_file` for a Go-sized buffer.
       Verify loaded data matches.
    e. **Version 1 backward compat**: Load a version-1 file and verify it reads correctly with
       `squares_per_plane=64`.
  - Update the existing constant assertion test (`kMaxBinaryPlanes` → `kMaxBinaryWords`).
  - Document WHY multi-word packing is needed (361 squares > 64 bits per word).

- **Struct size estimate after changes**:
  - `bitpacked_planes`: 117 × 8 = 936 bytes (unchanged array size)
  - New `num_binary_words` field: 2 bytes
  - Total: ~1229 bytes (within 1304 limit)
  - Go per-position: 96 words used × 8 + metadata ≈ ~1000 bytes effective
  - 5M positions: ~5.8 GB (fits 128 GB budget with headroom)

- **Implementation notes (2026-02-28)**:
  - Renamed compact binary-capacity semantics from `kMaxBinaryPlanes` to `kMaxBinaryWords` and added
    `num_binary_words` metadata to `CompactReplayPosition`.
  - Extended `compress_state`/`decompress_state` to accept `squares_per_plane` and support multi-word
    packing/unpacking (`ceil(squares/64)` words per binary plane) while preserving 64-square behavior.
  - Added board-size-aware shape handling in `CompactReplayBuffer` via `squares_per_plane_`,
    `words_per_plane_`, and `num_binary_words_`; all compression/decompression call sites now pass the
    configured plane size.
  - Upgraded compact replay serialization to version 2 with `squares_per_plane` in the header and
    implemented version-1 load compatibility for legacy chess files.
  - Exposed `squares_per_plane` in pybind for `CompactReplayBuffer` (default `64`) and updated
    `scripts/train.py` to select compact replay for Go by passing `rows * cols` instead of hardcoding
    a 64-square guard.
  - Restored Go replay capacity to `5_000_000` in `configs/go.yaml`.
  - Added regression coverage for:
    - 19×19 compact round-trip sampling
    - multi-word compression bitpacking/unpacking (including sixth-word high indices)
    - Go-sized file save/load round-trip
    - legacy version-1 replay-file loading
    - Python training-runtime compact-buffer wiring for Go
  - Verified Go constant plane encoding remains binary (`0.0/1.0` side-to-move plane), so
    `GO_CONFIG.float_plane_indices` correctly remains empty.
  - Validation run:
    - `cmake --build build --target alphazero_cpp -j$(nproc)`
    - `cmake --build build --target alphazero_cpp_tests -j$(nproc)`
    - `ctest --test-dir build --output-on-failure -R "(ReplayBufferTest\\.|ReplayCompressionTest\\.|CompactReplayBufferTest\\.)"`
    - `ctest --test-dir build --output-on-failure`
    - `python3 -m pytest tests/python/test_train_script.py`
    - `PYTHONPATH=build/src:$PYTHONPATH /home/hakan/miniconda3/envs/alphazero/bin/python -m pytest tests/python/test_bindings.py`
    - `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix --ignore-installed`
    - `python3 -m compileall scripts/train.py tests/python/test_bindings.py tests/python/test_train_script.py`
  - Tooling notes:
    - `ruff` unavailable in this sandbox (`ruff: command not found`)
    - `mypy` unavailable in this sandbox (`mypy: command not found`)
  - No additional follow-up tasks discovered while implementing TASK-003.

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
