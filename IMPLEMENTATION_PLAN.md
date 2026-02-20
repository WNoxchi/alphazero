# AlphaZero Implementation Plan

**Status**: FOUNDATION COMPLETE — TASK-001 through TASK-003, TASK-010 through TASK-014, and TASK-020 through TASK-025 complete; core implementation tasks remain.

**Generated**: 2026-02-19
**Specs analyzed**: `specs/overview.md`, `specs/game-interface.md`, `specs/neural-network.md`, `specs/mcts.md`, `specs/pipeline.md`, `specs/infrastructure.md`

---

## Priority Legend

- **P0 — Foundation**: Blocks all downstream work. Must be built first.
- **P1 — Core game logic**: Blocks MCTS, training, and self-play.
- **P2 — Neural network**: Blocks inference and training pipeline.
- **P3 — Search**: Blocks self-play.
- **P4 — Pipeline**: Blocks training runs.
- **P5 — Infrastructure**: Required for running and monitoring training.
- **P6 — Testing**: Validates correctness. Can partially overlap with implementation.

---

## P0 — Foundation (Blocking Everything)

### TASK-001: Create project directory structure and build system
- **Spec**: `infrastructure.md` §1, §2
- **State**: completed (2026-02-20)
- **Description**: Create the full directory tree per `infrastructure.md` §1. Create top-level `CMakeLists.txt` with C++20, CUDA, LibTorch, pybind11, Google Test dependencies. Create `pyproject.toml` with Python 3.11+, torch, tensorboard, pyyaml, numpy, pytest. Create `configs/chess_default.yaml` and `configs/go_default.yaml` per `pipeline.md` §9.
- **Priority rationale**: Every other task depends on the build system and directory structure existing.
- **Acceptance criteria**:
  - `cmake --build build` succeeds (even with empty source files / stubs)
  - `pip install -e ".[dev]"` succeeds
  - Directory tree matches `infrastructure.md` §1
- **Execution notes**:
  - Full directory scaffold created (C++/Python/tests/scripts/configs) and wired into CMake.
  - `cmake -S . -B build`, `cmake --build build`, `ctest --test-dir build`, and syntax checks succeeded.
  - `pip install -e ".[dev]"` could not complete in this sandbox due no package index access for `torch`; editable packaging was validated with `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`.

### TASK-002: Define abstract GameState and GameConfig interfaces (C++)
- **Spec**: `game-interface.md` §2
- **State**: completed (2026-02-20)
- **Description**: Create `src/games/game_state.h` with the abstract `GameState` class (pure virtual: `apply_action`, `legal_actions`, `is_terminal`, `outcome`, `current_player`, `encode`, `clone`, `hash`, `to_string`). Create `src/games/game_config.h` with the `GameConfig` struct (board geometry, encoding dimensions, action space, MCTS params, value head type, symmetry, factory method). Define `SymmetryTransform` interface and `get_symmetries()`.
- **Priority rationale**: All game implementations, MCTS, and the pipeline depend on these interfaces.
- **Acceptance criteria**:
  - Headers compile cleanly with C++20
  - Interfaces match spec signatures exactly
- **Execution notes**:
  - Added complete abstract `GameState` interface and complete `GameConfig`/`SymmetryTransform` contracts in `src/games/game_state.h` and `src/games/game_config.h`.
  - Added default identity symmetry implementation and default `get_symmetries()` returning identity transform.
  - Added C++ contract tests (`tests/cpp/test_game_interfaces.cpp`) verifying signature stability, default identity symmetry behavior, and `new_game()` factory/clone behavior.
  - Updated C++ test discovery to `POST_BUILD` so interface gtests run through `ctest`.
  - Validation passed: `cmake --build build --parallel`, `ctest --test-dir build --output-on-failure`, and `python3 -m compileall python scripts tests`.

### TASK-003: Define Python-side GameConfig dataclass
- **Spec**: `game-interface.md` §7
- **State**: completed (2026-02-20)
- **Description**: Create `python/alphazero/config.py` with the `GameConfig` dataclass and pre-defined `CHESS_CONFIG` and `GO_CONFIG` instances. Include YAML configuration loading per `pipeline.md` §9.
- **Priority rationale**: Python neural network, training loop, and pipeline depend on game config.
- **Acceptance criteria**:
  - `CHESS_CONFIG` and `GO_CONFIG` have correct values per spec
  - YAML config loading works for both `chess_default.yaml` and `go_default.yaml`
- **Execution notes**:
  - Implemented `GameConfig` as a frozen, slotted dataclass in `python/alphazero/config.py` with validation for dimensions, action-space size, value head type, and symmetry settings.
  - Added canonical `CHESS_CONFIG` and `GO_CONFIG` constants with spec-accurate values (`8x8x119/4672/WDL` for chess, `19x19x17/362/scalar` for Go).
  - Added `get_game_config()`, `load_yaml_config()`, and `load_game_config_from_yaml()` helpers to resolve pipeline YAML `game` selection into canonical configs, with explicit validation errors for malformed files.
  - Added resilient YAML parsing behavior: use `PyYAML` when installed, with a strict fallback parser for sandbox environments missing `yaml`.
  - Added `tests/python/test_config.py` with rationale-rich tests covering canonical presets, default YAML config resolution, tolerant game-name normalization, and error paths for malformed/unsupported configs.
  - Validation passed: `python3 -m unittest -q tests/python/test_config.py`, `python3 -m compileall -q python tests scripts`, `mypy python/alphazero/config.py tests/python/test_config.py`, and offline editable packaging check `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`.

---

## P1 — Game Implementations (Blocks MCTS, Training)

### TASK-010: Implement chess bitboard utilities
- **Spec**: `game-interface.md` §5 (Board Representation, Bitboards)
- **State**: completed (2026-02-20)
- **Description**: Create `src/games/chess/bitboard.h` and `bitboard.cpp`. Implement `ChessPosition` struct with 12 bitboards, side to move, castling rights, en passant, halfmove clock, fullmove number, repetition count. Implement bitboard operations: population count, bit scan forward/reverse, shift operations. Implement magic bitboards or kogge-stone for sliding piece attack generation. Implement precomputed attack tables for knights and kings. Implement Zobrist hashing for position comparison and repetition detection.
- **Priority rationale**: Chess move generation and state depend on bitboard infrastructure.
- **Acceptance criteria**:
  - Bitboard operations produce correct results for all piece types
  - Attack tables are correct for all squares
  - Zobrist hashing produces consistent hashes
- **Execution notes**:
  - Implemented a full chess bitboard utility layer in `src/games/chess/bitboard.h` and `src/games/chess/bitboard.cpp`, including `ChessPosition`, population count/bit scans, directional shifts, occupancy helpers, pawn/knight/king attack helpers, and ray-based sliding attacks for bishops/rooks/queens.
  - Added deterministic Zobrist hashing (piece-square keys, side-to-move key, castling-state keys, en-passant keys) for stable repetition hashing and position comparison.
  - Added `tests/cpp/test_chess_bitboard.cpp` and registered it in `tests/cpp/CMakeLists.txt`; tests include rationale comments and cover bit operations, edge-safe shifts, attack table correctness across all squares, sliding attacks against naive ray tracing under varied occupancies, occupancy aggregation, and hash determinism/sensitivity.
  - Validation passed: `cmake -S . -B build`, `cmake --build build --parallel`, `ctest --test-dir build --output-on-failure -R ChessBitboardTest`, `ctest --test-dir build --output-on-failure`, `python3 -m compileall -q python scripts tests`, and offline editable packaging check `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`.

### TASK-011: Implement chess move generation
- **Spec**: `game-interface.md` §5 (Move Generation)
- **State**: completed (2026-02-20)
- **Description**: Create `src/games/chess/movegen.h` and `movegen.cpp`. Implement pseudo-legal move generation for all piece types: sliding pieces (using magic bitboards), knights, kings, pawns (pushes, captures, en passant, promotion). Implement castling move generation (check rights, empty squares, non-attacked squares). Filter pseudo-legal moves to legal moves (remove moves that leave king in check). Implement bidirectional mapping between semantic moves and flat action indices per the 8x8x73 encoding scheme (`action_index = from_square * 73 + move_type_index`). Handle board flipping for black-to-move positions.
- **Priority rationale**: Chess state's `legal_actions()` and `apply_action()` depend on move generation.
- **Acceptance criteria**:
  - Perft tests pass at depths 1-6 for initial position, kiwipete, and standard endgame positions
  - Action index round-trips correctly for all move types
- **Execution notes**:
  - Implemented full chess move generation in `src/games/chess/movegen.h` and `src/games/chess/movegen.cpp`, including pseudo-legal generation for pawns/knights/sliders/king, special-move handling (castling, en passant, promotions), legal-move filtering via king-safety checks, and position-state updates in `apply_move`.
  - Added attack/check utilities (`is_square_attacked`, `is_in_check`) and legal action helpers (`legal_action_indices`) for downstream `ChessState` integration.
  - Implemented complete 8x8x73 action encoding/decoding with black-to-move canonical mirroring, queen-plane + knight-plane + underpromotion-plane mapping, and legality-validated action decode.
  - Replaced scaffold tests in `tests/cpp/test_chess_movegen.cpp` with rationale-rich coverage for move-family action round-trips, black-perspective mirroring, reference perft (initial, kiwipete, endgame), special-move king-safety edge cases, and illegal action decoding rejection.
  - Validation passed: `cmake --build build --parallel`, `./build/tests/cpp/alphazero_cpp_tests --gtest_filter=ChessMovegenTest.*`, `ctest --test-dir build --output-on-failure`, `python3 -m compileall -q python scripts tests`, `python3 -m unittest -q tests/python/test_config.py`, `mypy python/alphazero/config.py tests/python/test_config.py`, and offline editable packaging check `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`.
  - Lint status: attempted `ruff check python tests scripts`, but `ruff` is not installed in this environment (`/bin/bash: ruff: command not found`).

### TASK-012: Implement ChessState (GameState for chess)
- **Spec**: `game-interface.md` §2, §4, §5
- **State**: completed (2026-02-20)
- **Description**: Create `src/games/chess/chess_state.h` and `chess_state.cpp`. Implement `ChessState` inheriting from `GameState`. Implement `apply_action()` (make move, update bitboards, castling rights, en passant, halfmove clock, repetition tracking). Implement `legal_actions()` using movegen. Implement `is_terminal()` checking checkmate, stalemate, 50-move rule, threefold repetition, insufficient material, max game length (512). Implement `outcome()` returning +1/-1/0. Implement `current_player()`. Implement history management using copy-on-write or inline ring buffer (spec recommends inline buffer for chess). Implement `clone()` and `hash()` (Zobrist). Implement `to_string()`. Create `chess_config.cpp` with `ChessGameConfig` (board 8x8, 119 input planes, 4672 actions, WDL value head, no symmetry).
- **Priority rationale**: Needed by MCTS and self-play to play chess games.
- **Acceptance criteria**:
  - All terminal conditions detected correctly
  - History tracking supports T=8 positions
  - GameConfig values match spec exactly
- **Execution notes**:
  - Replaced chess state scaffolds with a full immutable `ChessState` implementation in `src/games/chess/chess_state.h` and `src/games/chess/chess_state.cpp`, including legal action application via action-index decode, repetition tracking via Zobrist-hash counts, inline T=8 history ring buffering, terminal detection, outcome scoring, cloning, hashing, human-readable rendering, and tensor encoding.
  - Added `src/games/chess/chess_config.h` and implemented `ChessGameConfig` in `src/games/chess/chess_config.cpp` with spec-accurate chess dimensions (`8x8`, `14*8+7=119` channels), action space (`4672`), value head (`WDL`), Dirichlet alpha (`0.3`), and max game length (`512`).
  - Added `tests/cpp/test_chess_state.cpp` with rationale-rich tests covering config correctness, immutable transition behavior, checkmate vs stalemate outcomes, draw-rule terminal paths (50-move, repetition, insufficient material, max-length), and T=8 history buffer behavior.
  - Updated `tests/cpp/CMakeLists.txt` to include `test_chess_state.cpp`.
  - Validation passed: `cmake --build build --parallel`, `./build/tests/cpp/alphazero_cpp_tests --gtest_filter=ChessStateTest.*`, `ctest --test-dir build --output-on-failure`, `python3 -m compileall -q python tests scripts`, `mypy python/alphazero/config.py tests/python/test_config.py`, and offline editable packaging check `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`.
  - Lint status: attempted `ruff check python tests scripts`, but `ruff` is not installed in this environment (`/bin/bash: ruff: command not found`).

### TASK-013: Implement chess input encoding
- **Spec**: `game-interface.md` §5 (Input Encoding)
- **State**: completed (2026-02-20)
- **Description**: Implement `ChessState::encode()` producing an 8x8x119 tensor. 14 planes per history step (6 P1 pieces + 6 P2 pieces + 2 repetition) x T=8 steps = 112 planes. 7 constant planes (color, total move count, 4 castling rights, no-progress count). Board orientation flipped for black-to-move. Zero-fill for history steps before game start.
- **Priority rationale**: Required for neural network inference on chess positions.
- **Acceptance criteria**:
  - Output tensor shape is (119, 8, 8)
  - Encoding matches spec for initial position and known mid-game positions
  - Board correctly flipped for black-to-move
- **Execution notes**:
  - Finalized chess encoding coverage by replacing scaffold `tests/cpp/test_chess_encoding.cpp` with rationale-rich tests that validate the full `(119, 8, 8)` contract, initial-position piece/repetition/constant planes, temporal history ordering and zero-fill behavior, repetition-plane semantics, and black-to-move canonical orientation.
  - Added tests for perspective-relative constant planes (color, castling rights, normalized total-move count, normalized no-progress count) using a targeted FEN position.
  - Validation passed: `cmake --build build --parallel`, `./build/tests/cpp/alphazero_cpp_tests --gtest_filter=ChessEncodingTest.*`, `ctest --test-dir build --output-on-failure`, `python3 -m compileall -q python scripts tests`, `mypy python/alphazero/config.py tests/python/test_config.py`, and offline editable packaging check `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`.
  - Lint status: attempted `ruff check python tests scripts`, but `ruff` is not installed in this environment (`/bin/bash: ruff: command not found`).

### TASK-014: Implement chess FEN/PGN serialization
- **Spec**: `game-interface.md` §5 (Serialization)
- **State**: completed (2026-02-20)
- **Description**: Implement FEN parsing and generation for `ChessState`. Implement PGN game record output for logged games. FEN round-trip must be identity.
- **Priority rationale**: Essential for debugging, testing (perft positions specified as FEN), and evaluation against external engines.
- **Acceptance criteria**:
  - FEN encode -> decode -> encode produces identical FEN
  - PGN output is valid and parseable by standard tools
- **Execution notes**:
  - Added chess serialization APIs in `src/games/chess/chess_state.h` and `src/games/chess/chess_state.cpp`: `ChessState::from_fen()`, `ChessState::to_fen()`, and `ChessState::actions_to_pgn()`.
  - Implemented strict FEN parsing/validation (field count, board layout, side-to-move, castling rights, en-passant square, halfmove/fullmove bounds, king-count sanity) and canonical FEN export.
  - Implemented PGN export from action history with SAN move text, check/checkmate suffixes, default PGN headers, and `SetUp`/`FEN` tags for non-initial starting positions.
  - Added rationale-rich tests in `tests/cpp/test_chess_serialization.cpp` and registered them in `tests/cpp/CMakeLists.txt`; coverage includes FEN round-trip identity, malformed FEN rejection, SAN/PGN output for checkmate games, non-initial start positions with black-to-move numbering, and invalid PGN input rejection.
  - Validation passed: `cmake --build build --parallel`, `ctest --test-dir build --output-on-failure -R ChessSerializationTest`, `ctest --test-dir build --output-on-failure`, `python3 -m compileall -q python tests scripts`, `mypy python/alphazero/config.py tests/python/test_config.py`, and offline editable packaging check `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`.
  - Lint status: attempted `ruff check python tests scripts`, but `ruff` is not installed in this environment (`/bin/bash: ruff: command not found`).

### TASK-020: Implement Go board representation and Zobrist hashing
- **Spec**: `game-interface.md` §6 (Board Representation, Zobrist Hashing)
- **State**: completed (2026-02-20)
- **Description**: Create `src/games/go/go_state.h`. Implement `GoPosition` struct with 19x19 board array, side to move, ko point, komi, move number, consecutive passes, position history (hash set for superko). Implement Zobrist hashing with pre-generated random values for each (intersection, color) pair. Incremental XOR updates on stone placement/capture.
- **Priority rationale**: Go rules and state depend on board representation.
- **Acceptance criteria**:
  - Board correctly represents empty, black, white stones
  - Zobrist hashing produces consistent hashes and detects identical positions
- **Execution notes**:
  - Implemented `src/games/go/go_state.h` and `src/games/go/go_state.cpp` with a full `GoPosition` representation (`19x19` board, side-to-move, ko point, komi, move number, consecutive passes, and positional superko hash history).
  - Added board/indexing helpers (`to_intersection`, row/column conversion, stone getters/setters, color/intersection validators) to make Go rules code consume a stable, bounds-safe representation API.
  - Implemented deterministic Zobrist hashing for Go with pre-generated keys for all `(intersection, color)` pairs, side-to-move keys, and ko-point keys.
  - Added both board-only hash (`zobrist_board_hash`) for positional superko workflows and full state hash (`zobrist_hash`) including side-to-move and ko point, plus incremental XOR update helpers for stone placement/capture.
  - Replaced scaffold `tests/cpp/test_go_rules.cpp` with rationale-rich tests covering default board semantics, coordinate/index round-trips, hash determinism/sensitivity, incremental hash parity with full recomputation, and position-history membership for repeated board hashes.
  - Validation passed: `cmake -S . -B build`, `cmake --build build --parallel`, `./build/tests/cpp/alphazero_cpp_tests --gtest_filter=GoStateRepresentationTest.*`, `ctest --test-dir build --output-on-failure`, `python3 -m compileall -q python scripts tests`, `mypy python/alphazero/config.py tests/python/test_config.py`, and offline editable packaging check `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`.
  - Lint status: attempted `ruff check python tests scripts`, but `ruff` is not installed in this environment (`/bin/bash: ruff: command not found`).

### TASK-021: Implement Go rules engine
- **Spec**: `game-interface.md` §6 (Go Rules Implementation, Liberty Tracking)
- **State**: completed (2026-02-20)
- **Description**: Create `src/games/go/go_rules.h` and `go_rules.cpp`. Implement liberty tracking using union-find (disjoint set) with `StoneGroup` struct (representative, liberty count, stone count). Implement stone placement: merge with adjacent same-color groups, subtract liberties from opponent groups, capture groups with zero liberties, verify no self-capture. Implement ko detection (single-stone recapture prohibition). Implement superko detection (positional — no repeated board positions using hash set). Implement pass logic (two consecutive passes end game). Implement self-capture prohibition.
- **Priority rationale**: Go state depends on correct rules engine.
- **Acceptance criteria**:
  - Simple and complex capture scenarios work correctly
  - Ko correctly detected and prohibited
  - Superko correctly detected
  - Self-capture correctly prohibited
  - Liberty counts accurate after all operations
- **Execution notes**:
  - Replaced scaffold Go rules files with a complete rules engine in `src/games/go/go_rules.h` and `src/games/go/go_rules.cpp`.
  - Added a concrete API for Go move processing (`MoveStatus`, `MoveResult`, `play_action`, `play_pass`, legality helpers) so future `GoState` code can consume validated rule transitions directly.
  - Implemented union-find-based board analysis (`StoneGroup`) to track connected groups, unique liberties, and stones-per-group; exposed `compute_stone_groups()` and `liberties_for_intersection()` for verification and downstream use.
  - Implemented full stone-placement semantics: same-color group merging, opponent group capture when liberties reach zero, self-capture rejection, ko-point detection on single-stone ko captures, positional superko checks via board-hash history, and pass handling with consecutive-pass termination signal.
  - Expanded `tests/cpp/test_go_rules.cpp` with rationale-rich rules tests covering liberty accounting, single/multi-stone captures, ko recapture blocking, positional superko rejection, self-capture rejection, and two-pass game termination while retaining existing Go representation/hash invariants.
  - Validation passed: `cmake --build build --parallel`, `./build/tests/cpp/alphazero_cpp_tests --gtest_filter=GoStateRepresentationTest.*:GoRulesEngineTest.*`, `ctest --test-dir build --output-on-failure`, `python3 -m compileall -q python scripts tests`, `mypy python/alphazero/config.py tests/python/test_config.py`, and offline editable packaging check `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`.
  - Lint status: attempted `ruff check python tests scripts`, but `ruff` is not installed in this environment (`/bin/bash: ruff: command not found`).

### TASK-022: Implement Tromp-Taylor scoring
- **Spec**: `game-interface.md` §6 (Scoring)
- **State**: completed (2026-02-20)
- **Description**: Create `src/games/go/scoring.h` and `scoring.cpp`. Implement Tromp-Taylor scoring: a point scores for a color if occupied by that color or reachable only by that color via empty intersections. Final score = black_points - white_points - komi. Implement area detection via flood fill from each empty intersection.
- **Priority rationale**: Required for Go game termination and outcome calculation.
- **Acceptance criteria**:
  - Scoring matches known game results
  - Handles all edge cases (seki, territory with dead stones, etc.)
- **Execution notes**:
  - Replaced scoring scaffolds with a complete Tromp-Taylor implementation in `src/games/go/scoring.h` and `src/games/go/scoring.cpp`.
  - Added a reusable `TrompTaylorScore` result contract (black points, white points, komi, final score, and winner helper) for downstream Go terminal/outcome integration.
  - Implemented empty-region flood fill to classify territory ownership by reachable boundary colors, awarding territory only for exclusive reachability and leaving shared/no-color regions neutral.
  - Expanded `tests/cpp/test_go_rules.cpp` with rationale-rich scoring tests covering occupied+territory accounting with komi, shared neutral-region behavior (seki/dame-style edge case), and known empty/full-board results.
  - Validation passed: `cmake --build build --parallel`, `./build/tests/cpp/alphazero_cpp_tests --gtest_filter=GoRulesEngineTest.*:GoScoringTest.*`, `ctest --test-dir build --output-on-failure`, `python3 -m compileall -q python scripts tests`, `mypy python/alphazero/config.py tests/python/test_config.py`, and offline editable packaging check `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`.
  - Lint status: attempted `ruff check python tests scripts`, but `ruff` is not installed in this environment (`/bin/bash: line 1: ruff: command not found`).

### TASK-023: Implement GoState (GameState for Go)
- **Spec**: `game-interface.md` §2, §4, §6
- **State**: completed (2026-02-20)
- **Description**: Create `src/games/go/go_state.cpp`. Implement `GoState` inheriting from `GameState`. Implement `apply_action()` (stone placement or pass, using go_rules). Implement `legal_actions()` (all empty intersections that don't violate ko/superko/self-capture, plus pass). Implement `is_terminal()` (two consecutive passes, max game length 722). Implement `outcome()` (Tromp-Taylor scoring). Implement history management via copy-on-write linked list (spec recommendation for Go). Implement `encode()`, `clone()`, `hash()`, `to_string()`. Create `go_config.cpp` with `GoGameConfig` (board 19x19, 17 input planes, 362 actions, scalar value head, 8 symmetries).
- **Priority rationale**: Needed by MCTS and self-play to play Go games.
- **Acceptance criteria**:
  - All terminal conditions detected correctly
  - History tracking supports T=8 positions
  - GameConfig values match spec exactly
- **Execution notes**:
  - Implemented a full `GoState` in `src/games/go/go_state.h` and `src/games/go/go_state.cpp` with immutable `apply_action()`, legality filtering through `go_rules`, terminal detection (`two passes` or `move_number >= 722`), Tromp-Taylor outcomes, copy-on-write linked-list ancestry for T=8 history, tensor encoding (`19x19x17`), cloning, hashing, and string rendering.
  - Added Go player-index helpers (`0=black`, `1=white`) at the `GameState` boundary while preserving internal Go stone-color representation (`1=black`, `2=white`) in `GoPosition`.
  - Added `src/games/go/go_config.h` and replaced the scaffold in `src/games/go/go_config.cpp` with `GoGameConfig`, including spec-accurate dimensions (`19x19`, `2*8+1=17`, `362` actions), scalar value head, symmetry flags (`supports_symmetry=true`, `num_symmetries=8`), and Go Dirichlet alpha (`0.03`).
  - Added `tests/cpp/test_go_state.cpp` (rationale-rich) and registered it in `tests/cpp/CMakeLists.txt`; tests cover config fidelity, immutable state transitions and illegal-move rejection, terminal/outcome semantics, T=8 history window behavior, and encode perspective/zero-fill guarantees.
  - Validation passed: `cmake --build build --parallel`, `ctest --test-dir build --output-on-failure -R GoStateTest`, `ctest --test-dir build --output-on-failure`, `python3 -m compileall -q python tests scripts`, `mypy python/alphazero/config.py tests/python/test_config.py`, and offline editable packaging check `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`.
  - Lint status: attempted `ruff check python tests scripts`, but `ruff` is not installed in this environment (`/bin/bash: line 1: ruff: command not found`).

### TASK-024: Implement Go input encoding
- **Spec**: `game-interface.md` §6 (Input Encoding)
- **State**: completed (2026-02-20)
- **Description**: Implement `GoState::encode()` producing a 19x19x17 tensor. 2 planes per history step (current player stones + opponent stones) x T=8 = 16 planes. 1 constant plane (color: all 1s if black to play). Zero-fill for history steps before game start.
- **Priority rationale**: Required for neural network inference on Go positions.
- **Acceptance criteria**:
  - Output tensor shape is (17, 19, 19)
  - Encoding matches spec for known positions
- **Execution notes**:
  - Verified `GoState::encode()` in `src/games/go/go_state.cpp` already satisfies the Go encoding contract: tensor layout `(17, 19, 19)`, per-step perspective-relative stone planes across `T=8` history, and black-to-move constant plane semantics.
  - Strengthened acceptance coverage in `tests/cpp/test_go_state.cpp` with rationale-rich checks for known black-to-move and white-to-move positions, explicit shape validation (`17*19*19`), and zero-filled history planes before game start.
  - Validation passed: `cmake --build build --parallel`, `ctest --test-dir build --output-on-failure -R GoStateTest`, `ctest --test-dir build --output-on-failure`, `python3 -m compileall -q python scripts tests`, `mypy python/alphazero/config.py tests/python/test_config.py`, and offline editable packaging check `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`.
  - Lint status: attempted `ruff check python tests scripts`, but `ruff` is not installed in this environment (`/bin/bash: line 1: ruff: command not found`).

### TASK-025: Implement Go symmetry transforms
- **Spec**: `game-interface.md` §2 (Symmetry Interface), §6 (Symmetry)
- **State**: completed (2026-02-20)
- **Description**: Implement `SymmetryTransform` for Go's 8 dihedral group symmetries (4 rotations x 2 reflections). `transform_board()` applies the symmetry to a (channels, 19, 19) tensor in-place. `transform_policy()` permutes the 362-element policy vector (361 intersections + invariant pass). Implement `get_symmetries()` returning all 8 transforms.
- **Priority rationale**: Required for Go training data augmentation.
- **Acceptance criteria**:
  - All 8 transforms produce equivalent game positions
  - Policy transforms are consistent with board transforms
  - Pass action (index 361) is invariant under all transforms
- **Execution notes**:
  - Added a Go-specific symmetry implementation in `src/games/go/go_config.cpp` and `src/games/go/go_config.h` via `GoGameConfig::get_symmetries()`, returning 8 `SymmetryTransform` instances covering the full D4 group (`4` quarter-turn rotations × reflected/unreflected variants).
  - Implemented in-place board tensor transforms for square `(channels, rows, cols)` layouts and policy permutation for the `362`-action Go space, with explicit invariant handling for pass action `361`.
  - Added defensive input validation for null pointers, non-square board tensors, and incorrect policy sizes to avoid undefined behavior when augmentation is wired incorrectly.
  - Replaced scaffold `tests/cpp/test_go_encoding.cpp` with rationale-rich tests that verify exact D4 policy permutations, board/policy permutation consistency across multiple channels, pass invariance, and invalid-input error paths.
  - Validation passed: `cmake --build build --parallel`, `ctest --test-dir build --output-on-failure -R GoEncodingTest`, `ctest --test-dir build --output-on-failure`, `python3 -m compileall -q python tests scripts`, `mypy python/alphazero/config.py tests/python/test_config.py`, and offline editable packaging check `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`.
  - Lint status: attempted `ruff check python tests scripts`, but `ruff` is not installed in this environment (`/bin/bash: line 1: ruff: command not found`).

### TASK-026: Implement Go SGF serialization
- **Spec**: `game-interface.md` §6 (Serialization)
- **State**: missing
- **Description**: Implement SGF (Smart Game Format) output for Go game records. Support reading SGF for debugging and analysis.
- **Priority rationale**: Useful for analysis and compatibility with Go tools, but not blocking.
- **Acceptance criteria**:
  - SGF output is valid and parseable by standard Go tools

---

## P2 — Neural Network (Blocks Inference and Training)

### TASK-030: Implement AlphaZeroNetwork base class
- **Spec**: `neural-network.md` §2 (Python Interface)
- **State**: missing
- **Description**: Create `python/alphazero/network/base.py` with abstract `AlphaZeroNetwork(nn.Module)` base class. Define `forward(x) -> (policy_logits, value)` interface. Accept `GameConfig` in constructor for input/output dimensions.
- **Priority rationale**: All network architectures depend on this interface.
- **Acceptance criteria**:
  - Abstract class with correct signature
  - Subclasses can be instantiated with GameConfig

### TASK-031: Implement ResNet + SE architecture
- **Spec**: `neural-network.md` §3
- **State**: missing
- **Description**: Create `python/alphazero/network/resnet_se.py`. Implement initial convolutional block (Conv2d -> BatchNorm2d -> ReLU). Implement SE-Residual block with Leela-style SE (scale + bias variant: FC -> ReLU -> FC -> split into scale/bias -> sigmoid(scale) * conv_output + bias + skip -> ReLU). Support configurable `num_blocks` (10/20/40) and `num_filters` (128/256). SE reduction ratio configurable (default 4). Weight initialization per `neural-network.md` §8: Kaiming He for conv/linear, Xavier for SE FC, zeros for final policy/value linear, standard for BatchNorm.
- **Priority rationale**: Core network architecture needed for all training and inference.
- **Acceptance criteria**:
  - Small (10 blocks, 128 filters), medium (20, 256), and large (40, 256) configs instantiate correctly
  - Parameter counts approximately match spec (~5M, ~25M, ~50M)
  - Weight initialization follows spec

### TASK-032: Implement policy and value heads
- **Spec**: `neural-network.md` §3 (Policy Head, Value Heads)
- **State**: missing
- **Description**: Create `python/alphazero/network/heads.py`. Implement policy head: Conv2d(F, 32, 1) -> BN -> ReLU -> Flatten -> Linear(32*H*W, action_space_size). Implement scalar value head (Go): Conv2d(F, 1, 1) -> BN -> ReLU -> Flatten -> Linear(H*W, 256) -> ReLU -> Linear(256, 1) -> Tanh. Implement WDL value head (Chess): same structure but Linear(256, 3) -> Softmax.
- **Priority rationale**: Network outputs depend on correct head implementations.
- **Acceptance criteria**:
  - Policy head output shape: (batch, action_space_size)
  - Scalar value output shape: (batch, 1), range [-1, 1]
  - WDL value output shape: (batch, 3), probabilities summing to 1

### TASK-033: Implement loss functions
- **Spec**: `neural-network.md` §4, `pipeline.md` §6 (Loss Computation)
- **State**: missing
- **Description**: Create `python/alphazero/training/loss.py`. Implement policy loss: cross-entropy between MCTS policy target and network logits (only over legal actions). Implement scalar value loss: MSE between tanh output and game outcome. Implement WDL value loss: cross-entropy between WDL target and network WDL output. Implement L2 regularization (c=1e-4, applied to all parameters). Combined loss: L_policy + L_value + c * L2.
- **Priority rationale**: Training loop depends on correct loss computation.
- **Acceptance criteria**:
  - Policy cross-entropy matches hand-computed values
  - Value losses (MSE and CE) match hand-computed values
  - L2 regularization includes all parameters
  - Policy and value losses weighted equally (unit-scaled)

### TASK-034: Implement learning rate schedule
- **Spec**: `neural-network.md` §5 (Training Configuration, LR Schedule)
- **State**: missing
- **Description**: Create `python/alphazero/training/lr_schedule.py`. Implement step-decay LR schedule: 0.2 for steps 0-200K, 0.02 for 200K-400K, 0.002 for 400K-600K, 0.0002 for 600K+. Support configurable milestones via YAML config.
- **Priority rationale**: Training loop depends on correct LR scheduling.
- **Acceptance criteria**:
  - LR values correct at each milestone boundary
  - Schedule configurable via YAML

### TASK-035: Implement batch norm folding utility
- **Spec**: `neural-network.md` §6 (Batch Normalization Folding)
- **State**: missing
- **Description**: Create `python/alphazero/network/bn_fold.py`. Implement BN folding: compute W_folded = W * γ / sqrt(σ² + ε) and b_folded = (b - μ) * γ / sqrt(σ² + ε) + β. Export a folded model copy (no BatchNorm layers, folded weights in Conv layers). Run after each training checkpoint.
- **Priority rationale**: Self-play inference performance depends on BN folding.
- **Acceptance criteria**:
  - Folded model produces identical outputs to original model (within 1e-5 tolerance)
  - Folded model has no BatchNorm layers

---

## P3 — MCTS (Blocks Self-Play)

### TASK-040: Implement MCTSNode data structure (SoA layout)
- **Spec**: `mcts.md` §3
- **State**: missing
- **Description**: Create `src/mcts/mcts_node.h`. Implement `MCTSNode` struct with SoA layout: `visit_count[MAX_ACTIONS]`, `total_value[MAX_ACTIONS]`, `mean_value[MAX_ACTIONS]`, `prior[MAX_ACTIONS]`, `actions[MAX_ACTIONS]`, `num_actions`, `total_visits`, `node_value`, `children[MAX_ACTIONS]`, `parent`, `parent_action`, `virtual_loss[MAX_ACTIONS]`. Define `NodeId` as `uint32_t` with `NULL_NODE = UINT32_MAX`. Define `MAX_ACTIONS` per game (218 for chess, 362 for Go) — use compile-time constant or template.
- **Priority rationale**: MCTS search, node store, and self-play all depend on this data structure.
- **Acceptance criteria**:
  - Struct compiles and has correct layout
  - SoA arrays are contiguous for vectorized PUCT

### TASK-041: Implement NodeStore interface and ArenaNodeStore
- **Spec**: `mcts.md` §4
- **State**: missing
- **Description**: Create `src/mcts/node_store.h` (interface: `allocate()`, `get()`, `release_subtree()`, `reset()`, `nodes_allocated()`, `memory_used_bytes()`). Create `src/mcts/arena_node_store.h` and `arena_node_store.cpp` with bump-pointer allocation from pre-allocated vector. Default capacity 8192 per game. Implement tree reuse: preserve chosen child's subtree, release siblings. Implement `reset()` as O(1) pointer reset.
- **Priority rationale**: MCTS search needs a node allocator.
- **Acceptance criteria**:
  - Allocation is O(1)
  - release_subtree correctly frees sibling trees
  - reset is O(1)
  - Memory tracking is accurate

### TASK-042: Implement MCTS search (PUCT, FPU, Dirichlet, virtual loss, backup)
- **Spec**: `mcts.md` §2, §6, §8, §9, §10, §11, §14
- **State**: missing
- **Description**: Create `src/mcts/mcts_search.h` and `mcts_search.cpp`. Implement the full MCTS simulation loop per `mcts.md` §14 pseudocode:
  - **Select**: PUCT score = Q(s,a) + c_puct * P(s,a) * sqrt(N_total) / (1 + N(s,a)). Apply virtual loss during selection.
  - **Expand**: Create child node, initialize edges with masked/renormalized NN priors.
  - **Evaluate**: Submit to eval queue or use terminal value.
  - **Backup**: Propagate negated value up the path, revert virtual loss, update visit counts/Q-values.
  - **FPU**: Leela-style FPU reduction: Q_fpu = V(parent) - c_fpu * sqrt(Σ visited P(s,a)), default c_fpu=0.25.
  - **Dirichlet noise**: At root only, P = (1-ε)*p + ε*η where η~Dir(α), ε=0.25, α=0.3 (chess) / 0.03 (Go).
  - **Temperature**: π(a) ∝ N(root,a)^(1/τ). τ=1.0 for moves 1-30, τ→0 (argmax) for moves 31+.
  - **Tree reuse**: After move selection, child becomes new root, siblings released.
  - **Resignation**: Resign if V(root) < v_resign AND max_child_V < v_resign (configurable, default -0.9). Disable in 10% of games.
  - **Synchronization**: Per-node atomic visit counts (relaxed ordering for reads, acquire-release for updates). Per-node spinlock/mutex for float value updates. Atomic virtual loss counters.
- **Priority rationale**: Core search algorithm. Blocks self-play.
- **Acceptance criteria**:
  - PUCT selects correct actions with mock NN (fixed policy/value)
  - Visit counts converge to expected distributions
  - Backup correctly negates at alternating levels
  - Virtual loss applied and reverted correctly
  - FPU formula matches spec
  - Dirichlet noise only at root
  - Temperature selection correct for both regimes
  - Tree reuse preserves statistics
  - All concurrency primitives are correct under contention

### TASK-043: Implement evaluation queue
- **Spec**: `mcts.md` §7
- **State**: missing
- **Description**: Create `src/mcts/eval_queue.h` and `eval_queue.cpp`. Implement `EvalQueue` with `submit_and_wait()` (MCTS threads submit encoded state, block on per-request semaphore) and `process_batch()` (GPU thread collects pending requests, runs batch inference, dispatches results). Implement flush triggers: immediate when pending >= batch_size (default 256), timeout after 100μs for partial batches. Thread-safe MPSC queue (lock-free or mutex-protected deque). Unified memory: input/output buffers directly accessible by CPU and GPU.
- **Priority rationale**: Decouples MCTS (CPU) from NN inference (GPU). Required for async self-play.
- **Acceptance criteria**:
  - Multiple producer threads, single consumer thread work correctly
  - All requests processed and results dispatched
  - Flush timeout triggers on partial batches
  - No deadlocks or data races under high contention

---

## P4 — Self-Play Pipeline (Blocks Training Runs)

### TASK-050: Implement replay buffer
- **Spec**: `pipeline.md` §5
- **State**: missing
- **Description**: Create `src/selfplay/replay_buffer.h` and `replay_buffer.cpp`. Implement ring buffer with `ReplayPosition` struct (encoded_state, policy, value/value_wdl, game_id, move_number). Capacity: configurable (default 1M positions). Thread-safe with readers-writer lock: `add_game()` (write, called by self-play), `sample()` (read, uniform random sampling for training mini-batches). Atomic write head and count. V1: store full NN input tensor (uncompressed). In unified memory for zero-copy GPU access.
- **Priority rationale**: Connects self-play (data generation) to training (data consumption).
- **Acceptance criteria**:
  - Concurrent writes from multiple game threads safe
  - Concurrent reads from training thread safe
  - Ring buffer wrapping works correctly
  - No data corruption or lost positions
  - Uniform random sampling is correct

### TASK-051: Implement self-play game lifecycle
- **Spec**: `pipeline.md` §4 (Game Lifecycle)
- **State**: missing
- **Description**: Create `src/selfplay/self_play_game.h` and `self_play_game.cpp`. Implement single game lifecycle: initialize GameState + reset MCTS arena + add Dirichlet noise. Move loop: K threads run simulations until budget (800) reached, compute move policy π(a) ∝ N^(1/τ), select move, store training sample (state, π, _), apply move, reuse subtree, add Dirichlet noise to new root. On terminal: compute outcome z for all stored positions, write (state, π, z) tuples to replay buffer. Handle resignation logic (disable in configurable fraction). Handle max game length adjudication.
- **Priority rationale**: Implements the core self-play loop for a single game.
- **Acceptance criteria**:
  - Game plays from start to natural/resigned/max-length termination
  - Training samples correctly include outcome from current player's perspective
  - Tree reuse works across moves
  - Resignation logic correct with disable fraction

### TASK-052: Implement self-play manager
- **Spec**: `pipeline.md` §4
- **State**: missing
- **Description**: Create `src/selfplay/self_play_manager.h` and `self_play_manager.cpp`. Maintain M concurrent game slots (default 32). Spawn K MCTS worker threads per game (default 8). When a game ends, write to replay buffer and start new game in that slot. Collect and report self-play metrics (game length, outcome, resignation, throughput).
- **Priority rationale**: Orchestrates concurrent self-play games feeding the eval queue.
- **Acceptance criteria**:
  - M games run concurrently with K threads each
  - Game slots recycle correctly on termination
  - Metrics collected per game completion

### TASK-053: Implement NeuralNetInference (C++ libtorch)
- **Spec**: `neural-network.md` §2 (C++ Inference Interface)
- **State**: missing
- **Description**: Create `src/nn/nn_inference.h` (abstract interface: `infer()`, `load_weights()`). Create `src/nn/libtorch_inference.h` and `libtorch_inference.cpp`. Implement batch inference using libtorch: load TorchScript model, run forward pass on GPU. Input/output in unified memory (no cudaMemcpy). Support loading new weights on checkpoint update.
- **Priority rationale**: Self-play eval queue needs C++ NN inference.
- **Acceptance criteria**:
  - Batch inference produces correct policy logits and value for known inputs
  - Weight loading from checkpoint file works
  - Unified memory — no explicit GPU memory transfers

### TASK-054: Implement pybind11 bindings
- **Spec**: `infrastructure.md` §1 (bindings/), `overview.md` §4
- **State**: missing
- **Description**: Create `src/bindings/python_bindings.cpp`. Expose to Python: GameState, GameConfig, ChessState, GoState, ReplayBuffer (sample method for training), SelfPlayManager (start/stop), EvalQueue. Bridge between C++ self-play engine and Python training loop.
- **Priority rationale**: Training loop (Python) needs to read from replay buffer and control self-play (C++).
- **Acceptance criteria**:
  - Python can create game states and call all interface methods
  - Python can sample from replay buffer
  - Python can start/stop self-play

---

## P5 — Training Pipeline and Infrastructure

### TASK-060: Implement training loop
- **Spec**: `pipeline.md` §6, `neural-network.md` §5
- **State**: missing
- **Description**: Create `python/alphazero/training/trainer.py`. Implement training loop: SGD with momentum 0.9, LR schedule, mixed precision (BF16 AMP + GradScaler), mini-batch sampling from replay buffer. Wait for min_buffer_size (10K) before training. Apply symmetry augmentation for Go. Log metrics every N steps. Checkpoint every 1K steps + export BN-folded weights. Milestone checkpoints every 50K steps.
- **Priority rationale**: Closes the self-play → train → improved network loop.
- **Acceptance criteria**:
  - Training step reduces loss on synthetic data
  - Mixed precision doesn't produce NaN
  - Gradients are non-zero
  - Checkpoint save/load round-trips correctly

### TASK-061: Implement GPU scheduling / pipeline orchestrator
- **Spec**: `pipeline.md` §3
- **State**: missing
- **Description**: Create `python/alphazero/pipeline/orchestrator.py`. Implement interleaved GPU scheduling: S inference batches (default 50) then T training steps (default 1). Coordinate self-play inference and training on single GPU. Weight updates visible immediately to next inference batch (unified memory). Support configurable S:T ratio.
- **Priority rationale**: Required for efficient single-GPU utilization.
- **Acceptance criteria**:
  - Inference and training interleave correctly
  - GPU utilization >80%
  - S:T ratio configurable

### TASK-062: Implement checkpointing
- **Spec**: `pipeline.md` §7
- **State**: missing
- **Description**: Create `python/alphazero/utils/checkpoint.py`. Save: model state_dict, optimizer state, training step, LR schedule state, replay buffer metadata. Every 1K steps (rolling, keep last 10). Milestone every 50K steps (permanent). Export BN-folded weights alongside each checkpoint. Support warm resume (load checkpoint, restart self-play with empty buffer).
- **Priority rationale**: Required for training run resumption and model deployment.
- **Acceptance criteria**:
  - Checkpoint save/load round-trips all state correctly
  - Rolling deletion keeps only last K checkpoints
  - Milestone checkpoints preserved
  - BN-folded weights exported correctly

### TASK-063: Implement TensorBoard logging
- **Spec**: `pipeline.md` §8, `infrastructure.md` §5
- **State**: missing
- **Description**: Create `python/alphazero/utils/logging.py`. Log all training metrics (loss/total, loss/policy, loss/value, loss/l2, lr, throughput, buffer size/games). Log self-play metrics (game length, outcome, resignation, moves/sec, games/hr). Write to `logs/<run_name>/`. Include periodic console summaries.
- **Priority rationale**: Required for monitoring training progress.
- **Acceptance criteria**:
  - All metrics from spec §8 are logged
  - TensorBoard can read the log files
  - Console output matches spec format

### TASK-064: Implement periodic Elo estimation
- **Spec**: `pipeline.md` §8 (Elo Estimation)
- **State**: missing
- **Description**: Create `python/alphazero/pipeline/evaluation.py`. Every 10K training steps, run evaluation match: current network vs milestone checkpoint, 100 sims/move, 50-100 games. Estimate Elo difference. Log as `eval/elo_vs_step_N`. Non-gating (monitoring only).
- **Priority rationale**: Provides human-readable training progress. Not blocking.
- **Acceptance criteria**:
  - Evaluation matches run and produce Elo estimates
  - Results logged to TensorBoard

---

## P6 — Scripts and Entry Points

### TASK-070: Implement main training script (train.py)
- **Spec**: `infrastructure.md` §6 (Running Training)
- **State**: missing
- **Description**: Create `scripts/train.py`. Accept `--config` (YAML path) and `--resume` (checkpoint path). Cold start: init random network, empty buffer, start self-play, begin training after min_buffer_size. Warm resume: load checkpoint, restart self-play. Graceful shutdown: signal threads, wait, save final checkpoint, flush metrics.
- **Priority rationale**: Main entry point for training runs.
- **Acceptance criteria**:
  - Cold start works for both chess and Go configs
  - Warm resume correctly continues from checkpoint
  - Graceful shutdown saves state

### TASK-071: Implement play script (play.py)
- **Spec**: `infrastructure.md` §7
- **State**: missing
- **Description**: Create `scripts/play.py`. Interactive human vs AI mode. AI vs external engine mode (e.g., Stockfish via UCI). Use MCTS with deterministic selection (τ→0), no Dirichlet noise, resignation enabled.
- **Priority rationale**: Useful for evaluation but not blocking training.
- **Acceptance criteria**:
  - Human can play against trained model interactively
  - Model can play against UCI engine

### TASK-072: Implement benchmark script (benchmark.py)
- **Spec**: `infrastructure.md` §6 (Performance Benchmarking)
- **State**: missing
- **Description**: Create `scripts/benchmark.py`. Benchmark inference throughput (positions/sec at various batch sizes). Benchmark training throughput (steps/sec). Benchmark MCTS throughput (sims/sec with various thread configs). Output results for tuning pipeline parameters.
- **Priority rationale**: Important for tuning but not blocking initial training.
- **Acceptance criteria**:
  - Reports inference, training, and MCTS throughput
  - Supports configurable batch sizes and thread counts

### TASK-073: Implement model export script (export_model.py)
- **Spec**: `infrastructure.md` §1 (scripts/)
- **State**: missing
- **Description**: Create `scripts/export_model.py`. Export trained model for deployment (TorchScript, ONNX, or similar).
- **Priority rationale**: Post-training utility. Lowest priority.
- **Acceptance criteria**:
  - Exports model in a format usable by inference engine

---

## P6 — Testing

### TASK-080: Implement chess perft tests
- **Spec**: `game-interface.md` §8 (Chess testing), `infrastructure.md` §4
- **State**: missing
- **Description**: Create `tests/cpp/test_chess_movegen.cpp`. Perft tests at depths 1-6 for initial position, kiwipete, and multiple endgame positions against known-correct counts.
- **Priority rationale**: Gold-standard correctness test for move generation. Should be written alongside TASK-011.
- **Acceptance criteria**: All perft counts match reference values

### TASK-081: Implement chess encoding tests
- **Spec**: `game-interface.md` §8, `infrastructure.md` §4
- **State**: missing
- **Description**: Create `tests/cpp/test_chess_encoding.cpp`. Verify input tensor for initial position and known mid-game positions. Verify action index <-> move round-trip. Verify board flipping for black-to-move. FEN round-trip tests.
- **Acceptance criteria**: All encoding/decoding tests pass

### TASK-082: Implement Go rules tests
- **Spec**: `game-interface.md` §8, `infrastructure.md` §4
- **State**: missing
- **Description**: Create `tests/cpp/test_go_rules.cpp`. Test capture scenarios (simple, snapback, large group). Ko detection and prohibition. Superko detection. Liberty counting. Tromp-Taylor scoring. Self-capture prohibition.
- **Acceptance criteria**: All Go rules tests pass

### TASK-083: Implement Go encoding tests
- **Spec**: `game-interface.md` §8, `infrastructure.md` §4
- **State**: missing
- **Description**: Create `tests/cpp/test_go_encoding.cpp`. Verify input tensor for known positions. Verify all 8 symmetry transforms. Verify policy vector transforms consistent with board transforms.
- **Acceptance criteria**: All Go encoding and symmetry tests pass

### TASK-084: Implement MCTS tests
- **Spec**: `infrastructure.md` §4
- **State**: missing
- **Description**: Create `tests/cpp/test_mcts.cpp`. Mock NN tests: verify visit count convergence, backup value negation, FPU computation, Dirichlet noise at root only, tree reuse statistics preservation.
- **Acceptance criteria**: All MCTS correctness tests pass

### TASK-085: Implement eval queue tests
- **Spec**: `infrastructure.md` §4
- **State**: missing
- **Description**: Create `tests/cpp/test_eval_queue.cpp`. Multi-producer single-consumer threading test. Verify all requests processed. Verify flush timeout. Stress test under high contention.
- **Acceptance criteria**: No deadlocks, all results dispatched correctly

### TASK-086: Implement arena node store tests
- **Spec**: `infrastructure.md` §4
- **State**: missing
- **Description**: Create `tests/cpp/test_arena.cpp`. Test allocation, release_subtree, reset, memory tracking.
- **Acceptance criteria**: All allocation and deallocation tests pass

### TASK-087: Implement replay buffer tests
- **Spec**: `infrastructure.md` §4
- **State**: missing
- **Description**: Create `tests/cpp/test_replay_buffer.cpp`. Concurrent write/read tests. Ring buffer wrapping. No data corruption.
- **Acceptance criteria**: All concurrency tests pass without data races

### TASK-088: Implement Python network tests
- **Spec**: `infrastructure.md` §4
- **State**: missing
- **Description**: Create `tests/python/test_network.py`. Instantiate ResNet+SE with various configs. Verify output shapes for chess and Go. Verify policy and value head dimensions.
- **Acceptance criteria**: All shape tests pass for all configs

### TASK-089: Implement Python loss function tests
- **Spec**: `infrastructure.md` §4
- **State**: missing
- **Description**: Create `tests/python/test_loss.py`. Verify policy cross-entropy, value MSE, value CE, and L2 regularization against hand-computed values.
- **Acceptance criteria**: All loss values match hand-computed references

### TASK-090: Implement Python BN folding tests
- **Spec**: `infrastructure.md` §4
- **State**: missing
- **Description**: Create `tests/python/test_bn_fold.py`. Run inference before and after BN folding. Verify outputs match within 1e-5.
- **Acceptance criteria**: Folded model outputs match within tolerance

### TASK-091: Implement Python training step test
- **Spec**: `infrastructure.md` §4
- **State**: missing
- **Description**: Create `tests/python/test_training.py`. Run single training step on synthetic data. Verify loss decreases, gradients non-zero, no NaN from mixed precision.
- **Acceptance criteria**: Training step completes without error

### TASK-092: Implement integration smoke test
- **Spec**: `infrastructure.md` §4 (Integration Tests — Smoke test)
- **State**: missing
- **Description**: Full pipeline (self-play + training) for 100 training steps. Verify replay buffer fills, checkpoints saved, metrics logged. No crash.
- **Priority rationale**: Validates the entire system end-to-end.
- **Acceptance criteria**: Pipeline runs 100 steps without crashing

### TASK-093: Implement Connect Four learning test
- **Spec**: `infrastructure.md` §4 (Integration Tests — Learning test)
- **State**: missing
- **Description**: Implement Connect Four as a simple GameState. Run AlphaZero training for short duration. Verify trained model beats random player >90%.
- **Priority rationale**: End-to-end algorithm validation on a tractable problem.
- **Acceptance criteria**: Trained model wins >90% against random

---

## Dependency Graph (Simplified)

```
TASK-001 (scaffolding)
  ├── TASK-002 (GameState/GameConfig interfaces)
  │     ├── TASK-010..014 (Chess implementation)
  │     └── TASK-020..026 (Go implementation)
  ├── TASK-003 (Python GameConfig)
  │     └── TASK-030..035 (Neural network)
  │
  TASK-010..014 + TASK-020..026 (Games)
  + TASK-030..035 (Network)
    ├── TASK-040..043 (MCTS)
    │     ├── TASK-050..054 (Self-play pipeline)
    │     │     ├── TASK-060..064 (Training pipeline)
    │     │     │     └── TASK-070..073 (Scripts)
    │     │     │           └── TASK-092..093 (Integration tests)

  Tests (TASK-080..091) can be written alongside their corresponding implementation tasks.
```

---

## Summary

| Priority | Tasks | Count | Description |
|----------|-------|-------|-------------|
| P0 | 001-003 | 3 | Project scaffolding, interfaces, config |
| P1 | 010-026 | 11 | Chess and Go game implementations |
| P2 | 030-035 | 6 | Neural network (PyTorch) |
| P3 | 040-043 | 4 | MCTS search engine |
| P4 | 050-054 | 5 | Self-play pipeline, inference, bindings |
| P5 | 060-064 | 5 | Training pipeline, monitoring |
| P6 | 070-073 | 4 | Scripts and entry points |
| P6 | 080-093 | 14 | Tests (unit, component, integration) |
| **Total** | | **52** | |
