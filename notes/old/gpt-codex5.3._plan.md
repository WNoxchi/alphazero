# IMPLEMENTATION PLAN

## Gap Analysis Summary

- Scope analyzed: `specs/overview.md`, `specs/game-interface.md`, `specs/neural-network.md`, `specs/mcts.md`, `specs/pipeline.md`, `specs/infrastructure.md`.
- Codebase reality: no `src/`, `python/`, `tests/`, `configs/`, or `scripts/` directories currently exist.
- Existing files are specification and prompt documents only.
- Fully implemented and passing tests: none identified.
- Partially implemented (stubs/placeholders/TODO): none identified in implementation code because implementation code is absent.
- Implemented but inconsistent with spec: none identified because implementation code is absent.

Evidence checks used before marking items missing:
- File-tree search: `find . -maxdepth 3 -type d`, `rg --files`, and path-filter checks for `src|python|tests|configs|scripts`.
- Exact-name search: symbols such as `GameState`, `GameConfig`, `AlphaZeroNetwork`, `MCTSNode`, `NodeStore`, `EvalQueue`, `ReplayBuffer`, `NeuralNetInference`.
- Related-term search: terms like `bitboard`, `movegen`, `c_puct`, `virtual_loss`, `dirichlet`, `resnet_se`, `bn_fold`, `checkpoint`, `self_play_manager`.
- TODO/stub search: `TODO|FIXME|HACK|PLACEHOLDER|stub|not implemented|minimal implementation|skip|skipped|ignored|xfail`.
- Result pattern: matches appear in spec/prompt docs only; no implementation artifacts found.

## Prioritized Tasks

- TASK-001: Establish project scaffolding and build/test entry points.
  - Spec trace: `specs/infrastructure.md` §1-3, `specs/overview.md` §4-6.
  - Current state: missing.
  - Priority rationale: blocking foundation for all implementation and validation.
  - Acceptance criteria summary: repository includes top-level `CMakeLists.txt`, `pyproject.toml`, `configs/`, `src/`, `python/`, `tests/`, and `scripts/` structure aligned with spec paths; build/test targets are declared.

- TASK-002: Implement cross-game abstractions (`GameState`, `GameConfig`, symmetry interface) and Python game config models.
  - Spec trace: `specs/game-interface.md` §2, §7.
  - Current state: missing.
  - Priority rationale: required contract for chess/go, MCTS, replay, NN, and pipeline integration.
  - Acceptance criteria summary: C++ abstract interfaces compile; Python config dataclasses for chess/go exist and are consumable by network and training code.

- TASK-003: Implement chess state engine, legal move generation, terminal logic, and encoding/action mapping.
  - Spec trace: `specs/game-interface.md` §5, §8 (Chess tests).
  - Current state: missing.
  - Priority rationale: core game backend dependency for chess self-play and training.
  - Acceptance criteria summary: chess module supports bitboard representation, legal move generation (including castling/en-passant/promotion), terminal adjudication rules, 8x8x119 encoding, 4672 action mapping, and FEN/PGN I/O; perft and encoding tests pass.

- TASK-004: Implement go state engine, liberties/capture/ko/superko, Tromp-Taylor scoring, encoding/action mapping, and symmetry transforms.
  - Spec trace: `specs/game-interface.md` §6, §8 (Go tests).
  - Current state: missing.
  - Priority rationale: core game backend dependency for go self-play and training.
  - Acceptance criteria summary: go module supports 19x19 state transitions, capture/self-capture legality, ko + superko enforcement, Tromp-Taylor scoring, 19x19x17 encoding, 362 action mapping, SGF export, and 8 symmetry transforms; go rules/encoding/symmetry tests pass.

- TASK-005: Implement neural network interface and ResNet+SE model with policy/value heads (scalar + WDL).
  - Spec trace: `specs/neural-network.md` §2-3, §9.
  - Current state: missing.
  - Priority rationale: required for both MCTS evaluation and training.
  - Acceptance criteria summary: `AlphaZeroNetwork` contract implemented, configurable ResNet+SE variants (small/medium/large), correct output dimensions for chess/go, and architecture factory wiring completed.

- TASK-006: Implement training losses, AMP/BF16 training path, LR schedule, and BN-fold export path.
  - Spec trace: `specs/neural-network.md` §4-8.
  - Current state: missing.
  - Priority rationale: required to turn self-play data into improving networks.
  - Acceptance criteria summary: policy + value losses and explicit L2 regularization implemented for scalar/WDL modes; SGD-momentum schedule works per milestones; AMP/BF16 training step stable; BN fold utility reproduces pre-fold outputs within tolerance.

- TASK-007: Implement C++ NN inference bridge and Python bindings (`NeuralNetInference`, libtorch/pybind integration).
  - Spec trace: `specs/neural-network.md` §2, `specs/infrastructure.md` §1-2.
  - Current state: missing.
  - Priority rationale: integration seam between C++ self-play/MCTS and Python-trained model.
  - Acceptance criteria summary: batched infer API works with expected tensor shapes, weight loading supported, and Python bindings build/load successfully.

- TASK-008: Implement MCTS node model, node store (arena), PUCT select/expand/evaluate/backup, FPU, root Dirichlet, temperature policy, and tree reuse.
  - Spec trace: `specs/mcts.md` §2-5, §8-11, §14.
  - Current state: missing.
  - Priority rationale: principal decision algorithm for self-play and gameplay quality.
  - Acceptance criteria summary: deterministic correctness against mock-network scenarios, backup sign handling correct, root-only Dirichlet verified, FPU behavior validated, and subtree reuse preserves expected statistics.

- TASK-009: Implement evaluation queue with batching/timeout and concurrent request handling.
  - Spec trace: `specs/mcts.md` §7, `specs/infrastructure.md` §4 (eval queue threading tests).
  - Current state: missing.
  - Priority rationale: critical performance and correctness component for asynchronous MCTS inference.
  - Acceptance criteria summary: MPSC submission + single-consumer batching works under contention; flush-on-size and flush-on-timeout behaviors verified; all submitted requests receive results without corruption.

- TASK-010: Implement tree-parallelism safeguards (virtual loss and synchronization primitives).
  - Spec trace: `specs/mcts.md` §5-6.
  - Current state: missing.
  - Priority rationale: needed to scale MCTS safely across K threads per game.
  - Acceptance criteria summary: virtual loss apply/revert sequencing is correct, race-free updates for visit/value statistics are validated, and threaded search does not deadlock.

- TASK-011: Implement self-play manager and single-game lifecycle orchestration.
  - Spec trace: `specs/pipeline.md` §4, `specs/mcts.md` §10-12.
  - Current state: missing.
  - Priority rationale: orchestrates game slots, move selection, resignation, and sample generation feeding replay/training.
  - Acceptance criteria summary: concurrent game slots run continuously, move policy targets derived from visit counts with temperature schedule, resignation calibration mode supported, terminal outcomes backfilled into stored trajectories.

- TASK-012: Implement replay buffer (thread-safe ring buffer) with sampling API.
  - Spec trace: `specs/pipeline.md` §5, `specs/infrastructure.md` §4 (replay tests).
  - Current state: missing.
  - Priority rationale: central data exchange between self-play and training.
  - Acceptance criteria summary: concurrent game writes + training reads are race-safe, wraparound semantics correct, uniform random sampling works, and minimum-fill gating supported.

- TASK-013: Implement asynchronous pipeline orchestrator (S:T interleaving), startup/resume/shutdown flows, and checkpoint lifecycle.
  - Spec trace: `specs/pipeline.md` §2-3, §6-7, §9-11.
  - Current state: missing.
  - Priority rationale: integrates all subsystems into an operational AlphaZero loop.
  - Acceptance criteria summary: interleaved inference/training cycle executes with configurable ratios; cold start and resume paths function; rolling and milestone checkpoints save/load correctly; graceful shutdown preserves final state.

- TASK-014: Implement operational scripts and runtime tooling (`train.py`, `play.py`, `benchmark.py`, export path).
  - Spec trace: `specs/infrastructure.md` §1, §6-7; `specs/pipeline.md` §9.
  - Current state: missing.
  - Priority rationale: required for running, evaluating, and benchmarking the system in practice.
  - Acceptance criteria summary: scripts accept config-driven parameters, execute intended mode, and integrate with engine/network modules.

- TASK-015: Implement full test suite (C++ unit/component + Python unit/component + integration smoke/learning tests) and connect to build targets.
  - Spec trace: `specs/infrastructure.md` §4, `specs/game-interface.md` §8.
  - Current state: missing.
  - Priority rationale: verification gate for correctness, regression prevention, and safe iteration.
  - Acceptance criteria summary: C++ and Python test targets are runnable from declared commands; coverage includes chess/go correctness, MCTS correctness, queue/buffer concurrency, network/loss/bn-fold/training-step checks, plus smoke and learning integration tests.

- TASK-016: Implement monitoring and evaluation telemetry (TensorBoard scalars, throughput, periodic Elo estimation).
  - Spec trace: `specs/pipeline.md` §8, `specs/infrastructure.md` §5.
  - Current state: missing.
  - Priority rationale: non-blocking for initial compile but required for controlled training operations and diagnostics.
  - Acceptance criteria summary: defined metric set is emitted to log directories, periodic Elo jobs run in non-gating mode, and console summaries report key health indicators.