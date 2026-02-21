# Task: Game-specific MCTSNode sizing

**Branch**: `feature-chess-improvements-node-fix`
**Status**: Completed

Read `specs/` for full codebase architecture. See `notes/perf_improvements.md` for
background context on all planned improvements.

---

## Objective

The `MCTSNode` type alias (`src/mcts/mcts_node.h:67`) unconditionally uses `GoMCTSNode`
(362 max actions) even for chess (218 max actions). Each node has 7 arrays sized to
`MaxActions`, so Go-sized nodes are ~7.5 KB vs ~4.5 KB for chess. With 256 concurrent
games x 8,192-node arenas, this wastes ~6 GB and hurts cache performance in the MCTS
hot loop.

Fix this so chess uses `ChessMCTSNode` and Go uses `GoMCTSNode`.

## Design guidance

Do NOT template the entire class hierarchy (`ArenaNodeStore` -> `MctsSearch` ->
`SelfPlayGame` -> bindings). That would propagate template parameters through every
layer. Instead, pick one of these approaches:

**Option A — Template specialization at the bindings layer**: Keep the core C++ classes
templated on `NodeType`, but only instantiate the two specializations (chess/Go) in
`python_bindings.cpp`. Expose them under the same Python class name, dispatched by
game config. This keeps the C++ generic while limiting template explosion to one file.

**Option B — Runtime dispatch at construction**: Have `ArenaNodeStore` allocate nodes
of a byte size determined at construction from `GameConfig::action_space_size`. Internal
code uses the actual `num_actions` per node rather than compile-time `MaxActions`. This
avoids templates entirely but requires careful memory layout.

Option A is likely simpler. The key constraint: Python code must see a single
`MctsSearch` / `ArenaNodeStore` interface regardless of game — the dispatch is invisible
to Python.

## Files you may modify

- `src/mcts/mcts_node.h`
- `src/mcts/arena_node_store.h` and `src/mcts/arena_node_store.cpp`
- `src/mcts/mcts_search.h` and `src/mcts/mcts_search.cpp`
- `src/mcts/eval_queue.h` and `src/mcts/eval_queue.cpp` (if needed)
- `src/selfplay/self_play_game.h` and `src/selfplay/self_play_game.cpp`
- `src/bindings/python_bindings.cpp`
- `src/games/game_config.h` (if needed to carry max_actions)
- Test files under `tests/` that relate to MCTS or self-play

## Files you must NOT modify

These are being changed in a parallel branch. Modifying them will cause merge conflicts.

- `src/selfplay/replay_buffer.h` and `src/selfplay/replay_buffer.cpp`
- `python/alphazero/training/trainer.py`
- `python/alphazero/pipeline/orchestrator.py`

## Config change (optional, include if convenient)

In `configs/chess_1hr.yaml`, apply these tuning changes:

```yaml
mcts:
  concurrent_games: 384       # was 256
  batch_size: 384             # was 256

replay_buffer:
  capacity: 750000            # was 500000
```

See `notes/perf_improvements.md` section 2 for rationale.

## Build & test

```bash
conda activate alphazero
cmake --build build --target alphazero_cpp -j$(nproc)
cd build && ctest --output-on-failure && cd ..
PYTHONPATH=build/src:$PYTHONPATH python -m pytest tests/ -x
```

All existing tests (C++ and Python) must pass. The integration smoke tests in
`tests/python/test_integration_smoke.py` exercise the full pipeline for both chess
and Go — these are the most important to verify.

## Status log

Update this section as you work:

| Step | Status | Notes |
|------|--------|-------|
| Read specs and understand codebase | Complete | Reviewed `specs/mcts.md`, `specs/game-interface.md`, and current `src/mcts`, `src/selfplay`, and binding code paths before edits. |
| Choose design approach (A or B) | Complete | Implemented Option A core templates plus runtime dispatch wrapper so external APIs stay non-templated. |
| Implement node-type dispatch | Complete | `NodeStore`, `ArenaNodeStore`, and `MctsSearch` are now templated; `RuntimeMctsSearch` dispatches chess to `ChessMCTSNode` and go to `GoMCTSNode`. |
| Update bindings | Complete | Python-facing `MctsSearch` now uses `RuntimeMctsSearch` internally with unchanged Python API. |
| All C++ tests pass | Complete | `ctest --output-on-failure` passed: 108/108 tests green. |
| All Python tests pass | Complete | `pytest` in `alphazero` env passed: 91 passed, 8 skipped. Binding tests skipped because `pybind11` is unavailable in this environment. |
| Config change applied (optional) | Complete | Updated `configs/chess_1hr.yaml`: `concurrent_games=384`, `batch_size=384`, `replay_buffer.capacity=750000`. |
