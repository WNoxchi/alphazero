# Implementation Plan: Watch Mode (Two AIs Play Each Other)

## Overview

Add a watch mode to the chess web app where two AI models (same or different checkpoints) play against each other automatically. The user selects models from dropdowns, controls speed, and can pause/resume.

**All changes are confined to `web/`.** No modifications to `scripts/`, `python/alphazero/`, or C++ code.

**Assumption:** All checkpoints share the same architecture (num_blocks, num_filters, se_reduction), specified once at server startup.

---

## File Change Summary

| File | Action | ~Lines |
|------|--------|--------|
| `web/model_manager.py` | CREATE | 90 |
| `web/watch_engine.py` | CREATE | 120 |
| `web/server.py` | MODIFY | +90 |
| `web/static/watch.html` | CREATE | 85 |
| `web/static/js/watch.js` | CREATE | 300 |
| `web/static/css/watch.css` | CREATE | 50 |
| `web/static/index.html` | MODIFY | +3 |
| `web/static/css/style.css` | MODIFY | +10 |

---

## Step 1: Create `web/model_manager.py`

This class scans a checkpoint directory and lazily loads/caches `PlayRuntime` objects.

### Imports and dependencies

```python
from __future__ import annotations

import argparse
import logging
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("alphazero.web.models")
```

Uses `build_play_runtime` and `load_runtime_dependencies` from `scripts/play.py` (already on sys.path via `web/engine.py`'s path setup — reuse that same pattern).

### Checkpoint scanning logic

The core alphazero code defines checkpoint filenames as `(checkpoint|milestone)_XXXXXXXX.pt` (regex: `^(checkpoint|milestone)_(\d{8})\.pt$`). Folded variants are `*_folded.pt`. The existing `alphazero.utils.checkpoint.list_checkpoints()` returns non-folded paths sorted by step, but we want to **prefer** folded variants for inference (smaller, faster).

```python
_CHECKPOINT_RE = re.compile(r"^(checkpoint|milestone)_(\d{8})\.pt$")
```

### Class: `ModelManager`

```python
class ModelManager:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        simulations: int = 800,
        num_blocks: int = 20,
        num_filters: int = 256,
        se_reduction: int = 4,
        device: str | None = None,
        fp32: bool = False,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir).resolve()
        self._simulations = simulations
        self._num_blocks = num_blocks
        self._num_filters = num_filters
        self._se_reduction = se_reduction
        self._device = device
        self._fp32 = fp32
        self._cache: dict[str, Any] = {}  # name -> PlayRuntime
        self._lock = threading.Lock()
        self._deps = None  # RuntimeDependencies, loaded once
```

### Method: `list_models()`

Scan `checkpoint_dir` for `.pt` files matching the pattern. For each non-folded checkpoint, check if a `_folded.pt` sibling exists and prefer it. Return sorted by step ascending.

```python
def list_models(self) -> list[dict[str, Any]]:
    if not self._checkpoint_dir.exists():
        return []
    entries = []
    for path in self._checkpoint_dir.glob("*.pt"):
        match = _CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        kind, step_digits = match.groups()
        step = int(step_digits)
        # Prefer folded variant for inference
        folded = path.parent / f"{kind}_{step_digits}_folded.pt"
        actual_path = folded if folded.exists() else path
        name = f"{kind}_{step_digits}"  # e.g. "checkpoint_00010000"
        entries.append({
            "name": name,
            "display_name": f"{kind} step {step:,}",
            "path": str(actual_path),
            "step": step,
        })
    entries.sort(key=lambda e: e["step"])
    # Deduplicate: if both checkpoint and milestone exist at same step, keep both
    return entries
```

### Method: `get_runtime(name)`

Load a `PlayRuntime` lazily, cache by name. Thread-safe.

```python
def get_runtime(self, name: str) -> Any:
    with self._lock:
        if name in self._cache:
            return self._cache[name]

    # Find the model entry
    models = self.list_models()
    entry = next((m for m in models if m["name"] == name), None)
    if entry is None:
        raise ValueError(f"Unknown model: {name}")

    # Lazy-import and build runtime
    from scripts.play import build_play_runtime, load_runtime_dependencies

    with self._lock:
        # Double-check after acquiring lock
        if name in self._cache:
            return self._cache[name]

        if self._deps is None:
            self._deps = load_runtime_dependencies()

        args = argparse.Namespace(
            game="chess",
            model=entry["path"],
            simulations=self._simulations,
            games=1,
            human_color="white",
            engine_time_ms=1000,
            device=self._device,
            num_blocks=self._num_blocks,
            num_filters=self._num_filters,
            se_reduction=self._se_reduction,
            fp32=self._fp32,
            c_puct=2.5,
            c_fpu=0.25,
            resign_threshold=-0.9,
            search_random_seed=0xC0FFEE1234567890,
            node_arena_capacity=262_144,
            opponent="human",
        )
        logger.info("Loading model %s from %s", name, entry["path"])
        runtime = build_play_runtime(args=args, dependencies=self._deps)
        self._cache[name] = runtime
        logger.info("Model %s loaded", name)
        return runtime
```

**Key detail:** `build_play_runtime` accepts a `dependencies` parameter. By passing the same `RuntimeDependencies` instance, we avoid re-importing heavy modules (torch, alphazero_cpp) on each load. Each call still creates a fresh model, loads weights, and builds a new evaluator.

**Same model for both sides:** When `white_model == black_model`, `get_runtime()` returns the exact same `PlayRuntime` from cache. This is correct — the model and evaluator are stateless (eval mode, `torch.no_grad()`). The `WatchEngine` creates separate MCTS search trees regardless.

---

## Step 2: Create `web/watch_engine.py`

Dual-MCTS game driver. Mirrors `ChessEngine` (engine.py:31-203) but manages two search trees.

### Imports

```python
from __future__ import annotations

from typing import Any
```

### Class: `WatchEngine`

```python
class WatchEngine:
    def __init__(
        self,
        white_runtime: Any,  # PlayRuntime
        black_runtime: Any,  # PlayRuntime
        *,
        node_arena_capacity: int = 262_144,
    ) -> None:
        self._white_runtime = white_runtime
        self._black_runtime = black_runtime
        self._node_arena_capacity = node_arena_capacity
        self._cpp = white_runtime.dependencies.cpp
        self._state: Any = None
        self._white_search: Any = None
        self._black_search: Any = None
        self._action_history: list[int] = []
```

### Method: `reset()`

```python
def reset(self) -> dict[str, Any]:
    self._state = self._cpp.ChessState()
    self._action_history = []
    self._white_search = self._cpp.MctsSearch(
        self._white_runtime.cpp_game_config,
        self._white_runtime.search_config,
        self._node_arena_capacity,
    )
    self._black_search = self._cpp.MctsSearch(
        self._black_runtime.cpp_game_config,
        self._black_runtime.search_config,
        self._node_arena_capacity,
    )
    return self._board_state()
```

### Method: `get_next_move()`

This is the core method. Called repeatedly by the auto-play loop.

```python
def get_next_move(self) -> dict[str, Any]:
    if bool(self._state.is_terminal()):
        return self._board_state()

    current_player = int(self._state.current_player())
    if current_player == 0:  # white
        search, runtime = self._white_search, self._white_runtime
    else:  # black
        search, runtime = self._black_search, self._black_runtime

    search.set_root_state(self._state)
    search.run_simulations(runtime.evaluator)

    # Check resignation
    if bool(search.should_resign()):
        result = "0-1" if current_player == 0 else "1-0"
        return {**self._board_state(), "resigned": True, "result": result}

    move_number = len(self._action_history) + 1
    action = int(search.select_action(move_number))
    uci = str(self._state.action_to_uci(action))

    # Eval score from selected edge
    eval_score = None
    try:
        stats = search.root_edge_stats(action)
        if stats is not None:
            eval_score = float(stats.mean_value)
    except Exception:
        pass

    self._state = self._state.apply_action(action)
    self._action_history.append(action)
    return self._board_state(eval_score=eval_score, last_uci=uci)
```

### Method: `_board_state()`

Duplicate of `ChessEngine._board_state()` from engine.py:91-124. Same dict shape.

```python
def _board_state(
    self, eval_score: float | None = None, last_uci: str | None = None
) -> dict[str, Any]:
    is_terminal = bool(self._state.is_terminal())
    result = None
    if is_terminal:
        outcome = float(self._state.outcome(0))
        if outcome > 0:
            result = "1-0"
        elif outcome < 0:
            result = "0-1"
        else:
            result = "1/2-1/2"

    legal_uci = []
    legal_dests: dict[str, list[str]] = {}
    if not is_terminal:
        for action_id, uci_str in self._state.legal_actions_uci():
            legal_uci.append(uci_str)
            src = uci_str[:2]
            dst = uci_str[2:4]
            legal_dests.setdefault(src, []).append(dst)

    return {
        "fen": str(self._state.to_fen()),
        "legal_moves": legal_uci,
        "legal_dests": legal_dests,
        "is_terminal": is_terminal,
        "result": result,
        "eval": eval_score,
        "last_move": last_uci,
        "move_number": len(self._action_history),
        "current_player": int(self._state.current_player()),
    }
```

### Method: `get_pgn()`

```python
def get_pgn(self, *, white_name: str = "White", black_name: str = "Black") -> str:
    if not self._action_history:
        return ""
    is_terminal = bool(self._state.is_terminal())
    if is_terminal:
        outcome = float(self._state.outcome(0))
        if outcome > 0:
            result = "1-0"
        elif outcome < 0:
            result = "0-1"
        else:
            result = "1/2-1/2"
    else:
        result = "*"
    pgn_body = str(self._cpp.ChessState.actions_to_pgn(self._action_history, result))
    # Prepend headers with model names
    headers = f'[White "{white_name}"]\n[Black "{black_name}"]\n\n'
    return headers + pgn_body
```

---

## Step 3: Modify `web/server.py`

### 3a. New imports (add near top)

```python
from web.model_manager import ModelManager
from web.watch_engine import WatchEngine
```

### 3b. New global (after `_engine` global on line 30)

```python
_model_manager: ModelManager | None = None
```

### 3c. New REST endpoint (after `app.mount` on line 44)

```python
@app.get("/api/models")
async def list_models():
    if _model_manager is None:
        return {"models": []}
    models = _model_manager.list_models()
    # Strip internal 'path' field, only send name/display_name/step
    return {"models": [{"name": m["name"], "display_name": m["display_name"], "step": m["step"]} for m in models]}
```

### 3d. New watch page route (after index route)

```python
@app.get("/watch")
async def watch_page():
    return FileResponse(STATIC_DIR / "watch.html")
```

### 3e. New WebSocket endpoint `/ws/watch` (after the existing `/ws/chess` handler)

```python
@app.websocket("/ws/watch")
async def watch_ws(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    watch_engine: WatchEngine | None = None
    auto_play_task: asyncio.Task | None = None
    paused = False
    delay_ms = 1000
    white_name = ""
    black_name = ""

    async def auto_play_loop():
        nonlocal watch_engine, paused
        try:
            while watch_engine is not None:
                if paused:
                    await asyncio.sleep(0.1)
                    continue
                state = await loop.run_in_executor(None, watch_engine.get_next_move)
                await websocket.send_json({"type": "watch_move", **state})
                if state.get("is_terminal") or state.get("resigned"):
                    pgn = watch_engine.get_pgn(
                        white_name=white_name, black_name=black_name
                    )
                    await websocket.send_json({
                        "type": "watch_game_over",
                        "result": state.get("result", "*"),
                        "pgn": pgn,
                    })
                    break
                await asyncio.sleep(delay_ms / 1000.0)
        except asyncio.CancelledError:
            pass

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type")

            if msg_type == "watch_start":
                if _model_manager is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No checkpoint directory configured (use --checkpoint-dir)",
                    })
                    continue

                # Cancel any existing auto-play
                if auto_play_task is not None and not auto_play_task.done():
                    auto_play_task.cancel()
                    await asyncio.sleep(0)  # Let cancellation propagate

                white_name = msg.get("white_model", "")
                black_name = msg.get("black_model", "")
                delay_ms = max(100, min(10000, msg.get("delay_ms", 1000)))
                paused = False

                try:
                    await websocket.send_json({"type": "watch_loading"})
                    white_rt = await loop.run_in_executor(
                        None, _model_manager.get_runtime, white_name
                    )
                    black_rt = await loop.run_in_executor(
                        None, _model_manager.get_runtime, black_name
                    )
                except Exception as exc:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Failed to load model: {exc}",
                    })
                    continue

                watch_engine = WatchEngine(white_rt, black_rt)
                state = watch_engine.reset()
                await websocket.send_json({"type": "watch_state", **state})
                auto_play_task = asyncio.create_task(auto_play_loop())

            elif msg_type == "watch_pause":
                paused = True
                await websocket.send_json({"type": "watch_paused"})

            elif msg_type == "watch_resume":
                paused = False
                await websocket.send_json({"type": "watch_resumed"})

            elif msg_type == "watch_set_speed":
                delay_ms = max(100, min(10000, msg.get("delay_ms", 1000)))
                await websocket.send_json({
                    "type": "watch_speed_ack",
                    "delay_ms": delay_ms,
                })

            elif msg_type == "watch_stop":
                if auto_play_task is not None and not auto_play_task.done():
                    auto_play_task.cancel()
                watch_engine = None
                await websocket.send_json({"type": "watch_stopped"})

            elif msg_type == "watch_pgn":
                if watch_engine is not None:
                    pgn = watch_engine.get_pgn(
                        white_name=white_name, black_name=black_name
                    )
                    await websocket.send_json({"type": "pgn", "pgn": pgn})

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        if auto_play_task is not None and not auto_play_task.done():
            auto_play_task.cancel()
        logger.info("Watch client disconnected")
```

### 3f. New CLI arg in `parse_args()` (server.py:119-130)

Add after the existing `--fp32` argument:

```python
parser.add_argument("--checkpoint-dir", default=None,
    help="Directory with checkpoints for watch mode model selection")
```

### 3g. ModelManager initialization in `main()` (server.py:133-152)

Add after `_engine` initialization (before `logger.info("Engine ready...")` line):

```python
if args.checkpoint_dir:
    global _model_manager
    _model_manager = ModelManager(
        checkpoint_dir=args.checkpoint_dir,
        simulations=args.simulations,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        se_reduction=args.se_reduction,
        device=args.device,
        fp32=args.fp32,
    )
    logger.info("Watch mode enabled with checkpoint dir: %s", args.checkpoint_dir)
```

---

## Step 4: Create `web/static/watch.html`

Same CDN imports as `index.html`. Key differences: nav bar, model dropdowns, speed slider, different controls, loads `watch.js`.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AlphaZero Chess — Watch</title>
  <link rel="stylesheet" href="https://unpkg.com/chessground@9.2.1/assets/chessground.base.css">
  <link rel="stylesheet" href="https://unpkg.com/chessground@9.2.1/assets/chessground.brown.css">
  <link rel="stylesheet" href="https://unpkg.com/chessground@9.2.1/assets/chessground.cburnett.css">
  <link rel="stylesheet" href="/static/css/style.css">
  <link rel="stylesheet" href="/static/css/watch.css">
</head>
<body>
  <div id="app">
    <header>
      <h1>AlphaZero Chess</h1>
      <nav>
        <a href="/">Play</a>
        <a href="/watch" class="active">Watch</a>
      </nav>
      <div id="status-bar">
        <span id="status-text">Select models to begin</span>
      </div>
    </header>

    <main>
      <div id="board-panel">
        <div id="player-top" class="player-info">
          <span class="player-name" id="black-label">Black</span>
        </div>
        <div id="board-container">
          <div id="board"></div>
        </div>
        <div id="player-bottom" class="player-info">
          <span class="player-name" id="white-label">White</span>
        </div>
      </div>

      <div id="side-panel">
        <div id="model-selection">
          <div class="model-picker">
            <label for="white-select">White</label>
            <select id="white-select"><option value="">Loading...</option></select>
          </div>
          <div class="model-picker">
            <label for="black-select">Black</label>
            <select id="black-select"><option value="">Loading...</option></select>
          </div>
        </div>

        <div id="speed-control">
          <label for="speed-slider">Delay: <span id="speed-value">1.0s</span></label>
          <input type="range" id="speed-slider" min="200" max="5000" step="100" value="1000">
        </div>

        <div id="eval-bar-container">
          <div id="eval-bar">
            <div id="eval-fill"></div>
          </div>
          <span id="eval-text">0.0</span>
        </div>

        <div id="move-history">
          <div id="move-list"></div>
        </div>

        <div id="controls">
          <button id="btn-start">Start</button>
          <button id="btn-pause" disabled>Pause</button>
          <button id="btn-stop" disabled>Stop</button>
          <button id="btn-pgn" disabled>PGN</button>
        </div>

        <div id="game-result" class="hidden">
          <span id="result-text"></span>
        </div>
      </div>
    </main>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
  <script type="module" src="/static/js/watch.js"></script>
</body>
</html>
```

---

## Step 5: Create `web/static/js/watch.js`

Follows `app.js` patterns closely. Key differences: `viewOnly: true` board, model fetching, auto-play message handling.

### State variables

```javascript
import { Chessground } from "https://unpkg.com/chessground@9.2.1/dist/chessground.min.js";

let cg = null;
let ws = null;
let currentFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
let moveList = [];       // [{number, white, black}, ...]
let gameActive = false;
let isPaused = false;
const chess = new Chess();
```

### DOM refs

Same pattern as app.js. Reference all elements by ID.

### `initBoard()` — read-only

```javascript
function initBoard() {
  cg = Chessground(boardEl, {
    orientation: "white",
    fen: currentFen,
    turnColor: "white",
    viewOnly: true,
    animation: { enabled: true, duration: 200 },
  });
}
```

### `loadModels()` — fetch from REST API

```javascript
async function loadModels() {
  try {
    const resp = await fetch("/api/models");
    const data = await resp.json();
    populateSelects(data.models);
  } catch (e) {
    setStatus("Failed to load model list", "");
  }
}

function populateSelects(models) {
  const whiteSelect = document.getElementById("white-select");
  const blackSelect = document.getElementById("black-select");
  whiteSelect.innerHTML = "";
  blackSelect.innerHTML = "";

  if (models.length === 0) {
    const opt = new Option("No models available", "");
    whiteSelect.add(opt);
    blackSelect.add(opt.cloneNode(true));
    document.getElementById("btn-start").disabled = true;
    return;
  }

  for (const m of models) {
    whiteSelect.add(new Option(m.display_name, m.name));
    blackSelect.add(new Option(m.display_name, m.name));
  }
  // Default: latest for both (same model vs itself)
  whiteSelect.selectedIndex = models.length - 1;
  blackSelect.selectedIndex = models.length - 1;
}
```

### `connect()` — WebSocket to `/ws/watch`

```javascript
function connect() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${location.host}/ws/watch`);
  ws.onopen = () => setStatus("Ready — select models and press Start", "connected");
  ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
  ws.onclose = () => setStatus("Disconnected — refresh to reconnect", "");
  ws.onerror = () => setStatus("Connection error", "");
}
```

### `handleMessage(msg)`

```javascript
function handleMessage(msg) {
  switch (msg.type) {
    case "watch_loading":
      setStatus("Loading models...", "thinking");
      break;

    case "watch_state":
      // New game started
      moveList = [];
      renderMoveList();
      hideResult();
      gameActive = true;
      isPaused = false;
      updateBoard(msg);
      enableControls(true);
      setStatus("Game in progress", "thinking");
      break;

    case "watch_move":
      if (msg.last_move) {
        addMoveToHistory(msg.last_move, msg.current_player);
      }
      updateBoard(msg);
      break;

    case "watch_game_over":
      gameActive = false;
      showResult(msg.result);
      setStatus(msg.result, "");
      enableControls(false);
      document.getElementById("btn-pgn").disabled = false;
      break;

    case "watch_paused":
      isPaused = true;
      document.getElementById("btn-pause").textContent = "Resume";
      setStatus("Paused", "connected");
      break;

    case "watch_resumed":
      isPaused = false;
      document.getElementById("btn-pause").textContent = "Pause";
      setStatus("Game in progress", "thinking");
      break;

    case "watch_speed_ack":
      break;

    case "watch_stopped":
      gameActive = false;
      setStatus("Stopped", "");
      enableControls(false);
      break;

    case "pgn":
      if (msg.pgn) {
        navigator.clipboard.writeText(msg.pgn).then(
          () => setStatus("PGN copied to clipboard", "connected"),
          () => window.prompt("PGN:", msg.pgn)
        );
      }
      break;

    case "error":
      setStatus("Error: " + msg.message, "");
      break;
  }
}
```

### `addMoveToHistory(uci, currentPlayer)`

`current_player` in the board state is the side **to move next** (after the move was applied). So the move was made by the opposite side.

```javascript
function addMoveToHistory(uci, currentPlayerNext) {
  // The move was made by the OTHER player
  const moverColor = currentPlayerNext === 0 ? "black" : "white";

  // Use chess.js to get SAN
  chess.load(currentFen);  // currentFen is still the OLD fen at this point
  const moveObj = chess.move({
    from: uci.slice(0, 2),
    to: uci.slice(2, 4),
    promotion: uci.length > 4 ? uci[4] : undefined,
  });
  const san = moveObj ? moveObj.san : uci;
  addMoveEntry(san, moverColor);
}
```

**Important subtlety:** `addMoveToHistory` must be called **before** `updateBoard` updates `currentFen`, because it needs the FEN from before the move was played to compute SAN with chess.js.

### `updateBoard(state)` — simplified (no movable)

```javascript
function updateBoard(state) {
  currentFen = state.fen;
  const turnColor = state.current_player === 0 ? "white" : "black";

  const cfg = {
    fen: currentFen,
    turnColor: turnColor,
    check: isInCheck(currentFen),
  };

  if (state.last_move) {
    cfg.lastMove = [state.last_move.slice(0, 2), state.last_move.slice(2, 4)];
  }

  cg.set(cfg);

  if (state.eval !== null && state.eval !== undefined) {
    updateEval(state.eval, state.current_player);
  }

  if (state.is_terminal || state.resigned) {
    gameActive = false;
  }
}
```

### `updateEval(score, currentPlayer)` — convert to white's perspective

The eval score from `root_edge_stats.mean_value` is from the **current mover's** perspective at time of selection. Since the move has already been applied, `current_player` in the state is the NEXT player. The eval was computed by the PREVIOUS player. So:

```javascript
function updateEval(score, currentPlayerNext) {
  // Score is from the perspective of the player who just moved
  // currentPlayerNext=0 means black just moved, score is from black's POV → negate for white
  // currentPlayerNext=1 means white just moved, score is from white's POV → keep as-is
  const whiteScore = currentPlayerNext === 0 ? -score : score;
  const pct = Math.max(2, Math.min(98, 50 + whiteScore * 50));
  evalFill.style.width = pct + "%";
  const display = whiteScore >= 0 ? "+" + whiteScore.toFixed(2) : whiteScore.toFixed(2);
  evalText.textContent = display;
}
```

### `addMoveEntry()`, `renderMoveList()`, `isInCheck()`, `showResult()`, `hideResult()`, `setStatus()`, `wsSend()`

Copy from `app.js` with minor adaptations. These are straightforward utility functions.

### Button handlers

```javascript
document.getElementById("btn-start").addEventListener("click", () => {
  const wm = document.getElementById("white-select").value;
  const bm = document.getElementById("black-select").value;
  if (!wm || !bm) return;

  document.getElementById("white-label").textContent = document.getElementById("white-select").selectedOptions[0].text;
  document.getElementById("black-label").textContent = document.getElementById("black-select").selectedOptions[0].text;

  const delayMs = parseInt(document.getElementById("speed-slider").value, 10);
  wsSend({ type: "watch_start", white_model: wm, black_model: bm, delay_ms: delayMs });
});

document.getElementById("btn-pause").addEventListener("click", () => {
  if (!gameActive) return;
  wsSend({ type: isPaused ? "watch_resume" : "watch_pause" });
});

document.getElementById("btn-stop").addEventListener("click", () => {
  wsSend({ type: "watch_stop" });
});

document.getElementById("btn-pgn").addEventListener("click", () => {
  wsSend({ type: "watch_pgn" });
});

document.getElementById("speed-slider").addEventListener("input", (e) => {
  const ms = parseInt(e.target.value, 10);
  document.getElementById("speed-value").textContent = (ms / 1000).toFixed(1) + "s";
  if (gameActive) {
    wsSend({ type: "watch_set_speed", delay_ms: ms });
  }
});
```

### `enableControls(active)`

```javascript
function enableControls(active) {
  document.getElementById("btn-pause").disabled = !active;
  document.getElementById("btn-stop").disabled = !active;
  document.getElementById("btn-pgn").disabled = !active;
  if (!active) {
    document.getElementById("btn-pause").textContent = "Pause";
  }
}
```

### Init

```javascript
initBoard();
loadModels();
connect();
```

---

## Step 6: Create `web/static/css/watch.css`

```css
/* Navigation */
nav {
  display: flex;
  gap: 0.75rem;
}

nav a {
  color: var(--text-muted);
  text-decoration: none;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  font-size: 0.85rem;
}

nav a:hover,
nav a.active {
  color: var(--text);
  background: var(--bg-card);
}

/* Model selection */
#model-selection {
  display: flex;
  gap: 1rem;
}

.model-picker {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.model-picker label {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text-muted);
}

.model-picker select {
  padding: 0.4rem;
  background: var(--bg-surface);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 4px;
  font-size: 0.82rem;
}

/* Speed control */
#speed-control {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.82rem;
}

#speed-control label {
  white-space: nowrap;
}

#speed-slider {
  flex: 1;
  accent-color: var(--accent);
}

#speed-value {
  font-family: monospace;
  min-width: 2.5rem;
}
```

---

## Step 7: Modify existing frontend files

### 7a. `web/static/index.html` — add nav link

Replace lines 17-21 (the `<header>` section):

```html
    <header>
      <h1>AlphaZero Chess</h1>
      <nav>
        <a href="/" class="active">Play</a>
        <a href="/watch">Watch</a>
      </nav>
      <div id="status-bar">
        <span id="status-text">Connecting...</span>
      </div>
    </header>
```

### 7b. `web/static/css/style.css` — add shared nav styles

The nav styles from `watch.css` are also needed in the play page. Either:
- (Option A) Move nav styles into `style.css` so both pages get them, and remove from `watch.css`
- (Option B) Keep nav styles only in `watch.css` and also link `watch.css` from `index.html`

**Recommended: Option A.** Add after the `header h1` rule (line 47 in style.css):

```css
nav {
  display: flex;
  gap: 0.75rem;
}

nav a {
  color: var(--text-muted);
  text-decoration: none;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  font-size: 0.85rem;
}

nav a:hover,
nav a.active {
  color: var(--text);
  background: var(--bg-card);
}
```

Then remove the nav section from `watch.css`.

---

## WebSocket Protocol Reference

### Client → Server (on `/ws/watch`)

| Type | Fields | Notes |
|------|--------|-------|
| `watch_start` | `white_model: str, black_model: str, delay_ms?: int` | `delay_ms` defaults to 1000, clamped 100–10000 |
| `watch_pause` | — | |
| `watch_resume` | — | |
| `watch_set_speed` | `delay_ms: int` | Live update, clamped 100–10000 |
| `watch_stop` | — | Cancels auto-play task |
| `watch_pgn` | — | |

### Server → Client

| Type | Fields | Notes |
|------|--------|-------|
| `watch_loading` | — | Sent while models are being loaded |
| `watch_state` | `fen, legal_moves, legal_dests, is_terminal, result, eval, last_move, move_number, current_player` | Game started, initial position |
| `watch_move` | Same as above + possibly `resigned: true` | A move was played |
| `watch_game_over` | `result: str, pgn: str` | |
| `watch_paused` | — | |
| `watch_resumed` | — | |
| `watch_speed_ack` | `delay_ms: int` | |
| `watch_stopped` | — | |
| `pgn` | `pgn: str` | Reuses existing type |
| `error` | `message: str` | Reuses existing type |

---

## Edge Cases & Notes

1. **Model loading latency:** First use of a model takes 2–5s (PyTorch load + GPU transfer). The `watch_loading` message lets the frontend show a loading state. Subsequent games with cached models are instant.

2. **GPU memory:** Each model is ~138MB (folded) or ~276MB (full). DGX Spark has 128GB unified memory. The cache is unbounded — fine for typical checkpoint dirs with <20 entries.

3. **Thread safety:** `get_next_move()` is called via `run_in_executor` sequentially (the async loop awaits each one). No concurrent access to the same search tree. If two browser tabs both use the same cached `PlayRuntime`, the evaluator's `model(input)` calls are serialized by CUDA on the default stream.

4. **Cancellation:** When `watch_stop` or disconnect occurs, `auto_play_task.cancel()` fires. The cancellation takes effect at the next `await` (either `sleep` or `run_in_executor`). MCTS mid-simulation cannot be interrupted, so worst case is waiting for one move to complete.

5. **Resignation:** Both engines have resignation thresholds. The auto-play loop detects `resigned: True` and sends `watch_game_over`. Could be toggled off in a future enhancement.

6. **PGN headers:** `WatchEngine.get_pgn()` prepends `[White "..."]` and `[Black "..."]` headers with model names before the move text from C++ `actions_to_pgn`.

---

## Verification Steps

1. **Start server:**
   ```bash
   PYTHONPATH=build/src:$PYTHONPATH python -m web.server \
     --model checkpoints/checkpoint_00010000.pt \
     --checkpoint-dir checkpoints/ \
     --simulations 100 \
     --num-blocks 10 --num-filters 128
   ```

2. **Check model list:** `curl http://localhost:8000/api/models` — should return JSON with available checkpoints

3. **Open watch page:** `http://localhost:8000/watch` — verify dropdowns are populated

4. **Same model game:** Select the same model for both, click Start. Moves should appear automatically.

5. **Different models:** Select two different checkpoints, verify game plays

6. **Controls:** Test pause/resume, speed slider, stop, PGN copy

7. **Play mode:** Navigate to `http://localhost:8000/` — existing play mode works as before

8. **Edge cases:** Disconnect during game (no server crash), start new game while one is running (old one cancelled), empty checkpoint dir (graceful "no models" message)
