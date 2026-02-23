import { Chessground } from "https://unpkg.com/chessground@9.2.1/dist/chessground.min.js";

// ── State ──────────────────────────────────────────────────────────────

let cg = null;           // Chessground instance
let ws = null;           // WebSocket connection
let boardFlipped = false;
let playerColor = "white";
let currentFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
let legalDests = new Map();
let moveList = [];       // [{number, white, black}, ...]
let gameOver = false;

// chess.js instance for SAN display (0.10.x global constructor)
const chess = new Chess();

let hasModelSelector = false; // whether /api/models returned models
let selectedModelName = "";  // display name for the AI player label

// ── DOM refs ───────────────────────────────────────────────────────────

const boardEl = document.getElementById("board");
const statusText = document.getElementById("status-text");
const moveListEl = document.getElementById("move-list");
const evalFill = document.getElementById("eval-fill");
const evalText = document.getElementById("eval-text");
const resultBox = document.getElementById("game-result");
const resultText = document.getElementById("result-text");
const modelSelect = document.getElementById("model-select");
const modelSelection = document.getElementById("model-selection");
const aiLabel = document.querySelector("#player-top .player-name");

// ── Chessground setup ──────────────────────────────────────────────────

function initBoard() {
  cg = Chessground(boardEl, {
    orientation: playerColor,
    fen: currentFen,
    turnColor: "white",
    movable: {
      color: playerColor,
      free: false,
      dests: legalDests,
      events: {
        after: onPlayerMove,
      },
    },
    draggable: { showGhost: true },
    premovable: { enabled: false },
    animation: { enabled: true, duration: 150 },
  });
}

function updateBoard(state) {
  currentFen = state.fen;
  const turnColor = state.current_player === 0 ? "white" : "black";

  // Build legal move dests map
  legalDests = new Map();
  if (state.legal_dests && !state.is_terminal) {
    for (const [src, dsts] of Object.entries(state.legal_dests)) {
      legalDests.set(src, dsts);
    }
  }

  const isMyTurn = turnColor === playerColor && !state.is_terminal;

  const cgConfig = {
    fen: currentFen,
    turnColor: turnColor,
    movable: {
      color: isMyTurn ? playerColor : undefined,
      dests: isMyTurn ? legalDests : new Map(),
    },
    check: isInCheck(currentFen),
  };

  // Highlight last move
  if (state.last_move) {
    cgConfig.lastMove = [state.last_move.slice(0, 2), state.last_move.slice(2, 4)];
  }

  cg.set(cgConfig);

  // Update eval bar
  if (state.eval !== null && state.eval !== undefined) {
    updateEval(state.eval);
  }

  // Handle game over
  if (state.is_terminal || state.resigned) {
    gameOver = true;
    const result = state.result || "Game over";
    showResult(result);
    setStatus(result, "");
    cg.set({ movable: { color: undefined, dests: new Map() } });
  }
}

function isInCheck(fen) {
  chess.load(fen);
  return chess.in_check() ? true : false;
}

// ── Move handling ──────────────────────────────────────────────────────

function onPlayerMove(orig, dest) {
  // Check if this is a pawn promotion
  const promotion = detectPromotion(orig, dest);
  let uci = orig + dest;
  if (promotion) {
    uci += promotion;
  }

  // Record the SAN before sending
  chess.load(currentFen);
  const moveObj = chess.move({ from: orig, to: dest, promotion: promotion || undefined });
  if (moveObj) {
    addMoveToHistory(moveObj.san, playerColor);
  }

  wsSend({ type: "move", uci: uci });
  setStatus("AlphaZero is thinking...", "thinking");
}

function detectPromotion(orig, dest) {
  // Check if a pawn is moving to the last rank
  chess.load(currentFen);
  const piece = chess.get(orig);
  if (!piece || piece.type !== "p") return null;

  const destRank = dest[1];
  if ((piece.color === "w" && destRank === "8") || (piece.color === "b" && destRank === "1")) {
    // Default to queen promotion — could add a dialog later
    return "q";
  }
  return null;
}

// ── Move history ───────────────────────────────────────────────────────

function addMoveToHistory(san, color) {
  if (color === "white") {
    const num = moveList.length + 1;
    moveList.push({ number: num, white: san, black: null });
  } else {
    if (moveList.length === 0) {
      moveList.push({ number: 1, white: "...", black: san });
    } else {
      moveList[moveList.length - 1].black = san;
    }
  }
  renderMoveList();
}

function renderMoveList() {
  moveListEl.innerHTML = "";
  for (const entry of moveList) {
    const numEl = document.createElement("span");
    numEl.className = "move-number";
    numEl.textContent = entry.number + ".";

    const whiteEl = document.createElement("span");
    whiteEl.className = "move";
    whiteEl.textContent = entry.white || "";

    const blackEl = document.createElement("span");
    blackEl.className = "move";
    blackEl.textContent = entry.black || "";

    moveListEl.appendChild(numEl);
    moveListEl.appendChild(whiteEl);
    moveListEl.appendChild(blackEl);
  }

  // Mark last move
  const allMoves = moveListEl.querySelectorAll(".move");
  if (allMoves.length > 0) {
    // Find last non-empty move
    for (let i = allMoves.length - 1; i >= 0; i--) {
      if (allMoves[i].textContent) {
        allMoves[i].classList.add("last-move");
        break;
      }
    }
  }

  // Scroll to bottom
  const moveHistory = document.getElementById("move-history");
  moveHistory.scrollTop = moveHistory.scrollHeight;
}

function addAiMoveToHistory(uci, fen) {
  // Convert UCI to SAN using chess.js
  const prevFen = getPreviousFen();
  if (prevFen) chess.load(prevFen);

  const moveObj = chess.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.length > 4 ? uci[4] : undefined });
  const san = moveObj ? moveObj.san : uci;

  const aiColor = playerColor === "white" ? "black" : "white";
  addMoveToHistory(san, aiColor);
}

function getPreviousFen() {
  // Reconstruct: we track the FEN before the AI move was applied
  // We can derive it from the move list. Simpler: just use the current fen before update.
  return currentFen;
}

// ── Eval bar ───────────────────────────────────────────────────────────

function updateEval(score) {
  // score is from the AI's perspective (current player at time of eval)
  // Convert to white's perspective for display
  const whiteScore = score; // Already from white's perspective after AI move
  const pct = Math.max(2, Math.min(98, 50 + whiteScore * 50));
  evalFill.style.width = pct + "%";

  const display = whiteScore >= 0 ? "+" + whiteScore.toFixed(2) : whiteScore.toFixed(2);
  evalText.textContent = display;
}

// ── Game result ────────────────────────────────────────────────────────

function showResult(result) {
  let text = result;
  if (result === "1-0") text = "White wins — 1-0";
  else if (result === "0-1") text = "Black wins — 0-1";
  else if (result === "1/2-1/2") text = "Draw — 1/2-1/2";

  resultText.textContent = text;
  resultBox.classList.remove("hidden");
}

function hideResult() {
  resultBox.classList.add("hidden");
}

// ── Status ─────────────────────────────────────────────────────────────

function setStatus(text, cls) {
  statusText.textContent = text;
  statusText.className = cls || "";
}

// ── WebSocket ──────────────────────────────────────────────────────────

function connect() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/ws/chess`);

  ws.onopen = () => {
    if (hasModelSelector) {
      setStatus("Select a model and click New Game", "connected");
    } else {
      setStatus("Your turn", "connected");
    }
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleMessage(msg);
  };

  ws.onclose = () => {
    setStatus("Disconnected — refresh to reconnect", "");
  };

  ws.onerror = () => {
    setStatus("Connection error", "");
  };
}

function wsSend(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function handleMessage(msg) {
  switch (msg.type) {
    case "game_state":
      // Full state reset (new game, undo)
      moveList = [];
      renderMoveList();
      hideResult();
      gameOver = false;
      updateBoard(msg);
      if (msg.last_move) {
        // If there's already a move (e.g. AI played first as white)
        addAiMoveToHistory(msg.last_move, msg.fen);
      }
      setStatus("Your turn", "connected");
      break;

    case "move_ack":
      // Our move was accepted, update board
      updateBoard(msg);
      break;

    case "loading":
      setStatus("Loading model...", "thinking");
      break;

    case "thinking":
      setStatus("AlphaZero is thinking...", "thinking");
      break;

    case "ai_move":
      if (msg.last_move) {
        addAiMoveToHistory(msg.last_move, msg.fen);
      }
      updateBoard(msg);
      if (!msg.is_terminal && !msg.resigned) {
        setStatus("Your turn", "connected");
      }
      break;

    case "game_over":
      gameOver = true;
      showResult(msg.result + (msg.reason ? " (" + msg.reason + ")" : ""));
      setStatus(msg.result, "");
      cg.set({ movable: { color: undefined, dests: new Map() } });
      break;

    case "pgn":
      if (msg.pgn) {
        navigator.clipboard.writeText(msg.pgn).then(
          () => setStatus("PGN copied to clipboard", "connected"),
          () => {
            // Fallback: show in prompt
            window.prompt("PGN:", msg.pgn);
          }
        );
      }
      break;

    case "error":
      setStatus("Error: " + msg.message, "");
      // Re-sync board state
      setTimeout(() => {
        if (!gameOver) setStatus("Your turn", "connected");
      }, 2000);
      break;
  }
}

// ── Button handlers ────────────────────────────────────────────────────

document.getElementById("btn-new-game").addEventListener("click", () => {
  playerColor = "white";
  boardFlipped = false;
  if (cg) cg.set({ orientation: "white" });
  const msg = { type: "new_game" };
  if (hasModelSelector && modelSelect.value) {
    msg.model = modelSelect.value;
    aiLabel.textContent = modelSelect.selectedOptions[0].text;
  }
  wsSend(msg);
});

document.getElementById("btn-flip").addEventListener("click", () => {
  playerColor = "black";
  boardFlipped = true;
  if (cg) cg.set({ orientation: "black" });
  const msg = { type: "new_game_as_black" };
  if (hasModelSelector && modelSelect.value) {
    msg.model = modelSelect.value;
    aiLabel.textContent = modelSelect.selectedOptions[0].text;
  }
  wsSend(msg);
});

document.getElementById("btn-undo").addEventListener("click", () => {
  if (!gameOver) {
    wsSend({ type: "undo" });
  }
});

document.getElementById("btn-resign").addEventListener("click", () => {
  if (!gameOver) {
    const player = playerColor === "white" ? 0 : 1;
    wsSend({ type: "resign", player: player });
  }
});

document.getElementById("btn-pgn").addEventListener("click", () => {
  wsSend({ type: "pgn" });
});

// ── Model loading ─────────────────────────────────────────────────────

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    const data = await response.json();
    const models = data.models || [];

    if (models.length === 0) {
      // No checkpoint-dir — hide selector (server has a --model)
      modelSelection.style.display = "none";
      return;
    }

    hasModelSelector = true;
    modelSelect.innerHTML = "";
    for (const model of models) {
      modelSelect.add(new Option(model.display_name, model.name));
    }
    modelSelect.selectedIndex = models.length - 1;
    selectedModelName = modelSelect.selectedOptions[0].text;
    aiLabel.textContent = selectedModelName;
  } catch (e) {
    console.error("loadModels failed:", e);
    modelSelection.style.display = "none";
  }
}

modelSelect.addEventListener("change", () => {
  selectedModelName = modelSelect.selectedOptions[0].text;
  aiLabel.textContent = selectedModelName;
});

// ── Init ───────────────────────────────────────────────────────────────

initBoard();
loadModels().then(() => connect());
