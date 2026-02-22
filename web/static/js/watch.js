import { Chessground } from "https://unpkg.com/chessground@9.2.1/dist/chessground.min.js";

let cg = null;
let ws = null;
let currentFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
let moveList = [];
let gameActive = false;
let isPaused = false;

const chess = new Chess();

const boardEl = document.getElementById("board");
const statusText = document.getElementById("status-text");
const moveListEl = document.getElementById("move-list");
const evalFill = document.getElementById("eval-fill");
const evalText = document.getElementById("eval-text");
const resultBox = document.getElementById("game-result");
const resultText = document.getElementById("result-text");
const whiteSelect = document.getElementById("white-select");
const blackSelect = document.getElementById("black-select");
const whiteLabel = document.getElementById("white-label");
const blackLabel = document.getElementById("black-label");
const btnStart = document.getElementById("btn-start");
const btnPause = document.getElementById("btn-pause");
const btnStop = document.getElementById("btn-stop");
const btnPgn = document.getElementById("btn-pgn");
const speedSlider = document.getElementById("speed-slider");
const speedValue = document.getElementById("speed-value");

function initBoard() {
  cg = Chessground(boardEl, {
    orientation: "white",
    fen: currentFen,
    turnColor: "white",
    viewOnly: true,
    animation: { enabled: true, duration: 200 },
  });
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    const data = await response.json();
    populateSelects(data.models || []);
  } catch (error) {
    setStatus("Failed to load model list", "");
    btnStart.disabled = true;
  }
}

function populateSelects(models) {
  whiteSelect.innerHTML = "";
  blackSelect.innerHTML = "";

  if (models.length === 0) {
    whiteSelect.add(new Option("No models available", ""));
    blackSelect.add(new Option("No models available", ""));
    btnStart.disabled = true;
    return;
  }

  for (const model of models) {
    whiteSelect.add(new Option(model.display_name, model.name));
    blackSelect.add(new Option(model.display_name, model.name));
  }
  whiteSelect.selectedIndex = models.length - 1;
  blackSelect.selectedIndex = models.length - 1;
  btnStart.disabled = false;
}

function connect() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/ws/watch`);

  ws.onopen = () => {
    setStatus("Ready - select models and press Start", "connected");
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleMessage(msg);
  };

  ws.onclose = () => {
    setStatus("Disconnected - refresh to reconnect", "");
  };

  ws.onerror = () => {
    setStatus("Connection error", "");
  };
}

function handleMessage(msg) {
  switch (msg.type) {
    case "watch_loading":
      setStatus("Loading models...", "thinking");
      break;

    case "watch_state":
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
        // SAN conversion uses pre-move FEN, so this must run before updateBoard().
        addMoveToHistory(msg.last_move, msg.current_player);
      }
      updateBoard(msg);
      break;

    case "watch_game_over":
      gameActive = false;
      showResult(msg.result);
      setStatus(msg.result, "");
      enableControls(false);
      btnPgn.disabled = false;
      break;

    case "watch_paused":
      isPaused = true;
      btnPause.textContent = "Resume";
      setStatus("Paused", "connected");
      break;

    case "watch_resumed":
      isPaused = false;
      btnPause.textContent = "Pause";
      setStatus("Game in progress", "thinking");
      break;

    case "watch_speed_ack":
      break;

    case "watch_stopped":
      gameActive = false;
      isPaused = false;
      enableControls(false);
      setStatus("Stopped", "");
      break;

    case "pgn":
      if (msg.pgn) {
        navigator.clipboard.writeText(msg.pgn).then(
          () => setStatus("PGN copied to clipboard", "connected"),
          () => {
            window.prompt("PGN:", msg.pgn);
          }
        );
      }
      break;

    case "error":
      setStatus("Error: " + msg.message, "");
      break;
  }
}

function updateBoard(state) {
  currentFen = state.fen;
  const turnColor = state.current_player === 0 ? "white" : "black";

  const config = {
    fen: currentFen,
    turnColor: turnColor,
    check: isInCheck(currentFen),
  };

  if (state.last_move) {
    config.lastMove = [state.last_move.slice(0, 2), state.last_move.slice(2, 4)];
  }

  cg.set(config);

  if (state.eval !== null && state.eval !== undefined) {
    updateEval(state.eval, state.current_player);
  }

  if (state.is_terminal || state.resigned) {
    gameActive = false;
  }
}

function addMoveToHistory(uci, currentPlayerNext) {
  const moverColor = currentPlayerNext === 0 ? "black" : "white";

  chess.load(currentFen);
  const moveObj = chess.move({
    from: uci.slice(0, 2),
    to: uci.slice(2, 4),
    promotion: uci.length > 4 ? uci[4] : undefined,
  });
  const san = moveObj ? moveObj.san : uci;
  addMoveEntry(san, moverColor);
}

function addMoveEntry(san, color) {
  if (color === "white") {
    const number = moveList.length + 1;
    moveList.push({ number: number, white: san, black: null });
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

  const allMoves = moveListEl.querySelectorAll(".move");
  if (allMoves.length > 0) {
    for (let i = allMoves.length - 1; i >= 0; i--) {
      if (allMoves[i].textContent) {
        allMoves[i].classList.add("last-move");
        break;
      }
    }
  }

  const moveHistory = document.getElementById("move-history");
  moveHistory.scrollTop = moveHistory.scrollHeight;
}

function updateEval(score, currentPlayerNext) {
  const whiteScore = currentPlayerNext === 0 ? -score : score;
  const pct = Math.max(2, Math.min(98, 50 + whiteScore * 50));
  evalFill.style.width = pct + "%";
  const display = whiteScore >= 0 ? "+" + whiteScore.toFixed(2) : whiteScore.toFixed(2);
  evalText.textContent = display;
}

function isInCheck(fen) {
  chess.load(fen);
  return chess.in_check() ? true : false;
}

function showResult(result) {
  let text = result;
  if (result === "1-0") text = "White wins - 1-0";
  else if (result === "0-1") text = "Black wins - 0-1";
  else if (result === "1/2-1/2") text = "Draw - 1/2-1/2";

  resultText.textContent = text;
  resultBox.classList.remove("hidden");
}

function hideResult() {
  resultBox.classList.add("hidden");
}

function setStatus(text, cls) {
  statusText.textContent = text;
  statusText.className = cls || "";
}

function wsSend(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function enableControls(active) {
  btnPause.disabled = !active;
  btnStop.disabled = !active;
  btnPgn.disabled = !active;
  if (!active) {
    btnPause.textContent = "Pause";
  }
}

btnStart.addEventListener("click", () => {
  const whiteModel = whiteSelect.value;
  const blackModel = blackSelect.value;
  if (!whiteModel || !blackModel) {
    return;
  }

  whiteLabel.textContent = whiteSelect.selectedOptions[0].text;
  blackLabel.textContent = blackSelect.selectedOptions[0].text;

  const delayMs = parseInt(speedSlider.value, 10);
  wsSend({
    type: "watch_start",
    white_model: whiteModel,
    black_model: blackModel,
    delay_ms: delayMs,
  });
});

btnPause.addEventListener("click", () => {
  if (!gameActive) {
    return;
  }
  wsSend({ type: isPaused ? "watch_resume" : "watch_pause" });
});

btnStop.addEventListener("click", () => {
  wsSend({ type: "watch_stop" });
});

btnPgn.addEventListener("click", () => {
  wsSend({ type: "watch_pgn" });
});

speedSlider.addEventListener("input", (event) => {
  const ms = parseInt(event.target.value, 10);
  speedValue.textContent = (ms / 1000).toFixed(1) + "s";
  if (gameActive) {
    wsSend({ type: "watch_set_speed", delay_ms: ms });
  }
});

initBoard();
loadModels();
connect();
