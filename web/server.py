"""FastAPI server for playing chess against AlphaZero in the browser."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in [ROOT, ROOT / "python", ROOT / "scripts", ROOT / "build" / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from web.engine import ChessEngine

logger = logging.getLogger("alphazero.web")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AlphaZero Chess")

# Global engine instance, initialized at startup
_engine: ChessEngine | None = None


def get_engine() -> ChessEngine:
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.websocket("/ws/chess")
async def chess_ws(websocket: WebSocket):
    await websocket.accept()
    engine = get_engine()
    loop = asyncio.get_running_loop()

    # Send initial board state
    state = engine.reset()
    await websocket.send_json({"type": "game_state", **state})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type")

            if msg_type == "move":
                uci = msg.get("uci", "")
                try:
                    state = engine.play_human_move(uci)
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue

                await websocket.send_json({"type": "move_ack", **state})

                # If game isn't over, get AI response
                if not state["is_terminal"]:
                    await websocket.send_json({"type": "thinking"})
                    ai_state = await loop.run_in_executor(None, engine.get_ai_move)
                    await websocket.send_json({"type": "ai_move", **ai_state})

            elif msg_type == "new_game":
                state = engine.reset()
                await websocket.send_json({"type": "game_state", **state})

            elif msg_type == "new_game_as_black":
                state = engine.reset()
                # AI plays first move as white
                await websocket.send_json({"type": "thinking"})
                ai_state = await loop.run_in_executor(None, engine.get_ai_move)
                await websocket.send_json({"type": "game_state", **ai_state})

            elif msg_type == "resign":
                player = msg.get("player", 0)
                result = "0-1" if player == 0 else "1-0"
                await websocket.send_json({
                    "type": "game_over",
                    "result": result,
                    "reason": "resignation",
                })

            elif msg_type == "undo":
                state = engine.undo()
                await websocket.send_json({"type": "game_state", **state})

            elif msg_type == "pgn":
                pgn = engine.get_pgn()
                await websocket.send_json({"type": "pgn", "pgn": pgn})

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaZero Chess Web UI")
    parser.add_argument("--model", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--simulations", type=int, default=800, help="MCTS simulations per move")
    parser.add_argument("--num-blocks", type=int, default=20, help="ResNet blocks")
    parser.add_argument("--num-filters", type=int, default=256, help="ResNet filters")
    parser.add_argument("--se-reduction", type=int, default=4, help="SE reduction ratio")
    parser.add_argument("--device", default=None, help="Torch device (cpu/cuda)")
    parser.add_argument("--fp32", action="store_true", help="Disable mixed precision")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.info("Loading AlphaZero model from %s ...", args.model)

    global _engine
    _engine = ChessEngine(
        model_path=args.model,
        simulations=args.simulations,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        se_reduction=args.se_reduction,
        device=args.device,
        fp32=args.fp32,
    )
    logger.info("Engine ready. Starting server on %s:%d", args.host, args.port)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
