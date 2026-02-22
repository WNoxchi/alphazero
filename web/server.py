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
from web.model_manager import ModelManager, infer_architecture
from web.watch_engine import WatchEngine

logger = logging.getLogger("alphazero.web")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AlphaZero Chess")

# Global engine instance, initialized at startup
_engine: ChessEngine | None = None
_model_manager: ModelManager | None = None


def get_engine() -> ChessEngine:
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/watch")
async def watch_page():
    return FileResponse(STATIC_DIR / "watch.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/models")
async def list_models():
    if _model_manager is None:
        return {"models": []}
    models = _model_manager.list_models()
    return {
        "models": [
            {
                "name": model["name"],
                "display_name": model["display_name"],
                "step": model["step"],
            }
            for model in models
        ]
    }


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

    def clamp_delay(raw: object) -> int:
        try:
            delay = int(raw)
        except (TypeError, ValueError):
            delay = 1000
        return max(100, min(10000, delay))

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
                    pgn = watch_engine.get_pgn(white_name=white_name, black_name=black_name)
                    await websocket.send_json(
                        {
                            "type": "watch_game_over",
                            "result": state.get("result", "*"),
                            "pgn": pgn,
                        }
                    )
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
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "No checkpoint directory configured (use --checkpoint-dir)",
                        }
                    )
                    continue

                if auto_play_task is not None and not auto_play_task.done():
                    auto_play_task.cancel()
                    await asyncio.sleep(0)

                white_name = str(msg.get("white_model", ""))
                black_name = str(msg.get("black_model", ""))
                delay_ms = clamp_delay(msg.get("delay_ms", 1000))
                paused = False

                try:
                    await websocket.send_json({"type": "watch_loading"})
                    white_rt = await loop.run_in_executor(None, _model_manager.get_runtime, white_name)
                    black_rt = await loop.run_in_executor(None, _model_manager.get_runtime, black_name)
                except Exception as exc:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Failed to load model: {exc}",
                        }
                    )
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
                delay_ms = clamp_delay(msg.get("delay_ms", 1000))
                await websocket.send_json({"type": "watch_speed_ack", "delay_ms": delay_ms})

            elif msg_type == "watch_stop":
                if auto_play_task is not None and not auto_play_task.done():
                    auto_play_task.cancel()
                watch_engine = None
                await websocket.send_json({"type": "watch_stopped"})

            elif msg_type == "watch_pgn":
                if watch_engine is not None:
                    pgn = watch_engine.get_pgn(white_name=white_name, black_name=black_name)
                    await websocket.send_json({"type": "pgn", "pgn": pgn})

            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    }
                )

    except WebSocketDisconnect:
        if auto_play_task is not None and not auto_play_task.done():
            auto_play_task.cancel()
        logger.info("Watch client disconnected")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaZero Chess Web UI")
    parser.add_argument("--model", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--simulations", type=int, default=800, help="MCTS simulations per move")
    parser.add_argument("--num-blocks", type=int, default=None, help="ResNet blocks (auto-detected from checkpoint)")
    parser.add_argument("--num-filters", type=int, default=None, help="ResNet filters (auto-detected from checkpoint)")
    parser.add_argument("--se-reduction", type=int, default=None, help="SE reduction ratio (auto-detected from checkpoint)")
    parser.add_argument("--device", default=None, help="Torch device (cpu/cuda)")
    parser.add_argument("--fp32", action="store_true", help="Disable mixed precision")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory with checkpoints for watch mode model selection",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.info("Loading AlphaZero model from %s ...", args.model)

    # Auto-detect architecture from checkpoint if not explicitly provided
    if args.num_blocks is None or args.num_filters is None or args.se_reduction is None:
        arch = infer_architecture(args.model)
        if args.num_blocks is None:
            args.num_blocks = arch["num_blocks"]
        if args.num_filters is None:
            args.num_filters = arch["num_filters"]
        if args.se_reduction is None:
            args.se_reduction = arch["se_reduction"]
        logger.info(
            "Auto-detected architecture: %d blocks, %d filters, SE reduction %d",
            args.num_blocks, args.num_filters, args.se_reduction,
        )

    global _engine, _model_manager
    _engine = ChessEngine(
        model_path=args.model,
        simulations=args.simulations,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        se_reduction=args.se_reduction,
        device=args.device,
        fp32=args.fp32,
    )

    if args.checkpoint_dir:
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

    logger.info("Engine ready. Starting server on %s:%d", args.host, args.port)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
