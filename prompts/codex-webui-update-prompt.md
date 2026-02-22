
  Task: Implement the Watch Mode feature for the chess web UI, exactly as specified in notes/webui-update-plan.md.

  Context files to read first: web/server.py, web/engine.py, web/static/index.html, web/static/js/app.js,
  web/static/css/style.css, scripts/play.py

  Critical notes:
  - In watch.js, addMoveToHistory() MUST be called before updateBoard() — it needs the pre-move FEN
  - Match existing code patterns exactly (async executor pattern in server.py, board state dict shape in engine.py)
  - All changes confined to web/ — do not modify any other directories

  Deliverable: All 8 files from the plan, fully implemented, no stubs.
