# AlphaZero

Single-machine AlphaZero for Chess and Go. Specs are in `specs/` — read these for requirements.

You may use tmux to spawn a tmux session for scraping logs and checking your work. Settings are in `~/.tmux.conf`

## Sandbox Notes

- In restricted/offline sandbox sessions, validate editable packaging with:
  `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`
- If `pytest` is unavailable in the current interpreter, run Python tests with `python3 -m unittest` for task-level validation.
