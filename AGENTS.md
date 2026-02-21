# AlphaZero

Single-machine AlphaZero for Chess and Go. Specs are in `specs/` — read these for requirements.

You may use tmux to spawn a tmux session for scraping logs and checking your work. Settings are in `~/.tmux.conf`

## Sandbox Notes

- In restricted/offline sandbox sessions, validate editable packaging with:
  `python3 -m pip install -e . --no-build-isolation --no-deps --prefix /tmp/alphazero-prefix`
  If pip attempts to uninstall a non-writable environment install, rerun with `--ignore-installed`.
- The C++ extension currently builds against the `alphazero` conda env (Python 3.11). If default `python3` is 3.13, run binding tests with:
  `PYTHONPATH=build/src:$PYTHONPATH /home/hakan/miniconda3/envs/alphazero/bin/python -m pytest tests/python/test_bindings.py`
- If `pytest` is unavailable in the current interpreter, run Python tests with `python3 -m unittest` for task-level validation.
- `ruff` may be unavailable in sandbox interpreters; when missing, run available static checks (for example `mypy` and `python3 -m compileall`) and note the missing linter in the task log.
