# Feature Notes

## Auto-resume training from latest checkpoint
`scripts/train.py` requires an explicit `--resume <path>` to continue from a checkpoint. There is no auto-discovery of the latest checkpoint. Consider adding an `--auto-resume` flag that calls `find_latest_checkpoint(checkpoint_dir)` when no explicit path is given.
