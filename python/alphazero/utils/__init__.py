"""Utility helpers for checkpointing and runtime logging."""

__all__: list[str] = []

from alphazero.utils.logging import (
    DEFAULT_CONSOLE_SUMMARY_INTERVAL_STEPS,
    DEFAULT_LOG_DIR,
    REQUIRED_SELFPLAY_SCALARS,
    REQUIRED_TRAINING_SCALARS,
    ScalarWriter,
    TensorBoardMetricsLogger,
    WriterFactory,
    build_run_name,
    create_metrics_logger,
    load_log_dir_from_config,
)

__all__.extend(
    [
        "DEFAULT_CONSOLE_SUMMARY_INTERVAL_STEPS",
        "DEFAULT_LOG_DIR",
        "REQUIRED_SELFPLAY_SCALARS",
        "REQUIRED_TRAINING_SCALARS",
        "ScalarWriter",
        "TensorBoardMetricsLogger",
        "WriterFactory",
        "build_run_name",
        "create_metrics_logger",
        "load_log_dir_from_config",
    ]
)

try:
    from alphazero.utils.checkpoint import (
        DEFAULT_ROLLING_CHECKPOINT_KEEP_LAST,
        CheckpointPaths,
        LoadedCheckpoint,
        extract_replay_buffer_metadata,
        find_latest_checkpoint,
        list_checkpoints,
        load_checkpoint,
        load_latest_checkpoint,
        normalize_lr_schedule_entries,
        normalize_replay_buffer_metadata,
        save_checkpoint,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional torch install.
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "DEFAULT_ROLLING_CHECKPOINT_KEEP_LAST",
            "CheckpointPaths",
            "LoadedCheckpoint",
            "extract_replay_buffer_metadata",
            "find_latest_checkpoint",
            "list_checkpoints",
            "load_checkpoint",
            "load_latest_checkpoint",
            "normalize_lr_schedule_entries",
            "normalize_replay_buffer_metadata",
            "save_checkpoint",
        ]
    )
