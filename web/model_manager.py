"""Checkpoint discovery and cached runtime loading for watch mode."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import threading
from typing import Any

logger = logging.getLogger("alphazero.web.models")

_CHECKPOINT_RE = re.compile(r"^(checkpoint|milestone)_(\d{8})\.pt$")


def infer_architecture(checkpoint_path: str | Path) -> dict[str, int]:
    """Infer num_blocks, num_filters, se_reduction from checkpoint weights."""
    import torch

    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    state_dict = payload["model_state_dict"]

    num_filters = int(state_dict["input_conv.weight"].shape[0])

    num_blocks = 0
    while f"residual_blocks.{num_blocks}.conv_1.weight" in state_dict:
        num_blocks += 1

    se_fc1 = state_dict.get("residual_blocks.0.se_fc_1.weight")
    if se_fc1 is not None:
        se_reduction = num_filters // int(se_fc1.shape[0])
    else:
        se_reduction = 4

    return {"num_blocks": num_blocks, "num_filters": num_filters, "se_reduction": se_reduction}


class ModelManager:
    """Discover checkpoints and lazily build cached PlayRuntime instances."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        simulations: int = 800,
        num_blocks: int | None = None,
        num_filters: int | None = None,
        se_reduction: int | None = None,
        device: str | None = None,
        fp32: bool = False,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir).resolve()
        self._simulations = simulations
        self._device = device
        self._fp32 = fp32
        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._deps = None

        if num_blocks is not None and num_filters is not None and se_reduction is not None:
            self._num_blocks = num_blocks
            self._num_filters = num_filters
            self._se_reduction = se_reduction
        else:
            arch = self._infer_architecture()
            self._num_blocks = arch["num_blocks"]
            self._num_filters = arch["num_filters"]
            self._se_reduction = arch["se_reduction"]
            logger.info(
                "Inferred architecture: %d blocks, %d filters, SE reduction %d",
                self._num_blocks, self._num_filters, self._se_reduction,
            )

    def _infer_architecture(self) -> dict[str, int]:
        """Infer architecture from the first checkpoint found."""
        models = self.list_models()
        if not models:
            raise ValueError(f"No checkpoints found in {self._checkpoint_dir}")
        return infer_architecture(models[0]["path"])

    def list_models(self) -> list[dict[str, Any]]:
        """Return checkpoints sorted by step, preferring folded variants."""
        if not self._checkpoint_dir.exists():
            return []

        entries: list[dict[str, Any]] = []
        for path in self._checkpoint_dir.glob("*.pt"):
            match = _CHECKPOINT_RE.match(path.name)
            if match is None:
                continue
            kind, step_digits = match.groups()
            step = int(step_digits)
            name = f"{kind}_{step_digits}"
            entries.append(
                {
                    "name": name,
                    "display_name": f"{kind} step {step:,}",
                    "path": str(path),
                    "step": step,
                }
            )

        entries.sort(key=lambda e: e["step"])
        return entries

    def get_runtime(self, name: str) -> Any:
        """Return a cached PlayRuntime for the given model name."""
        with self._lock:
            if name in self._cache:
                return self._cache[name]

        models = self.list_models()
        entry = next((model for model in models if model["name"] == name), None)
        if entry is None:
            raise ValueError(f"Unknown model: {name}")

        from scripts.play import build_play_runtime, load_runtime_dependencies

        with self._lock:
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
