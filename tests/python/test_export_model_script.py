"""Tests for scripts/export_model.py deployment artifact export behavior."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Literal, cast
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "export_model.py"

_SPEC = importlib.util.spec_from_file_location("alphazero_export_script", SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import bootstrap guard.
    raise RuntimeError(f"Unable to load export script module from {SCRIPT_PATH}")
export_script = cast(Any, importlib.util.module_from_spec(_SPEC))
sys.modules[_SPEC.name] = export_script
_SPEC.loader.exec_module(export_script)


class _FakeDevice:
    def __init__(self, device_type: str) -> None:
        self.type = device_type


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...], device: _FakeDevice) -> None:
        self.shape = shape
        self.device = device


class _FakeNoGrad:
    def __enter__(self) -> "_FakeNoGrad":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        del exc_type, exc, tb
        return False


class _FakeJitArtifact:
    def __init__(self) -> None:
        self.saved_path: Path | None = None

    def eval(self) -> "_FakeJitArtifact":
        return self

    def save(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("torchscript", encoding="utf-8")
        self.saved_path = path


class _FakeJit:
    def __init__(self) -> None:
        self.trace_calls: list[tuple[Any, Any, bool]] = []
        self.freeze_calls: list[Any] = []

    def trace(self, model: Any, sample_input: Any, *, strict: bool) -> _FakeJitArtifact:
        self.trace_calls.append((model, sample_input, bool(strict)))
        return _FakeJitArtifact()

    def freeze(self, artifact: _FakeJitArtifact) -> _FakeJitArtifact:
        self.freeze_calls.append(artifact)
        return artifact


class _FakeOnnx:
    def __init__(self) -> None:
        self.export_calls: list[dict[str, Any]] = []

    def export(self, model: Any, sample_input: Any, output_path: str, **kwargs: Any) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("onnx", encoding="utf-8")
        payload = {
            "model": model,
            "sample_input": sample_input,
            "output_path": path,
        }
        payload.update(kwargs)
        self.export_calls.append(payload)


class _FakeTorch:
    float32 = object()

    def __init__(self) -> None:
        self.jit = _FakeJit()
        self.onnx = _FakeOnnx()
        self.zeros_calls: list[tuple[tuple[int, ...], Any, _FakeDevice]] = []

    def device(self, spec: str) -> _FakeDevice:
        return _FakeDevice(spec)

    def zeros(self, shape: tuple[int, ...], *, dtype: Any, device: _FakeDevice) -> _FakeTensor:
        self.zeros_calls.append((shape, dtype, device))
        return _FakeTensor(shape, device)

    def no_grad(self) -> _FakeNoGrad:
        return _FakeNoGrad()


class _FakeModel:
    def __init__(
        self,
        game_config: Any,
        *,
        num_blocks: int,
        num_filters: int,
        se_reduction: int,
    ) -> None:
        self.game_config = game_config
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.se_reduction = se_reduction
        self.device: _FakeDevice | None = None
        self.eval_calls = 0

    def to(self, *, device: _FakeDevice) -> "_FakeModel":
        self.device = device
        return self

    def eval(self) -> "_FakeModel":
        self.eval_calls += 1
        return self


@dataclass(frozen=True, slots=True)
class _FakeLoadCheckpoint:
    calls: list[tuple[Path, Any, Any, str]]
    step: int

    def __call__(self, path: Path, model: Any, optimizer: Any, *, map_location: str) -> Any:
        self.calls.append((Path(path), model, optimizer, map_location))
        return SimpleNamespace(step=self.step)


@dataclass(frozen=True, slots=True)
class _FakeFoldModel:
    calls: list[Any]

    def __call__(self, model: Any) -> Any:
        self.calls.append(model)
        return model


class ExportModelScriptTests(unittest.TestCase):
    def _make_args(self, **overrides: object) -> Any:
        values: dict[str, object] = {
            "checkpoint": "checkpoint.pt",
            "output": None,
            "format": "torchscript",
            "config": None,
            "game": "chess",
            "num_blocks": None,
            "num_filters": None,
            "se_reduction": None,
            "batch_size": 1,
            "device": "cpu",
            "fold_bn": False,
            "dynamic_batch": False,
            "opset": 17,
            "overwrite": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _make_dependencies(
        self,
        *,
        torch_module: _FakeTorch,
        loader: _FakeLoadCheckpoint,
        fold_model: _FakeFoldModel,
    ) -> Any:
        return export_script.RuntimeDependencies(
            torch=torch_module,
            ResNetSE=_FakeModel,
            load_checkpoint=loader,
            export_folded_model=fold_model,
        )

    def test_build_runtime_uses_config_game_and_network_defaults(self) -> None:
        """WHY: Export must reconstruct the exact trained architecture from config so checkpoint loading is valid."""
        torch_module = _FakeTorch()
        loader = _FakeLoadCheckpoint(calls=[], step=4321)
        fold_model = _FakeFoldModel(calls=[])

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint_00004321.pt"
            checkpoint_path.write_text("placeholder", encoding="utf-8")

            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("game: \"go\"\n", encoding="utf-8")
            parsed_config = {
                "game": "go",
                "network": {
                    "architecture": "resnet_se",
                    "num_blocks": 12,
                    "num_filters": 96,
                    "se_reduction": 3,
                },
            }

            original_loader = export_script.load_yaml_config
            export_script.load_yaml_config = lambda _path: parsed_config
            try:
                runtime = export_script.build_export_runtime(
                    args=self._make_args(
                        checkpoint=str(checkpoint_path),
                        config=str(config_path),
                        game=None,
                    ),
                    dependencies=self._make_dependencies(
                        torch_module=torch_module,
                        loader=loader,
                        fold_model=fold_model,
                    ),
                )
            finally:
                export_script.load_yaml_config = original_loader

        self.assertEqual(runtime.game_config.name, "go")
        self.assertEqual(runtime.model.num_blocks, 12)
        self.assertEqual(runtime.model.num_filters, 96)
        self.assertEqual(runtime.model.se_reduction, 3)
        self.assertEqual(runtime.checkpoint_step, 4321)
        self.assertEqual(runtime.output_path.name, "checkpoint_00004321_torchscript.ts")

        self.assertEqual(len(loader.calls), 1)
        loaded_path, _loaded_model, loaded_optimizer, map_location = loader.calls[0]
        self.assertEqual(loaded_path, checkpoint_path)
        self.assertIsNone(loaded_optimizer)
        self.assertEqual(map_location, "cpu")

        self.assertEqual(torch_module.zeros_calls[0][0], (1, 17, 19, 19))

    def test_run_from_args_torchscript_exports_and_applies_bn_folding(self) -> None:
        """WHY: Deployment export must emit a TorchScript artifact and optionally fold BN layers for inference efficiency."""
        torch_module = _FakeTorch()
        loader = _FakeLoadCheckpoint(calls=[], step=100)
        fold_model = _FakeFoldModel(calls=[])

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint_00000100.pt"
            checkpoint_path.write_text("placeholder", encoding="utf-8")
            output_path = Path(temp_dir) / "exported_model"

            summary = export_script.run_from_args(
                self._make_args(
                    checkpoint=str(checkpoint_path),
                    output=str(output_path),
                    fold_bn=True,
                    overwrite=True,
                ),
                dependencies=self._make_dependencies(
                    torch_module=torch_module,
                    loader=loader,
                    fold_model=fold_model,
                ),
            )

            expected_output = output_path.with_suffix(".ts")
            self.assertTrue(expected_output.exists())

        self.assertEqual(summary.export_format, "torchscript")
        self.assertEqual(summary.output_path, expected_output)
        self.assertEqual(summary.checkpoint_step, 100)
        self.assertTrue(summary.fold_batch_norm)
        self.assertEqual(len(fold_model.calls), 1)
        self.assertEqual(len(torch_module.jit.trace_calls), 1)
        self.assertEqual(len(torch_module.jit.freeze_calls), 1)

    def test_run_from_args_onnx_emits_dynamic_batch_metadata_when_requested(self) -> None:
        """WHY: ONNX consumers need dynamic batch axes metadata for variable-size inference batches."""
        torch_module = _FakeTorch()
        loader = _FakeLoadCheckpoint(calls=[], step=250)
        fold_model = _FakeFoldModel(calls=[])

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint_00000250.pt"
            checkpoint_path.write_text("placeholder", encoding="utf-8")
            output_path = Path(temp_dir) / "model_export.onnx"

            summary = export_script.run_from_args(
                self._make_args(
                    checkpoint=str(checkpoint_path),
                    output=str(output_path),
                    format="onnx",
                    batch_size=4,
                    dynamic_batch=True,
                    overwrite=True,
                ),
                dependencies=self._make_dependencies(
                    torch_module=torch_module,
                    loader=loader,
                    fold_model=fold_model,
                ),
            )

            self.assertTrue(output_path.exists())

        self.assertEqual(summary.export_format, "onnx")
        self.assertEqual(summary.output_path, output_path)
        self.assertEqual(summary.checkpoint_step, 250)

        self.assertEqual(len(torch_module.onnx.export_calls), 1)
        onnx_call = torch_module.onnx.export_calls[0]
        self.assertEqual(
            onnx_call["dynamic_axes"],
            {
                "input": {0: "batch"},
                "policy_logits": {0: "batch"},
                "value": {0: "batch"},
            },
        )
        self.assertEqual(onnx_call["opset_version"], 17)

    def test_build_runtime_requires_game_context_without_config(self) -> None:
        """WHY: Export cannot reconstruct model dimensions safely unless game selection is explicitly provided."""
        torch_module = _FakeTorch()
        loader = _FakeLoadCheckpoint(calls=[], step=10)
        fold_model = _FakeFoldModel(calls=[])

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint_00000010.pt"
            checkpoint_path.write_text("placeholder", encoding="utf-8")

            args = self._make_args(
                checkpoint=str(checkpoint_path),
                config=None,
                game=None,
            )

            with self.assertRaisesRegex(ValueError, "Either --game or --config"):
                export_script.build_export_runtime(
                    args=args,
                    dependencies=self._make_dependencies(
                        torch_module=torch_module,
                        loader=loader,
                        fold_model=fold_model,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
