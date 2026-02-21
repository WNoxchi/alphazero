"""Scaffold validation tests for TASK-001."""

from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class ScaffoldLayoutTests(unittest.TestCase):
    def test_project_layout_includes_required_scaffold_paths(self) -> None:
        """Guard against accidental deletion of directories/files that unblock later tasks."""
        required_paths = [
            ROOT / "configs" / "chess_default.yaml",
            ROOT / "configs" / "go_default.yaml",
            ROOT / "src" / "CMakeLists.txt",
            ROOT / "src" / "games" / "game_state.h",
            ROOT / "src" / "mcts" / "mcts_node.h",
            ROOT / "python" / "alphazero" / "network" / "resnet_se.py",
            ROOT / "tests" / "cpp" / "CMakeLists.txt",
            ROOT / "scripts" / "train.py",
        ]

        missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.exists()]
        self.assertEqual(missing, [], msg=f"Missing scaffold paths: {missing}")

    def test_default_configs_include_expected_pipeline_keys(self) -> None:
        """Ensure default configs keep the mandatory top-level sections required by the runtime."""
        common_required_keys = [
            "game:",
            "network:",
            "mcts:",
            "training:",
            "pipeline:",
            "replay_buffer:",
            "evaluation:",
            "system:",
        ]

        chess_text = (ROOT / "configs" / "chess_default.yaml").read_text(encoding="utf-8")
        go_text = (ROOT / "configs" / "go_default.yaml").read_text(encoding="utf-8")

        for key in common_required_keys:
            self.assertIn(key, chess_text, msg=f"chess config missing key marker: {key}")
            self.assertIn(key, go_text, msg=f"go config missing key marker: {key}")

        # This catches accidental cross-game config swaps.
        self.assertIn("dirichlet_alpha: 0.3", chess_text)
        self.assertIn("dirichlet_alpha: 0.03", go_text)


if __name__ == "__main__":
    unittest.main()
