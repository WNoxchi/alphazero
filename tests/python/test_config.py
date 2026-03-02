"""Tests for Python game configuration and YAML loading behavior."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from alphazero.config import (  # noqa: E402
    CHESS_CONFIG,
    GO_CONFIG,
    GameConfig,
    get_game_config,
    load_game_config_from_yaml,
    load_yaml_config,
)


class GameConfigTests(unittest.TestCase):
    def test_predefined_configs_match_spec_dimensions_and_heads(self) -> None:
        """Guards the canonical chess/go tensor/action dimensions so network wiring cannot drift."""
        self.assertEqual(CHESS_CONFIG.name, "chess")
        self.assertEqual(CHESS_CONFIG.board_shape, (8, 8))
        self.assertEqual(CHESS_CONFIG.input_channels, 119)
        self.assertEqual(CHESS_CONFIG.action_space_size, 4672)
        self.assertEqual(CHESS_CONFIG.value_head_type, "wdl")
        self.assertFalse(CHESS_CONFIG.supports_symmetry)
        self.assertFalse(CHESS_CONFIG.supports_ownership)
        self.assertEqual(CHESS_CONFIG.num_symmetries, 1)
        self.assertEqual(CHESS_CONFIG.float_plane_indices, (113, 118))
        self.assertEqual(CHESS_CONFIG.num_float_planes, 2)
        self.assertEqual(CHESS_CONFIG.num_binary_planes, 117)

        self.assertEqual(GO_CONFIG.name, "go")
        self.assertEqual(GO_CONFIG.board_shape, (19, 19))
        self.assertEqual(GO_CONFIG.input_channels, 17)
        self.assertEqual(GO_CONFIG.action_space_size, 362)
        self.assertEqual(GO_CONFIG.value_head_type, "scalar")
        self.assertTrue(GO_CONFIG.supports_symmetry)
        self.assertTrue(GO_CONFIG.supports_ownership)
        self.assertEqual(GO_CONFIG.num_symmetries, 8)
        self.assertEqual(GO_CONFIG.float_plane_indices, ())
        self.assertEqual(GO_CONFIG.num_float_planes, 0)
        self.assertEqual(GO_CONFIG.num_binary_planes, 17)

    def test_game_config_rejects_invalid_float_plane_indices(self) -> None:
        """Protects compact-buffer wiring by validating float-plane metadata shape and uniqueness."""
        with self.assertRaisesRegex(ValueError, "duplicate"):
            GameConfig(
                name="toy",
                board_shape=(3, 3),
                input_channels=4,
                action_space_size=9,
                value_head_type="scalar",
                supports_symmetry=False,
                num_symmetries=1,
                float_plane_indices=(1, 1),
            )

        with self.assertRaisesRegex(ValueError, "in range"):
            GameConfig(
                name="toy",
                board_shape=(3, 3),
                input_channels=4,
                action_space_size=9,
                value_head_type="scalar",
                supports_symmetry=False,
                num_symmetries=1,
                float_plane_indices=(4,),
            )

    def test_default_yaml_files_resolve_to_expected_game_configs(self) -> None:
        """Ensures runtime YAML selection maps exactly to predefined game constants."""
        self.assertEqual(
            load_game_config_from_yaml(ROOT / "configs" / "chess_default.yaml"), CHESS_CONFIG
        )
        self.assertEqual(load_game_config_from_yaml(ROOT / "configs" / "go_default.yaml"), GO_CONFIG)

    def test_chess_runtime_config_uses_expanded_replay_capacity(self) -> None:
        """WHY: throughput scaling relies on a larger replay buffer cap to retain far more compact positions."""
        chess_runtime_config = (ROOT / "configs" / "chess.yaml").read_text(encoding="utf-8")
        self.assertIn("replay_buffer:", chess_runtime_config)
        self.assertIn("capacity: 5000000", chess_runtime_config)

    def test_chess_runtime_config_extends_opening_temperature_window(self) -> None:
        """WHY: self-play diversity tuning requires stochastic move sampling to remain active through move 40."""
        chess_runtime_config = (ROOT / "configs" / "chess.yaml").read_text(encoding="utf-8")
        self.assertIn("temperature_moves: 40", chess_runtime_config)

    def test_get_game_config_is_case_insensitive_and_strips_whitespace(self) -> None:
        """Prevents fragile CLI/config plumbing from failing due to harmless casing or spacing."""
        self.assertEqual(get_game_config(" Chess "), CHESS_CONFIG)
        self.assertEqual(get_game_config("GO"), GO_CONFIG)

    def test_yaml_loader_rejects_missing_or_invalid_game_key(self) -> None:
        """Protects startup by failing fast when required game selection is malformed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_game_path = pathlib.Path(tmp_dir) / "missing_game.yaml"
            missing_game_path.write_text("network: {}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "'game' key"):
                load_game_config_from_yaml(missing_game_path)

            unsupported_game_path = pathlib.Path(tmp_dir) / "unsupported_game.yaml"
            unsupported_game_path.write_text('game: "checkers"\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported game"):
                load_game_config_from_yaml(unsupported_game_path)

    def test_yaml_loader_requires_top_level_mapping(self) -> None:
        """Validates pipeline config shape so accidental list/scalar files fail with a clear error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_path = pathlib.Path(tmp_dir) / "invalid.yaml"
            invalid_path.write_text(
                textwrap.dedent(
                    """
                    - game: chess
                    - game: go
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "top-level mapping"):
                load_yaml_config(invalid_path)


if __name__ == "__main__":
    unittest.main()
