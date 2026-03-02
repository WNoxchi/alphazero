"""Game-specific configuration and YAML loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


_VALID_VALUE_HEAD_TYPES: Final[frozenset[str]] = frozenset({"scalar", "wdl"})


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Static dimensions and behavior flags for a supported game."""

    name: str
    board_shape: tuple[int, int]
    input_channels: int
    action_space_size: int
    value_head_type: str
    supports_symmetry: bool
    num_symmetries: int
    supports_ownership: bool = False
    float_plane_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        rows, cols = self.board_shape
        if rows <= 0 or cols <= 0:
            raise ValueError("board_shape must contain positive dimensions")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.action_space_size <= 0:
            raise ValueError("action_space_size must be positive")
        if self.value_head_type not in _VALID_VALUE_HEAD_TYPES:
            raise ValueError(
                f"value_head_type must be one of {sorted(_VALID_VALUE_HEAD_TYPES)}; "
                f"got {self.value_head_type!r}"
            )
        if self.num_symmetries <= 0:
            raise ValueError("num_symmetries must be positive")
        if not self.supports_symmetry and self.num_symmetries != 1:
            raise ValueError("num_symmetries must be 1 when supports_symmetry is False")
        seen: set[int] = set()
        for plane_index in self.float_plane_indices:
            if isinstance(plane_index, bool) or not isinstance(plane_index, int):
                raise TypeError(
                    "float_plane_indices entries must be integers, "
                    f"got {type(plane_index).__name__}"
                )
            if plane_index < 0 or plane_index >= self.input_channels:
                raise ValueError(
                    "float_plane_indices entries must be in range "
                    f"[0, {self.input_channels}), got {plane_index}"
                )
            if plane_index in seen:
                raise ValueError(
                    f"float_plane_indices entries must be unique, got duplicate {plane_index}"
                )
            seen.add(plane_index)
        if self.input_channels - len(self.float_plane_indices) <= 0:
            raise ValueError(
                "input_channels must include at least one binary plane for replay compression"
            )

    @property
    def num_float_planes(self) -> int:
        return len(self.float_plane_indices)

    @property
    def num_binary_planes(self) -> int:
        return self.input_channels - self.num_float_planes


CHESS_CONFIG = GameConfig(
    name="chess",
    board_shape=(8, 8),
    input_channels=119,
    action_space_size=4672,
    value_head_type="wdl",
    supports_symmetry=False,
    supports_ownership=False,
    num_symmetries=1,
    float_plane_indices=(113, 118),
)

GO_CONFIG = GameConfig(
    name="go",
    board_shape=(19, 19),
    input_channels=17,
    action_space_size=362,
    value_head_type="scalar",
    supports_symmetry=True,
    supports_ownership=True,
    num_symmetries=8,
    float_plane_indices=(),
)


_GAME_CONFIGS: Final[dict[str, GameConfig]] = {
    CHESS_CONFIG.name: CHESS_CONFIG,
    GO_CONFIG.name: GO_CONFIG,
}


def _parse_scalar_token(token: str) -> Any:
    stripped = token.strip()
    if not stripped:
        return ""
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]

    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(stripped)
    except ValueError:
        pass

    try:
        return float(stripped)
    except ValueError:
        return stripped


def _load_yaml_with_fallback(raw: str, path: Path) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        parsed: dict[str, Any] = {}
        saw_top_level_mapping = False

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip(" "))
            if indent == 0 and stripped.startswith("-"):
                raise ValueError(
                    f"Config file {path} must contain a top-level mapping, got sequence item"
                )
            if indent != 0:
                continue

            if ":" not in line:
                raise ValueError(f"Invalid top-level YAML line in {path}: {line!r}")

            key, value = line.split(":", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Invalid empty key in {path}: {line!r}")

            saw_top_level_mapping = True
            value = value.split("#", 1)[0].strip()
            parsed[key] = {} if value == "" else _parse_scalar_token(value)

        if not saw_top_level_mapping:
            raise ValueError(f"Config file {path} must contain a top-level mapping")
        return parsed

    return yaml.safe_load(raw)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a pipeline YAML file and return its top-level mapping."""

    path = Path(config_path)
    raw = path.read_text(encoding="utf-8")
    loaded = _load_yaml_with_fallback(raw, path)

    if loaded is None:
        raise ValueError(f"Config file {path} is empty")
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Config file {path} must contain a top-level mapping, got {type(loaded).__name__}"
        )
    return loaded


def get_game_config(game_name: str) -> GameConfig:
    """Return the canonical game config for a game name."""

    normalized = game_name.strip().lower()
    try:
        return _GAME_CONFIGS[normalized]
    except KeyError as exc:
        supported_games = ", ".join(sorted(_GAME_CONFIGS))
        raise ValueError(
            f"Unsupported game {game_name!r}; supported games: {supported_games}"
        ) from exc


def load_game_config_from_yaml(config_path: str | Path) -> GameConfig:
    """Resolve `game` from a pipeline YAML file into a predefined ``GameConfig``."""

    config = load_yaml_config(config_path)
    game_name = config.get("game")
    if not isinstance(game_name, str) or not game_name.strip():
        raise ValueError(
            f"Config file {config_path} must define a non-empty string for the 'game' key"
        )
    return get_game_config(game_name)


__all__ = [
    "GameConfig",
    "CHESS_CONFIG",
    "GO_CONFIG",
    "get_game_config",
    "load_yaml_config",
    "load_game_config_from_yaml",
]
