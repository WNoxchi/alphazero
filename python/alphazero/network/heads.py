"""Policy and value head modules for AlphaZero network architectures."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_board_shape(board_shape: tuple[int, int]) -> tuple[int, int]:
    rows, cols = board_shape
    _validate_positive_int("board_shape rows", rows)
    _validate_positive_int("board_shape cols", cols)
    return rows, cols


class PolicyHead(nn.Module):
    """Project shared tower features to policy logits over the full action space."""

    def __init__(
        self,
        *,
        num_filters: int,
        board_shape: tuple[int, int],
        action_space_size: int,
    ) -> None:
        super().__init__()
        _validate_positive_int("num_filters", num_filters)
        _validate_positive_int("action_space_size", action_space_size)
        board_rows, board_cols = _validate_board_shape(board_shape)

        self.conv = nn.Conv2d(num_filters, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.linear = nn.Linear(32 * board_rows * board_cols, action_space_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = functional.relu(self.bn(self.conv(x)))
        return self.linear(logits.flatten(start_dim=1))


class _ValueHeadBase(nn.Module):
    """Shared value-head stem before scalar/WDL output activation."""

    def __init__(self, *, num_filters: int, board_shape: tuple[int, int], output_dim: int) -> None:
        super().__init__()
        _validate_positive_int("num_filters", num_filters)
        _validate_positive_int("output_dim", output_dim)
        board_rows, board_cols = _validate_board_shape(board_shape)

        self.conv = nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.hidden = nn.Linear(board_rows * board_cols, 256)
        self.linear = nn.Linear(256, output_dim)

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        value = functional.relu(self.bn(self.conv(x)))
        value = functional.relu(self.hidden(value.flatten(start_dim=1)))
        return self.linear(value)


class ScalarValueHead(_ValueHeadBase):
    """Scalar value head for Go; emits tanh-bounded value in [-1, 1]."""

    def __init__(self, *, num_filters: int, board_shape: tuple[int, int]) -> None:
        super().__init__(num_filters=num_filters, board_shape=board_shape, output_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self._forward_logits(x))


class WDLValueHead(_ValueHeadBase):
    """WDL value head for chess; emits normalized win/draw/loss probabilities."""

    def __init__(self, *, num_filters: int, board_shape: tuple[int, int]) -> None:
        super().__init__(num_filters=num_filters, board_shape=board_shape, output_dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self._forward_logits(x), dim=-1)


__all__ = ["PolicyHead", "ScalarValueHead", "WDLValueHead"]
