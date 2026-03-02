"""Base interface for AlphaZero neural network architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from alphazero.config import GameConfig


class AlphaZeroNetwork(nn.Module, ABC):
    """Abstract interface shared by all AlphaZero network architectures."""

    def __init__(self, game_config: GameConfig) -> None:
        super().__init__()
        if not isinstance(game_config, GameConfig):
            raise TypeError(
                "game_config must be an instance of alphazero.config.GameConfig, "
                f"got {type(game_config).__name__}"
            )
        self.game_config = game_config

    def validate_input_shape(self, x: torch.Tensor) -> None:
        """Validate a model input tensor against the configured game dimensions."""

        if x.ndim != 4:
            raise ValueError(f"expected input shape (batch, C, H, W), got {tuple(x.shape)}")

        _, channels, height, width = x.shape
        expected_height, expected_width = self.game_config.board_shape
        if channels != self.game_config.input_channels:
            raise ValueError(
                f"expected {self.game_config.input_channels} input channels, got {channels}"
            )
        if (height, width) != (expected_height, expected_width):
            raise ValueError(
                f"expected board shape {(expected_height, expected_width)}, "
                f"got {(height, width)}"
            )

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            x: Board state tensor with shape `(batch, C_in, H, W)`.

        Returns:
            A tuple of `(policy_logits, value)` or
            `(policy_logits, value, ownership_logits)` where:
            - `policy_logits` has shape `(batch, action_space_size)`.
            - `value` has shape `(batch, 1)` for scalar heads or `(batch, 3)` for WDL heads.
            - `ownership_logits` has shape `(batch, board_area)` when present.
        """
