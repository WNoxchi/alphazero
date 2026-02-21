"""ResNet + Squeeze-and-Excitation network for AlphaZero."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from alphazero.config import GameConfig
from alphazero.network.base import AlphaZeroNetwork
from alphazero.network.heads import PolicyHead, ScalarValueHead, WDLValueHead


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


class SEResidualBlock(nn.Module):
    """Residual block with Leela-style SE scale+bias modulation."""

    def __init__(self, num_filters: int, se_reduction: int) -> None:
        super().__init__()
        _validate_positive_int("num_filters", num_filters)
        _validate_positive_int("se_reduction", se_reduction)

        squeezed_channels = max(1, num_filters // se_reduction)

        self.conv_1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(num_filters)
        self.conv_2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(num_filters)

        self.se_fc_1 = nn.Linear(num_filters, squeezed_channels)
        self.se_fc_2 = nn.Linear(squeezed_channels, 2 * num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = functional.relu(self.bn_1(self.conv_1(x)))
        out = self.bn_2(self.conv_2(out))

        squeezed = out.mean(dim=(2, 3))
        se_projection = self.se_fc_2(functional.relu(self.se_fc_1(squeezed)))
        scale, bias = torch.chunk(se_projection, chunks=2, dim=1)
        scale = torch.sigmoid(scale).unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        out = scale * out + bias

        return functional.relu(out + residual)


class ResNetSE(AlphaZeroNetwork):
    """AlphaZero ResNet with Leela-style SE residual blocks."""

    def __init__(
        self,
        game_config: GameConfig,
        *,
        num_blocks: int = 20,
        num_filters: int = 256,
        se_reduction: int = 4,
    ) -> None:
        super().__init__(game_config)
        _validate_positive_int("num_blocks", num_blocks)
        _validate_positive_int("num_filters", num_filters)
        _validate_positive_int("se_reduction", se_reduction)

        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.se_reduction = se_reduction

        self.input_conv = nn.Conv2d(
            self.game_config.input_channels,
            num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.input_bn = nn.BatchNorm2d(num_filters)

        self.residual_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters=num_filters, se_reduction=se_reduction) for _ in range(num_blocks)]
        )

        self.policy_head = PolicyHead(
            num_filters=num_filters,
            board_shape=self.game_config.board_shape,
            action_space_size=self.game_config.action_space_size,
        )

        self.value_head: ScalarValueHead | WDLValueHead
        if self.game_config.value_head_type == "scalar":
            self.value_head = ScalarValueHead(num_filters=num_filters, board_shape=self.game_config.board_shape)
        else:
            self.value_head = WDLValueHead(num_filters=num_filters, board_shape=self.game_config.board_shape)

        self._initialize_weights()

    @classmethod
    def small(cls, game_config: GameConfig) -> ResNetSE:
        """Construct the small dev profile (10 blocks, 128 filters)."""

        return cls(game_config, num_blocks=10, num_filters=128, se_reduction=4)

    @classmethod
    def medium(cls, game_config: GameConfig) -> ResNetSE:
        """Construct the medium default profile (20 blocks, 256 filters)."""

        return cls(game_config, num_blocks=20, num_filters=256, se_reduction=4)

    @classmethod
    def large(cls, game_config: GameConfig) -> ResNetSE:
        """Construct the large profile (40 blocks, 256 filters)."""

        return cls(game_config, num_blocks=40, num_filters=256, se_reduction=4)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

        for block in self.residual_blocks:
            nn.init.xavier_uniform_(block.se_fc_1.weight)
            nn.init.zeros_(block.se_fc_1.bias)
            nn.init.xavier_uniform_(block.se_fc_2.weight)
            nn.init.zeros_(block.se_fc_2.bias)

        nn.init.zeros_(self.policy_head.linear.weight)
        nn.init.zeros_(self.policy_head.linear.bias)
        nn.init.zeros_(self.value_head.linear.weight)
        nn.init.zeros_(self.value_head.linear.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate_input_shape(x)

        features = functional.relu(self.input_bn(self.input_conv(x)))
        for block in self.residual_blocks:
            features = block(features)

        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value


__all__ = ["SEResidualBlock", "ResNetSE"]
