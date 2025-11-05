from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from yonokuni.core import ACTION_VECTOR_SIZE, GameState
from yonokuni.features import (
    AUX_VECTOR_SIZE,
    BOARD_CHANNELS,
    state_to_torch,
)


@dataclass
class YonokuniNetConfig:
    channels: int = 128
    num_blocks: int = 10
    dropout: float = 0.0


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = F.relu(x)
        return x


class YonokuniNet(nn.Module):
    """ResNet-style policy/value network."""

    def __init__(self, config: YonokuniNetConfig = YonokuniNetConfig()) -> None:
        super().__init__()
        self.config = config
        c = config.channels

        self.input_conv = nn.Conv2d(BOARD_CHANNELS, c, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(c)

        self.aux_linear = nn.Linear(AUX_VECTOR_SIZE, c)

        blocks = []
        for _ in range(config.num_blocks):
            blocks.append(ResidualBlock(c, dropout=config.dropout))
        self.res_blocks = nn.Sequential(*blocks)

        # Policy head
        self.policy_conv = nn.Conv2d(c, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_linear = nn.Linear(2 * 8 * 8, ACTION_VECTOR_SIZE)

        # Value head
        self.value_conv = nn.Conv2d(c, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_linear1 = nn.Linear(8 * 8, c)
        self.value_linear2 = nn.Linear(c, 1)

    def forward(self, board: torch.Tensor, aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            board: Tensor shaped (batch, C=8, H=8, W=8)
            aux:   Tensor shaped (batch, 8)
        Returns:
            policy logits (batch, 1792), value (batch, 1)
        """
        x = self.input_conv(board)
        x = self.input_bn(x)
        x = F.relu(x)

        aux_embed = self.aux_linear(aux)
        aux_embed = aux_embed.unsqueeze(-1).unsqueeze(-1)  # (batch, C, 1, 1)
        x = x + aux_embed

        x = self.res_blocks(x)

        # Policy
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_linear(p)

        # Value
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_linear1(v))
        value = torch.tanh(self.value_linear2(v))

        return policy_logits, value.squeeze(-1)


class YonokuniEvaluator:
    """Convenience wrapper to run the network on GameState objects."""

    def __init__(
        self,
        model: YonokuniNet,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model.to(device=device, dtype=dtype)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.model.eval()

    @torch.no_grad()
    def predict(self, state: GameState) -> Tuple[np.ndarray, float]:
        board_t, aux_t = state_to_torch(state, device=self.device, dtype=self.dtype)
        board_t = board_t.unsqueeze(0)
        aux_t = aux_t.unsqueeze(0)
        policy_logits, value = self.model(board_t, aux_t)
        policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
        value_scalar = float(value.cpu().numpy()[0])
        return policy, value_scalar

    @torch.no_grad()
    def predict_batch(self, states: Sequence[GameState]) -> Tuple[np.ndarray, np.ndarray]:
        boards: List[torch.Tensor] = []
        auxes: List[torch.Tensor] = []
        for state in states:
            board_t, aux_t = state_to_torch(state, device=self.device, dtype=self.dtype)
            boards.append(board_t)
            auxes.append(aux_t)
        board_tensor = torch.stack(boards, dim=0)
        aux_tensor = torch.stack(auxes, dim=0)
        policy_logits, values = self.model(board_tensor, aux_tensor)
        policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()
        values_np = values.cpu().numpy()
        return policy, values_np
