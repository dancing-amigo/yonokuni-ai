from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from yonokuni.models.network import YonokuniEvaluator, YonokuniNet, YonokuniNetConfig


@dataclass
class InferenceConfig:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    enable_amp: bool = torch.cuda.is_available()


class InferenceRunner:
    def __init__(self, model: YonokuniNet, config: Optional[InferenceConfig] = None) -> None:
        self.config = config or InferenceConfig()
        self.model = model.to(device=self.config.device, dtype=self.config.dtype)
        self.model.eval()
        self.scaler = torch.cuda.amp.autocast if self.config.enable_amp else _nullcontext

    @torch.no_grad()
    def run(self, board_tensor: torch.Tensor, aux_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        board_tensor = board_tensor.to(device=self.config.device, dtype=self.config.dtype)
        aux_tensor = aux_tensor.to(device=self.config.device, dtype=self.config.dtype)
        with self.scaler():
            policy_logits, value = self.model(board_tensor, aux_tensor)
        return policy_logits.float().cpu(), value.float().cpu()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
