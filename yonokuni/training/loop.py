from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from yonokuni.models import YonokuniNet
from yonokuni.selfplay import ReplayBuffer


@dataclass
class TrainingConfig:
    batch_size: int = 128
    policy_weight: float = 1.0
    value_weight: float = 1.0
    l2_weight: float = 1e-4
    grad_clip: Optional[float] = None
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32
    apply_symmetry: bool = True


@dataclass
class TrainingStepOutput:
    policy_loss: float
    value_loss: float
    l2_loss: float
    total_loss: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "l2_loss": self.l2_loss,
            "total_loss": self.total_loss,
        }


class Trainer:
    def __init__(
        self,
        model: YonokuniNet,
        optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        config: TrainingConfig = TrainingConfig(),
    ) -> None:
        self.model = model.to(device=config.device, dtype=config.dtype)
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.config = config
        self.device = config.device or torch.device("cpu")
        self.dtype = config.dtype

    def train_step(self) -> TrainingStepOutput:
        if len(self.replay_buffer) == 0:
            raise ValueError("Replay buffer is empty; cannot perform training step.")

        boards_np, aux_np, policy_np, value_np = self.replay_buffer.sample(
            self.config.batch_size,
            apply_symmetry=self.config.apply_symmetry,
        )

        board_tensor = torch.from_numpy(boards_np).to(device=self.device, dtype=self.dtype)
        aux_tensor = torch.from_numpy(aux_np).to(device=self.device, dtype=self.dtype)
        policy_target = torch.from_numpy(policy_np).to(device=self.device, dtype=self.dtype)
        value_target = torch.from_numpy(value_np).to(device=self.device, dtype=self.dtype)

        self.model.train()
        policy_logits, value_pred = self.model(board_tensor, aux_tensor)

        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(policy_target * log_probs).sum(dim=-1).mean()

        value_loss = F.mse_loss(value_pred, value_target)

        l2_loss = torch.zeros((), device=self.device, dtype=self.dtype)
        if self.config.l2_weight > 0:
            for param in self.model.parameters():
                l2_loss = l2_loss + param.pow(2).sum()
            l2_loss = l2_loss * self.config.l2_weight

        total_loss = (
            self.config.policy_weight * policy_loss
            + self.config.value_weight * value_loss
            + l2_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        if self.config.grad_clip and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return TrainingStepOutput(
            policy_loss=float(policy_loss.detach().cpu().item()),
            value_loss=float(value_loss.detach().cpu().item()),
            l2_loss=float(l2_loss.detach().cpu().item()),
            total_loss=float(total_loss.detach().cpu().item()),
        )
