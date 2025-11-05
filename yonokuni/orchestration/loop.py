from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from yonokuni import YonokuniEnv, Transform
from yonokuni.mcts import MCTSConfig
from yonokuni.models import YonokuniEvaluator, YonokuniNet, YonokuniNetConfig
from yonokuni.selfplay import ReplayBuffer, SelfPlayManager, make_mcts_policy_from_model
from yonokuni.training import Trainer, TrainingConfig


@dataclass
class SelfPlayTrainerConfig:
    episodes_per_iteration: int = 16
    training_steps_per_iteration: int = 32
    buffer_capacity: int = 500_000
    self_play_transforms: Optional[list] = None
    mcts_config: MCTSConfig = field(default_factory=lambda: MCTSConfig(num_simulations=64))
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    model_config: YonokuniNetConfig = field(default_factory=YonokuniNetConfig)
    temperature: float = 1.0
    seed: Optional[int] = None
    env_factory: Callable[[], YonokuniEnv] = YonokuniEnv


class SelfPlayTrainer:
    def __init__(
        self,
        config: SelfPlayTrainerConfig = SelfPlayTrainerConfig(),
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device or config.training_config.device or torch.device("cpu")
        rng = np.random.default_rng(config.seed)

        self.model = YonokuniNet(config.model_config).to(device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            transforms=config.self_play_transforms,
            seed=config.seed,
        )

        self.training_config = config.training_config
        self.training_config.device = self.device

        self.trainer = Trainer(
            self.model,
            self.optimizer,
            self.replay_buffer,
            config=self.training_config,
        )

        policy = make_mcts_policy_from_model(
            self.model,
            config=config.mcts_config,
            device=self.device,
            rng=rng,
        )
        self.self_play = SelfPlayManager(
            self.replay_buffer,
            policy=policy,
            env_factory=config.env_factory,
            temperature=config.temperature,
            seed=config.seed,
        )

    def iteration(self) -> dict:
        self.self_play.generate(self.config.episodes_per_iteration)

        metrics = []
        for _ in range(self.config.training_steps_per_iteration):
            metrics.append(self.trainer.train_step().as_dict())
        return {
            "buffer_size": len(self.replay_buffer),
            "training_metrics": metrics,
        }

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_config": self.training_config,
            "self_play_config": self.config,
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
