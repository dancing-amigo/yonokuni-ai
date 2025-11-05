from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from yonokuni import Transform, YonokuniEnv
from yonokuni.mcts import MCTSConfig
from yonokuni.models import YonokuniNet, YonokuniNetConfig
from yonokuni.selfplay import ReplayBuffer, SelfPlayManager, make_mcts_policy_from_model
from yonokuni.training import Trainer, TrainingConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


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
    log_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 10
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None


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

        transforms = config.self_play_transforms
        if transforms is None:
            transforms = list(Transform)

        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            transforms=transforms,
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

        self.writer: Optional[SummaryWriter] = None
        if self.config.log_dir and SummaryWriter is not None:
            os.makedirs(self.config.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.config.log_dir)

        self.wandb_run = None
        self._wandb = None
        if self.config.wandb_project:
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError('wandb is not installed but wandb_project is set') from exc
            self._wandb = wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                entity=self.config.wandb_entity,
                config={
                    'episodes_per_iteration': self.config.episodes_per_iteration,
                    'training_steps_per_iteration': self.config.training_steps_per_iteration,
                    'buffer_capacity': self.config.buffer_capacity,
                    'mcts': vars(self.config.mcts_config),
                    'training': self.config.training_config.__dict__,
                },
            )

        self.iteration_index = 0
        self.training_step_count = 0

    def iteration(self) -> Dict[str, object]:
        self_play_stats = self.self_play.generate(self.config.episodes_per_iteration)

        metrics = []
        for _ in range(self.config.training_steps_per_iteration):
            result = self.trainer.train_step().as_dict()
            metrics.append(result)
            if self.writer:
                for key, value in result.items():
                    self.writer.add_scalar(f"train/{key}", value, self.training_step_count)
            self.training_step_count += 1

        buffer_size = len(self.replay_buffer)
        avg_metrics = {}
        if metrics:
            keys = metrics[0].keys()
            avg_metrics = {k: float(np.mean([m[k] for m in metrics])) for k in keys}
        if self.writer:
            self.writer.add_scalar("buffer/size", buffer_size, self.iteration_index)
            if self_play_stats["games_played"]:
                team_a_winrate = self_play_stats["team_a_wins"] / self_play_stats["games_played"]
                team_b_winrate = self_play_stats["team_b_wins"] / self_play_stats["games_played"]
                self.writer.add_scalar("self_play/team_a_winrate", team_a_winrate, self.iteration_index)
                self.writer.add_scalar("self_play/team_b_winrate", team_b_winrate, self.iteration_index)
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f"train_avg/{key}", value, self.iteration_index)
            self.writer.flush()
        team_a_winrate = (self_play_stats["team_a_wins"] / self_play_stats["games_played"]) if self_play_stats["games_played"] else 0.0
        team_b_winrate = (self_play_stats["team_b_wins"] / self_play_stats["games_played"]) if self_play_stats["games_played"] else 0.0
        if self.wandb_run and self._wandb:
            log_data = {f"train_avg/{k}": v for k, v in avg_metrics.items()}
            log_data.update(
                {
                    "buffer/size": buffer_size,
                    "self_play/team_a_winrate": team_a_winrate,
                    "self_play/team_b_winrate": team_b_winrate,
                    "iteration": self.iteration_index,
                }
            )
            if self_play_stats["games_played"]:
                log_data["self_play/draw_rate"] = self_play_stats["draws"] / self_play_stats["games_played"]
                log_data["self_play/avg_moves"] = self_play_stats.get("average_moves", 0.0)
            self._wandb.log(log_data, step=self.iteration_index)

        self.iteration_index += 1

        if (
            self.config.checkpoint_dir
            and self.config.checkpoint_interval > 0
            and self.iteration_index % self.config.checkpoint_interval == 0
        ):
            self.save_checkpoint()

        return {
            "buffer_size": buffer_size,
            "training_metrics": metrics,
            "training_metrics_avg": avg_metrics,
            "iteration": self.iteration_index,
            "self_play": self_play_stats,
        }

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        if not path:
            if not self.config.checkpoint_dir:
                raise ValueError("checkpoint_dir is not configured.")
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_{self.iteration_index:05d}.pt",
            )

        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.to_state(),
            "iteration_index": self.iteration_index,
            "training_step_count": self.training_step_count,
        }
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        buffer_state = state.get("replay_buffer")
        if buffer_state:
            self.replay_buffer.load_state(buffer_state)
        self.iteration_index = state.get("iteration_index", 0)
        self.training_step_count = state.get("training_step_count", 0)

    def close(self) -> None:
        if self.writer:
            self.writer.close()
