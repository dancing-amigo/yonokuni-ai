from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from yonokuni.env import YonokuniEnv
from yonokuni.core import PlayerColor
from yonokuni.selfplay.replay_buffer import ReplayBuffer, ReplaySample


def team_of_color(color: PlayerColor) -> str:
    return "A" if color in (PlayerColor.RED, PlayerColor.YELLOW) else "B"


class Policy:
    """Policy interface producing action probabilities over legal moves."""

    def act(
        self,
        board: np.ndarray,
        aux: np.ndarray,
        legal_mask: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng()

    def act(self, board: np.ndarray, aux: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        logits = legal_mask.astype(np.float64)
        if logits.sum() == 0:
            return logits
        probs = logits / logits.sum()
        return probs.astype(np.float32, copy=True)


def select_action(probabilities: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if probabilities.sum() == 0:
        raise ValueError("Policy produced zero probability over legal actions.")
    probs = probabilities.astype(np.float64, copy=True)
    if temperature <= 1e-6:
        return int(np.argmax(probs))
    adjusted = probs ** (1.0 / temperature)
    adjusted /= adjusted.sum()
    return int(rng.choice(len(adjusted), p=adjusted))


@dataclass
class TrajectoryStep:
    board: np.ndarray
    aux: np.ndarray
    policy: np.ndarray
    player: PlayerColor


class SelfPlayManager:
    def __init__(
        self,
        buffer: ReplayBuffer,
        policy: Optional[Policy] = None,
        *,
        env_factory: Callable[[], YonokuniEnv] = YonokuniEnv,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.buffer = buffer
        self.policy = policy or RandomPolicy()
        self.env_factory = env_factory
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def generate(self, episodes: int) -> None:
        for _ in range(episodes):
            self._play_single_episode()

    # ------------------------------------------------------------------
    def _play_single_episode(self) -> None:
        env = self.env_factory()
        obs, info = env.reset()

        trajectory: List[TrajectoryStep] = []

        terminated = False
        final_reward = 0.0

        while not terminated:
            board = obs["board"]
            aux = obs["aux"]
            legal_mask = info["legal_action_mask"]

            policy_probs = self.policy.act(board, aux, legal_mask)
            if policy_probs.shape != legal_mask.shape:
                raise ValueError("Policy output shape mismatch.")
            policy_probs = policy_probs * legal_mask
            if policy_probs.sum() <= 0:
                raise ValueError("Policy assigned zero probability to legal moves.")
            policy_probs = policy_probs / policy_probs.sum()

            current_player_idx = int(np.argmax(aux[:4]))
            current_player = PlayerColor(current_player_idx + 1)

            trajectory.append(
                TrajectoryStep(
                    board=board.copy(),
                    aux=aux.copy(),
                    policy=policy_probs.astype(np.float32),
                    player=current_player,
                )
            )

            action_index = select_action(policy_probs, self.temperature, self.rng)
            obs, reward, terminated, truncated, info = env.step(action_index)
            final_reward = reward
            if truncated:
                terminated = True

        final_values = self._final_values(final_reward, trajectory)

        for step, value in zip(trajectory, final_values):
            self.buffer.add(
                ReplaySample(
                    board=step.board,
                    aux=step.aux,
                    policy=step.policy,
                    value=value,
                )
            )
        env.close()

    def _final_values(self, final_reward: float, trajectory: Sequence[TrajectoryStep]) -> List[float]:
        if final_reward > 0:
            winning_team = "A"
        elif final_reward < 0:
            winning_team = "B"
        else:
            return [0.0 for _ in trajectory]

        winner_sign = 1.0 if winning_team == "A" else -1.0
        values: List[float] = []
        for step in trajectory:
            player_team = team_of_color(step.player)
            if player_team == "A":
                value = winner_sign
            else:
                value = -winner_sign
            values.append(value)
        return values
