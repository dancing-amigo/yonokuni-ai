from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from yonokuni.core import GameResult, GameState, PlayerColor, decode_action
from yonokuni.env import YonokuniEnv
from yonokuni.selfplay.self_play import Policy


@dataclass
class EvaluationResult:
    games_played: int
    team_a_wins: int
    team_b_wins: int
    draws: int
    average_length: float

    def winrate_team_a(self) -> float:
        return self.team_a_wins / max(1, self.games_played)

    def winrate_team_b(self) -> float:
        return self.team_b_wins / max(1, self.games_played)


class RuleBasedPolicy(Policy):
    """Very simple heuristic policy to act as evaluation baseline."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng()

    def act(self, state: GameState, legal_mask: np.ndarray) -> np.ndarray:
        indices = np.flatnonzero(legal_mask)
        if len(indices) == 0:
            return legal_mask.astype(np.float32)

        scores = []
        center = np.array([3.5, 3.5])
        for idx in indices:
            action = decode_action(int(idx))
            dest = np.array([action.to_row, action.to_col], dtype=np.float32)
            score = -np.linalg.norm(dest - center)
            scores.append(score)

        scores = np.array(scores)
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()

        result = np.zeros_like(legal_mask, dtype=np.float32)
        result[indices] = probs
        return result


def evaluate_policies(
    policy_team_a: Policy,
    policy_team_b: Policy,
    *,
    episodes: int,
    env_factory: Optional[callable] = None,
) -> EvaluationResult:
    env_factory = env_factory or YonokuniEnv

    team_a_wins = 0
    team_b_wins = 0
    draws = 0
    total_ply = 0

    for _ in range(episodes):
        env = env_factory()
        obs, info = env.reset()
        terminated = False
        ply = 0

        while not terminated:
            state_snapshot: GameState = env._state.copy()
            legal_mask = info["legal_action_mask"]
            current_player = state_snapshot.current_player
            policy = policy_team_a if current_player.team == "A" else policy_team_b
            probs = policy.act(state_snapshot, legal_mask)
            if probs.sum() <= 0:
                probs = legal_mask.astype(np.float32)
                probs /= probs.sum()
            action_index = int(np.random.choice(len(probs), p=probs))
            obs, reward, terminated, truncated, info = env.step(action_index)
            ply += 1
            if truncated:
                terminated = True

        total_ply += ply
        if env._state.result == GameResult.TEAM_A_WIN:
            team_a_wins += 1
        elif env._state.result == GameResult.TEAM_B_WIN:
            team_b_wins += 1
        else:
            draws += 1

    average_length = total_ply / max(1, episodes)
    return EvaluationResult(
        games_played=episodes,
        team_a_wins=team_a_wins,
        team_b_wins=team_b_wins,
        draws=draws,
        average_length=average_length,
    )
