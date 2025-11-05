from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from yonokuni.env import YonokuniEnv
from yonokuni.core import GameState, PlayerColor, GameResult
from yonokuni.mcts import MCTS, MCTSConfig
from yonokuni.models import YonokuniEvaluator
from yonokuni.selfplay.replay_buffer import ReplayBuffer, ReplaySample

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def team_of_color(color: PlayerColor) -> str:
    return "A" if color in (PlayerColor.RED, PlayerColor.YELLOW) else "B"


class Policy:
    """Policy interface producing action probabilities over legal moves."""

    def act(self, state: GameState, legal_mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng()

    def act(self, state: GameState, legal_mask: np.ndarray) -> np.ndarray:
        logits = legal_mask.astype(np.float64)
        if logits.sum() == 0:
            return logits
        probs = logits / logits.sum()
        return probs.astype(np.float32, copy=True)


class MCTSPolicy(Policy):
    def __init__(
        self,
        evaluator: Callable[[GameState], Tuple[np.ndarray, float]],
        config: Optional[MCTSConfig] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.mcts = MCTS(evaluator, config=config, rng=rng)

    def act(self, state: GameState, legal_mask: np.ndarray) -> np.ndarray:
        result = self.mcts.run(state)
        policy = result.policy.astype(np.float32, copy=True)
        policy *= legal_mask
        total = policy.sum()
        if total > 0:
            policy /= total
        return policy


def make_mcts_policy_from_model(
    model,
    config: Optional[MCTSConfig] = None,
    *,
    device: Optional["torch.device"] = None,
    dtype: "torch.dtype" = torch.float32,
    rng: Optional[np.random.Generator] = None,
) -> MCTSPolicy:
    if torch is None:
        raise RuntimeError("PyTorch is required to use make_mcts_policy_from_model.")
    evaluator = YonokuniEvaluator(model, device=device, dtype=dtype)

    def eval_fn(state: GameState) -> Tuple[np.ndarray, float]:
        return evaluator.predict(state)

    return MCTSPolicy(eval_fn, config=config, rng=rng)


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

    def generate(self, episodes: int) -> Dict[str, object]:
        results: Dict[str, object] = {
            "games_played": 0,
            "team_a_wins": 0,
            "team_b_wins": 0,
            "draws": 0,
            "total_moves": 0,
        }
        for _ in range(episodes):
            game_result, move_count = self._play_single_episode()
            results["games_played"] += 1
            results["total_moves"] += move_count
            if game_result == GameResult.TEAM_A_WIN:
                results["team_a_wins"] += 1
            elif game_result == GameResult.TEAM_B_WIN:
                results["team_b_wins"] += 1
            else:
                results["draws"] += 1
        if results["games_played"]:
            results["average_moves"] = results["total_moves"] / results["games_played"]
        else:
            results["average_moves"] = 0.0
        return results

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

            state_snapshot = env._state.copy()

            policy_probs = self.policy.act(state_snapshot, legal_mask)
            if policy_probs.shape != legal_mask.shape:
                raise ValueError("Policy output shape mismatch.")
            policy_probs = policy_probs * legal_mask
            if policy_probs.sum() <= 0:
                raise ValueError("Policy assigned zero probability to legal moves.")
            policy_probs = policy_probs / policy_probs.sum()

            current_player = state_snapshot.current_player

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
        final_result = env._state.result

        for step, value in zip(trajectory, final_values):
            self.buffer.add(
                ReplaySample(
                    board=step.board,
                    aux=step.aux,
                    policy=step.policy,
                    value=value,
                )
            )
        move_count = len(trajectory)
        env.close()
        return final_result, move_count

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
