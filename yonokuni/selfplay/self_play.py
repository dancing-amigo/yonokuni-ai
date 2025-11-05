from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from concurrent.futures import ThreadPoolExecutor

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

    def spawn(self, seed: Optional[int] = None) -> "Policy":
        """Return a copy of this policy for parallel execution."""
        return self


class RandomPolicy(Policy):
    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng()

    def act(self, state: GameState, legal_mask: np.ndarray) -> np.ndarray:
        logits = legal_mask.astype(np.float64)
        if logits.sum() == 0:
            return logits
        probs = logits / logits.sum()
        return probs.astype(np.float32, copy=True)

    def spawn(self, seed: Optional[int] = None) -> "RandomPolicy":
        rng = np.random.default_rng(seed)
        return RandomPolicy(rng)


class MCTSPolicy(Policy):
    def __init__(
        self,
        evaluator: Callable[[GameState], Tuple[np.ndarray, float]],
        config: Optional[MCTSConfig] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._config = deepcopy(config) if config else MCTSConfig()
        self._evaluator = evaluator
        self.mcts = MCTS(evaluator, config=self._config, rng=rng or np.random.default_rng())

    def act(self, state: GameState, legal_mask: np.ndarray) -> np.ndarray:
        result = self.mcts.run(state)
        policy = result.policy.astype(np.float32, copy=True)
        policy *= legal_mask
        total = policy.sum()
        if total > 0:
            policy /= total
        return policy

    def spawn(self, seed: Optional[int] = None) -> "MCTSPolicy":
        rng = np.random.default_rng(seed)
        return MCTSPolicy(self._evaluator, config=self._config, rng=rng)


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
        temperature_schedule: Optional[Sequence[Tuple[int, float]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.buffer = buffer
        self.policy = policy or RandomPolicy()
        self.env_factory = env_factory
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.temperature_schedule = self._normalize_schedule(temperature_schedule, default=temperature)

    def _normalize_schedule(
        self,
        schedule: Optional[Sequence[Tuple[int, float]]],
        default: float,
    ) -> List[Tuple[int, float]]:
        if not schedule:
            return []
        normalized: List[Tuple[int, float]] = []
        for entry in schedule:
            ply, temp = entry
            normalized.append((int(ply), float(temp)))
        normalized.sort(key=lambda x: x[0])
        if normalized[0][0] != 0:
            normalized.insert(0, (0, default))
        return normalized

    def _temperature_for_move(self, move_index: int) -> float:
        if not self.temperature_schedule:
            return self.temperature
        current_temp = self.temperature_schedule[0][1]
        for threshold, temp in self.temperature_schedule:
            if move_index < threshold:
                return current_temp
            current_temp = temp
        return current_temp


    def generate(self, episodes: int, workers: int = 1) -> Dict[str, object]:
        results: Dict[str, object] = {
            "games_played": 0,
            "team_a_wins": 0,
            "team_b_wins": 0,
            "draws": 0,
            "total_moves": 0,
        }
        if workers <= 1:
            for _ in range(episodes):
                game_result, move_count = self._play_single_episode(self.policy)
                self._accumulate_results(results, game_result, move_count)
        else:
            counts = [episodes // workers] * workers
            for i in range(episodes % workers):
                counts[i] += 1
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                for count in counts:
                    if count == 0:
                        continue
                    seed = int(self.rng.integers(2**32))
                    policy = self.policy.spawn(seed)
                    futures.append(executor.submit(self._run_worker, policy, count, seed))
                for future in futures:
                    worker_result = future.result()
                    for key in ("games_played", "team_a_wins", "team_b_wins", "draws", "total_moves"):
                        results[key] += worker_result[key]
        if results["games_played"]:
            results["average_moves"] = results["total_moves"] / results["games_played"]
        else:
            results["average_moves"] = 0.0
        return results

    def _accumulate_results(self, results: Dict[str, object], game_result: GameResult, move_count: int) -> None:
        results["games_played"] += 1
        results["total_moves"] += move_count
        if game_result == GameResult.TEAM_A_WIN:
            results["team_a_wins"] += 1
        elif game_result == GameResult.TEAM_B_WIN:
            results["team_b_wins"] += 1
        else:
            results["draws"] += 1

    def _run_worker(self, policy: Policy, episodes: int, seed: Optional[int] = None) -> Dict[str, object]:
        worker_results = {
            "games_played": 0,
            "team_a_wins": 0,
            "team_b_wins": 0,
            "draws": 0,
            "total_moves": 0,
        }
        rng = np.random.default_rng(seed)
        for _ in range(episodes):
            game_result, move_count = self._play_single_episode(policy, rng)
            self._accumulate_results(worker_results, game_result, move_count)
        return worker_results


    # ------------------------------------------------------------------
    def _play_single_episode(self, policy: Policy, rng: Optional[np.random.Generator] = None) -> Tuple[GameResult, int]:
        rng = rng or self.rng
        env = self.env_factory()
        obs, info = env.reset()

        trajectory: List[TrajectoryStep] = []

        terminated = False
        final_reward = 0.0
        move_index = 0

        while not terminated:
            board = obs["board"]
            aux = obs["aux"]
            legal_mask = info["legal_action_mask"]

            state_snapshot = env._state.copy()

            policy_probs = policy.act(state_snapshot, legal_mask)
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

            temperature = self._temperature_for_move(move_index)
            action_index = select_action(policy_probs, temperature, rng)
            obs, reward, terminated, truncated, info = env.step(action_index)
            final_reward = reward
            move_index += 1
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
