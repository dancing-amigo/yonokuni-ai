from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from yonokuni.env import YonokuniEnv
from yonokuni.core import GameResult, GameState, PlayerColor
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


@dataclass
class EarlyTerminationConfig:
    enable_resign: bool = False
    resign_threshold: float = 0.9
    resign_min_moves: int = 60
    resign_consecutive: int = 8
    resign_disable_ratio: float = 0.0  # fraction of games to ignore resign (0.0-1.0)

    enable_stagnation: bool = False
    stagnation_no_death: int = 50  # consecutive moves without a new death to trigger draw

    enable_value_draw: bool = False
    value_draw_band: float = 0.05
    value_draw_consecutive: int = 20

    enable_repetition: bool = False
    repetition_count: int = 3  # repetitions to declare draw

    enable_soft_maxply: bool = False
    soft_value_band: float = 0.1  # near-even value band
    soft_remaining_moves: int = 20  # if within band and remaining moves below this, declare draw


def _init_endgame_state(state: GameState, *, rng: np.random.Generator) -> GameState:
    """Mutate a GameState to a late-game random setup with 4 pieces per color."""
    board = state.board
    board[:] = 0
    # choose 4 positions per color (16 total), prefer central area for quicker finishes
    centers = [(r, c) for r in range(2, 6) for c in range(2, 6)]
    outer = [(r, c) for r in range(8) for c in range(8) if (r, c) not in centers]
    positions = []
    positions += rng.choice(len(centers), size=min(len(centers), 8), replace=False).tolist()
    # ensure enough positions: take from outer if needed
    remaining = 16 - len(positions)
    if remaining > 0:
        positions += [len(centers) + idx for idx in rng.choice(len(outer), size=remaining, replace=False).tolist()]
    coords: List[Tuple[int, int]] = []
    for idx in positions:
        if idx < len(centers):
            coords.append(centers[idx])
        else:
            coords.append(outer[idx - len(centers)])

    colors = [PlayerColor.RED, PlayerColor.BLUE, PlayerColor.YELLOW, PlayerColor.GREEN]
    coord_iter = iter(coords)
    for color in colors:
        for _ in range(4):
            r, c = next(coord_iter)
            board[r, c] = int(color)

    state.board = board
    state.dead_mask[:] = False
    state.dead_players[:] = False
    state.captured_counts[:] = 0
    state.current_player = rng.choice(list(PlayerColor))
    state.ply_count = 0
    state.result = GameResult.ONGOING
    state.last_action = None
    return state


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
        evaluator: Optional[YonokuniEvaluator] = None,
        early_termination: EarlyTerminationConfig = EarlyTerminationConfig(),
        step_penalty: float = 0.0,
        endgame_start: bool = False,
    ) -> None:
        self.buffer = buffer
        self.policy = policy or RandomPolicy()
        self.env_factory = env_factory
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.temperature_schedule = self._normalize_schedule(temperature_schedule, default=temperature)
        self.evaluator = evaluator
        self.early_termination = early_termination
        self.step_penalty = float(step_penalty)
        self.endgame_start = endgame_start

        self.value_fn: Optional[Callable[[GameState], float]] = None
        if self.evaluator:
            self.value_fn = lambda state: self.evaluator.predict(state)[1]

    def _normalize_schedule(
        self,
        schedule: Optional[Sequence[Tuple[int, float]]],
        default: float,
    ) -> List[Tuple[int, float]]:
        if not schedule:
            return []
        normalized: List[Tuple[int, float]] = []
        for ply, temp in schedule:
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
            "draws_max_ply": 0,
            "draws_early": 0,
            "draws_other": 0,
            "total_moves": 0,
            "center_wins_team_a": 0,
            "center_wins_team_b": 0,
            "center_wins_none": 0,
            "death_turn_sum": [0.0, 0.0, 0.0, 0.0],
            "death_turn_count": [0, 0, 0, 0],
        }
        if workers <= 1:
            for _ in range(episodes):
                game_result, move_count, stats = self._play_single_episode(self.policy)
                self._accumulate_results(results, game_result, move_count, stats)
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
                    for key in (
                        "games_played",
                        "team_a_wins",
                        "team_b_wins",
                        "draws",
                        "draws_max_ply",
                        "draws_early",
                        "draws_other",
                        "total_moves",
                    ):
                        results[key] += worker_result[key]
                    results["center_wins_team_a"] += worker_result["center_wins_team_a"]
                    results["center_wins_team_b"] += worker_result["center_wins_team_b"]
                    results["center_wins_none"] += worker_result["center_wins_none"]
                    for idx in range(4):
                        results["death_turn_sum"][idx] += worker_result["death_turn_sum"][idx]
                        results["death_turn_count"][idx] += worker_result["death_turn_count"][idx]
        if results["games_played"]:
            results["average_moves"] = results["total_moves"] / results["games_played"]
        else:
            results["average_moves"] = 0.0
        results["average_death_turns"] = [
            (results["death_turn_sum"][idx] / results["death_turn_count"][idx])
            if results["death_turn_count"][idx] > 0
            else None
            for idx in range(4)
        ]
        return results

    def _play_single_episode(
        self,
        policy: Policy,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[GameResult, int, Dict[str, object]]:
        rng = rng or self.rng
        env = self.env_factory()
        obs, info = env.reset()

        if self.endgame_start:
            env._state = _init_endgame_state(env._state, rng=rng)
            obs = env._build_observation()
            info = env._build_info()

        trajectory: List[TrajectoryStep] = []

        terminated = False
        final_reward = 0.0
        move_index = 0
        death_turns = [-1, -1, -1, -1]
        last_dead_count = int(np.sum(env._state.dead_players))

        repetition_counts: Dict[bytes, int] = {}
        resign_streak = 0
        value_draw_streak = 0
        stagnation_streak = 0
        early_reason: Optional[str] = None

        while not terminated:
            board = obs["board"]
            aux = obs["aux"]
            legal_mask = info["legal_action_mask"]

            state_snapshot = env._state.copy()
            dead_before = state_snapshot.dead_players.copy()

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

            dead_after = env._state.dead_players
            for idx in range(4):
                if dead_after[idx] and not dead_before[idx] and death_turns[idx] < 0:
                    death_turns[idx] = move_index

            # Early termination checks (post-step)
            early_cfg = self.early_termination
            value_pred: Optional[float] = None
            if self.value_fn and (
                early_cfg.enable_resign or early_cfg.enable_value_draw or early_cfg.enable_soft_maxply
            ):
                value_pred = float(self.value_fn(env._state))

            # Repetition draw
            if early_cfg.enable_repetition:
                board_key = env._state.board.tobytes()
                repetition_counts[board_key] = repetition_counts.get(board_key, 0) + 1
                if repetition_counts[board_key] >= early_cfg.repetition_count:
                    env._state.result = GameResult.DRAW
                    final_reward = 0.0
                    terminated = True
                    early_reason = "repetition"

            # Stagnation draw (no new deaths)
            if not terminated and early_cfg.enable_stagnation:
                dead_count = int(np.sum(env._state.dead_players))
                if dead_count == last_dead_count:
                    stagnation_streak += 1
                else:
                    stagnation_streak = 0
                last_dead_count = dead_count
                if stagnation_streak >= early_cfg.stagnation_no_death:
                    env._state.result = GameResult.DRAW
                    final_reward = 0.0
                    terminated = True
                    early_reason = "stagnation"

            # Value-based draw (near-even for long)
            if not terminated and early_cfg.enable_value_draw and value_pred is not None:
                if abs(value_pred) < early_cfg.value_draw_band:
                    value_draw_streak += 1
                else:
                    value_draw_streak = 0
                if value_draw_streak >= early_cfg.value_draw_consecutive:
                    env._state.result = GameResult.DRAW
                    final_reward = 0.0
                    terminated = True
                    early_reason = "value_draw"

            # Resign if clearly losing for consecutive moves
            if not terminated and early_cfg.enable_resign and value_pred is not None:
                if move_index >= early_cfg.resign_min_moves and abs(value_pred) >= early_cfg.resign_threshold:
                    resign_streak += 1
                else:
                    resign_streak = 0
                if early_cfg.resign_disable_ratio > 0.0:
                    if rng.random() < early_cfg.resign_disable_ratio:
                        resign_streak = 0
                if resign_streak >= early_cfg.resign_consecutive:
                    if value_pred > 0:
                        env._state.result = GameResult.TEAM_A_WIN
                        final_reward = 1.0
                    else:
                        env._state.result = GameResult.TEAM_B_WIN
                        final_reward = -1.0
                    terminated = True
                    early_reason = "resign"

            # Soft max-ply: near-even and close to cap -> draw
            if not terminated and early_cfg.enable_soft_maxply and value_pred is not None:
                remaining = getattr(env, "_max_ply", None)
                if remaining is not None:
                    remaining = env._max_ply - (move_index + 1)
                    if remaining <= early_cfg.soft_remaining_moves and abs(value_pred) < early_cfg.soft_value_band:
                        env._state.result = GameResult.DRAW
                        final_reward = 0.0
                        terminated = True
                        early_reason = "soft_max_ply"

            move_index += 1
            if truncated:
                terminated = True

        final_values = self._final_values(final_reward, trajectory)
        final_result = env._state.result
        center_team = self._center_team(env._state)
        draw_reason: Optional[str] = None
        if final_result == GameResult.DRAW:
            if early_reason is not None:
                draw_reason = early_reason
            elif env._state.ply_count >= env._state.max_ply:
                draw_reason = "max_ply"
            else:
                draw_reason = "other"

        move_count = len(trajectory)
        for step, value in zip(trajectory, final_values):
            adjusted_value = value - (self.step_penalty * move_count) if self.step_penalty != 0.0 else value
            self.buffer.add(
                ReplaySample(
                    board=step.board,
                    aux=step.aux,
                    policy=step.policy,
                    value=adjusted_value,
                )
            )
        env.close()
        game_stats = {
            "center_team": center_team,
            "death_turns": death_turns,
            "early_termination": early_reason,
            "draw_reason": draw_reason,
        }
        return final_result, move_count, game_stats

    def _center_team(self, state: GameState) -> Optional[str]:
        centres = [(3, 3), (3, 4), (4, 3), (4, 4)]
        occupants: List[PlayerColor] = []
        for r, c in centres:
            val = state.board[r][c]
            if val == 0:
                return None
            occupants.append(PlayerColor(int(val)))
        team = team_of_color(occupants[0])
        if all(team_of_color(piece) == team for piece in occupants):
            return team
        return None

    def _accumulate_results(
        self,
        results: Dict[str, object],
        game_result: GameResult,
        move_count: int,
        game_stats: Dict[str, object],
    ) -> None:
        results["games_played"] += 1
        results["total_moves"] += move_count
        if game_result == GameResult.TEAM_A_WIN:
            results["team_a_wins"] += 1
        elif game_result == GameResult.TEAM_B_WIN:
            results["team_b_wins"] += 1
        else:
            results["draws"] += 1
            reason = game_stats.get("draw_reason")
            if reason == "max_ply":
                results["draws_max_ply"] += 1
            elif reason in ("repetition", "stagnation", "value_draw", "soft_max_ply"):
                results["draws_early"] += 1
            else:
                results["draws_other"] += 1

        center_team = game_stats.get("center_team")
        if center_team == "A":
            results["center_wins_team_a"] += 1
        elif center_team == "B":
            results["center_wins_team_b"] += 1
        else:
            results["center_wins_none"] += 1

        death_turns = game_stats.get("death_turns", [])
        for idx, turn in enumerate(death_turns):
            if turn is not None and turn >= 0:
                results["death_turn_sum"][idx] += float(turn)
                results["death_turn_count"][idx] += 1

    def _run_worker(self, policy: Policy, episodes: int, seed: Optional[int] = None) -> Dict[str, object]:
        worker_results = {
            "games_played": 0,
            "team_a_wins": 0,
            "team_b_wins": 0,
            "draws": 0,
            "draws_max_ply": 0,
            "draws_early": 0,
            "draws_other": 0,
            "total_moves": 0,
            "center_wins_team_a": 0,
            "center_wins_team_b": 0,
            "center_wins_none": 0,
            "death_turn_sum": [0.0, 0.0, 0.0, 0.0],
            "death_turn_count": [0, 0, 0, 0],
        }
        rng = np.random.default_rng(seed)
        for _ in range(episodes):
            game_result, move_count, game_stats = self._play_single_episode(policy, rng)
            self._accumulate_results(worker_results, game_result, move_count, game_stats)
        return worker_results

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
