from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from yonokuni.core import (
    ACTION_VECTOR_SIZE,
    Action,
    GameResult,
    GameState,
    PlayerColor,
    apply_action,
    decode_action,
    encode_action,
    enumerate_legal_actions,
)


def team_sign(color: PlayerColor) -> int:
    return 1 if color in (PlayerColor.RED, PlayerColor.YELLOW) else -1


@dataclass
class MCTSConfig:
    num_simulations: int = 256
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    temperature: float = 1.0


class Node:
    __slots__ = ("player", "prior", "visit_count", "value_sum", "children", "legal_actions")

    def __init__(self, player: PlayerColor, prior: float = 0.0) -> None:
        self.player: PlayerColor = player
        self.prior: float = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[int, "Node"] = {}
        self.legal_actions: List[int] = []

    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return bool(self.children)

    def total_children_visits(self) -> int:
        return sum(child.visit_count for child in self.children.values())


EvaluationFn = Callable[[GameState], Tuple[np.ndarray, float]]


@dataclass
class MCTSResult:
    visit_counts: np.ndarray
    policy: np.ndarray
    value: float


class MCTS:
    def __init__(
        self,
        evaluator: EvaluationFn,
        config: Optional[MCTSConfig] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    def run(self, state: GameState) -> MCTSResult:
        root_state = state.copy()
        if root_state.result != GameResult.ONGOING:
            value = self._value_from_result(root_state.result, root_state.current_player)
            visit_counts = np.zeros(ACTION_VECTOR_SIZE, dtype=np.float32)
            policy = np.zeros(ACTION_VECTOR_SIZE, dtype=np.float32)
            return MCTSResult(visit_counts=visit_counts, policy=policy, value=value)
        root = Node(root_state.current_player)

        policy, value = self.evaluator(root_state)
        legal_actions = self._expand(root, root_state, policy)
        root.value_sum += value
        root.visit_count += 1

        if legal_actions and self.config.dirichlet_alpha > 0.0:
            self._apply_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            self._simulate(root, root_state.copy())

        visit_counts = np.zeros(ACTION_VECTOR_SIZE, dtype=np.float32)
        for action_index, child in root.children.items():
            visit_counts[action_index] = child.visit_count

        temperature = self.config.temperature
        if temperature <= 1e-6:
            policy_target = np.zeros_like(visit_counts)
            best = int(np.argmax(visit_counts))
            policy_target[best] = 1.0
        else:
            adjusted = visit_counts ** (1.0 / temperature)
            if adjusted.sum() > 0:
                policy_target = adjusted / adjusted.sum()
            else:
                policy_target = np.ones_like(adjusted) / len(adjusted)

        return MCTSResult(visit_counts=visit_counts, policy=policy_target, value=root.q_value())

    # ------------------------------------------------------------------
    def _simulate(self, root: Node, state: GameState) -> None:
        node = root
        path: List[Tuple[Node, int]] = []

        while True:
            if state.result != GameResult.ONGOING:
                value = self._value_from_result(state.result, node.player)
                self._backpropagate(path, node, value)
                return

            if not node.is_expanded():
                policy, value = self.evaluator(state)
                self._expand(node, state, policy)
                self._backpropagate(path, node, value)
                return

            action_index, child = self._select_child(node)
            path.append((node, action_index))
            action = decode_action(action_index)
            state = apply_action(state, action, in_place=False)
            node = child

    def _select_child(self, node: Node) -> Tuple[int, Node]:
        total_visits = node.total_children_visits()
        sqrt_total = np.sqrt(total_visits + 1)
        best_score = -np.inf
        best_action = -1
        best_child: Optional[Node] = None

        for action_index, child in node.children.items():
            q = child.q_value()
            u = self.config.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action_index
                best_child = child

        if best_child is None:
            raise RuntimeError("Failed to select child node.")
        return best_action, best_child

    def _expand(self, node: Node, state: GameState, policy: np.ndarray) -> List[int]:
        if state.result != GameResult.ONGOING:
            node.legal_actions = []
            return []
        legal_actions = enumerate_legal_actions(state, state.current_player)

        action_indices: List[int] = []
        priors = []
        for action in legal_actions:
            idx = encode_action(action)
            action_indices.append(idx)
            priors.append(max(0.0, float(policy[idx])))

        if not action_indices:
            node.legal_actions = []
            return []

        priors_array = np.array(priors, dtype=np.float64)
        if priors_array.sum() <= 0:
            priors_array = np.ones_like(priors_array) / len(priors_array)
        else:
            priors_array = priors_array / priors_array.sum()

        for idx, prior in zip(action_indices, priors_array):
            next_state = apply_action(state, decode_action(idx), in_place=False)
            node.children[idx] = Node(next_state.current_player, prior=prior)
        node.legal_actions = action_indices
        return action_indices

    def _apply_dirichlet_noise(self, node: Node) -> None:
        if not node.legal_actions:
            return
        alpha = self.config.dirichlet_alpha
        frac = self.config.dirichlet_fraction
        noise = self.rng.dirichlet([alpha] * len(node.legal_actions))
        for action_index, noise_value in zip(node.legal_actions, noise):
            child = node.children[action_index]
            child.prior = child.prior * (1 - frac) + noise_value * frac

    def _backpropagate(
        self,
        path: Sequence[Tuple[Node, int]],
        leaf: Node,
        value: float,
    ) -> None:
        leaf.visit_count += 1
        leaf.value_sum += value
        last_player = leaf.player
        last_value = value

        for parent, action_index in reversed(path):
            child = parent.children[action_index]
            if child is not leaf:
                child.visit_count += 1
                child.value_sum += last_value

            if team_sign(parent.player) != team_sign(last_player):
                last_value = -last_value
            last_player = parent.player
            parent.visit_count += 1
            parent.value_sum += last_value

    def _value_from_result(self, result: GameResult, player: PlayerColor) -> float:
        if result == GameResult.DRAW:
            return 0.0
        winner_sign = 1 if result == GameResult.TEAM_A_WIN else -1
        player_sign = team_sign(player)
        return float(winner_sign * player_sign)
