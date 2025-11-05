import numpy as np

from yonokuni.core import (
    ACTION_VECTOR_SIZE,
    GameResult,
    PlayerColor,
    encode_action,
    enumerate_legal_actions,
    initialize_game_state,
)
from yonokuni.mcts import MCTS, MCTSConfig


def uniform_evaluator(state):
    legal = enumerate_legal_actions(state)
    policy = np.zeros(ACTION_VECTOR_SIZE, dtype=np.float32)
    for action in legal:
        policy[encode_action(action)] = 1.0
    if legal:
        policy /= len(legal)
    return policy, 0.0


def test_mcts_returns_probability_distribution():
    state = initialize_game_state()
    config = MCTSConfig(num_simulations=32, temperature=1.0, dirichlet_alpha=0.0)
    mcts = MCTS(uniform_evaluator, config=config, rng=np.random.default_rng(0))

    result = mcts.run(state)
    assert result.policy.shape[0] == ACTION_VECTOR_SIZE
    assert np.isclose(result.policy.sum(), 1.0)
    legal = enumerate_legal_actions(state)
    assert np.isclose(result.policy[[encode_action(a) for a in legal]].sum(), 1.0)


def test_mcts_handles_terminal_state():
    state = initialize_game_state()
    state.result = GameResult.TEAM_A_WIN
    state.current_player = PlayerColor.RED
    config = MCTSConfig(num_simulations=8, dirichlet_alpha=0.0)
    mcts = MCTS(uniform_evaluator, config=config, rng=np.random.default_rng(1))

    result = mcts.run(state)
    assert result.value == 1.0
