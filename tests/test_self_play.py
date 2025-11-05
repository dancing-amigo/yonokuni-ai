import numpy as np

from yonokuni import YonokuniEnv
from yonokuni.core import ACTION_VECTOR_SIZE, encode_action, enumerate_legal_actions
from yonokuni.features import Transform
from yonokuni.mcts import MCTSConfig
from yonokuni.selfplay import MCTSPolicy, RandomPolicy, ReplayBuffer, SelfPlayManager


def test_self_play_generates_replay_samples():
    buffer = ReplayBuffer(512, transforms=[Transform.IDENTITY], seed=0)
    policy = RandomPolicy(np.random.default_rng(42))
    manager = SelfPlayManager(
        buffer,
        policy=policy,
        env_factory=lambda: YonokuniEnv(max_ply=20),
        temperature=1.0,
        seed=123,
    )

    manager.generate(episodes=1)

    assert len(buffer) > 0
    boards, aux, policies, values = buffer.sample(1, apply_symmetry=False)
    assert boards.shape[1:] == (8, 8, 8)
    assert aux.shape[1:] == (8,)
    assert policies.shape[1] > 0
    assert np.all(np.abs(values) <= 1.0)


def uniform_evaluator(state):
    legal = enumerate_legal_actions(state)
    policy = np.zeros(ACTION_VECTOR_SIZE, dtype=np.float32)
    for action in legal:
        policy[encode_action(action)] = 1.0
    if legal:
        policy /= len(legal)
    return policy, 0.0


def test_self_play_with_mcts_policy():
    buffer = ReplayBuffer(256, transforms=[Transform.IDENTITY], seed=1)
    policy = MCTSPolicy(uniform_evaluator, config=MCTSConfig(num_simulations=8, dirichlet_alpha=0.0))
    manager = SelfPlayManager(
        buffer,
        policy=policy,
        env_factory=lambda: YonokuniEnv(max_ply=10),
        temperature=1.0,
        seed=321,
    )

    manager.generate(episodes=1)
    assert len(buffer) > 0
