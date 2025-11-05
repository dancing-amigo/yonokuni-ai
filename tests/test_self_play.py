import numpy as np

from yonokuni import YonokuniEnv
from yonokuni.features import Transform
from yonokuni.selfplay import RandomPolicy, ReplayBuffer, SelfPlayManager


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
