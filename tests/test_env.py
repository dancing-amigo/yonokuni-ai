import numpy as np

from yonokuni import YonokuniEnv
from yonokuni.core import ACTION_VECTOR_SIZE, encode_action, enumerate_legal_actions


def test_reset_returns_valid_observation():
    env = YonokuniEnv()
    obs, info = env.reset()

    assert obs["board"].shape == (8, 8, 8)
    assert obs["aux"].shape == (8,)
    assert "legal_action_mask" in info
    assert info["legal_action_mask"].shape == (ACTION_VECTOR_SIZE,)


def test_legal_mask_matches_enumeration():
    env = YonokuniEnv()
    env.reset()
    mask = env.legal_action_mask()
    legal = enumerate_legal_actions(env._state)
    ones = np.count_nonzero(mask)
    assert ones == len(legal)
    for action in legal:
        assert mask[encode_action(action)] == 1


def test_step_advances_state_and_returns_reward():
    env = YonokuniEnv()
    obs, info = env.reset()
    legal_actions = np.flatnonzero(info["legal_action_mask"])
    action = int(legal_actions[0])

    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert np.any(next_obs["board"] != obs["board"])
    assert next_info["legal_action_mask"].shape == (ACTION_VECTOR_SIZE,)
