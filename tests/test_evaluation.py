import numpy as np

from yonokuni.evaluation import RuleBasedPolicy, evaluate_policies
from yonokuni.selfplay import RandomPolicy


def test_evaluate_random_vs_random_small():
    policy_a = RandomPolicy(np.random.default_rng(0))
    policy_b = RandomPolicy(np.random.default_rng(1))
    result = evaluate_policies(policy_a, policy_b, episodes=2)
    assert result.games_played == 2
    assert result.team_a_wins + result.team_b_wins + result.draws == 2
    assert result.average_length > 0



def test_rule_based_policy_distribution():
    policy = RuleBasedPolicy()
    from yonokuni.core import initialize_game_state
    state = initialize_game_state()
    mask = np.zeros(1792, dtype=np.int8)
    mask[:10] = 1
    probs = policy.act(state, mask)
    assert np.all(probs >= 0)
    assert np.isclose(probs.sum(), 1.0)
