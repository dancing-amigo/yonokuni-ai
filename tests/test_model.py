import torch

from yonokuni.core import initialize_game_state
from yonokuni.models import YonokuniEvaluator, YonokuniNet, YonokuniNetConfig


def test_network_forward_shapes():
    model = YonokuniNet(YonokuniNetConfig(channels=32, num_blocks=2))
    board = torch.randn(4, 8, 8, 8)
    aux = torch.randn(4, 8)
    policy_logits, value = model(board, aux)
    assert policy_logits.shape == (4, 1792)
    assert value.shape == (4,)


def test_evaluator_outputs_distribution():
    model = YonokuniNet(YonokuniNetConfig(channels=32, num_blocks=2))
    evaluator = YonokuniEvaluator(model)
    state = initialize_game_state()
    policy, value = evaluator.predict(state)
    assert policy.shape == (1792,)
    assert abs(policy.sum() - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0
