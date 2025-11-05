import torch

from yonokuni.models import InferenceRunner, YonokuniNet, YonokuniNetConfig


def test_inference_runner_shapes():
    model = YonokuniNet(YonokuniNetConfig(channels=16, num_blocks=1))
    runner = InferenceRunner(model)
    board = torch.zeros(2, 8, 8, 8)
    aux = torch.zeros(2, 8)
    policy, value = runner.run(board, aux)
    assert policy.shape == (2, 1792)
    assert value.shape == (2,)
