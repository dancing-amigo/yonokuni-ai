import numpy as np
import torch

from yonokuni.features import Transform
from yonokuni.models import YonokuniNet, YonokuniNetConfig
from yonokuni.selfplay import ReplayBuffer, ReplaySample
from yonokuni.training import Trainer, TrainingConfig


def make_random_sample() -> ReplaySample:
    board = np.zeros((8, 8, 8), dtype=np.float32)
    # place one random piece
    channel = np.random.randint(0, 8)
    row = np.random.randint(0, 8)
    col = np.random.randint(0, 8)
    board[channel, row, col] = 1.0
    aux = np.zeros((8,), dtype=np.float32)
    aux[np.random.randint(0, 4)] = 1.0
    policy = np.random.random(1792).astype(np.float32)
    policy /= policy.sum()
    value = float(np.random.uniform(-1, 1))
    return ReplaySample(board=board, aux=aux, policy=policy, value=value)


def test_trainer_runs_one_step_and_updates_params():
    buffer = ReplayBuffer(256, transforms=[Transform.IDENTITY], seed=0)
    for _ in range(64):
        buffer.add(make_random_sample())

    model = YonokuniNet(YonokuniNetConfig(channels=32, num_blocks=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model,
        optimizer,
        buffer,
        TrainingConfig(batch_size=16, l2_weight=1e-4, grad_clip=1.0, apply_symmetry=False),
    )

    initial_params = [param.clone().detach() for param in model.parameters()]
    output = trainer.train_step()
    assert output.total_loss > 0

    updated = False
    for param, initial in zip(model.parameters(), initial_params):
        if not torch.allclose(param, initial):
            updated = True
            break
    assert updated, "Model parameters did not change after training step"
