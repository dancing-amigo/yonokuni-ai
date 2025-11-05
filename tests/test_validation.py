import numpy as np
import pytest

from yonokuni.selfplay import ReplayBuffer, ReplaySample
from yonokuni.validation import ReplayDataError, validate_batch, validate_buffer_sample


def make_sample():
    board = np.zeros((8, 8, 8), dtype=np.float32)
    aux = np.zeros((8,), dtype=np.float32)
    aux[0] = 1.0
    policy = np.ones(1792, dtype=np.float32) / 1792
    value = 0.0
    return ReplaySample(board=board, aux=aux, policy=policy, value=value)


def test_validate_batch_ok():
    boards = np.zeros((2, 8, 8, 8), dtype=np.float32)
    aux = np.zeros((2, 8), dtype=np.float32)
    policies = np.ones((2, 1792), dtype=np.float32) / 1792
    values = np.zeros((2,), dtype=np.float32)
    validate_batch(boards, aux, policies, values)


def test_validate_batch_nan():
    boards = np.zeros((1, 8, 8, 8), dtype=np.float32)
    aux = np.zeros((1, 8), dtype=np.float32)
    policies = np.ones((1, 1792), dtype=np.float32) / 1792
    values = np.zeros((1,), dtype=np.float32)
    boards[0, 0, 0, 0] = np.nan
    with pytest.raises(ReplayDataError):
        validate_batch(boards, aux, policies, values)


def test_validate_buffer_sample():
    buffer = ReplayBuffer(10, seed=0)
    buffer.add(make_sample())
    validate_buffer_sample(buffer, 1)
