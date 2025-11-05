from __future__ import annotations

import numpy as np

from yonokuni.selfplay.replay_buffer import ReplayBuffer


class ReplayDataError(ValueError):
    pass


def validate_batch(boards: np.ndarray, aux: np.ndarray, policies: np.ndarray, values: np.ndarray) -> None:
    if not np.isfinite(boards).all():
        raise ReplayDataError("boards contain non-finite values")
    if not np.isfinite(aux).all():
        raise ReplayDataError("aux contains non-finite values")
    if not np.isfinite(policies).all():
        raise ReplayDataError("policies contain non-finite values")
    if not np.isfinite(values).all():
        raise ReplayDataError("values contain non-finite values")
    if (values < -1.0).any() or (values > 1.0).any():
        raise ReplayDataError("values out of [-1,1] range")
    policy_sums = policies.sum(axis=-1)
    if not np.allclose(policy_sums, 1.0, atol=1e-3):
        raise ReplayDataError("policy vectors must sum to 1")
    if (policies < 0).any():
        raise ReplayDataError("policy contains negative probabilities")

def validate_buffer_sample(buffer: ReplayBuffer, sample_size: int) -> None:
    if len(buffer) == 0 or sample_size <= 0:
        return
    size = min(sample_size, len(buffer))
    boards, aux, policies, values = buffer.sample(size, apply_symmetry=False, replace=False)
    validate_batch(boards, aux, policies, values)
