from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from yonokuni.core import ACTION_VECTOR_SIZE
from yonokuni.features import (
    Transform,
    all_transforms,
    apply_policy_transform,
    team_flipped,
    transform_aux_vector,
    transform_board_tensor,
)


@dataclass
class ReplaySample:
    board: np.ndarray  # (8, 8, 8) channel-first
    aux: np.ndarray  # (8,)
    policy: np.ndarray  # (1792,)
    value: float

    def copy(self) -> "ReplaySample":
        return ReplaySample(
            board=self.board.copy(),
            aux=self.aux.copy(),
            policy=self.policy.copy(),
            value=float(self.value),
        )


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        transforms: Optional[Sequence[Transform]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = capacity
        self._buffer: Deque[ReplaySample] = deque(maxlen=capacity)
        self.transforms = list(transforms) if transforms is not None else list(all_transforms())
        if Transform.IDENTITY not in self.transforms:
            self.transforms.append(Transform.IDENTITY)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def add(self, sample: ReplaySample) -> None:
        self._validate_sample(sample)
        self._buffer.append(sample.copy())

    def extend(self, samples: Iterable[ReplaySample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample(
        self,
        batch_size: int,
        *,
        apply_symmetry: bool = True,
        replace: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._buffer:
            raise ValueError("Cannot sample from an empty ReplayBuffer.")

        indices = self._sample_indices(batch_size, replace=replace)

        boards: List[np.ndarray] = []
        aux_list: List[np.ndarray] = []
        policies: List[np.ndarray] = []
        values: List[float] = []

        for idx in indices:
            sample = self._buffer[idx]
            if apply_symmetry:
                transform = self.rng.choice(self.transforms)
                board = transform_board_tensor(sample.board, transform)
                aux = transform_aux_vector(sample.aux, transform)
                policy = apply_policy_transform(sample.policy, transform)
                value = -sample.value if team_flipped(transform) else sample.value
            else:
                board = sample.board
                aux = sample.aux
                policy = sample.policy
                value = sample.value

            boards.append(board.astype(np.float32, copy=False))
            aux_list.append(aux.astype(np.float32, copy=False))
            policies.append(policy.astype(np.float32, copy=False))
            values.append(np.float32(value))

        return (
            np.stack(boards, axis=0),
            np.stack(aux_list, axis=0),
            np.stack(policies, axis=0),
            np.asarray(values, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    def to_state(self) -> dict:
        transforms = [
            t.name if isinstance(t, Transform) else t for t in self.transforms
        ]
        return {
            "capacity": self.capacity,
            "transforms": transforms,
            "rng_state": self.rng.bit_generator.state,
            "samples": [sample.copy() for sample in self._buffer],
        }

    def load_state(self, state: dict) -> None:
        transforms = state.get("transforms")
        if transforms:
            self.transforms = [
                Transform[name] if isinstance(name, str) else name for name in transforms
            ]
            if Transform.IDENTITY not in self.transforms:
                self.transforms.append(Transform.IDENTITY)

        samples = state.get("samples", [])
        self._buffer = deque(maxlen=self.capacity)
        for sample in samples:
            if isinstance(sample, ReplaySample):
                self._buffer.append(sample.copy())
            else:
                raise ValueError("Replay buffer state contains invalid sample type.")

        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = rng_state

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self.to_state(), fh)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        capacity: Optional[int] = None,
        transforms: Optional[Sequence[Transform]] = None,
        seed: Optional[int] = None,
    ) -> "ReplayBuffer":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        buffer_capacity = capacity or state.get("capacity", 0)
        buffer = cls(
            capacity=buffer_capacity,
            transforms=transforms,
            seed=seed,
        )
        buffer.load_state(state)
        return buffer

    # ------------------------------------------------------------------
    def _sample_indices(self, batch_size: int, *, replace: bool) -> np.ndarray:
        buffer_len = len(self._buffer)
        if not replace and batch_size > buffer_len:
            raise ValueError("Cannot sample without replacement when batch_size > buffer size.")
        if replace:
            return self.rng.integers(0, buffer_len, size=batch_size)
        return self.rng.choice(buffer_len, size=batch_size, replace=False)

    @staticmethod
    def _validate_sample(sample: ReplaySample) -> None:
        if sample.board.shape != (8, 8, 8):
            raise ValueError(f"board tensor must be (8, 8, 8), got {sample.board.shape}")
        if sample.aux.shape != (8,):
            raise ValueError("aux vector must be (8,)")
        if sample.policy.shape != (ACTION_VECTOR_SIZE,):
            raise ValueError("policy vector has incorrect shape.")
