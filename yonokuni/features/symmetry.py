from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Dict, Iterable, Tuple

import numpy as np

from yonokuni.core import ACTION_VECTOR_SIZE, Action, PlayerColor, decode_action, encode_action

BOARD_DIM = 8


class Transform(Enum):
    IDENTITY = auto()
    ROT90 = auto()
    ROT180 = auto()
    ROT270 = auto()
    FLIP_H = auto()
    FLIP_V = auto()
    FLIP_MAIN_DIAG = auto()
    FLIP_ANTI_DIAG = auto()


@dataclass(frozen=True)
class SymmetrySpec:
    name: str
    transform: Transform
    position_fn: staticmethod
    color_map: Dict[PlayerColor, PlayerColor]
    team_flipped: bool


def _identity(r: int, c: int) -> Tuple[int, int]:
    return r, c


def _rot90(r: int, c: int) -> Tuple[int, int]:
    return c, BOARD_DIM - 1 - r


def _rot180(r: int, c: int) -> Tuple[int, int]:
    return BOARD_DIM - 1 - r, BOARD_DIM - 1 - c


def _rot270(r: int, c: int) -> Tuple[int, int]:
    return BOARD_DIM - 1 - c, r


def _flip_h(r: int, c: int) -> Tuple[int, int]:
    return r, BOARD_DIM - 1 - c


def _flip_v(r: int, c: int) -> Tuple[int, int]:
    return BOARD_DIM - 1 - r, c


def _flip_main_diag(r: int, c: int) -> Tuple[int, int]:
    return c, r


def _flip_anti_diag(r: int, c: int) -> Tuple[int, int]:
    return BOARD_DIM - 1 - c, BOARD_DIM - 1 - r


def _color_map_from_cycle(
    mapping: Dict[PlayerColor, PlayerColor], colors: Iterable[PlayerColor]
) -> Dict[PlayerColor, PlayerColor]:
    return {color: mapping[color] for color in colors}


_COLOURS = list(PlayerColor)

_SPECS: Dict[Transform, SymmetrySpec] = {
    Transform.IDENTITY: SymmetrySpec(
        "identity",
        Transform.IDENTITY,
        staticmethod(_identity),
        {
            PlayerColor.RED: PlayerColor.RED,
            PlayerColor.BLUE: PlayerColor.BLUE,
            PlayerColor.YELLOW: PlayerColor.YELLOW,
            PlayerColor.GREEN: PlayerColor.GREEN,
        },
        team_flipped=False,
    ),
    Transform.ROT90: SymmetrySpec(
        "rot90",
        Transform.ROT90,
        staticmethod(_rot90),
        {
            PlayerColor.RED: PlayerColor.BLUE,
            PlayerColor.BLUE: PlayerColor.YELLOW,
            PlayerColor.YELLOW: PlayerColor.GREEN,
            PlayerColor.GREEN: PlayerColor.RED,
        },
        team_flipped=True,
    ),
    Transform.ROT180: SymmetrySpec(
        "rot180",
        Transform.ROT180,
        staticmethod(_rot180),
        {
            PlayerColor.RED: PlayerColor.YELLOW,
            PlayerColor.BLUE: PlayerColor.GREEN,
            PlayerColor.YELLOW: PlayerColor.RED,
            PlayerColor.GREEN: PlayerColor.BLUE,
        },
        team_flipped=False,
    ),
    Transform.ROT270: SymmetrySpec(
        "rot270",
        Transform.ROT270,
        staticmethod(_rot270),
        {
            PlayerColor.RED: PlayerColor.GREEN,
            PlayerColor.BLUE: PlayerColor.RED,
            PlayerColor.YELLOW: PlayerColor.BLUE,
            PlayerColor.GREEN: PlayerColor.YELLOW,
        },
        team_flipped=True,
    ),
    Transform.FLIP_H: SymmetrySpec(
        "flip_h",
        Transform.FLIP_H,
        staticmethod(_flip_h),
        {
            PlayerColor.RED: PlayerColor.RED,
            PlayerColor.BLUE: PlayerColor.GREEN,
            PlayerColor.YELLOW: PlayerColor.YELLOW,
            PlayerColor.GREEN: PlayerColor.BLUE,
        },
        team_flipped=False,
    ),
    Transform.FLIP_V: SymmetrySpec(
        "flip_v",
        Transform.FLIP_V,
        staticmethod(_flip_v),
        {
            PlayerColor.RED: PlayerColor.YELLOW,
            PlayerColor.BLUE: PlayerColor.BLUE,
            PlayerColor.YELLOW: PlayerColor.RED,
            PlayerColor.GREEN: PlayerColor.GREEN,
        },
        team_flipped=False,
    ),
    Transform.FLIP_MAIN_DIAG: SymmetrySpec(
        "flip_main_diag",
        Transform.FLIP_MAIN_DIAG,
        staticmethod(_flip_main_diag),
        {
            PlayerColor.RED: PlayerColor.GREEN,
            PlayerColor.BLUE: PlayerColor.YELLOW,
            PlayerColor.YELLOW: PlayerColor.BLUE,
            PlayerColor.GREEN: PlayerColor.RED,
        },
        team_flipped=True,
    ),
    Transform.FLIP_ANTI_DIAG: SymmetrySpec(
        "flip_anti_diag",
        Transform.FLIP_ANTI_DIAG,
        staticmethod(_flip_anti_diag),
        {
            PlayerColor.RED: PlayerColor.BLUE,
            PlayerColor.BLUE: PlayerColor.RED,
            PlayerColor.YELLOW: PlayerColor.GREEN,
            PlayerColor.GREEN: PlayerColor.YELLOW,
        },
        team_flipped=True,
    ),
}


def get_spec(transform: Transform) -> SymmetrySpec:
    return _SPECS[transform]


def all_transforms() -> Iterable[Transform]:
    return list(_SPECS.keys())


def transform_position(transform: Transform, row: int, col: int) -> Tuple[int, int]:
    spec = get_spec(transform)
    return spec.position_fn(row, col)


def transform_colour(transform: Transform, colour: PlayerColor) -> PlayerColor:
    return get_spec(transform).color_map[colour]


@lru_cache(maxsize=None)
def _policy_permutation_cached(transform: Transform) -> np.ndarray:
    perm = np.zeros(ACTION_VECTOR_SIZE, dtype=np.int32)
    for idx in range(ACTION_VECTOR_SIZE):
        action = decode_action(idx)
        transformed = transform_action(transform, action)
        new_idx = encode_action(transformed)
        perm[new_idx] = idx
    return perm


def policy_permutation(transform: Transform) -> np.ndarray:
    """Return permutation array P such that new_policy = old_policy[P]."""
    return _policy_permutation_cached(transform)


def transform_action(transform: Transform, action: Action) -> Action:
    fr, fc = transform_position(transform, action.from_row, action.from_col)
    tr, tc = transform_position(transform, action.to_row, action.to_col)
    return Action(fr, fc, tr, tc)


def apply_policy_transform(policy: np.ndarray, transform: Transform) -> np.ndarray:
    perm = policy_permutation(transform)
    return policy[perm]


def team_flipped(transform: Transform) -> bool:
    return get_spec(transform).team_flipped


def transform_aux_vector(aux: np.ndarray, transform: Transform) -> np.ndarray:
    spec = get_spec(transform)
    result = np.zeros_like(aux)
    for colour in PlayerColor:
        new_colour = spec.color_map[colour]
        old_index = int(colour) - 1
        new_index = int(new_colour) - 1
        result[new_index] = aux[old_index]
        result[new_index + 4] = aux[old_index + 4]
    return result


def transform_board_tensor(board: np.ndarray, transform: Transform) -> np.ndarray:
    result = np.zeros_like(board)
    for colour in PlayerColor:
        alive_channel = (int(colour) - 1) * 2
        dead_channel = alive_channel + 1
        alive_positions = np.argwhere(board[alive_channel] > 0.5)
        dead_positions = np.argwhere(board[dead_channel] > 0.5)

        mapped_colour = transform_colour(transform, colour)
        mapped_alive_channel = (int(mapped_colour) - 1) * 2
        mapped_dead_channel = mapped_alive_channel + 1

        for row, col in alive_positions:
            nr, nc = transform_position(transform, int(row), int(col))
            result[mapped_alive_channel, nr, nc] = 1.0

        for row, col in dead_positions:
            nr, nc = transform_position(transform, int(row), int(col))
            result[mapped_dead_channel, nr, nc] = 1.0
    return result
