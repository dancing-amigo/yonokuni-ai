from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from yonokuni.core import GameState, PlayerColor

BOARD_DIM = 8
BOARD_CHANNELS = 8  # alive/dead channels per colour
AUX_VECTOR_SIZE = 8  # current player one-hot (4) + dead flags (4)


def build_board_tensor(state: GameState) -> np.ndarray:
    """Return board tensor with shape (8, 8, 8) channel-first."""
    tensor = np.zeros((BOARD_CHANNELS, BOARD_DIM, BOARD_DIM), dtype=np.float32)
    for (row, col), value in np.ndenumerate(state.board):
        if value == 0:
            continue
        color = PlayerColor(int(value))
        dead = bool(state.dead_mask[row, col])
        channel_offset = (int(color) - 1) * 2
        channel = channel_offset + (1 if dead else 0)
        tensor[channel, row, col] = 1.0
    return tensor


def build_aux_vector(state: GameState) -> np.ndarray:
    aux = np.zeros((AUX_VECTOR_SIZE,), dtype=np.float32)
    aux[int(state.current_player) - 1] = 1.0
    aux[4:] = state.dead_players.astype(np.float32)
    return aux


def state_to_numpy(state: GameState) -> Tuple[np.ndarray, np.ndarray]:
    return build_board_tensor(state), build_aux_vector(state)


def state_to_torch(
    state: GameState,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    board_np, aux_np = state_to_numpy(state)
    board = torch.from_numpy(board_np).to(device=device, dtype=dtype)
    aux = torch.from_numpy(aux_np).to(device=device, dtype=dtype)
    return board, aux
