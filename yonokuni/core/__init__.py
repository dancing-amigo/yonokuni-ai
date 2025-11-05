"""Core game logic for Yonokuni AI."""

from .state import GameResult, GameState, PlayerColor
from .rules import (
    Action,
    ActionVector,
    BOARD_SIZE,
    DEAD_THRESHOLD,
    ACTION_VECTOR_SIZE,
    STARTING_PIECES,
    apply_action,
    enumerate_legal_actions,
    encode_action,
    decode_action,
    initialize_game_state,
)

__all__ = [
    "GameState",
    "GameResult",
    "PlayerColor",
    "Action",
    "ActionVector",
    "ACTION_VECTOR_SIZE",
    "BOARD_SIZE",
    "DEAD_THRESHOLD",
    "STARTING_PIECES",
    "apply_action",
    "enumerate_legal_actions",
    "encode_action",
    "decode_action",
    "initialize_game_state",
]
