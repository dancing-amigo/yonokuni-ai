from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

BoardArray = NDArray[np.int8]
BoolArray = NDArray[np.bool_]


class PlayerColor(IntEnum):
    RED = 1
    BLUE = 2
    YELLOW = 3
    GREEN = 4

    @property
    def team(self) -> Team:
        return Team.A if self in (PlayerColor.RED, PlayerColor.YELLOW) else Team.B

    def next(self) -> "PlayerColor":
        return PlayerColor(((int(self) - 1 + 1) % 4) + 1)


class Team(Enum):
    A = "A"
    B = "B"


class GameResult(Enum):
    ONGOING = "ongoing"
    TEAM_A_WIN = "team_a_win"
    TEAM_B_WIN = "team_b_win"
    DRAW = "draw"


def colours_for_team(team: Team) -> Tuple[PlayerColor, PlayerColor]:
    if team == Team.A:
        return (PlayerColor.RED, PlayerColor.YELLOW)
    return (PlayerColor.BLUE, PlayerColor.GREEN)


@dataclass(frozen=True)
class Action:
    from_row: int
    from_col: int
    to_row: int
    to_col: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.from_row, self.from_col, self.to_row, self.to_col)


@dataclass(frozen=True)
class ActionRecord:
    action: Action
    captured_positions: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    captured_colors: Tuple[PlayerColor, ...] = field(default_factory=tuple)
    eliminated_players: Tuple[PlayerColor, ...] = field(default_factory=tuple)
    resulted_in: GameResult = GameResult.ONGOING


@dataclass
class GameState:
    board: BoardArray  # shape (8, 8), dtype=np.int8, values 0 (empty) or 1..4 player colour
    dead_mask: BoolArray  # shape (8, 8), dtype=bool
    captured_counts: np.ndarray  # shape (4,), dtype=np.int16
    dead_players: BoolArray  # shape (4,), dtype=bool
    current_player: PlayerColor
    ply_count: int = 0
    max_ply: int = 400
    result: GameResult = GameResult.ONGOING
    last_action: Optional[ActionRecord] = None

    def copy(self) -> "GameState":
        return GameState(
            board=self.board.copy(),
            dead_mask=self.dead_mask.copy(),
            captured_counts=self.captured_counts.copy(),
            dead_players=self.dead_players.copy(),
            current_player=self.current_player,
            ply_count=self.ply_count,
            max_ply=self.max_ply,
            result=self.result,
            last_action=self.last_action,
        )

    def set_last_action(self, record: ActionRecord) -> None:
        self.last_action = record

    @property
    def is_terminal(self) -> bool:
        return self.result != GameResult.ONGOING

    def team_dead(self, team: Team) -> bool:
        colours = colours_for_team(team)
        return bool(self.dead_players[int(colours[0]) - 1] and self.dead_players[int(colours[1]) - 1])

    def alive_players(self) -> Sequence[PlayerColor]:
        return [color for color in PlayerColor if not self.dead_players[int(color) - 1]]

    def occupied_positions(self, color: PlayerColor) -> Iterable[Tuple[int, int]]:
        positions = np.argwhere(self.board == int(color))
        for r, c in positions:
            yield int(r), int(c)

    def has_any_piece(self, color: PlayerColor) -> bool:
        return np.any(self.board == int(color))

    def __repr__(self) -> str:
        board_str = "\n".join(" ".join(str(cell) for cell in row) for row in self.board)
        return (
            f"GameState(current={self.current_player}, result={self.result}, ply={self.ply_count})\n"
            f"{board_str}"
        )


# Convenient tuple aliases used across modules
Position = Tuple[int, int]
