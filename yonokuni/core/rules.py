from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .state import Action, ActionRecord, GameResult, GameState, PlayerColor, Position, Team

BOARD_SIZE = 8
STARTING_PIECES = 6
DEAD_THRESHOLD = 3
MAX_DISTANCE = BOARD_SIZE - 1
DIRECTIONS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_VECTOR_SIZE = BOARD_SIZE * BOARD_SIZE * len(DIRECTIONS) * MAX_DISTANCE


@dataclass(frozen=True)
class ActionVector:
    origin: Tuple[int, int]
    direction_index: int
    distance: int

    def to_action(self) -> Action:
        dr, dc = DIRECTIONS[self.direction_index]
        to_row = self.origin[0] + dr * self.distance
        to_col = self.origin[1] + dc * self.distance
        return Action(self.origin[0], self.origin[1], to_row, to_col)

    @staticmethod
    def from_action(action: Action) -> "ActionVector":
        dr = action.to_row - action.from_row
        dc = action.to_col - action.from_col
        if dr != 0 and dc != 0:
            raise ValueError("Action is not orthogonal.")
        if dr == 0 and dc == 0:
            raise ValueError("Action has zero movement.")
        if dr != 0:
            direction = (1, 0) if dr > 0 else (-1, 0)
            distance = abs(dr)
            direction_index = DIRECTIONS.index(direction)
        else:
            direction = (0, 1) if dc > 0 else (0, -1)
            distance = abs(dc)
            direction_index = DIRECTIONS.index(direction)
        if distance < 1 or distance > MAX_DISTANCE:
            raise ValueError("Action distance out of range.")
        return ActionVector((action.from_row, action.from_col), direction_index, distance)

    def to_index(self) -> int:
        base = self.origin[0] * BOARD_SIZE + self.origin[1]
        base = base * len(DIRECTIONS) + self.direction_index
        return base * MAX_DISTANCE + (self.distance - 1)

    @staticmethod
    def from_index(index: int) -> "ActionVector":
        if not 0 <= index < ACTION_VECTOR_SIZE:
            raise ValueError("Action index out of range.")
        distance = (index % MAX_DISTANCE) + 1
        index //= MAX_DISTANCE
        direction_index = index % len(DIRECTIONS)
        index //= len(DIRECTIONS)
        origin_row = index // BOARD_SIZE
        origin_col = index % BOARD_SIZE
        return ActionVector((origin_row, origin_col), direction_index, distance)


def encode_action(action: Action) -> int:
    return ActionVector.from_action(action).to_index()


def decode_action(index: int) -> Action:
    return ActionVector.from_index(index).to_action()


def initialize_game_state(max_ply: int = 400) -> GameState:
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    dead_mask = np.zeros_like(board, dtype=bool)
    captured_counts = np.zeros(4, dtype=np.int16)
    dead_players = np.zeros(4, dtype=bool)

    # Place pieces along the edges excluding corners.
    for col in range(1, BOARD_SIZE - 1):
        board[0, col] = PlayerColor.RED
        board[BOARD_SIZE - 1, col] = PlayerColor.YELLOW
    for row in range(1, BOARD_SIZE - 1):
        board[row, BOARD_SIZE - 1] = PlayerColor.BLUE
        board[row, 0] = PlayerColor.GREEN

    return GameState(
        board=board,
        dead_mask=dead_mask,
        captured_counts=captured_counts,
        dead_players=dead_players,
        current_player=PlayerColor.RED,
        ply_count=0,
        max_ply=max_ply,
        result=GameResult.ONGOING,
    )


def enumerate_legal_actions(state: GameState, player: Optional[PlayerColor] = None) -> List[Action]:
    if player is None:
        player = state.current_player

    legal: List[Action] = []
    for row, col in state.occupied_positions(player):
        for direction_index, (dr, dc) in enumerate(DIRECTIONS):
            next_row, next_col = row + dr, col + dc
            distance = 1
            while _in_bounds(next_row, next_col) and state.board[next_row, next_col] == 0:
                legal.append(Action(row, col, next_row, next_col))
                distance += 1
                next_row += dr
                next_col += dc
    return legal


def apply_action(state: GameState, action: Action, *, in_place: bool = False) -> GameState:
    source_state = state if in_place else state.copy()
    if source_state.is_terminal:
        raise ValueError("Cannot apply action to a terminal state.")

    mover_color_value = source_state.board[action.from_row, action.from_col]
    if mover_color_value == 0:
        raise ValueError("No piece at the action origin.")

    mover_color = PlayerColor(int(mover_color_value))
    if mover_color != source_state.current_player:
        raise ValueError("Action does not belong to the current player.")

    if source_state.board[action.to_row, action.to_col] != 0:
        raise ValueError("Destination cell must be empty.")

    if not _validate_path_clear(source_state.board, action):
        raise ValueError("Action path is obstructed.")

    mover_dead = bool(source_state.dead_players[int(mover_color) - 1])

    # Execute movement.
    source_state.board[action.to_row, action.to_col] = source_state.board[action.from_row, action.from_col]
    source_state.dead_mask[action.to_row, action.to_col] = source_state.dead_mask[action.from_row, action.from_col]
    source_state.board[action.from_row, action.from_col] = 0
    source_state.dead_mask[action.from_row, action.from_col] = False

    captured_positions: List[Position] = []
    captured_colors: List[PlayerColor] = []

    if not mover_dead:
        captured_positions = _collect_captures(source_state.board, action.to_row, action.to_col, mover_color)
        captured_positions = sorted({pos for pos in captured_positions})

    if captured_positions:
        for row, col in captured_positions:
            captured_color_value = source_state.board[row, col]
            if captured_color_value == 0:
                continue
            captured_color = PlayerColor(int(captured_color_value))
            captured_colors.append(captured_color)
            source_state.board[row, col] = 0
            source_state.dead_mask[row, col] = False
            source_state.captured_counts[int(captured_color) - 1] += 1

    newly_dead: List[PlayerColor] = []
    for color in PlayerColor:
        idx = int(color) - 1
        if not source_state.dead_players[idx] and source_state.captured_counts[idx] >= DEAD_THRESHOLD:
            source_state.dead_players[idx] = True
            newly_dead.append(color)
    if newly_dead:
        _synchronise_dead_mask(source_state)

    # Check victory conditions after captures and death propagation.
    result = _evaluate_terminal(source_state)
    source_state.ply_count += 1

    if result == GameResult.ONGOING and source_state.ply_count >= source_state.max_ply:
        result = GameResult.DRAW

    source_state.result = result

    record = ActionRecord(
        action=action,
        captured_positions=tuple(captured_positions),
        captured_colors=tuple(captured_colors),
        eliminated_players=tuple(newly_dead),
        resulted_in=result,
    )
    source_state.set_last_action(record)

    if source_state.result == GameResult.ONGOING:
        next_player = _select_next_player(source_state, mover_color)
        if next_player is None:
            source_state.result = GameResult.DRAW
        else:
            source_state.current_player = next_player
    return source_state


def _select_next_player(state: GameState, mover_color: PlayerColor) -> Optional[PlayerColor]:
    candidate = mover_color
    for _ in range(len(PlayerColor)):
        candidate = candidate.next()
        if enumerate_legal_actions(state, candidate):
            return candidate
    return None


def _collect_captures(board: np.ndarray, row: int, col: int, color: PlayerColor) -> List[Position]:
    captures: List[Position] = []
    captures.extend(_sandwich_captures(board, row, col, color))
    captures.extend(_surrounded_captures(board, row, col, color))
    return captures


def _sandwich_captures(board: np.ndarray, row: int, col: int, color: PlayerColor) -> List[Position]:
    captures: List[Position] = []
    for dr, dc in DIRECTIONS:
        positions: List[Position] = []
        r, c = row + dr, col + dc
        while _in_bounds(r, c) and board[r, c] != 0:
            occupant = PlayerColor(int(board[r, c]))
            if occupant == color:
                captures.extend(positions)
                break
            if occupant.team == color.team and occupant != color:
                break  # friendly different colour blocks the capture
            positions.append((r, c))
            r += dr
            c += dc
        # hitting empty or boundary without own colour terminates with no capture
    return captures


def _surrounded_captures(board: np.ndarray, row: int, col: int, color: PlayerColor) -> List[Position]:
    captures: List[Position] = []
    visited: set[Position] = set()
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        if not _in_bounds(r, c):
            continue
        occupant_val = board[r, c]
        if occupant_val == 0:
            continue
        occupant_color = PlayerColor(int(occupant_val))
        if occupant_color == color:
            continue
        if (r, c) in visited:
            continue
        group = _bfs_group(board, (r, c), forbid_color=color)
        visited.update(group)
        if not group:
            continue
        if all(not _can_piece_move(board, pos) for pos in group):
            captures.extend(group)
    return captures


def _bfs_group(board: np.ndarray, start: Position, forbid_color: PlayerColor) -> List[Position]:
    stack = [start]
    group: List[Position] = []
    seen: set[Position] = set()
    while stack:
        position = stack.pop()
        if position in seen:
            continue
        seen.add(position)
        r, c = position
        if not _in_bounds(r, c):
            continue
        occupant_val = board[r, c]
        if occupant_val == 0:
            continue
        occupant_color = PlayerColor(int(occupant_val))
        if occupant_color == forbid_color:
            continue
        group.append(position)
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if _in_bounds(nr, nc) and (nr, nc) not in seen:
                if board[nr, nc] != 0:
                    stack.append((nr, nc))
    return group


def _can_piece_move(board: np.ndarray, position: Position) -> bool:
    r, c = position
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        while _in_bounds(nr, nc):
            if board[nr, nc] == 0:
                return True
            # Hit a piece; this direction is blocked.
            break
    return False


def _synchronise_dead_mask(state: GameState) -> None:
    for color in PlayerColor:
        if state.dead_players[int(color) - 1]:
            state.dead_mask[state.board == int(color)] = True


def _evaluate_terminal(state: GameState) -> GameResult:
    centre_result = _check_centre_control(state.board)
    if centre_result is not None:
        return GameResult.TEAM_A_WIN if centre_result == Team.A else GameResult.TEAM_B_WIN

    if state.team_dead(Team.A):
        return GameResult.TEAM_B_WIN
    if state.team_dead(Team.B):
        return GameResult.TEAM_A_WIN
    return GameResult.ONGOING


def _check_centre_control(board: np.ndarray) -> Optional[Team]:
    centres = [(3, 3), (3, 4), (4, 3), (4, 4)]
    occupants: List[PlayerColor] = []
    for r, c in centres:
        val = board[r, c]
        if val == 0:
            return None
        occupants.append(PlayerColor(int(val)))
    team = occupants[0].team
    if all(piece.team == team for piece in occupants):
        return team
    return None


def _validate_path_clear(board: np.ndarray, action: Action) -> bool:
    vertical_delta = action.to_row - action.from_row
    horizontal_delta = action.to_col - action.from_col
    dr = 0 if vertical_delta == 0 else (1 if vertical_delta > 0 else -1)
    dc = 0 if horizontal_delta == 0 else (1 if horizontal_delta > 0 else -1)
    if dr != 0 and dc != 0:
        return False
    if dr == 0 and dc == 0:
        return False
    r, c = action.from_row + dr, action.from_col + dc
    while (r, c) != (action.to_row, action.to_col):
        if board[r, c] != 0:
            return False
        r += dr
        c += dc
    return True


def _in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE
