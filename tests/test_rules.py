import pytest

from yonokuni.core import (
    Action,
    GameState,
    GameResult,
    PlayerColor,
    apply_action,
    initialize_game_state,
)


def fresh_state() -> GameState:
    state = initialize_game_state()
    state.board[:, :] = 0
    state.dead_mask[:, :] = False
    state.captured_counts[:] = 0
    state.dead_players[:] = False
    state.current_player = PlayerColor.RED
    return state


def test_invalid_move_path_blocked() -> None:
    state = fresh_state()
    state.board[3, 3] = PlayerColor.RED
    state.board[3, 4] = PlayerColor.BLUE

    action = Action(3, 3, 3, 5)

    with pytest.raises(ValueError):
        apply_action(state, action)


def test_sandwich_capture_succeeds() -> None:
    state = fresh_state()
    state.board[3, 1] = PlayerColor.RED
    state.board[3, 3] = PlayerColor.BLUE
    state.board[3, 4] = PlayerColor.RED

    action = Action(3, 1, 3, 2)
    next_state = apply_action(state, action)

    assert next_state.board[3, 3] == 0
    assert next_state.captured_counts[int(PlayerColor.BLUE) - 1] == 1
    assert next_state.last_action is not None
    assert (3, 3) in next_state.last_action.captured_positions


def test_sandwich_capture_blocked_by_allied_colour() -> None:
    state = fresh_state()
    state.board[3, 1] = PlayerColor.RED
    state.board[3, 3] = PlayerColor.BLUE
    state.board[3, 4] = PlayerColor.YELLOW  # same team different color blocks capture

    action = Action(3, 1, 3, 2)
    next_state = apply_action(state, action)

    assert next_state.board[3, 3] == PlayerColor.BLUE
    assert next_state.captured_counts[int(PlayerColor.BLUE) - 1] == 0
    assert not next_state.last_action.captured_positions


def test_surrounded_capture_corner_group() -> None:
    state = fresh_state()
    state.board[1, 0] = PlayerColor.RED
    state.board[1, 1] = PlayerColor.RED
    state.board[0, 0] = PlayerColor.BLUE
    state.board[0, 1] = PlayerColor.GREEN
    state.board[2, 2] = PlayerColor.RED

    action = Action(2, 2, 0, 2)
    next_state = apply_action(state, action)

    assert next_state.board[0, 0] == 0
    assert next_state.board[0, 1] == 0
    assert next_state.captured_counts[int(PlayerColor.BLUE) - 1] == 1
    assert next_state.captured_counts[int(PlayerColor.GREEN) - 1] == 1
    assert set(next_state.last_action.captured_positions) == {(0, 0), (0, 1)}


def test_dead_piece_cannot_capture() -> None:
    state = fresh_state()
    state.board[3, 1] = PlayerColor.RED
    state.board[3, 3] = PlayerColor.BLUE
    state.board[3, 4] = PlayerColor.RED
    state.dead_players[int(PlayerColor.RED) - 1] = True
    state.dead_mask[3, 1] = True

    action = Action(3, 1, 3, 2)
    next_state = apply_action(state, action)

    assert next_state.board[3, 3] == PlayerColor.BLUE
    assert not next_state.last_action.captured_positions


def test_centre_control_triggers_win() -> None:
    state = fresh_state()
    # Place Team A pieces in the centre and ensure current player moves elsewhere.
    centres = [(3, 3), (3, 4), (4, 3), (4, 4)]
    for r, c in centres:
        state.board[r, c] = PlayerColor.RED if (r + c) % 2 == 0 else PlayerColor.YELLOW

    state.board[1, 1] = PlayerColor.RED
    action_legit = Action(1, 1, 1, 2)
    next_state = apply_action(state, action_legit)

    assert next_state.result == GameResult.TEAM_A_WIN


def test_turn_skips_player_without_legal_moves() -> None:
    state = fresh_state()
    state.board[2, 1] = PlayerColor.RED
    state.board[0, 1] = PlayerColor.BLUE
    state.board[0, 0] = PlayerColor.GREEN
    state.board[0, 2] = PlayerColor.YELLOW
    state.board[1, 1] = PlayerColor.GREEN

    action = Action(2, 1, 2, 2)
    next_state = apply_action(state, action)

    assert next_state.current_player == PlayerColor.YELLOW


def test_dead_threshold_marks_player_dead() -> None:
    state = fresh_state()
    state.board[3, 1] = PlayerColor.RED
    state.board[3, 4] = PlayerColor.RED
    state.board[3, 3] = PlayerColor.BLUE
    state.board[4, 4] = PlayerColor.BLUE
    state.captured_counts[int(PlayerColor.BLUE) - 1] = 2

    action = Action(3, 1, 3, 2)
    next_state = apply_action(state, action)

    blue_idx = int(PlayerColor.BLUE) - 1
    assert next_state.dead_players[blue_idx]
    assert next_state.dead_mask[4, 4]


def test_max_ply_results_in_draw() -> None:
    state = fresh_state()
    state.max_ply = 1
    state.board[3, 1] = PlayerColor.RED

    action = Action(3, 1, 3, 2)
    next_state = apply_action(state, action)

    assert next_state.result == GameResult.DRAW
