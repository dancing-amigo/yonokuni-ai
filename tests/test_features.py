import numpy as np

from yonokuni.core import (
    ACTION_VECTOR_SIZE,
    PlayerColor,
    encode_action,
    enumerate_legal_actions,
    initialize_game_state,
)
from yonokuni.features import (
    Transform,
    apply_policy_transform,
    build_board_tensor,
    state_to_numpy,
    transform_action,
    transform_board_tensor,
)


def test_state_to_numpy_initial_board_counts():
    state = initialize_game_state()
    board, aux = state_to_numpy(state)

    assert board.shape == (8, 8, 8)
    assert aux.shape == (8,)
    # 24 initial pieces
    assert board.sum() == 24
    # Top edge should be red pieces (channel 0)
    assert board[0, 0, 1] == 1.0


def test_transform_board_tensor_rot90_relabels_colour():
    state = initialize_game_state()
    state.board[:, :] = 0
    state.board[0, 1] = PlayerColor.RED
    board = build_board_tensor(state)

    transformed = transform_board_tensor(board, Transform.ROT90)
    # Red -> Blue, piece moves from (0,1) -> (1,7)
    blue_alive_channel = (int(PlayerColor.BLUE) - 1) * 2
    assert transformed[blue_alive_channel, 1, 7] == 1.0


def test_policy_transform_rot90_matches_transformed_action():
    state = initialize_game_state()
    action = enumerate_legal_actions(state)[0]
    index = encode_action(action)
    policy = np.zeros((ACTION_VECTOR_SIZE,), dtype=np.float32)
    policy[index] = 1.0

    transformed_policy = apply_policy_transform(policy, Transform.ROT90)
    transformed_index = transformed_policy.argmax()

    transformed_action = transform_action(Transform.ROT90, action)
    assert transformed_index == encode_action(transformed_action)
