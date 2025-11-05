import numpy as np

from yonokuni.core import ACTION_VECTOR_SIZE, PlayerColor, encode_action, enumerate_legal_actions, initialize_game_state
from yonokuni.features import Transform
from yonokuni.features.symmetry import get_spec
from yonokuni.selfplay import ReplayBuffer, ReplaySample


def make_simple_sample() -> ReplaySample:
    state = initialize_game_state()
    board = np.zeros((8, 8, 8), dtype=np.float32)
    aux = np.zeros((8,), dtype=np.float32)
    aux[int(state.current_player) - 1] = 1.0
    policy = np.zeros((ACTION_VECTOR_SIZE,), dtype=np.float32)
    action = enumerate_legal_actions(state)[0]
    policy[encode_action(action)] = 1.0
    board_tensor = np.zeros_like(board)
    board_tensor[0, 0, 1] = 1.0  # Single red piece for traceability
    return ReplaySample(board=board_tensor, aux=aux, policy=policy, value=1.0)


def test_replay_buffer_symmetry_transform_flips_value_when_team_flipped():
    sample = make_simple_sample()
    buffer = ReplayBuffer(10, transforms=[Transform.ROT90], seed=123)
    buffer.add(sample)

    boards, aux, policies, values = buffer.sample(1, apply_symmetry=True)

    # ROT90 swaps teams; value should flip sign.
    assert values.shape == (1,)
    assert values[0] == -1.0

    # Current player should map from Red -> Blue.
    current_player_index = int(np.argmax(aux[0, :4]))
    assert current_player_index == int(get_spec(Transform.ROT90).color_map[PlayerColor.RED]) - 1

    # Policy probability mass should remain 1.
    assert np.isclose(policies.sum(), 1.0)


def test_replay_buffer_sample_no_symmetry_returns_original():
    sample = make_simple_sample()
    buffer = ReplayBuffer(10, transforms=[Transform.IDENTITY], seed=0)
    buffer.add(sample)

    boards, aux, policies, values = buffer.sample(1, apply_symmetry=False)
    np.testing.assert_array_equal(boards[0], sample.board)
    np.testing.assert_array_equal(aux[0], sample.aux)
    np.testing.assert_array_equal(policies[0], sample.policy)
    assert values[0] == sample.value


def test_replay_buffer_state_roundtrip(tmp_path):
    sample = make_simple_sample()
    buffer = ReplayBuffer(10, transforms=[Transform.IDENTITY], seed=0)
    buffer.add(sample)
    state = buffer.to_state()

    new_buffer = ReplayBuffer(10, transforms=[Transform.IDENTITY], seed=1)
    new_buffer.load_state(state)
    assert len(new_buffer) == len(buffer)

    save_path = tmp_path / 'buffer.pkl'
    buffer.save(save_path.as_posix())
    restored = ReplayBuffer.load(save_path.as_posix(), capacity=10, transforms=[Transform.IDENTITY], seed=2)
    assert len(restored) == len(buffer)
