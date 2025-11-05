import json
from pathlib import Path

from yonokuni import YonokuniEnv
from yonokuni.core import Action, encode_action

from scripts.play_vs_ai import replay_logged_game


def create_sample_log(path: Path) -> None:
    env = YonokuniEnv()
    env.reset()
    moves = []
    action1 = encode_action(Action(0, 1, 1, 1))
    env.step(action1)
    moves.append({
        "move_index": 0,
        "actor": "human",
        "team": "A",
        "action_index": action1,
        "from": [0, 1],
        "to": [1, 1],
    })
    action2 = encode_action(Action(1, 7, 1, 6))
    moves.append({
        "move_index": 1,
        "actor": "ai",
        "team": "B",
        "action_index": action2,
        "from": [1, 7],
        "to": [1, 6],
    })
    log = {"metadata": {}, "moves": moves}
    path.write_text(json.dumps(log))


def test_replay_logged_game(tmp_path):
    log_path = tmp_path / "game.json"
    create_sample_log(log_path)
    summary = replay_logged_game(log_path, verbose=False)
    assert summary["moves"] == 2
    board = summary["board"]
    assert board[1][1] == 1
    assert board[1][6] == 2
