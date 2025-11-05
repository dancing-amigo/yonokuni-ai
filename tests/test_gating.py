import numpy as np

from yonokuni.evaluation import EvaluationResult, gate_model


def make_result(winrate: float, games: int) -> EvaluationResult:
    team_a_wins = int(winrate * games)
    team_b_wins = games - team_a_wins
    return EvaluationResult(
        games_played=games,
        team_a_wins=team_a_wins,
        team_b_wins=team_b_wins,
        draws=0,
        average_length=10.0,
    )


def test_gate_model_promote():
    result = make_result(0.6, 50)
    decision = gate_model(result, threshold=0.55, min_games=20)
    assert decision.promote
    assert decision.winrate >= 0.55


def test_gate_model_reject_due_to_games():
    result = make_result(0.8, 10)
    decision = gate_model(result, threshold=0.55, min_games=20)
    assert not decision.promote


def test_gate_model_reject_due_to_winrate():
    result = make_result(0.5, 40)
    decision = gate_model(result, threshold=0.55, min_games=20)
    assert not decision.promote
