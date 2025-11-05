from __future__ import annotations

from dataclasses import dataclass

from .match import EvaluationResult


@dataclass
class GatingDecision:
    promote: bool
    winrate: float
    threshold: float
    min_games: int
    result: EvaluationResult


def gate_model(result: EvaluationResult, *, threshold: float, min_games: int) -> GatingDecision:
    winrate = result.winrate_team_a()
    promote = result.games_played >= min_games and winrate >= threshold
    return GatingDecision(
        promote=promote,
        winrate=winrate,
        threshold=threshold,
        min_games=min_games,
        result=result,
    )
