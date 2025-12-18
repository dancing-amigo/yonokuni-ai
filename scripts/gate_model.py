#!/usr/bin/env python3
"""Run gating evaluation between current and baseline models."""

import argparse
import json
import torch

from yonokuni.evaluation import RuleBasedPolicy, evaluate_policies, gate_model
from yonokuni.mcts import MCTSConfig
from yonokuni.models import YonokuniNet, YonokuniNetConfig
from yonokuni.selfplay import RandomPolicy, make_mcts_policy_from_model


def load_model(checkpoint_path: str, device: torch.device) -> YonokuniNet:
    model = YonokuniNet(YonokuniNetConfig())
    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.to(device=device)
    model.eval()
    return model


def policy_from_identifier(
    identifier: str,
    config: MCTSConfig,
    device: torch.device,
):
    key = identifier.lower()
    if key == "random":
        return RandomPolicy()
    if key == "rule":
        return RuleBasedPolicy()
    model = load_model(identifier, device)
    return make_mcts_policy_from_model(model, config=config, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate a model against a baseline."
    )
    parser.add_argument(
        "current",
        help="Current model checkpoint path or 'random'/'rule'",
    )
    parser.add_argument(
        "baseline",
        help="Baseline model checkpoint path or 'random'/'rule'",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-games", type=int, default=40)
    parser.add_argument(
        "--both-sides",
        action="store_true",
        help=(
            "Also run a second match with sides swapped "
            "and aggregate results."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcts_config = MCTSConfig(
        num_simulations=args.mcts_simulations,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_fraction=0.0,
    )

    policy_current = policy_from_identifier(
        args.current,
        mcts_config,
        device,
    )
    policy_baseline = policy_from_identifier(
        args.baseline,
        mcts_config,
        device,
    )

    if not args.both_sides:
        result = evaluate_policies(
            policy_current,
            policy_baseline,
            episodes=args.episodes,
        )
        decision = gate_model(
            result,
            threshold=args.threshold,
            min_games=args.min_games,
        )
        current_wins = result.team_a_wins
        baseline_wins = result.team_b_wins
        draws = result.draws
        games = result.games_played
        avg_len = result.average_length
    else:
        # Match 1: current is Team A, baseline is Team B
        r1 = evaluate_policies(
            policy_current,
            policy_baseline,
            episodes=args.episodes,
        )
        # Match 2: swap sides (baseline is Team A, current is Team B)
        r2 = evaluate_policies(
            policy_baseline,
            policy_current,
            episodes=args.episodes,
        )
        current_wins = r1.team_a_wins + r2.team_b_wins
        baseline_wins = r1.team_b_wins + r2.team_a_wins
        draws = r1.draws + r2.draws
        games = r1.games_played + r2.games_played
        avg_len = (
            (r1.average_length * r1.games_played)
            + (r2.average_length * r2.games_played)
        ) / max(1, games)
        # For gating, treat "current" as Team A in an aggregated result.
        result = type(r1)(
            games_played=games,
            team_a_wins=current_wins,
            team_b_wins=baseline_wins,
            draws=draws,
            average_length=avg_len,
        )
        decision = gate_model(
            result,
            threshold=args.threshold,
            min_games=args.min_games,
        )

    output = {
        "games": games,
        "current_wins": current_wins,
        "baseline_wins": baseline_wins,
        "draws": draws,
        "average_length": avg_len,
        "current_winrate": current_wins / max(1, games),
        "threshold": decision.threshold,
        "min_games": decision.min_games,
        "promote": decision.promote,
        "both_sides": args.both_sides,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
