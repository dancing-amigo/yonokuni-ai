#!/usr/bin/env python3
"""Evaluate a saved model checkpoint against a baseline policy."""

import argparse
import json
import torch

from yonokuni.core import GameResult
from yonokuni.evaluation import RuleBasedPolicy, evaluate_policies
from yonokuni.mcts import MCTSConfig
from yonokuni.models import YonokuniNet, YonokuniNetConfig
from yonokuni.selfplay import RandomPolicy, make_mcts_policy_from_model


def load_model(checkpoint_path: str, device: torch.device) -> YonokuniNet:
    model = YonokuniNet(YonokuniNetConfig())
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.to(device=device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--baseline", choices=["random", "rule"], default="rule")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    mcts_config = MCTSConfig(
        num_simulations=args.mcts_simulations,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_fraction=0.0,
    )
    policy_model = make_mcts_policy_from_model(model, config=mcts_config, device=device)
    if args.baseline == "random":
        baseline_policy = RandomPolicy()
    else:
        baseline_policy = RuleBasedPolicy()

    result = evaluate_policies(policy_model, baseline_policy, episodes=args.episodes)

    output = {
        "games": result.games_played,
        "team_a_wins": result.team_a_wins,
        "team_b_wins": result.team_b_wins,
        "draws": result.draws,
        "average_length": result.average_length,
        "team_a_winrate": result.winrate_team_a(),
        "team_b_winrate": result.winrate_team_b(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
