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
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.to(device=device)
    model.eval()
    return model


def policy_from_identifier(identifier: str, config: MCTSConfig, device: torch.device):
    key = identifier.lower()
    if key == "random":
        return RandomPolicy()
    if key == "rule":
        return RuleBasedPolicy()
    model = load_model(identifier, device)
    return make_mcts_policy_from_model(model, config=config, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate a model against a baseline.")
    parser.add_argument("current" , help="Current model checkpoint path or 'random'/'rule'")
    parser.add_argument("baseline", help="Baseline model checkpoint path or 'random'/'rule'")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-games", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcts_config = MCTSConfig(
        num_simulations=args.mcts_simulations,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_fraction=0.0,
    )

    policy_current = policy_from_identifier(args.current, mcts_config, device)
    policy_baseline = policy_from_identifier(args.baseline, mcts_config, device)

    result = evaluate_policies(policy_current, policy_baseline, episodes=args.episodes)
    decision = gate_model(result, threshold=args.threshold, min_games=args.min_games)

    output = {
        "games": result.games_played,
        "team_a_wins": result.team_a_wins,
        "team_b_wins": result.team_b_wins,
        "draws": result.draws,
        "average_length": result.average_length,
        "winrate_team_a": result.winrate_team_a(),
        "winrate_team_b": result.winrate_team_b(),
        "threshold": decision.threshold,
        "min_games": decision.min_games,
        "promote": decision.promote,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
