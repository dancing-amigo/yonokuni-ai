#!/usr/bin/env python3
"""Evaluate current model against a saved legacy model."""

import argparse
import json
import torch

from yonokuni.evaluation import RuleBasedPolicy, evaluate_policies
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


def build_policy_from_checkpoint(path: str, config: MCTSConfig, device: torch.device):
    if path.lower() == "random":
        return RandomPolicy()
    if path.lower() == "rule":
        return RuleBasedPolicy()
    model = load_model(path, device)
    return make_mcts_policy_from_model(model, config=config, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("current", help="Path to current model checkpoint")
    parser.add_argument("baseline", help="Path to baseline model checkpoint or 'random'/'rule'")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcts_config = MCTSConfig(
        num_simulations=args.mcts_simulations,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_fraction=0.0,
    )

    policy_current = build_policy_from_checkpoint(args.current, mcts_config, device)
    policy_baseline = build_policy_from_checkpoint(args.baseline, mcts_config, device)

    result = evaluate_policies(policy_current, policy_baseline, episodes=args.episodes)

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
