#!/usr/bin/env python3
"""Run a single self-play/train iteration for smoke testing."""

import argparse
import json

from yonokuni.orchestration import SelfPlayTrainer, SelfPlayTrainerConfig
from yonokuni.training import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=16)
    args = parser.parse_args()

    config = SelfPlayTrainerConfig(
        episodes_per_iteration=args.episodes,
        training_steps_per_iteration=args.train_steps,
        training_config=TrainingConfig(batch_size=32, apply_symmetry=True),
    )

    trainer = SelfPlayTrainer(config)
    result = trainer.iteration()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
