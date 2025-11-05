#!/usr/bin/env python3
"""Run a single self-play/train iteration for smoke testing."""

import argparse
import json
from pathlib import Path

import yaml

try:
    from tqdm.auto import trange
except ImportError:
    trange = range

from yonokuni.mcts import MCTSConfig
from yonokuni.orchestration import SelfPlayTrainer, SelfPlayTrainerConfig
from yonokuni.training import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/self_play.yaml")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--log-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--mcts-simulations", type=int)
    parser.add_argument("--dirichlet-alpha", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--self-play-workers", type=int)
    parser.add_argument("--validation-sample-size", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()

    cfg = {}
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
    episodes = args.episodes if args.episodes is not None else cfg.get("episodes_per_iteration", 8)
    train_steps = args.train_steps if args.train_steps is not None else cfg.get("training_steps_per_iteration", 16)
    log_dir = args.log_dir if args.log_dir is not None else cfg.get("log_dir")
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else cfg.get("checkpoint_dir")
    checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval is not None else cfg.get("checkpoint_interval", 10)
    temperature = args.temperature if args.temperature is not None else cfg.get("temperature", 1.0)
    workers = args.self_play_workers if args.self_play_workers is not None else cfg.get("self_play_workers", 1)
    validation_sample_size = args.validation_sample_size if args.validation_sample_size is not None else cfg.get("validation_sample_size", 0)
    temperature_schedule = cfg.get("temperature_schedule")

    mcts_cfg = cfg.get("mcts", {})
    if args.mcts_simulations is not None:
        mcts_cfg["num_simulations"] = args.mcts_simulations
    if args.dirichlet_alpha is not None:
        mcts_cfg["dirichlet_alpha"] = args.dirichlet_alpha
    mcts = MCTSConfig(**mcts_cfg)
    training_cfg = cfg.get("training", {})
    training = TrainingConfig(**training_cfg)

    config = SelfPlayTrainerConfig(
        episodes_per_iteration=episodes,
        training_steps_per_iteration=train_steps,
        training_config=training,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        temperature=temperature,
        temperature_schedule=temperature_schedule,
        self_play_workers=workers,
        validation_sample_size=validation_sample_size,
        mcts_config=mcts,
        wandb_project=args.wandb_project or cfg.get("wandb_project"),
        wandb_run_name=args.wandb_run_name or cfg.get("wandb_run_name"),
        wandb_entity=args.wandb_entity or cfg.get("wandb_entity"),
    )

    trainer = SelfPlayTrainer(config)
    try:
        iterator = trange(args.iterations, desc="Iterations") if trange is not range else range(args.iterations)
        for i in iterator:
            result = trainer.iteration()
            print(json.dumps({"iteration": i + 1, **result}, indent=2))
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
