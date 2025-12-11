#!/usr/bin/env python3
"""
Profile a self-play + training iteration to see where time is spent.

Example (Windows GPU preset):
  python scripts/profile_iteration.py --config configs/self_play_windows_gpu.yaml --iterations 1
Override knobs as needed:
  --episodes 24 --train-steps 128 --mcts-simulations 16 --self-play-workers 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import yaml

from yonokuni.env import YonokuniEnv
from yonokuni.mcts import MCTSConfig
from yonokuni.orchestration import SelfPlayTrainer, SelfPlayTrainerConfig
from yonokuni.selfplay import EarlyTerminationConfig
from yonokuni.training import TrainingConfig
from yonokuni.validation import validate_buffer_sample


def load_yaml_config(path_str: str) -> Dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _make_env_factory(env_max_ply: Optional[int]) -> YonokuniEnv:
    if env_max_ply is None:
        return YonokuniEnv  # type: ignore[return-value]

    def factory() -> YonokuniEnv:
        return YonokuniEnv(max_ply=env_max_ply)

    return factory  # type: ignore[return-value]


def build_config(args: argparse.Namespace, cfg: Dict) -> SelfPlayTrainerConfig:
    episodes = args.episodes if args.episodes is not None else cfg.get("episodes_per_iteration", 8)
    train_steps = args.train_steps if args.train_steps is not None else cfg.get("training_steps_per_iteration", 16)
    log_dir = args.log_dir if args.log_dir is not None else cfg.get("log_dir")
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else cfg.get("checkpoint_dir")
    checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval is not None else cfg.get(
        "checkpoint_interval", 10
    )
    temperature = args.temperature if args.temperature is not None else cfg.get("temperature", 1.0)
    workers = args.self_play_workers if args.self_play_workers is not None else cfg.get("self_play_workers", 1)
    validation_sample_size = (
        args.validation_sample_size if args.validation_sample_size is not None else cfg.get("validation_sample_size", 0)
    )
    temperature_schedule = cfg.get("temperature_schedule")

    mcts_cfg = cfg.get("mcts", {})
    if args.mcts_simulations is not None:
        mcts_cfg["num_simulations"] = args.mcts_simulations
    if args.dirichlet_alpha is not None:
        mcts_cfg["dirichlet_alpha"] = args.dirichlet_alpha
    mcts = MCTSConfig(**mcts_cfg)

    training_cfg = cfg.get("training", {})
    training = TrainingConfig(**training_cfg)
    early_stop_cfg = cfg.get("early_termination", {})
    early_stop = EarlyTerminationConfig(**early_stop_cfg)

    return SelfPlayTrainerConfig(
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
        env_factory=_make_env_factory(args.env_max_ply),
        early_termination=early_stop,
        wandb_project=args.wandb_project or cfg.get("wandb_project"),
        wandb_run_name=args.wandb_run_name or cfg.get("wandb_run_name"),
        wandb_entity=args.wandb_entity or cfg.get("wandb_entity"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a self-play + training iteration.")
    parser.add_argument("--config", type=str, default="configs/self_play_windows_gpu.yaml")
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
    parser.add_argument("--env-max-ply", type=int, help="Override YonokuniEnv max_ply (default 400)")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--resume-from", type=str, help="Checkpoint path to resume from")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation check during profiling")
    parser.add_argument("--print-gpu", action="store_true", help="Print CUDA device info once at start")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    config = build_config(args, cfg)

    trainer = SelfPlayTrainer(config)
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    # Optional CUDA info (helps confirm GPU + dtype)
    if args.print_gpu:
        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None
        if torch is not None and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            print(json.dumps({"cuda": True, "device": device_name, "capability": capability}, indent=2))
        else:
            print(json.dumps({"cuda": False}, indent=2))

    try:
        for i in range(args.iterations):
            iter_index = trainer.iteration_index + 1

            t0 = time.perf_counter()
            self_play_stats = trainer.self_play.generate(
                config.episodes_per_iteration,
                workers=config.self_play_workers,
            )
            t1 = time.perf_counter()

            metrics = []
            for _ in range(config.training_steps_per_iteration):
                result = trainer.trainer.train_step().as_dict()
                metrics.append(result)
                trainer.training_step_count += 1
            t2 = time.perf_counter()

            if (
                not args.skip_validation
                and config.validation_sample_size > 0
                and len(trainer.replay_buffer) >= config.validation_sample_size
            ):
                validate_buffer_sample(trainer.replay_buffer, config.validation_sample_size)

            buffer_size = len(trainer.replay_buffer)
            avg_metrics = {}
            if metrics:
                keys = metrics[0].keys()
                avg_metrics = {k: float(sum(m[k] for m in metrics) / len(metrics)) for k in keys}

            trainer.iteration_index += 1

            self_play_time = t1 - t0
            train_time = t2 - t1
            total_time = t2 - t0

            games = self_play_stats.get("games_played", 0)
            games_per_sec = games / self_play_time if self_play_time > 0 else 0.0
            train_steps_per_sec = config.training_steps_per_iteration / train_time if train_time > 0 else 0.0

            result = {
                "iteration": iter_index,
                "buffer_size": buffer_size,
                "self_play_time_sec": self_play_time,
                "train_time_sec": train_time,
                "total_time_sec": total_time,
                "games_played": games,
                "games_per_sec": games_per_sec,
                "train_steps_per_sec": train_steps_per_sec,
                "self_play": self_play_stats,
                "train_avg": avg_metrics,
            }
            print(json.dumps(result, indent=2))
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
