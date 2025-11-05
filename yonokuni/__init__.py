"""Yonokuni AI core package."""

from . import core, env, features, mcts, selfplay, models, training, evaluation
from .env import YonokuniEnv
from .features import (
    AUX_VECTOR_SIZE,
    BOARD_CHANNELS,
    Transform,
    all_transforms,
    apply_policy_transform,
    policy_permutation,
    build_aux_vector,
    build_board_tensor,
    state_to_numpy,
    state_to_torch,
    team_flipped,
    transform_aux_vector,
    transform_board_tensor,
    transform_action,
)
from .mcts import MCTS, MCTSConfig
from .models import YonokuniEvaluator, YonokuniNet, YonokuniNetConfig
from .selfplay import (
    MCTSPolicy,
    RandomPolicy,
    ReplayBuffer,
    ReplaySample,
    SelfPlayManager,
    make_mcts_policy_from_model,
)
from .training import TrainingConfig, Trainer, TrainingStepOutput
from .evaluation import EvaluationResult, evaluate_policies
from .orchestration import SelfPlayTrainer, SelfPlayTrainerConfig

__all__ = [
    "core",
    "env",
    "features",
    "models",
    "training",
    "evaluation",
    "mcts",
    "selfplay",
    "YonokuniEnv",
    "AUX_VECTOR_SIZE",
    "BOARD_CHANNELS",
    "Transform",
    "all_transforms",
    "apply_policy_transform",
    "policy_permutation",
    "build_aux_vector",
    "build_board_tensor",
    "state_to_numpy",
    "state_to_torch",
    "team_flipped",
    "transform_aux_vector",
    "transform_board_tensor",
    "transform_action",
    "MCTS",
    "MCTSConfig",
    "YonokuniNet",
    "YonokuniNetConfig",
    "YonokuniEvaluator",
    "ReplayBuffer",
    "ReplaySample",
    "MCTSPolicy",
    "make_mcts_policy_from_model",
    "RandomPolicy",
    "SelfPlayManager",
    "TrainingConfig",
    "Trainer",
    "TrainingStepOutput",
    "SelfPlayTrainer",
    "SelfPlayTrainerConfig",
    "EvaluationResult",
    "evaluate_policies",
]
