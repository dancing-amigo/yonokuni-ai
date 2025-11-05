import os

from yonokuni.mcts import MCTSConfig
from yonokuni.orchestration import SelfPlayTrainer, SelfPlayTrainerConfig
from yonokuni.training import TrainingConfig


def test_self_play_trainer_checkpoint(tmp_path):
    config = SelfPlayTrainerConfig(
        episodes_per_iteration=1,
        training_steps_per_iteration=1,
        buffer_capacity=128,
        mcts_config=MCTSConfig(num_simulations=1, dirichlet_alpha=0.0),
        training_config=TrainingConfig(batch_size=8, apply_symmetry=False),
        checkpoint_dir=tmp_path.as_posix(),
        checkpoint_interval=1,
    )

    trainer = SelfPlayTrainer(config)
    try:
        result = trainer.iteration()
        assert "buffer_size" in result
        path = trainer.save_checkpoint()
        assert os.path.exists(path)
    finally:
        trainer.close()

    new_trainer = SelfPlayTrainer(config)
    try:
        new_trainer.load_checkpoint(path)
        assert len(new_trainer.replay_buffer) > 0
    finally:
        new_trainer.close()
