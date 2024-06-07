import torch

from discrete.lib.main import run_training
from discrete.run.env.rock_collector import rock_collector_rew_per_step_env_config
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(rock_collector_rew_per_step_env_config, "rock_collector_teacher_rew_per_step", device, max_training_steps=int(2e5))

if __name__ == '__main__':
    run_training(config)
