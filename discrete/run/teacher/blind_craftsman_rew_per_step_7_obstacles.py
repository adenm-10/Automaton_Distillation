import torch

from discrete.lib.main import run_training
from discrete.run.env.blind_craftsman_7_obstacles import blind_craftsman_rew_per_step_env_config_7_obstacles
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(blind_craftsman_rew_per_step_env_config_7_obstacles, "blind_craftsman_rew_per_step_7_obstacles", device)

if __name__ == '__main__':
    run_training(config)
