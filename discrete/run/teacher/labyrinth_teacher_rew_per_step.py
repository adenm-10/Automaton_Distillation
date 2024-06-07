import torch

from discrete.lib.main import run_training
from discrete.run.env.labyrinth import labyrinth_rew_per_step_env_config
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(labyrinth_rew_per_step_env_config, "labyrinth_teacher_rew_per_step", device)

if __name__ == '__main__':
    run_training(config)
