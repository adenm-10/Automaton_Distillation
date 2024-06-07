import torch

from discrete.lib.main import run_training
from discrete.run.env.super_easy import super_easy_env_config
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(super_easy_env_config, "super_easy", device)

if __name__ == '__main__':
    run_training(config)
