import torch

from discrete.lib.main import run_training
from discrete.run.env.blind_craftsman_7 import blind_craftsman_exp_env_config_7
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(blind_craftsman_exp_env_config_7, "blind_craftsman_teacher_exp_7", device)

if __name__ == '__main__':
    run_training(config)
