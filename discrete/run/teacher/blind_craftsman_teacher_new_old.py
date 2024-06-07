import torch

from discrete.lib.main import run_training
from discrete.run.env.blind_craftsman_new import blind_craftsman_env_config_new
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(blind_craftsman_env_config_new, "blind_craftsman_teacher_new", device)

if __name__ == '__main__':
    run_training(config)
