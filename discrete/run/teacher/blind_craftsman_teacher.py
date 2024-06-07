import torch

from discrete.lib.main import run_training
from discrete.run.env.blind_craftsman import blind_craftsman_env_config
from discrete.run.utils import teacher_config_v1
from discrete.run.env.blind_craftsman import blind_craftsman_aps, blind_craftsman_ltlf

device = torch.device("cpu")
config = teacher_config_v1(blind_craftsman_env_config, "blind_craftsman_teacher", device)

if __name__ == '__main__':
    run_training(config)
