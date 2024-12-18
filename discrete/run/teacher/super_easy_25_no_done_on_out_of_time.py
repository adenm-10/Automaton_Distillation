import torch

from discrete.lib.main import run_training
from discrete.run.env.super_easy_25 import super_easy_env_config_25
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(super_easy_env_config_25, "super_easy_25_no_done_on_out_of_time", device,
                           no_done_on_out_of_time=True)

if __name__ == '__main__':
    run_training(config)
