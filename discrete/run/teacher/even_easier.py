import torch

from discrete.lib.main import run_training
from discrete.run.env.even_easier import even_easier_env_config
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(even_easier_env_config, "even_easier_2", device)

if __name__ == '__main__':
    run_training(config)
