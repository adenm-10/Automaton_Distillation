import torch

from discrete.lib.main import run_training
from discrete.run.env.space_invaders_difficult import space_invaders_config
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(space_invaders_config, "space_invaders_difficult_teacher",
                           device, max_training_steps=int(5e6))

if __name__ == '__main__':
    run_training(config)
