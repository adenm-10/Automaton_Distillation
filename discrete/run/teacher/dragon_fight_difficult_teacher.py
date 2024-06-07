import torch

from discrete.lib.main import run_training
from discrete.run.env.dragon_fight_difficult import dragon_fight_config
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(dragon_fight_config, "dragon_fight_difficult_teacher",
                           device, max_training_steps=int(2e6))

if __name__ == '__main__':
    run_training(config)
