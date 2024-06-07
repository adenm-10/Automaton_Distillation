import torch

from discrete.lib.main import run_training
from discrete.run.env.dragon_fight_easy import dragon_fight_config, dragon_fight_aps, dragon_fight_ltlf
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(dragon_fight_config, "dragon_fight_easy_teacher",
                           device, max_training_steps=int(2e6), aps=dragon_fight_aps, ltlf=dragon_fight_ltlf)

if __name__ == '__main__':
    run_training(config)
