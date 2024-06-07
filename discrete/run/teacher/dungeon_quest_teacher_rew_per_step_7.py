import torch

from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest_7 import dungeon_quest_rew_per_step_env_config_7
from discrete.run.utils import teacher_config_v1
from discrete.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf

device = torch.device("cpu")
config = teacher_config_v1(dungeon_quest_rew_per_step_env_config_7, "dungeon_quest_teacher_rew_per_step_7", device, aps=dungeon_quest_aps,
                           ltlf=dungeon_quest_ltlf)

if __name__ == '__main__':
    run_training(config)
