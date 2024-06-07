import torch

from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest import dungeon_quest_env_config
from discrete.run.utils import teacher_config_v1
from discrete.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf

device = torch.device("cpu")
config = teacher_config_v1(dungeon_quest_env_config, "dungeon_quest_teacher", device)



if __name__ == '__main__':
    run_training(config)
