import torch

"""
original
from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest import dungeon_quest_rew_per_step_env_config
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(
    dungeon_quest_rew_per_step_env_config,
    "dungeon_quest_teacher_rew_per_step",
    device,
    max_training_steps=int(2e6))

if __name__ == '__main__':
    run_training(config)
"""

from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest_7 import dungeon_quest_rew_per_step_env_config
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("\n==============\nCuda detected!\n==============\n")
else:
    print("No CUDA detected, using CPU...\n")


config = teacher_config_v1(
    dungeon_quest_rew_per_step_env_config,
    "dungeon_quest_teacher_rew_per_step",
    device,
    max_training_steps=int(5e5))

if __name__ == '__main__':
    print("\n\n============================================")
    print(f"Training Teacher / Independent Dueling-DQN Agent")
    print(f"Max Training Steps: {max_training_steps}")
    print(f"LTLF: {dungeon_quest_ltlf}")
    print("============================================\n\n")

    run_training(config)

