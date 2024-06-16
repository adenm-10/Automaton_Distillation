

print("entered training run...")

import torch
import time

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest_7 import dungeon_quest_rew_per_step_env_config_7, dungeon_quest_rew_per_step_env_config_7_cont
from discrete.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf
from discrete.run.utils import teacher_config_v1
from discrete.lib.agent.DDPG_Agent import DDPG_Agent

from discrete.lib.agent.AC_Agent import AC_Agent

print("imported all dependencies, checking for cuda")

#device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("\n==============\nCuda detected!\n==============\n")
else:
    print("No CUDA detected, using CPU...\n")
    # assert False

max_training_steps=int(2000)

config = teacher_config_v1(dungeon_quest_rew_per_step_env_config_7_cont, "dungeon_quest_rew_per_step_env_config_7_cont",
                           device, aps=dungeon_quest_aps, agent_cls=DDPG_Agent,
                           ltlf=dungeon_quest_ltlf, max_training_steps=max_training_steps)


if __name__ == '__main__':
    print("\n\n============================================")
    print(f"DDPG AGENT TRAINING")
    print(f"Steps Before Total Run Termination: {max_training_steps}")
    print("============================================\n\n")
    start_time = time.time()
    run_training(config)
    print(f"Total elapsed time: {time.time() - start_time}")
