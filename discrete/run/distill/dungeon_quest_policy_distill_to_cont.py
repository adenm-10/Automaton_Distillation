import torch

from discrete.lib.main import run_policy_distillation
from discrete.lib.agent.TD3_Agent import TD3_Agent
from discrete.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from discrete.run.env.dungeon_quest_7 import dungeon_quest_rew_per_step_env_config_7_cont, dungeon_quest_aps, dungeon_quest_ltlf
from discrete.run.env.dungeon_quest_7_obstacles import dungeon_quest_rew_per_step_env_config_7_obstacles_cont, dungeon_quest_aps_obstacles, dungeon_quest_ltlf_obstacles
from discrete.run.utils import teacher_config_v1, student_config_v1
import argparse
import time

import torch


device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
#     print("\n==============\nCuda detected!\n==============\n")
# else:
#     print("No CUDA detected, using CPU...\n")
#     # assert False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to handle command line arguments for ALR, CLR, and Gamma.")
    
    # Add arguments
    parser.add_argument('--total-steps', type=int, default=int(2000), help='Buffer Batch Size')
    parser.add_argument('--path-to-out', type=str, default=None, help='Path to place plots')
    
    # Parse arguments from command line
    args = parser.parse_args()
    
    # Assign parsed values to variables
    max_training_steps = int(args.total_steps)
    path_to_out = args.path_to_out
    # dungeon_quest_config_7.placements[-1].tile.reward = args.dragon_reward

    teacher_config = teacher_config_v1(dungeon_quest_rew_per_step_env_config_7_cont, 
                           "dungeon_quest_teacher_rew_per_step_7_productMDP",
                           device, 
                           agent_cls=TD3_Agent, 
                           aps=dungeon_quest_aps,
                           ltlf=dungeon_quest_ltlf, 
                           max_training_steps=max_training_steps,
                           path_to_out=path_to_out)
    
    student_config = teacher_config_v1(dungeon_quest_rew_per_step_env_config_7_obstacles_cont, 
                               "blind_craftsman_7_obstacles_target_rew_per_step_productMDP",
                               device, 
                               agent_cls=TD3_Agent,
                               max_training_steps=max_training_steps, 
                            #    gamma=gamma, alr=alr, clr=clr, batch_size=batch_size, tau=tau, 
                               path_to_out=path_to_out,
                               aps=dungeon_quest_aps_obstacles, 
                               ltlf=dungeon_quest_ltlf_obstacles)

    print("\n\n============================================")
    print("Training Teacher / Independent TD3 Agent")
    print(f"Max Training Steps: {max_training_steps}")
    # print(f"LTLF: {dungeon_quest_ltlf}")
    # print(f"Hyperparameters: {}")
    print("============================================\n\n")
    start_time = time.time()
    run_policy_distillation(teacher_config, student_config)
    print(f"Total elapsed time: {time.time() - start_time}")