import torch

from discrete.lib.main import run_policy_distillation
from discrete.lib.agent.normal_agent import DuelingQNetworkAgent
# from discrete.lib.agent.one_hot_automaton_agent import DuelingQNetworkAgent
from discrete.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from discrete.run.env.gold_mine_7 import gold_mine_rew_per_step_env_config_7, gold_mine_automaton, gold_mine_ap_extractor
from discrete.run.env.gold_mine_7_obstacles import gold_mine_rew_per_step_env_config_7_obstacles
from discrete.run.env.dungeon_quest_7_obstacles import dungeon_quest_rew_per_step_env_config_7_obstacles, dungeon_quest_aps_obstacles, dungeon_quest_ltlf_obstacles
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

    teacher_config = teacher_config_v1(gold_mine_rew_per_step_env_config_7, 
                           "gold_mine_rew_per_step_env_config_7",
                           device, 
                           agent_cls=DuelingQNetworkAgent, 
                        #    aps=dungeon_quest_aps,
                        #    ltlf=dungeon_quest_ltlf, 
                           max_training_steps=max_training_steps,
                           path_to_out=path_to_out
                           )
    
    student_config = teacher_config_v1(gold_mine_rew_per_step_env_config_7_obstacles, 
                               "gold_mine_rew_per_step_env_config_7_obstacles",
                               device, 
                               agent_cls=DuelingQNetworkAgent,
                               max_training_steps=max_training_steps, 
                            #    gamma=gamma, alr=alr, clr=clr, batch_size=batch_size, tau=tau, 
                               path_to_out=path_to_out,
                            #    aps=dungeon_quest_aps_obstacles, 
                            #    ltlf=dungeon_quest_ltlf_obstacles
                            )

    teacher_config = teacher_config._replace(automaton=gold_mine_automaton, ap_extractor=gold_mine_ap_extractor)
    student_config = student_config._replace(automaton=gold_mine_automaton, ap_extractor=gold_mine_ap_extractor)

    print("\n\n============================================")
    print("Gold mine policy distillation run")
    print(f"Max Training Steps: {max_training_steps}")
    # print(f"LTLF: {dungeon_quest_ltlf}")
    # print(f"Hyperparameters: {}")
    print("============================================\n\n")
    start_time = time.time()
    run_policy_distillation(teacher_config, student_config)
    print(f"Total elapsed time: {time.time() - start_time}")