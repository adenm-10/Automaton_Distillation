import torch

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.main import run_training
from discrete.run.env.blind_craftsman_7 import blind_craftsman_rew_per_step_env_config_7
from discrete.run.env.blind_craftsman import blind_craftsman_aps, blind_craftsman_ltlf
from discrete.run.utils import teacher_config_v1
import argparse
import time

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
    parser.add_argument('--total-steps', type=int, default=int(5e5), help='Buffer Batch Size')
    parser.add_argument('--path-to-out', type=str, default=None, help='Path to place plots')

    
    # Parse arguments from command line
    args = parser.parse_args()
    
    # Assign parsed values to variables
    max_training_steps = int(args.total_steps)
    path_to_out = args.path_to_out
    # dungeon_quest_config_7.placements[-1].tile.reward = args.dragon_reward

    config = teacher_config_v1(blind_craftsman_rew_per_step_env_config_7, 
                               "blind_craftsman_teacher_rew_per_step_7_productMDP",
                               device, 
                               agent_cls=OneHotAutomatonAfterFeatureExtractorAgent,
                            #    aps=dungeon_quest_aps, 
                            #    ltlf=dungeon_quest_ltlf, 
                               max_training_steps=max_training_steps, 
                            #    gamma=gamma, alr=alr, clr=clr, batch_size=batch_size, tau=tau, 
                               path_to_out=path_to_out,
                               aps=blind_craftsman_aps, 
                               ltlf=blind_craftsman_ltlf)

    print("\n\n============================================")
    print("Training Teacher / Independent TD3 Agent")
    print(f"Max Training Steps: {max_training_steps}")
    # print(f"LTLF: {dungeon_quest_ltlf}")
    # print(f"Hyperparameters: {}")
    print("============================================\n\n")
    start_time = time.time()
    run_training(config)
    print(f"Total elapsed time: {time.time() - start_time}")
