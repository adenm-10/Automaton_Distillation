import torch

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.main import run_training
from discrete.run.env.gold_mine_7 import gold_mine_rew_per_step_env_config_7, gold_mine_ltlf
from discrete.run.env.gold_mine_7 import gold_mine_automaton, gold_mine_ap_extractor
from discrete.run.utils import teacher_config_v1
import argparse
import time

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("\n==============\nCuda detected!\n==============\n")
else:
    print("No CUDA detected, using CPU...\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to handle command line arguments for ALR, CLR, and Gamma.")
    
    # Add arguments
    parser.add_argument('--alr', type=float, default=0.0001, help='Actor Learning Rate')
    parser.add_argument('--clr', type=float, default=0.0001, help='Critic Learning Rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount Factor (Gamma)')
    parser.add_argument('--batch-size', type=int, default=64, help='Buffer Batch Size')
    parser.add_argument('--tau', type=float, default=0.005, help='Target Transfer Tau')
    parser.add_argument('--total-steps', type=int, default=int(1e6), help='Buffer Batch Size')
    parser.add_argument('--path-to-out', type=str, default="", help='Path to place plots')

    
    # Parse arguments from command line
    args = parser.parse_args()
    
    # Assign parsed values to variables
    alr = args.alr
    clr = args.clr
    gamma = args.gamma
    batch_size = args.batch_size
    tau = args.tau
    max_training_steps = int(args.total_steps)
    path_to_out = args.path_to_out

    config = teacher_config_v1(gold_mine_rew_per_step_env_config_7, 
                               "gold_mine_teacher_rew_per_step_7_productMDP",
                               device, 
                               agent_cls=OneHotAutomatonAfterFeatureExtractorAgent,
                               max_training_steps=max_training_steps, 
                               gamma=gamma, alr=alr, clr=clr, batch_size=batch_size, tau=tau, 
                               path_to_out=path_to_out)
    
    
    # Add automaton to config
    config = config._replace(automaton=gold_mine_automaton, ap_extractor=gold_mine_ap_extractor)

    print("\n\n============================================")
    print("Training Teacher / Independent TD3 Agent")
    print(f"Max Training Steps: {max_training_steps}")
    print(f"LTLF: {gold_mine_ltlf}")
    print(f"Environment: Discrete Gold Mine 7x7")
    print("============================================\n\n")
    start_time = time.time()
    run_training(config)
    print(f"Total elapsed time: {time.time() - start_time}")
