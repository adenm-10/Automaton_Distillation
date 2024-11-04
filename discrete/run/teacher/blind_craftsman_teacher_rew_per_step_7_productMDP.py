import time
import torch
import argparse

from discrete.lib.main import run_training
from discrete.run.utils import teacher_config_v1
from discrete.run.env.blind_craftsman_7 import blind_craftsman_rew_per_step_env_config_7
from discrete.run.env.blind_craftsman import blind_craftsman_aps, blind_craftsman_ltlf
from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("\n==============\nCuda detected!\n==============\n")
else:
    print("No CUDA detected, using CPU...\n")
    # assert False

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

    config = teacher_config_v1(blind_craftsman_rew_per_step_env_config_7, 
                               "blind_craftsman_teacher_rew_per_step_7_productMDP",
                               device, 
                               aps=blind_craftsman_aps, 
                               ltlf=blind_craftsman_ltlf,
                               agent_cls=OneHotAutomatonAfterFeatureExtractorAgent,
                               max_training_steps=max_training_steps, 
                               gamma=gamma, alr=alr, clr=clr, batch_size=batch_size, tau=tau, 
                               path_to_out=path_to_out)

    print("\n\n============================================")
    print("Training Teacher / Independent TD3 Agent")
    print(f"Max Training Steps: {max_training_steps}")
    print(f"LTLF: {blind_craftsman_ltlf}")
    print(f"Environment: Discrete Blind Craftsman 7x7")
    print("============================================\n\n")
    start_time = time.time()
    run_training(config)
    print(f"Total elapsed time: {time.time() - start_time}")
