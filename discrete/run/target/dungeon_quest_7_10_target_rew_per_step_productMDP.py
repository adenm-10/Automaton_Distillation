import torch
import argparse

from discrete.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest import dungeon_quest_rew_per_step_env_config
from discrete.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf
from discrete.run.utils import student_config_v1

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device('cuda:0')




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to handle command line arguments for ALR, CLR, and Gamma.")
    parser.add_argument('--total-steps', type=int, default=int(2e6), help='Buffer Batch Size')
    parser.add_argument('--path-to-out', type=str, default="", help='Path to place plots')

    # Parse arguments from command line
    args = parser.parse_args()
    
    max_training_steps = int(args.total_steps)
    path_to_out = args.path_to_out

    config = student_config_v1(
        env_config=dungeon_quest_rew_per_step_env_config,
        teacher_run_name="dungeon_quest_teacher_rew_per_step_7_productMDP",
        student_run_name="dungeon_quest_7_10_target_rew_per_step_productMDP",
        device=device,
        anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
        anneal_target_aut_kwargs={
            "exponent_base": 0.999
        },
        aps=dungeon_quest_aps,
        ltlf=dungeon_quest_ltlf,
        max_training_steps=int(max_training_steps),
        path_to_out=path_to_out
    )

    run_training(config)