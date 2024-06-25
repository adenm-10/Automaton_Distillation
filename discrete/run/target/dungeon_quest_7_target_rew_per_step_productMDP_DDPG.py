import torch

from discrete.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest_7 import dungeon_quest_rew_per_step_env_config_7_cont, dungeon_quest_aps, dungeon_quest_ltlf
from discrete.run.utils import student_config_v1
from discrete.lib.agent.DDPG_Agent import DDPG_Agent

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("\n==============\nCuda detected!\n==============\n")
else:
    print("No CUDA detected, using CPU...\n")

max_training_steps = int(2e6)

config = student_config_v1(
    env_config=dungeon_quest_rew_per_step_env_config_7_cont,
    teacher_run_name="dungeon_quest_teacher_rew_per_step_7_productMDP",
    student_run_name="dungeon_quest_7_target_rew_per_step_productMDP_DDPG",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.999
    },
    agent_cls=DDPG_Agent,
    aps=dungeon_quest_aps,
    ltlf=dungeon_quest_ltlf,
    max_training_steps=max_training_steps
)

if __name__ == '__main__':
    print("\n\n============================================")
    print(f"Training Student / Dependent DDPG Agent")
    print(f"Max Training Steps: {max_training_steps}")
    print(f"LTLF: {dungeon_quest_ltlf}")
    # print(f"Hyperparameters: {}")
    print("============================================\n\n")

    run_training(config)
