import torch

from discrete.lib.automaton.target_automaton import RewardShapingTargetAutomaton
from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest import dungeon_quest_rew_per_step_env_config
from discrete.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf
from discrete.run.utils import student_config_v1

device = torch.device("cpu")
config = student_config_v1(
    env_config=dungeon_quest_rew_per_step_env_config,
    teacher_run_name="dungeon_quest_machine_rew_per_step",
    student_run_name="dungeon_quest_target_machine_shaping_rew_per_step_productMDP",
    device=device,
    anneal_target_aut_class=RewardShapingTargetAutomaton,
    anneal_target_aut_kwargs={
        "gamma": 0.999
    },
    aps=dungeon_quest_aps,
    ltlf=dungeon_quest_ltlf
)

if __name__ == '__main__':
    run_training(config)
