import torch

from discrete.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest import dungeon_quest_rew_per_step_env_config
from discrete.run.utils import student_config_v1

device = torch.device("cpu")
config = student_config_v1(
    env_config=dungeon_quest_rew_per_step_env_config,
    teacher_run_name="dungeon_quest_teacher_rew_per_step_7",
    student_run_name="dungeon_quest_7_10_target_rew_per_step",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.999
    }
)

if __name__ == '__main__':
    run_training(config)
