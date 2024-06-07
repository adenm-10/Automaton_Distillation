import torch

from discrete.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from discrete.lib.main import run_training
from discrete.run.env.blind_craftsman_7_obstacles import blind_craftsman_rew_per_step_env_config_7_obstacles
from discrete.run.env.blind_craftsman import blind_craftsman_aps, blind_craftsman_ltlf
from discrete.run.utils import student_config_v1

device = torch.device("cpu")
config = student_config_v1(
    env_config=blind_craftsman_rew_per_step_env_config_7_obstacles,
    teacher_run_name="blind_craftsman_teacher_rew_per_step_7_productMDP",
    student_run_name="blind_craftsman_7_obstacles_target_rew_per_step_productMDP",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.999
    },
    aps=blind_craftsman_aps,
    ltlf=blind_craftsman_ltlf
)

if __name__ == '__main__':
    run_training(config)
