import torch

from discrete.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from discrete.lib.main import run_training
from discrete.run.env.blind_craftsman import blind_craftsman_exp_env_config
from discrete.run.utils import student_config_v1

device = torch.device("cpu")
config = student_config_v1(
    env_config=blind_craftsman_exp_env_config,
    teacher_run_name="blind_craftsman_teacher_exp_7",
    student_run_name="blind_craftsman_7_10_target_exp",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.999
    }
)

if __name__ == '__main__':
    run_training(config)
