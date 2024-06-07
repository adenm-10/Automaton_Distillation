import torch

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.automaton.target_automaton import RewardShapingTargetAutomaton
from discrete.lib.main import run_training
from discrete.run.env.blind_craftsman_7_obstacles import blind_craftsman_rew_per_step_env_config_7_obstacles
from discrete.run.env.blind_craftsman import blind_craftsman_aps, blind_craftsman_ltlf
from discrete.run.utils import student_config_reward_machine

device = torch.device("cpu")
config = student_config_reward_machine(
    env_config=blind_craftsman_rew_per_step_env_config_7_obstacles,
    teacher_run_name="blind_craftsman_machine_rew_per_step",
    student_run_name="blind_craftsman_obstacles_target_machine_rew_per_step_CRM",
    agent_cls=OneHotAutomatonAfterFeatureExtractorAgent,
    device=device,
    aps=blind_craftsman_aps,
    ltlf=blind_craftsman_ltlf
)

if __name__ == '__main__':
    run_training(config)
