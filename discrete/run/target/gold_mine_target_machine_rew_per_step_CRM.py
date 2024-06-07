import torch

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.automaton.target_automaton import RewardShapingTargetAutomaton
from discrete.lib.main import run_training
from discrete.run.env.gold_mine import gold_mine_rew_per_step_env_config
from discrete.run.env.gold_mine import gold_mine_automaton
from discrete.run.utils import student_config_reward_machine

device = torch.device("cpu")
config = student_config_reward_machine(
    env_config=gold_mine_rew_per_step_env_config,
    teacher_run_name="gold_mine_machine_rew_per_step",
    student_run_name="gold_mine_target_machine_rew_per_step_CRM",
    agent_cls=OneHotAutomatonAfterFeatureExtractorAgent,
    device=device,
    automaton=gold_mine_automaton,
    max_training_steps=int(2e6)
)

if __name__ == '__main__':
    run_training(config)
