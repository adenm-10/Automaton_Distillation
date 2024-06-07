import torch

from discrete.lib.main import run_training
from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.run.env.blind_craftsman_7_obstacles import blind_craftsman_rew_per_step_env_config_7_obstacles
from discrete.run.env.blind_craftsman import blind_craftsman_aps, blind_craftsman_ltlf
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(blind_craftsman_rew_per_step_env_config_7_obstacles, "blind_craftsman_obstacles_teacher_rew_per_step_productMDP", device,
                           agent_cls=OneHotAutomatonAfterFeatureExtractorAgent, aps=blind_craftsman_aps, 
                           ltlf=blind_craftsman_ltlf)

if __name__ == '__main__':
    run_training(config)
