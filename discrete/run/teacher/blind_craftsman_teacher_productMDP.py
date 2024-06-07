import torch

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.main import run_training
from discrete.run.env.blind_craftsman import blind_craftsman_env_config
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(blind_craftsman_env_config, "blind_craftsman_teacher_productMDP", device, 
							agent_cls=OneHotAutomatonAfterFeatureExtractorAgent)

if __name__ == '__main__':
	run_training(config)
