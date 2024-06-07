import torch

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.main import run_distill
from discrete.run.env.blind_craftsman import blind_craftsman_rew_per_step_env_config
from discrete.run.utils import teacher_config_v1
from discrete.run.env.blind_craftsman import blind_craftsman_aps, blind_craftsman_ltlf

device = torch.device("cpu")
config = teacher_config_v1(blind_craftsman_rew_per_step_env_config, "blind_craftsman_teacher_rew_per_step", device, 
                            aps=blind_craftsman_aps, ltlf=blind_craftsman_ltlf)

if __name__ == '__main__':
    run_distill(config)
