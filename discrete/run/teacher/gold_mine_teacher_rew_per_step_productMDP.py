import torch

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.main import run_training
from discrete.run.env.gold_mine import gold_mine_rew_per_step_env_config
from discrete.run.env.gold_mine import gold_mine_automaton, gold_mine_ap_extractor
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(gold_mine_rew_per_step_env_config,
                           "gold_mine_teacher_rew_per_step_productMDP",
                           device, agent_cls=OneHotAutomatonAfterFeatureExtractorAgent,
                           max_training_steps=int(2e6))

# Add automaton to config
config = config._replace(automaton=gold_mine_automaton, ap_extractor=gold_mine_ap_extractor)

if __name__ == '__main__':
    run_training(config)
