import torch

from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest import dungeon_quest_rew_per_step_env_config
from discrete.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf
from discrete.run.utils import teacher_config_v1

device = torch.device("cpu")
config = teacher_config_v1(dungeon_quest_rew_per_step_env_config, "dungeon_quest_teacher_rew_per_step_productMDP",
                           device, agent_cls=OneHotAutomatonAfterFeatureExtractorAgent, aps=dungeon_quest_aps,
                           ltlf=dungeon_quest_ltlf,
                           max_training_steps=int(2e6))

if __name__ == '__main__':
    run_training(config)
