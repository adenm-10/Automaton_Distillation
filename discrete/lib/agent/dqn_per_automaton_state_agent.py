from typing import Tuple

import torch
import torch.nn as nn

from discrete.lib.agent.agent import Agent, TargetAgent
from discrete.lib.agent.easy_target_agent import EasyTargetAgent
from discrete.lib.agent.feature_extractor import FeatureExtractor
from discrete.lib.agent.multi_agent_layer import MultiAgentLinearLayer


class DQNPerAutomatonStateAgent(Agent):
    """An dueling q-agent with a shared feature extractor and a different linear layer for each automaton state"""

    def __init__(self, input_shape: Tuple, num_automaton_states: int, num_actions: int):
        super().__init__()

        self.input_shape = input_shape
        self.feature_extractor = FeatureExtractor(input_shape=input_shape)
        self.half_feat_extractor_output_size = self.feature_extractor.output_size // 2
        self.val_branch = MultiAgentLinearLayer(self.half_feat_extractor_output_size, 1, num_automaton_states)
        self.adv_branch = MultiAgentLinearLayer(self.half_feat_extractor_output_size, num_actions, num_automaton_states)
        self.flattener = nn.Flatten()
        self.num_automaton_states = num_automaton_states
        self.num_actions = num_actions

    @classmethod
    def create_agent(cls, input_shape: Tuple, num_automaton_states: int, num_actions: int) -> "Agent":
        return cls(input_shape, num_automaton_states, num_actions)

    def forward(self, obs, automaton_states):
        features = self.feature_extractor(obs)
        val_stream, adv_stream = torch.split(features, self.half_feat_extractor_output_size, dim=1)

        # Val and adv are because of dueling Q-network architecture, not actor critic
        val_stream = self.flattener(val_stream)
        adv_stream = self.flattener(adv_stream)

        val = self.val_branch(automaton_states, val_stream)
        adv = self.adv_branch(automaton_states, adv_stream)

        mean_adv = adv.mean(dim=1)

        q_vals = val + (adv - mean_adv.unsqueeze(1))  # Unsqueeze necessary so that it broadcasts correctly
        return q_vals

    def calc_q_values_batch(self, observation: torch.Tensor, automaton_states: torch.Tensor) -> torch.Tensor:
        return self(observation, automaton_states)
    
    def calc_v_values_batch(self, observation: torch.Tensor, automaton_state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(obs)
        val_stream, _ = torch.split(features, self.half_feat_extractor_output_size, dim=1)

        # Val and adv are because of dueling Q-network architecture, not actor critic
        val_stream = self.flattener(val_stream)

        val = self.val_branch(automaton_states, val_stream)
        
        return val

    def create_target_agent(self) -> "TargetAgent":
        eta = EasyTargetAgent(self, DQNPerAutomatonStateAgent(self.input_shape, self.num_automaton_states,
                                                              self.num_actions)).to(self.adv_branch.weight.device)
        eta.update_weights()
        return eta
