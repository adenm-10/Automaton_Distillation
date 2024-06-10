'''
Created on Feb 6, 2024

@author: diegobenalcazar
'''

import abc
from abc import ABC
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
import os

from discrete.lib.agent.AC_Agent import AC_Agent, AC_TargetAgent
from discrete.lib.agent.AC_easy_target_agent import AC_EasyTargetAgent
# from discrete.lib.agent.feature_extractor import FeatureExtractor


class Residual(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input):
        output = self.inner(input)
        return input + output

class FeatureExtractor(nn.Module):
    """
    A basic feature extractor designed to work on stacked atari frames
    Heavily based on architecture from DeepSynth and AlphaGo
    """

    def __init__(self, input_shape):
        super().__init__()

        num_blocks = 3
        num_intermediate_filters = 32
        # kernel_size = (3, 3)
        kernel_size = (3, 3)
        padding_amount = 1

        num_channels, *input_shape_single = input_shape

        # print(f"Continuous FE input shape: {[1, *input_shape]}")
        # print(f"num_channels: {num_channels}")
        # print(f"input shape single: {input_shape_single}")

        grid_size = 1
        for dim in input_shape_single:
            grid_size *= dim

        # Basically the architecture from AlphaGo
        def generate_common():
            init_conv = nn.Sequential(
                nn.Conv2d(num_channels, num_intermediate_filters, kernel_size=kernel_size, padding=padding_amount),
                nn.BatchNorm2d(num_intermediate_filters),
                nn.LeakyReLU()
            )

            blocks = [nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(in_channels=num_intermediate_filters,
                                  out_channels=num_intermediate_filters,
                                  kernel_size=kernel_size,
                                  padding=padding_amount),
                        nn.BatchNorm2d(num_intermediate_filters),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=num_intermediate_filters,
                                  out_channels=num_intermediate_filters,
                                  kernel_size=kernel_size,
                                  padding=padding_amount),
                        nn.BatchNorm2d(num_intermediate_filters))
                ),
                nn.LeakyReLU()
            ) for _ in range(num_blocks)]

            return nn.Sequential(
                init_conv, *blocks
            )

        self.net = generate_common()
        self.flattener = nn.Flatten()

        # test_zeros = torch.zeros((1, *input_shape))
        test_zeros = torch.zeros((1, *input_shape, padding_amount))
        # print(f"test zeros: {test_zeros}")
        # print(f"test zeros shape: {test_zeros.shape}")
        # assert False
        self.output_size = int(self.net(test_zeros).numel())

    def forward(self, input):
        all_features = self.net(input)
        return self.flattener(all_features)

    def clone(self):
        other_featext = FeatureExtractor(self.input_shape).to(self.config.device)
        other_featext.load_state_dict(self.state_dict())
        return other_featext

class OUActionNoise(object):
    # def __init__(self, mu, sigma=0.5, theta=1, dt=1e-1, x0=None):
    def __init__(self, mu, sigma=0.5, theta=0.3, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        # print(f"Self.Mu: {self.mu}")
        self.sigma = sigma # standard deviation: variance = stdev^2
        self.dt = dt
        self.x0 = x0
        self.reset()
        
    def __call__(self):
        x = self.x_prev + (self.theta * (self.mu - self.x_prev) * self.dt) + (self.sigma * np.sqrt(self.dt)*np.random.normal(size=self.mu.shape))
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, 
                chkpt_dir='tmp/ddpg'):

        super(CriticNetwork, self).__init__()
        # print(f"input dims critic: {input_dims}")
        self.flattener = nn.Flatten()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'ddpg')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        # f1 = 0.003
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        # f2 = 0.003
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        # print("n_actions")
        # print(self.n_actions)
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        f3 = 0.003
        self.q = nn.Linear(fc2_dims, 1)
        torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.q.bias.data, -f3, f3)
        
        # self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0')
        
        self.to(self.device)
        
    def forward(self, state, action):
        # print(f"critic state input: {state}")
        state = self.flattener(state)
        # print(f"critic state input: {state}")
        # print("Critic forward")
        # print(state.shape)
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = torch.nn.functional.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        
        # print("State value")
        # print(state_value)
        # print(state_value.shape)
        # print(f"\nCritic Action:\n{action}\n")
        # print(action)
        action = action.view(-1, self.n_actions)
        # print(f"\nCritic Action:\n{action}\n")
        # print(action.shape)
        
        
        action_value_flag = self.action_value(action)
        # print("Ici")
        action_value = torch.nn.functional.relu(action_value_flag)
        state_action_value = torch.nn.functional.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
        
        # print(f"\nState Value:\n{state_value}\n{state_value.shape}\n")
        # print(f"\nAction Value: \n{action_value}\n{action_value.shape}\n")
        # print(f"\nState Action Value:\n{state_action_value}\n{state_action_value.shape}\n")
        
        # print(state_action_value)
        # print(state_action_value.shape)
        
        return state_action_value.squeeze()
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
        
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, name, fc1_dims = 400, fc2_dims = 300, 
                chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()

        # print(f"actor input dims: {input_dims}")
        self.flattener = nn.Flatten() #added
        # print(f"actor input dims: {input_dims}")
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'ddpg')

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc1=nn.Linear(self.input_dims, self.fc1_dims)
        
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        # f1 = 0.003
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        # f2 = 0.003
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        f3 = 0.003
        # print(f"\nfc2dims: {fc2_dims}, n_actions: {n_actions}\n")

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr =alpha)
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda:0')
        self.to(self.device)
        
    def forward(self, state):
        # print(f"actor state input: {state}")
        # print(f"actor state input shape: {state.shape}")
        
        x = self.flattener(state)
        # print(f"actor state input: {x}")
        # assert False
        # print(f"after flatten shape: {x.shape}")
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = torch.tanh(self.mu(x))
        x = x.squeeze()
        # print(f"actor op: {x}")
        # assert False
        return x
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
        
class DDPG_Agent(AC_Agent):
    """
    Represents a DDPG-learning agent that may choose to incorporate the current automaton state into its q-values
    """
    
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.input_shape = input_shape
        self.name = "DDPG Agent"
        print(f"input shape: {input_shape}")
        # print(f"\nnum_actions: \n{num_actions}\n")

        # assert False
        # self.feature_extractor = FeatureExtractor(input_shape=input_shape)
        self.num_actions = num_actions
        # print("DDPG Agent num actions")
        # print(num_actions)
        # self.half_feat_extractor_output_size = self.feature_extractor.output_size // 2
        # self.val_branch = nn.Linear(self.half_feat_extractor_output_size, 1)
        # self.adv_branch = nn.Linear(self.half_feat_extractor_output_size, self.num_actions)
        self.flattener = nn.Flatten()
        # self.num_actions = num_actions
        
        
        # self.actor = ActorNetwork(input_dims, n_actions=num_actions, name = 'Actor')
        # self.target_actor = ActorNetwork(input_dims, n_actions=num_actions, name = 'TargetActor')        
        # self.noise = OUActionNoise(mu=np.zeros(num_actions))
        self.actor = ActorNetwork(alpha=0.00025, input_dims = np.prod(input_shape), n_actions=self.num_actions, name = 'Actor')
        # self.actor = ActorNetwork(alpha=0.000025, input_dims = np.prod(input_shape), n_actions=num_actions, name = 'Actor')
        # self.target_actor = ActorNetwork(alpha=0.000025, input_dims = np.prod(input_shape), n_actions=num_actions, name = 'TargetActor')
        # self.noise = OUActionNoise(mu=np.zeros(num_actions))
        self.noise = OUActionNoise(mu=np.zeros(1))
        self.device = torch.device('cuda:0')


        
        # self.gamma = gamma
        # self.tau = tau
        # self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        # self.batch_size = batch_size
        #
        # self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Actor')
        # self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,layer2_size, n_actions=n_actions, name='TargetActor')        
        self.critic = CriticNetwork(beta=0.00025, input_dims = np.prod(input_shape), fc1_dims= 400,fc2_dims=300, n_actions=self.num_actions, name='Critic')
        # self.target_critic = CriticNetwork(beta, input_dims, layer1_size,layer2_size, n_actions=n_actions, name='TargetCritic')
        # self.noise = OUActionNoise(mu=np.zeros(n_actions))
        #
        # self.update_network_parameters(tau=1)
        #

    @classmethod
    def create_agent(cls, input_shape: Tuple, num_automaton_states: int, num_actions: int) -> "AC_Agent":
        return cls(input_shape, num_actions)
    
    def choose_action(self, observation: torch.Tensor, automaton_states: torch.Tensor) -> torch.tensor:
        
        self.actor.eval()
        # print(f"Continuous Observation / DDPG Input: {observation}")
        # assert False
        mu = self.actor.forward(observation).to(self.actor.device)
        # print(f"\nmu: {mu}\n")
        # assert False

        # noise = torch.tensor(self.noise(),dtype=torch.float).to(self.actor.device)
        noise = torch.tensor(np.random.normal(scale=0.3, size=1)).to(self.actor.device)
        mu_prime = mu + noise
        # print(f"Mu: {mu}, \nNoise: {noise}")
        # print(f"Forward Action: {mu_prime}")
        # print(f"Mu Prime: {mu_prime}")

        self.actor.train()
        action = mu_prime.cpu().detach().numpy()
        # print(f"\naction: {action}\n")
        # return action.flatten()
        return action
    
    # def remember(self, state, action, reward, new_state, done):
    #     self.memory.store_transition(state, action, reward, new_state, done)
     
    def forward(self, obs):
        print("DDPG forward taken")
        assert False
        # print()
        features = self.feature_extractor(obs)
        val_stream, adv_stream = torch.split(features, self.half_feat_extractor_output_size, dim=1)

        # Val and adv are because of dueling Q-network architecture, not actor critic
        val_stream = self.flattener(val_stream)
        adv_stream = self.flattener(adv_stream)

        val = self.val_branch(val_stream)
        adv = self.adv_branch(adv_stream)
        mean_adv = adv.mean(dim=1)

        q_vals = val + (adv - mean_adv.unsqueeze(1))  # Unsqueeze necessary so that it broadcasts correctly
        return q_vals

    def calc_q_values_single(self, observation: torch.Tensor, automaton_state: int) -> torch.Tensor:
        """
        Calculate the q values for a single sample
        Default implementation just calls calc_q_values_batch
        """
        return self.calc_q_values_batch(observation.unsqueeze(0), torch.as_tensor([automaton_state], dtype=torch.long,
                                                                                  device=observation.device)).view(
            (-1,))

    def calc_q_values_batch(self, observation: torch.Tensor, automaton_states: torch.Tensor) -> torch.Tensor:
        return self(observation)
    
    def calc_v_values_batch(self, observation: torch.Tensor, automaton_state: torch.Tensor) -> torch.Tensor:
        return self.calc_q_values_batch(observation, automaton_state).amax(dim=-1)

    def create_target_agent(self) -> "AC_TargetAgent":
        # print(f"self.adv_branch.weight.device: {self.adv_branch.weight.device}")
        # assert False
        # return AC_EasyTargetAgent(self, DDPG_Agent(self.input_shape, self.num_actions)).to(
        #     self.adv_branch.weight.device)
        return AC_EasyTargetAgent(self, DDPG_Agent(self.input_shape, self.num_actions)).to(self.device)
    
        