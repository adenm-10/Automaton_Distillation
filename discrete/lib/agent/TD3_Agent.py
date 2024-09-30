'''
Created on Feb 6, 2024

@author: diegobenalcazar
'''

import abc
from abc import ABC
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from discrete.lib.agent.AC_Agent import AC_Agent, AC_TargetAgent
from discrete.lib.agent.AC_easy_target_agent import AC_EasyTargetAgent
from discrete.lib.agent.feature_extractor import FeatureExtractor

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, 
                chkpt_dir='tmp/ddpg', device='cpu'):

        super(CriticNetwork, self).__init__()

        self.flattener = nn.Flatten()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'ddpg')

        ############
        # CRITIC 1 #
        ############

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        #f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        #torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        #torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.bn1 = nn.LayerNorm(self.fc1_dims)
        
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        #torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        #torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.action_1_value = nn.Linear(self.n_actions, self.fc2_dims)

        f3 = 0.003
        self.q1 = nn.Linear(fc2_dims, 1)
        #torch.nn.init.uniform_(self.q1.weight.data, -f3, f3)
        #torch.nn.init.uniform_(self.q1.bias.data, -f3, f3)

        ############
        # CRITIC 2 #
        ############

        self.fc3 = nn.Linear(self.input_dims, self.fc1_dims)
        #f4 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        #torch.nn.init.uniform_(self.fc3.weight.data, -f4, f4)
        #torch.nn.init.uniform_(self.fc3.bias.data, -f4, f4)
        #self.bn3 = nn.LayerNorm(self.fc1_dims)
        
        self.fc4 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #f5 = 1 / np.sqrt(self.fc4.weight.data.size()[0])
        #torch.nn.init.uniform_(self.fc4.weight.data, -f5, f5)
        #torch.nn.init.uniform_(self.fc4.bias.data, -f5, f5)
        #self.bn4 = nn.LayerNorm(self.fc2_dims)
        self.action_2_value = nn.Linear(self.n_actions, self.fc2_dims)

        #f6 = 0.003
        self.q2 = nn.Linear(fc2_dims, 1)
        #torch.nn.init.uniform_(self.q2.weight.data, -f6, f6)
        #torch.nn.init.uniform_(self.q2.bias.data, -f6, f6)
        
        self.device = device
        
        self.to(self.device)
        
    def forward(self, state, action):

        state = self.flattener(state)

        ############
        # CRITIC 1 #
        ############
        state_value = F.relu(self.fc1(state))
        #state_value = self.bn1(state_value)
        #state_value = torch.nn.functional.relu(state_value)
        state_value_s = self.fc2(state_value)
        #state_value = self.bn2(state_value)
        
        action = action.view(-1, self.n_actions)
          
        action_value_flag = self.action_1_value(action)
        s11 = torch.mm(state_value, self.fc2.weight.data.t())
        s12 = torch.mm(action, self.action_1_value.weight.data.t())
        #action_value = torch.nn.functional.relu(action_value_flag)
        state_action_value = F.relu(s11 + s12 + self.action_1_value.bias.data)
        q1 = self.q1(state_action_value)

        ############
        # CRITIC 2 #
        ############
        state_value1 = F.relu(self.fc3(state))
        #state_value = self.bn3(state_value)
        #state_value = torch.nn.functional.relu(state_value)
        state_value_s1 = self.fc4(state_value)
        #state_value = self.bn4(state_value)
        
        action = action.view(-1, self.n_actions)
          
        action_value_flag1 = self.action_2_value(action)
        #action_value = torch.nn.functional.relu(action_value_flag)
        s21 = torch.mm(state_value1, self.fc4.weight.data.t())
        s22 = torch.mm(action, self.action_2_value.weight.data.t())
        #state_action_value = torch.nn.functional.relu(torch.add(state_value, action_value))
        s2 = F.relu(s21 + s22 + self.action_2_value.bias.data)
        q2 = self.q2(s2)
        
        # print(f"\nState Value:\n{state_value}\n{state_value.shape}\n")
        # print(f"\nAction Value: \n{action_value}\n{action_value.shape}\n")
        # print(f"\nState Action Value:\n{state_action_value}\n{state_action_value.shape}\n")
        
        #q1.squeeze(), q2.squeeze()
        
        return q1.squeeze(), q2.squeeze()
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
        
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, name, fc1_dims = 800, fc2_dims = 600, 
                chkpt_dir='tmp/ddpg', device='cpu'):
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
        
        #f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        # f1 = 0.003
        #torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        #torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        #f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        # f2 = 0.003
        #torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        #torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        #f3 = 0.003
        # print(f"\nfc2dims: {fc2_dims}, n_actions: {n_actions}\n")

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        #torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        #torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)
        
        # self.optimizer = torch.optim.Adam(self.parameters(), lr =alpha)
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tanh = nn.Tanh()
        self.device = device
        self.to(self.device)
        
    def forward(self, state):
        # print(f"actor state input: {state}")
        # print(f"actor state input shape: {state.shape}")
        
        x = self.flattener(state)
        # print(f"after flatten shape: {x.shape}")
        x = self.fc1(x)
        #x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.relu(x)
        a = self.tanh(self.mu(x))
        #x = x.squeeze()

        return a
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
        
class TD3_Agent(AC_Agent):
    """
    Represents a TD3 agent that utilizes the integrated into the other tools chosen for the replay buffer and parallelization
    """
    
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.name = "TD3 Agent"
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.noise_clip = 0.5
        self.policy_noise_stddev = 0.2
        self.max_action = 1.0
        if os.getenv("NOISE_CLIP") is not None:
            self.noise_clip = float(os.getenv("NOISE_CLIP"))
            self.policy_noise_stddev = float(os.getenv("NOISE_STDDEV"))

        # Update Critic every (self.d) timesteps
        self.d = 2 # update timesteps
        if os.getenv("POLICY_FREQ") is not None:
            self.d = int(os.getenv("POLICY_FREQ"))

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')

        self.flattener = nn.Flatten()
        
        self.actor = ActorNetwork(alpha=0.005, input_dims = np.prod(input_shape), n_actions=self.num_actions, name = 'Actor', device=self.device)
        self.actor.to(self.device)

        self.critic = CriticNetwork(beta=0.005, input_dims = np.prod(input_shape), fc1_dims= 800,fc2_dims=600, n_actions=self.num_actions, name='Critic', device=self.device)
        self.critic.to(self.device)

    def noise(self, action):
        return (torch.randn_like(action) * self.policy_noise_stddev).clamp(-self.noise_clip, self.noise_clip)

    @classmethod
    def create_agent(cls, input_shape: Tuple, num_automaton_states: int, num_actions: int) -> "AC_Agent":
        return cls(input_shape, num_actions)
    
    def choose_action(self, observation: torch.Tensor, automaton_states: torch.Tensor) -> torch.tensor:

        self.actor.eval()
        mu = self.actor.forward(observation).to(self.actor.device)

        # noise = torch.tensor(self.noise(),dtype=torch.float).to(self.actor.device)
        noise = self.noise(mu)
        mu_prime = (mu + noise).clamp(-self.max_action, self.max_action)

        self.actor.train()
        action = mu_prime.cpu().detach().numpy()

        return action

    def calc_q_values_single(self, observation: torch.Tensor, automaton_state: int) -> torch.Tensor:
        """
        Calculate the q values for a single sample
        Default implementation just calls calc_q_values_batch
        """
        return self.calc_q_values_batch(observation.unsqueeze(0), torch.as_tensor([automaton_state], dtype=torch.long,
                                                                                  device=observation.device)).view((-1,))

    def calc_q_values_batch(self, observation: torch.Tensor, automaton_states: torch.Tensor) -> torch.Tensor:
        return self(observation)
    
    def calc_v_values_batch(self, observation: torch.Tensor, automaton_state: torch.Tensor) -> torch.Tensor:
        return self.calc_q_values_batch(observation, automaton_state).amax(dim=-1)

    def create_target_agent(self, tau=1) -> "AC_TargetAgent":
        return AC_EasyTargetAgent(self, TD3_Agent(self.input_shape, self.num_actions), tau=tau).to(self.device)
    
        