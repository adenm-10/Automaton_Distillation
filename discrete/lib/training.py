import copy
import time
from typing import Tuple, Dict, List
from datetime import datetime
import os
import pickle
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Union
import math

from discrete.lib.agent.agent import Agent, TargetAgent
from discrete.lib.agent.AC_Agent import AC_Agent, AC_TargetAgent
from discrete.lib.automaton.ap_extractor import APExtractor
from discrete.lib.automaton.automaton import Automaton
from discrete.lib.automaton.target_automaton import TargetAutomaton
from discrete.lib.automaton.reward_machine import RewardMachine
from discrete.lib.checkpoint import save_checkpoint, Checkpoint
from discrete.lib.config import Configuration
from discrete.lib.create_training_state import create_training_state
from discrete.lib.env.util import make_vec_env, make_env
from discrete.lib.intrinsic_reward import IntrinsicRewardCalculatorBatchWrapper
from discrete.lib.rollout_buffer import VecRolloutBufferHelper, RolloutBuffer, CircularRolloutBuffer
from discrete.lib.updater import Updater
from discrete.lib.agent.DDPG_Agent import DDPG_Agent
from discrete.run.utils import get_wasserstein, get_kl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

curr_crit_loss = None 
curr_act_loss = None

top_loss = [(0, 0, 0, 0, 0, 0, 0)] * 10

def DQN_learn(config: Configuration, optim: Optimizer, agent: Agent, target_agent: TargetAgent,
          rollout_buffer: RolloutBuffer, automaton: Automaton, logger: SummaryWriter, iter_num: int, reward_machine: RewardMachine = None):
    """
	Perform double Q-network gradient descent on a batch of samples from the rollout buffer (from deepsynth)
	"""
    optim.zero_grad()
    
    # if isinstance(automaton, RewardMachine):
    #     reward_machine = automaton
    # else:
    #     reward_machine = rm
    
    rollout_sample, indices, importance = rollout_buffer.sample(config.agent_train_batch_size,
                                                                automaton.num_states,
                                                                priority_scale=config.rollout_buffer_config.priority_scale,
                                                                reward_machine=reward_machine)

    importance = torch.pow(importance, 1 - config.epsilon)  # So that high-priority states aren't _too_ overrepresented
    
    # Estimate best action in new states using main Q network
    q_max = agent.calc_q_values_batch(rollout_sample.next_states, rollout_sample.next_aut_states)
    arg_q_max = torch.argmax(q_max, dim=1)
    
    # print("arg q max")
    # print(arg_q_max)
    # print(arg_q_max.shape)

    # Target DQN estimates q-values
    future_q_values = target_agent.calc_q_values_batch(rollout_sample.next_states, rollout_sample.next_aut_states)
    double_q = future_q_values[range(config.agent_train_batch_size), arg_q_max]
    
    # Calculate targets (bellman equation)
    target_q = rollout_sample.rewards + (config.gamma * double_q * (~rollout_sample.dones).float())
    target_q = target_q.detach()

    if isinstance(automaton, TargetAutomaton):
        # Q_teacher
        target_automaton_q = automaton.target_q_values(rollout_sample.aut_states, rollout_sample.aps, iter_num)
        # Beta
        target_automaton_q_weights = automaton.target_q_weights(rollout_sample.aut_states, rollout_sample.aps, iter_num)
        
        target_q = (target_automaton_q * target_automaton_q_weights) + (target_q * (1 - target_automaton_q_weights))

        

        automaton.update_training_observed_count(rollout_sample.aut_states, rollout_sample.aps)

    # What are the q-values that the current agent predicts for the actions it took
    q_values = agent.calc_q_values_batch(rollout_sample.states, rollout_sample.aut_states)
    
    # ADEN WAS HERE
    # action_q_values = q_values[range(config.agent_train_batch_size), rollout_sample.actions]
    # print(f"Rollout Sample Actions: \n{rollout_sample.actions}\n\nShape: \n{rollout_sample.actions.shape}")
    action_q_values = q_values[range(config.agent_train_batch_size), (rollout_sample.actions).long()]

    # Sample q values that we get wrong more often
    error = action_q_values - target_q
    rollout_buffer.set_priorities(indices=indices, errors=error.detach())

    # Actually train the neural network
    loss = F.mse_loss(input=action_q_values, target=target_q, reduction='none')
    loss = (loss * importance).mean()
    loss.backward()

    logger.add_scalar("training/loss", float(loss), global_step=iter_num)

    optim.step()

    return float(loss)
    
def DDPG_learn(config: Configuration, actor_optim: Optimizer, critic_optim: Optimizer, agent: AC_Agent, target_agent: AC_TargetAgent,
          rollout_buffer: RolloutBuffer, automaton: Automaton, logger: SummaryWriter, iter_num: int, reward_machine: RewardMachine = None):
    """
    Perform AC gradient descent on a batch of samples from the rollout buffer (from deepsynth)
    """
    
    # print("AC learn")

    target_agent.target.actor.eval()
    target_agent.target.critic.eval()
    agent.critic.eval()
    
    rollout_sample, indices, importance = rollout_buffer.sample(config.agent_train_batch_size,
                                                                automaton.num_states,
                                                                priority_scale=config.rollout_buffer_config.priority_scale,
                                                                reward_machine=reward_machine)
    
    # importance = torch.pow(importance, 1 - config.epsilon)  # So that high-priority states aren't to overrepresented

    states = rollout_sample.states
    next_states = rollout_sample.next_states
    rewards = rollout_sample.rewards
    actions = rollout_sample.actions
    dones = rollout_sample.dones
    
    target_actions = target_agent.target.actor.forward(next_states)
    
    # Q'
    target_critic_value = target_agent.target.critic.forward(next_states, target_actions)
    # print('target_critic_value')
    # print(target_critic_value)
    
    # Q
    critic_value = agent.critic.forward(states, actions)
    # print(critic_value)
    # print(critic_value.size())
    
    # print('critic_value')
    # print(critic_value)
    
    target_q = []

    for j in range(config.agent_train_batch_size):
        target_q.append(rewards[j] + config.gamma*target_critic_value[j]*dones[j])
        # target_q.append(rewards[j] + config.gamma*target_critic_value[j])
    target_q = torch.tensor(target_q).to(config.device)
    target_q = target_q.view(config.agent_train_batch_size, 1)
    target_q = target_q.squeeze()

    if isinstance(automaton, TargetAutomaton):
        # print("Automaton Distillation to Affecting Target Q Value\n\n\n")
        # Q_teacher
        target_automaton_q = automaton.target_q_values(rollout_sample.aut_states, rollout_sample.aps, iter_num)

        # print(f"Aut States: {rollout_sample.aut_states}")
        # print(f"APS': {rollout_sample.aps}")
        # print(f"target q automaton: {target_automaton_q}")

        # Beta
        target_automaton_q_weights = automaton.target_q_weights(rollout_sample.aut_states, rollout_sample.aps, iter_num)
        
        # print('target_automaton_q')
        # print(target_automaton_q.shape)
        
        target_q = (target_automaton_q * target_automaton_q_weights) + (target_q * (1 - target_automaton_q_weights))

        # print(f"Target_Q: {target_q}")

        automaton.update_training_observed_count(rollout_sample.aut_states, rollout_sample.aps)

    # print(f"States: {states}")
    # print(f"Actions: {actions}")

    
    agent.critic.train()
    critic_optim.zero_grad()
    critic_loss = F.mse_loss(target_q, critic_value)
    # print(f"Critic Loss: {critic_loss}")
    curr_crit_loss = critic_loss
    critic_loss.backward()
    critic_optim.step()
    
    agent.critic.eval()
    actor_optim.zero_grad()
    mu = agent.actor.forward(states)
    agent.actor.train()
    actor_loss = -agent.critic.forward(states,mu)
    actor_loss = torch.mean(actor_loss)
    # print(f"Actor Loss: {actor_loss}\n\n")
    curr_act_loss = actor_loss
    actor_loss.backward()
    actor_optim.step()

    # for i in range(len(top_loss)):
    #     if critic_loss.item() + actor_loss.item() > top_loss[i][0]:
    #         top_loss[i] = (curr_crit_loss.item() + curr_act_loss.item(), curr_crit_loss, curr_act_loss, states, actions, next_states, rewards)

    # What are the q-values that the current agent predicts for the actions it took
    # q_values = agent.calc_q_values_batch(rollout_sample.states, rollout_sample.aut_states)
    # action_q_values = q_values[range(config.agent_train_batch_size), rollout_sample.actions]
    
    # Sample q values that we get wrong more often
    # print(f"\n\ntarget q: {target_q}\n\n")
    # print(f"\n\critic_value: {critic_value}\n\n")
    # assert False

    error = critic_value - target_q
    rollout_buffer.set_priorities(indices=indices, errors=error.detach())

    # Actually train the neural network
    # loss = F.mse_loss(input=action_q_values, target=target_q, reduction='none')
    # loss = (loss * importance).mean()
    # loss.backward()

    # print(f"Critic loss: {float(critic_loss)}")
    # print(f"Actor  loss: {float(actor_loss)}")

    # logger.add_scalar("training / actor loss", float(actor_loss), global_step=iter_num)
    logger.add_scalar("training / critic loss", float(critic_loss), global_step=iter_num)

    # optim.step()

    return float(critic_loss)

def TD3_learn(config: Configuration, actor_optim: Optimizer, critic_optim: Optimizer, agent: AC_Agent, target_agent: AC_TargetAgent,
          rollout_buffer: RolloutBuffer, automaton: Automaton, logger: SummaryWriter, iter_num: int, reward_machine: RewardMachine = None):
    """
    Perform AC gradient descent on a batch of samples from the rollout buffer (from deepsynth)
    """

    cur_timestep = iter_num
    
    # print("AC learn")

    target_agent.target.actor.eval()
    target_agent.target.critic.eval()
    agent.critic.eval()
    
    rollout_sample, indices, importance = rollout_buffer.sample(config.agent_train_batch_size,
                                                                automaton.num_states,
                                                                priority_scale=config.rollout_buffer_config.priority_scale,
                                                                reward_machine=reward_machine)
    
    # importance = torch.pow(importance, 1 - config.epsilon)  # So that high-priority states aren't to overrepresented

    states = rollout_sample.states
    next_states = rollout_sample.next_states
    rewards = rollout_sample.rewards
    actions = rollout_sample.actions
    dones = rollout_sample.dones

    
    with torch.no_grad():
        target_actions = target_agent.target.actor.forward(next_states)

        # Add noise to the actions for target smoothing
        noise = agent.noise(target_actions)
        
        target_actions = (target_actions + noise).clamp(-agent.max_action, agent.max_action)
        
            
        # Q'
        target_critic_1_value, target_critic_2_value = target_agent.target.critic.forward(next_states, target_actions)
        
        # y = r + gamma * min(q1, q2)
        target_q = []
        for j in range(config.agent_train_batch_size):
           
            target_q.append(rewards[j] + config.gamma * min(target_critic_1_value[j], target_critic_2_value[j]) * (~dones[j]).float())
            # target_q.append(rewards[j] + config.gamma*target_critic_value[j])
        target_q = torch.tensor(target_q).to(config.device)
        target_q = target_q.view(config.agent_train_batch_size, 1)
        target_q = target_q.squeeze()
        
        

        if isinstance(automaton, TargetAutomaton):
            # Q_teacher
            target_automaton_q = automaton.target_q_values(rollout_sample.aut_states, rollout_sample.aps, iter_num)

            # Beta
            target_automaton_q_weights = automaton.target_q_weights(rollout_sample.aut_states, rollout_sample.aps, iter_num)
            
            target_q = (target_automaton_q * target_automaton_q_weights) + (target_q * (1 - target_automaton_q_weights))

            automaton.update_training_observed_count(rollout_sample.aut_states, rollout_sample.aps)
 
    # Q
    critic_1_value, critic_2_value = agent.critic.forward(states, actions)

    agent.critic.train()
    critic_loss = F.mse_loss(target_q, critic_1_value) + F.mse_loss(target_q, critic_2_value)

    logger.add_scalar("Target_Q", float(target_q.float().mean()), global_step=iter_num)


    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    agent.critic.eval()

    # Delayed Target Update
    # if t mod d:
    #   Update Actor weights
    #   Update Both Critic Target Weights
    #   Update Actor Target Weights

    actor_loss = 0

    if (cur_timestep + 1) % agent.d == 0:

        mu = agent.actor.forward(states)

        # Only want Q1 prediction
        actor_loss, _ = agent.critic.forward(states, mu)
        actor_loss = torch.mean(-actor_loss)

        agent.actor.train()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()
        agent.actor.eval()

        target_agent.update_weights()
    

    error = critic_1_value - target_q
    rollout_buffer.set_priorities(indices=indices, errors=error.detach())

    logger.add_scalar("training / actor loss", float(actor_loss), global_step=iter_num)

    return float(critic_loss)

def Policy_Distill_learn(student_config: Configuration, teacher_config: Configuration, optim: Optimizer, student_agent: Agent, teacher_rollout_buffer: Agent,
                         logger: SummaryWriter, iter_num: int, current_aut_states,
                         loss_metric='kl', reward_machine: RewardMachine = None):
    
    optim.zero_grad()
    
    ##################
    # batch = random.sample(expert_data, self.training_batch_size)
    # from the teacher ?
    rollout_sample, indices, _ = teacher_rollout_buffer.sample(teacher_config.agent_train_batch_size,
                                                                # automaton.num_states,
                                                                priority_scale=teacher_config.rollout_buffer_config.priority_scale,
                                                                reward_machine=reward_machine)
    
    

    # print(batch[0])
    # states = torch.stack([x[0] for x in batch]) # states
    states = torch.stack([x for x in rollout_sample.states])
    # print(f"states shape: {states.shape}")

    # means_teacher = torch.stack([x[1] for x in batch]) # actions
    means_teacher = torch.stack([x for x in rollout_sample.actions])

    fake_std = torch.from_numpy(np.array([1e-6]*len(means_teacher))) # for deterministic
    
    # stds_teacher = torch.stack([fake_std for x in batch])
    stds_teacher = torch.stack([fake_std for _ in rollout_sample.actions])
    ####################

    # means_student = self.policy.mean_action(states)
    q_values = student_agent.calc_q_values_batch(torch.as_tensor(states, device=student_config.device, dtype=torch.float), rollout_sample.aut_states)
    means_student = take_eps_greedy_action_from_q_values(q_values, student_config.epsilon)
    means_student = torch.stack([torch.tensor(x) for x in means_student])

    # stds_student = self.policy.get_std(states)
    sigma = torch.tensor(0.5, requires_grad=True)  # Replace with your desired scalar value
    scale = torch.exp(torch.clamp(sigma, min=math.log(1e-6)))
    stds_student = torch.stack([scale for _ in rollout_sample.states])

    if loss_metric == 'kl':
        loss = torch.tensor((stds_teacher.log() - stds_student.log() + (stds_student.pow(2) + (means_teacher - means_student).pow(2)) / (2 * stds_student.pow(2)) - 0.5).mean(), requires_grad=True)
    elif loss_metric == 'wasserstein':
        loss = get_wasserstein([means_teacher, stds_teacher], [means_student, stds_student])

    
    loss.backward()
    optim.step()

    logger.add_scalar("training/loss", float(loss), global_step=iter_num)

    return loss

def distill(config: Configuration, optim: Optimizer, teacher: Agent, student: Agent,
            rollout_buffer: RolloutBuffer, automaton: Automaton, logger: SummaryWriter, iter_num: int):
    """
    Perform policy distillation on a batch of samples from the rollout buffer
    """
    optim.zero_grad()

    rollout_sample, indices, importance = rollout_buffer.sample(config.agent_train_batch_size,
                                                                automaton.num_states,
                                                                priority_scale=config.rollout_buffer_config.priority_scale)
    
    # Teacher q-values
    teacher_q_values = teacher.calc_q_values_batch(rollout_sample.states, rollout_sample.aut_states)
    teacher_q_values_softmax = F.log_softmax(teacher_q_values / config.temperature, dim=1)
    
    # Student q-values
    student_q_values = student.calc_q_values_batch(rollout_sample.states, rollout_sample.aut_states)
    student_q_values_softmax = F.log_softmax(student_q_values / config.temperature, dim=1)

    # Train student
    loss = F.kl_div(input=student_q_values_softmax, target=teacher_q_values_softmax, log_target=True, reduction='batchmean')
    loss.backward()

    logger.add_scalar("training/loss", float(loss), global_step=iter_num)

    optim.step()

def take_eps_greedy_action_from_q_values(q_values: torch.Tensor, epsilon: float) -> np.ndarray:
    num_actions = q_values.shape[1]
    greedy_actions = torch.argmax(q_values, dim=1)
    modified_actions = torch.where(torch.rand_like(greedy_actions, dtype=torch.float32) > epsilon, greedy_actions,
                                   torch.randint_like(greedy_actions, num_actions))
    return modified_actions.detach().cpu().numpy()

def vec_env_distinct_episodes(states: torch.Tensor, infos: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
	The main annoyance of vecenv is that it automatically resets the environment after encountering a done
	The last observation is buried in the info dict for the vec.
	This function produces a vector of states that represent the step after the previous states,
	and a separate vector of states that represent the input to the next step
	"""
    states_after_current = states.clone()
    for i, info in enumerate(infos):
        if "terminal_observation" in info:
            states_after_current[i] = torch.as_tensor(info["terminal_observation"], device=states.device)

    return states_after_current, states

def reset_done_aut_states(aut_states_after_previous: torch.Tensor, dones: torch.Tensor, automaton: Automaton, device: str) -> torch.tensor:
    """
	Reset the automaton state for resetted environments
	:param aut_states_after_previous: The automaton state of the environment, possibly of the terminal state
	:param dones: Which are actually terminal states
	"""

    """
	The orginal code is as follows but it gets error on my computer so I modified the code.
	return torch.where(dones, automaton.default_state, aut_states_after_previous)
	"""



    a = automaton.default_state
    a = np.int64(a)
    b = torch.tensor(aut_states_after_previous.cpu().numpy(), dtype=torch.int64, device=device)

    return torch.where(dones, a, b)

class TraceHelper:
    """
	Keep track of all AP traces that haven't yet been used for synthesis- mostly an abstraction for vecenv
	"""

    def __init__(self, num_vec_envs: int):
        self.num_vec_envs = num_vec_envs
        self.completed_traces = []
        self.in_progress_traces = [[] for _ in range(num_vec_envs)]
        self.next_step = None  # Need to keep track of the most recent step separately

    def add_aps(self, aps):
        assert self.next_step is None, "Must finalize step before adding APs again"
        self.next_step = aps.tolist()

    def finalize_step(self, dones):
        # See note in train_agent about add_aps vs finalize_step.
        # Short version is that we want a way to recalculate the current state (without including the last AP) if the
        # automaton changes, but include the last AP when updating the automaton
        for i in range(len(dones)):
            self.in_progress_traces[i].append(int(self.next_step[i]))
            if dones[i]:
                self.completed_traces.append(self.in_progress_traces[i])
                self.in_progress_traces[i] = []

        self.next_step = None

    def get_traces_and_clear_completed(self):
        ret_traces = self.completed_traces
        self.completed_traces = []
        in_progress_traces_incl_next = copy.deepcopy(self.in_progress_traces)

        if self.next_step is not None:
            for i, in_progress_trace in enumerate(in_progress_traces_incl_next):
                in_progress_trace.append(self.next_step[i])

        ret_traces.extend(in_progress_traces_incl_next)
        ret_traces = [ret_trace for ret_trace in ret_traces if len(ret_trace) > 0]
        return ret_traces

def distill_agent(config: Configuration,
                  teacher: Agent,
                  student: Agent,
                  automaton: Automaton,
                  ap_extractor: APExtractor,
                  teacher_buffer: RolloutBuffer,
                  student_buffer: RolloutBuffer,
                  logger: SummaryWriter,
                  start_iter_num: int) -> Agent:
    """
    Distill knowledge from a teacher to a student
    :param teacher: The teacher agent for policy distillation
    :param student: The student agent for policy distillation
    :param config: Configuration for the whole training run
    :param automaton: The automaton to use during training. The states and transitions of the input will be updated
    :param ap_extractor: The weights of this will not be updated
    :param teacher_buffer: Teacher experience buffer
    :param student_buffer: Student experience buffer (only for logging purposes)
    :return: The trained agent
    """
    # TODO clarify ndarrays vs Tensors & devices
    env = make_vec_env(config.env_config, config.num_parallel_envs)
    
    buff_helper = VecRolloutBufferHelper(config.num_parallel_envs, student_buffer, logger,
                                         no_done_on_out_of_time=config.no_done_on_out_of_time)

    current_states = torch.as_tensor(env.reset(), device=config.device)
    current_aut_states = torch.tensor([automaton.default_state] * config.num_parallel_envs,
                                      device=config.device, dtype=torch.long)

    optimizer = torch.optim.Adam(student.parameters())

    trace_helper = TraceHelper(config.num_parallel_envs)
    batch_intrins_rew_calculator = IntrinsicRewardCalculatorBatchWrapper(config.intrinsic_reward_calculator,
                                                                         device=config.device)
    batch_intrins_reward_state = batch_intrins_rew_calculator.create_state(config.num_parallel_envs)

    checkpoint_updater = Updater(lambda: save_checkpoint(config, Checkpoint(
        iter_num=i,
        ap_extractor_state=ap_extractor.state_dict(),
        automaton_state=automaton.state_dict(),
        rollout_buffer_state=student_buffer.state_dict(),
        agent_state=student.state_dict()
    )))

    for i in range(start_iter_num, config.max_training_steps):
        # Generate experience
        q_values = student.calc_q_values_batch(torch.as_tensor(current_states, device=config.device, dtype=torch.float),
                                             current_aut_states)
        actions = take_eps_greedy_action_from_q_values(q_values, config.epsilon)
        obs, rewards, dones, infos = env.step(actions)
        obs = torch.as_tensor(obs, device=config.device)
        rewards = torch.as_tensor(rewards, device=config.device)
        dones = torch.as_tensor(dones, device=config.device)
        states_after_current, next_states = vec_env_distinct_episodes(obs, infos)

        aps_after_current = ap_extractor.extract_aps_batch(states_after_current, infos)

        # If dfa_updater changes the automaton, we need to recalculate the current automaton state
        # Since aps_after_current shouldn't be included in this calculation, trace_helper is "two-phase"
        # First, we add the aps to a special staging area where if the current automaton state must be recalculated,
        # these aps aren't included. Then, the finalize_step call merges in the newest aps.
        # The new aps are still included in the traces for the purposes of automaton synthesis.
        trace_helper.add_aps(aps_after_current)

        aut_states_after_current = automaton.step_batch(current_aut_states, aps_after_current)
        assert aut_states_after_current.min() != -1, "Automaton stepping failed"
        
        if isinstance(automaton, TargetAutomaton):
            rewards += automaton.target_reward_shaping(current_aut_states, aut_states_after_current)
        
        trace_helper.finalize_step(dones)

        intr_rewards = batch_intrins_rew_calculator.calc_intr_rewards_batch(batch_intrins_reward_state,
                                                                            current_states,
                                                                            actions,
                                                                            states_after_current,
                                                                            rewards,
                                                                            dones,
                                                                            current_aut_states,
                                                                            aps_after_current,
                                                                            aut_states_after_current)

        next_aut_states = reset_done_aut_states(aut_states_after_current, dones, automaton, device=config.device)
        
        # All of these are part of the same episode. next_states and next_aut_states may be part of a different episode
        buff_helper.add_vec_experiences(current_states=current_states,
                                        actions_after_current=actions,
                                        ext_rewards_after_current=rewards,
                                        intr_rewards_after_current=intr_rewards,
                                        dones_after_current=dones,
                                        states_after_current=states_after_current,
                                        current_aut_states=current_aut_states,
                                        aut_states_after_current=aut_states_after_current,
                                        aps_after_current=aps_after_current,
                                        infos=infos,
                                        global_step=i)

        current_states = next_states
        current_aut_states = next_aut_states

        logger.add_scalar("experience_generation/extrinsic_reward", float(rewards.float().mean()), global_step=i)
        logger.add_scalar("experience_generation/intrinsic_reward", float(intr_rewards.float().mean()), global_step=i)

        # Policy distillation
        distill(config=config, optim=optimizer, teacher=teacher, student=student, rollout_buffer=teacher_buffer,
                automaton=automaton, logger=logger, iter_num=i)

        checkpoint_updater.update_every(config.checkpoint_every_steps)

    return student

def moving_average(input_list, window_size=100):
    output_list = []
    sum_so_far = 0
    
    for i in range(len(input_list)):
        sum_so_far += input_list[i]
        if i >= window_size:
            sum_so_far -= input_list[i - window_size]
            output_list.append(sum_so_far / window_size)
        else:
            output_list.append(sum_so_far / (i + 1))
    
    return output_list

def train_agent(config: Configuration,
                agent: Union[Agent, AC_Agent],
                automaton: Automaton,
                ap_extractor: APExtractor,
                rollout_buffer: RolloutBuffer,
                logger: SummaryWriter,
                start_iter_num: int,
                reward_machine: RewardMachine = None,
                run_name=None,
                teacher_rollout_buffer: RolloutBuffer=None,
                policy_distill_teacher_config: Configuration=None,
                ) -> Union[Agent,AC_Agent]:
    """
	Train the agent for an entire generation
	:param agent: The agent to train
	:param config: Configuration for the whole training run
	:param automaton: The automaton to use during training. The states and transitions of the input will be updated
	:param ap_extractor: The weights of this will not be updated
	:param rollout_buffer: Assumed to already be labeled with the correct automaton states and intrinsic rewards, if any states are present
    :param reward_machine: The reward machine used for CRM, if any
	:return: The trained agent
	"""
    
    # Setup logging
    path_out = "./logger"  # You can customize this
    logged = setup_logger(path_out)
    logged.info("Starting training process...")

    if run_name is not None:
        config = config._replace(run_name=run_name)

    # TODO clarify ndarrays vs Tensors & devices
    # env = make_vec_env(config.env_config, config.num_parallel_envs)
    env = make_vec_env(config.env_config, config.num_parallel_envs)

    buff_helper = VecRolloutBufferHelper(config.num_parallel_envs, rollout_buffer, logger,
                                         no_done_on_out_of_time=config.no_done_on_out_of_time)

    # Create the target agent for the AC Agent Architecture
    target_agent = agent.create_target_agent()
    if isinstance(agent, AC_Agent):
      target_agent = agent.create_target_agent(tau=config.tau)

    # Initialize current environment and automaton states
    current_states = torch.as_tensor(env.reset(), device=config.device)
    current_aut_states = torch.tensor([automaton.default_state] * config.num_parallel_envs,
                                      device=config.device, dtype=torch.long)

    # Set optimizer(s) parameters
    if isinstance(agent, AC_Agent):
        actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr = config.actor_lr)
        critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr = config.critic_lr)
    else:
        optimizer = torch.optim.Adam(agent.parameters())

    trace_helper = TraceHelper(config.num_parallel_envs)
    batch_intrins_rew_calculator = IntrinsicRewardCalculatorBatchWrapper(config.intrinsic_reward_calculator,
                                                                         device=config.device)
    batch_intrins_reward_state = batch_intrins_rew_calculator.create_state(config.num_parallel_envs)

    # The next few functions keep the main training loop concise by moving out some counting tasks
    target_agent_updater = Updater(lambda: target_agent.update_weights())

    checkpoint_updater = Updater(lambda: save_checkpoint(config, Checkpoint(
        iter_num=i,
        ap_extractor_state=ap_extractor.state_dict(),
        automaton_state=automaton.state_dict(),
        rollout_buffer_state=rollout_buffer.state_dict(),
        agent_state=agent.state_dict()
    )))

    ep_counter = 0
    training_iterations = []

    rewards_per_step,       rewards_per_episode,       rewards_per_ep_current    = [], [], [0]*config.num_parallel_envs
    steps_to_term_per_step, steps_to_term_per_episode, steps_to_terminal_current = [], [], [0]*config.num_parallel_envs

    reward_mav, steps_mav = [], []
    
    # Decaying exploration noise parameters
    expl_noise = 0.3
    action_shape = env.action_space.shape
    max_action = 1.0

    path_to_out = ""
    if config.path_to_out:
        path_to_out = config.path_to_out
    else:
        now = datetime.now().strftime("%m-%d_%H-%M-%S")
        dirname = os.path.dirname(__file__)
        hard_path = f"./test_output/test_output_{now}"

        if isinstance(agent, AC_Agent):
            hard_path = hard_path + "_cont/"
        else:
            hard_path = hard_path + "_disc/"

        path_to_out = os.path.join(hard_path)
        
        try:
            os.mkdir(path_to_out)
        except:
            print("Output Directory Already Exists")
    # print(f"\n\nExporting Plots to:\n{path_to_out}\n\n")

    for i in range(start_iter_num, config.max_training_steps):
        # print(f"\nStep {i}")
    
        # (1) Choose action based off current state
        if isinstance(agent, AC_Agent):
            actions = agent.choose_action(torch.as_tensor(current_states, device=config.device, dtype=torch.float),
                                             current_aut_states) # DDPG doesnt use current aut states in action selection
	    actions = (actions + np.random.normal(0, max_action * expl_noise, size=action_shape)).clip(-max_action, max_action)
        else:
            q_values = agent.calc_q_values_batch(torch.as_tensor(current_states, device=config.device, dtype=torch.float),
                                             current_aut_states)
            actions = take_eps_greedy_action_from_q_values(q_values, config.epsilon)
        
            
        # (2) Take a step in the environment, process returned state, reward, and information
        obs, rewards, dones, infos = env.step(actions)
      
        # Graphing operations
        steps_to_terminal_current = [x+1 for x in steps_to_terminal_current]
        rewards_per_ep_current =    [x+r for x, r in zip(rewards_per_ep_current, rewards)]
        for index, done in enumerate(dones):
            if done:
                steps_to_term_per_episode.append(steps_to_terminal_current[index])
                steps_to_terminal_current[index] = 0

                rewards_per_episode.append(rewards_per_ep_current[index])
                rewards_per_ep_current[index] = 0

                ep_counter = ep_counter + 1

        infos_discrete = copy.deepcopy(infos)
        for info in infos_discrete:
            info['position'] = (int(info['position'][0]), int(info['position'][1]))

        obs = torch.as_tensor(obs, device=config.device)
        rewards = torch.as_tensor(rewards, device=config.device)
        dones = torch.as_tensor(dones, device=config.device)

        rewards_per_step.append(float(rewards.float().mean()))
        if len(steps_to_term_per_episode) > 0:
            steps_to_term_per_step.append(steps_to_term_per_episode[-1])
        else:
            steps_to_term_per_step.append(0)

        states_after_current, next_states = vec_env_distinct_episodes(obs, infos)
        logger.add_scalar("reward", float(rewards.float().mean()), global_step=i)
        #logged.info(f"Step {i}: Action taken, Reward: {rewards.mean():.3f}, Done: {dones}")


        aps_after_current = []
        if isinstance(agent, AC_Agent):
            aps_after_current = ap_extractor.extract_aps_batch(states_after_current, infos_discrete)
        else:
            aps_after_current = ap_extractor.extract_aps_batch(states_after_current, infos)

        # If dfa_updater changes the automaton, we need to recalculate the current automaton state
        # Since aps_after_current shouldn't be included in this calculation, trace_helper is "two-phase"
        # First, we add the aps to a special staging area where if the current automaton state must be recalculated,
        # these aps aren't included. Then, the finalize_step call merges in the newest aps.
        # The new aps are still included in the traces for the purposes of automaton synthesis.
        trace_helper.add_aps(aps_after_current)

        aut_states_after_current = automaton.step_batch(current_aut_states, aps_after_current)
        assert aut_states_after_current.min() != -1, "Automaton stepping failed"
        
        # Reward shaping
        if isinstance(automaton, TargetAutomaton):
            rewards += automaton.target_reward_shaping(current_aut_states, aut_states_after_current)

        trace_helper.finalize_step(dones)

        intr_rewards = batch_intrins_rew_calculator.calc_intr_rewards_batch(batch_intrins_reward_state,
                                                                            current_states,
                                                                            actions,
                                                                            states_after_current,
                                                                            rewards,
                                                                            dones,
                                                                            current_aut_states,
                                                                            aps_after_current,
                                                                            aut_states_after_current)

        next_aut_states = reset_done_aut_states(aut_states_after_current, dones, automaton, device=config.device)

        # All of these are part of the same episode. next_states and next_aut_states may be part of a different episode
        buff_helper.add_vec_experiences(current_states=current_states,
                                        actions_after_current=actions,
                                        ext_rewards_after_current=rewards,
                                        intr_rewards_after_current=intr_rewards,
                                        dones_after_current=dones,
                                        states_after_current=states_after_current,
                                        current_aut_states=current_aut_states,
                                        aut_states_after_current=aut_states_after_current,
                                        aps_after_current=aps_after_current,
                                        infos=infos,
                                        global_step=i)

        current_states = next_states
        current_aut_states = next_aut_states
        
        # (3) Update the agents to learn based if rollout buffer has enough experiences
        if rollout_buffer.num_filled_approx() >= config.rollout_buffer_config.min_size_before_training:
            # print("entered training block")
            # Train off-policy
            if teacher_rollout_buffer != None:
                loss = Policy_Distill_learn(student_config=config, teacher_config=policy_distill_teacher_config, optim=optimizer, student_agent=agent, 
                                            teacher_rollout_buffer=teacher_rollout_buffer, logger=logger, iter_num=i, current_aut_states=current_aut_states, loss_metric='kl', reward_machine=reward_machine)
            elif isinstance(agent, AC_Agent):
                loss = TD3_learn(config=config, actor_optim=actor_optimizer, critic_optim=critic_optimizer, agent=agent, target_agent=target_agent, rollout_buffer=rollout_buffer,
                automaton=automaton, logger=logger, iter_num=i, reward_machine=reward_machine)
            else:
                loss = DQN_learn(config=config, optim=optimizer, agent=agent, target_agent=target_agent, rollout_buffer=rollout_buffer,
                  automaton=automaton, logger=logger, iter_num=i, reward_machine=reward_machine)
            
            training_iterations.append(i)
            
            if agent.name != "TD3 Agent":
                target_agent_updater.update_every(config.target_agent_update_every_steps)
        checkpoint_updater.update_every(config.checkpoint_every_steps)

        # (4) Process returned data for live logging and postprocessing for visualization
        if i % 1000 == 0 and i != 0:
            reward_mav = moving_average(rewards_per_step)
            reward_ep_mav = moving_average(rewards_per_episode)
            steps_mav = moving_average(steps_to_term_per_episode)

            try:
                print(f"Completed Steps: {i:8} || Avg Steps: {int(steps_mav[-1]):4} || Avg Rew: {reward_ep_mav[-1]:.3f}")
                logged.info(f"Step {i}: Avg Reward: {reward_mav[-1]:.3f}, Episode Reward: {rewards_per_ep_current}")
            except:
                print(f"Completed Steps: {i:8}")

    plot_results(rewards_per_episode, steps_to_term_per_episode,
                 rewards_per_step, steps_to_term_per_step,
                 path_to_out, agent,
                 displayed_steps=None, displayed_episodes=None)

    plot_results(rewards_per_episode, steps_to_term_per_episode,
                 rewards_per_step, steps_to_term_per_step,
                 path_to_out, agent,
                 displayed_steps=100000, displayed_episodes=10000)

    export_results(rewards_per_episode, steps_to_term_per_episode,
                   rewards_per_step, steps_to_term_per_step,
                   path_to_out)

    return agent, rollout_buffer

def export_results(rewards_per_episode, steps_to_term_per_episode,
                   rewards_per_step, steps_to_term_per_step,
                   path_to_out):

    data_dict = {
        'rewards_per_episode': rewards_per_episode, 
        'steps_to_term_per_episode': steps_to_term_per_episode,
        'rewards_per_step': rewards_per_step, 
        'steps_to_term_per_step': steps_to_term_per_step
        }

    filepath = f"{path_to_out}/rew_and_steps_lists.pkl"
    with open(filepath, 'wb') as file:
        pickle.dump(data_dict, file)

    return   

def plot_results(rewards_per_episode, steps_to_term_per_episode,
                 rewards_per_step, steps_to_term_per_step, 
                 path_to_out, agent,
                 displayed_steps=None, displayed_episodes=None):

    def plot_details(iteration_list, raw_data, moving_average_data,  # Data specifics
                     blue_label, red_label, x_label, y_label,        # Labels
                     save_figure_name):                              # File save name

        plt.plot(iteration_list, raw_data,            color='blue', label=blue_label)
        plt.plot(iteration_list, moving_average_data, color='red',  label=red_label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="upper right")
        
        if isinstance(agent, AC_Agent):
            plt.savefig(f'{path_to_out}/Student_{save_figure_name}.png')
        else:
            plt.savefig(f'{path_to_out}/Teacher_{save_figure_name}.png')

        plt.clf()

    reward_mav = moving_average(rewards_per_step)
    reward_ep_mav = moving_average(rewards_per_episode)
    steps_mav = moving_average(steps_to_term_per_step)
    steps_ep_mav = moving_average(steps_to_term_per_episode)

    ep_filename, step_filename = "Full", "Full"

    if displayed_steps and displayed_steps < len(rewards_per_step) and displayed_steps < len(steps_to_term_per_step):
        reward_mav =             reward_mav[0:displayed_steps]
        steps_mav  =             steps_mav[0:displayed_steps]
        rewards_per_step =       rewards_per_step[0:displayed_steps]
        steps_to_term_per_step = steps_to_term_per_step[0:displayed_steps]

        step_filename = f"{displayed_steps}_Steps"

    if displayed_episodes and displayed_episodes < len(rewards_per_episode) and displayed_episodes < len(steps_to_term_per_episode):
        reward_ep_mav =             reward_ep_mav[0:displayed_episodes]
        steps_ep_mav  =             steps_ep_mav[0:displayed_episodes]
        rewards_per_episode =       rewards_per_episode[0:displayed_episodes]
        steps_to_term_per_episode = steps_to_term_per_episode[0:displayed_episodes]

        ep_filename   = f"{displayed_episodes}_Episodes"

    plt.clf()

    episode_iterations = [i for i in range(len(reward_ep_mav))]
    steps_iterations =   [i for i in range(len(reward_mav))]

    # Rewards per Episode
    plot_details(episode_iterations, rewards_per_episode, reward_ep_mav,
                 'Raw Rewards', 'Moving Average Rewards', 'Episodes', 'Rewards per Episode',
                 f"Reward_{ep_filename}")

    # Rewards per Step
    plot_details(steps_iterations, rewards_per_step, reward_mav,
                 'Raw Rewards', 'Moving Average Rewards', 'Timesteps', 'Rewards per Step',
                 f"Reward_{step_filename}")

    # Steps to Reach Terminal State per Episode
    plot_details(episode_iterations, steps_to_term_per_episode, steps_ep_mav,
                 'Raw Steps', 'Moving Average Steps', 'Episodes', 'Steps To Terminal State per Episode',
                 f"Steps_{ep_filename}")

    # Steps to Reach Terminal State per Step
    plot_details(steps_iterations, steps_to_term_per_step, steps_mav,
                 'Raw Steps', 'Moving Average Steps', 'Timesteps', 'Steps To Terminal State per Step',
                 f"Steps_{step_filename}")

    return


def setup_logger(path_to_out):
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)  # Set the logging level
    
    # Create formatter that matches the structure of the image output (timestamp, step, message)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Ensure the directory exists
    os.makedirs(path_to_out, exist_ok=True)

    # Set up the logger
    file_handler = logging.FileHandler(os.path.join(path_to_out, 'log.txt'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    
    
    # Optional: also log to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
