import copy
import time
from typing import Tuple, Dict, List
from datetime import datetime
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Union

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

curr_crit_loss = None 
curr_act_loss = None

top_loss = [(0, 0, 0, 0, 0, 0, 0)] * 10

def learn(config: Configuration, optim: Optimizer, agent: Agent, target_agent: TargetAgent,
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
    
    # print("states")
    # print(states)
    # print(states.shape)
    # print("actions")
    # print(actions)
    # print(actions.shape)
    # print("target_actions")
    # print(target_actions)
    # print(target_actions.shape)
    # print("next_states")
    # print(next_states)
    # print(next_states.shape)
    
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

    
    # print("target")
    # print(target_q.shape)
    
    #update actor parameters
    
    # if isinstance(automaton, RewardMachine):
    #     reward_machine = automaton
    # else:
    #     reward_machine = rm

    # Estimate best action in new states using main Q network
    # q_max = agent.calc_q_values_batch(rollout_sample.next_states, rollout_sample.next_aut_states)
    # arg_q_max = torch.argmax(q_max, dim=1)
    #
    # # Target DQN estimates q-values
    # future_q_values = target_agent.calc_q_values_batch(rollout_sample.next_states, rollout_sample.next_aut_states)
    # double_q = future_q_values[range(config.agent_train_batch_size), arg_q_max]
    #
    # # Calculate targets (bellman equation)
    # target_q = rollout_sample.rewards + (config.gamma * double_q * (~rollout_sample.dones).float())
    # target_q = target_q.detach()

    # print(f"Automaton type: {type(automaton)}\n\n")
    # print(f"Automaton: {automaton}\n\n")

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
        
        # Q'
        target_critic_1_value, target_critic_2_value = target_agent.target.critic.forward(next_states, target_actions)
        
        # y = r + gamma * min(q1, q2)
        target_q = []
        for j in range(config.agent_train_batch_size):
            target_q.append(rewards[j] + config.gamma * min(target_critic_1_value[j], target_critic_2_value[j]) * dones[j])
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
 
    # Q
    critic_1_value, critic_2_value = agent.critic.forward(states, actions)

    agent.critic.train()
    critic_loss = F.mse_loss(target_q, critic_1_value) + F.mse_loss(target_q, critic_2_value)

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
    
    # Sample q values that we get wrong more often
    # print(f"\n\ntarget q: {target_q}\n\n")
    # print(f"\n\critic_value: {critic_value}\n\n")
    # assert False

    error = critic_1_value - target_q
    rollout_buffer.set_priorities(indices=indices, errors=error.detach())

    # print(f"Critic loss: {float(critic_loss)}")
    # print(f"Actor  loss: {float(actor_loss)}")

    logger.add_scalar("training / actor loss", float(actor_loss), global_step=iter_num)

    return float(critic_loss)

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


def reset_done_aut_states(aut_states_after_previous: torch.Tensor, dones: torch.Tensor,
                          automaton: Automaton) -> torch.tensor:
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
    # print(dones)
    # print(a)
    b = torch.tensor(aut_states_after_previous.cpu().numpy(), dtype=torch.int64, device="cpu")
    
    #precious reverted to the old code

    return torch.where(dones, automaton.default_state, aut_states_after_previous)


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

        next_aut_states = reset_done_aut_states(aut_states_after_current, dones, automaton)
        
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
                run_name=None) -> Union[Agent,AC_Agent]:
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
    if run_name is not None:
        config = config._replace(run_name=run_name)

    # TODO clarify ndarrays vs Tensors & devices
    # env = make_vec_env(config.env_config, config.num_parallel_envs)
    env = make_vec_env(config.env_config, config.num_parallel_envs)

    buff_helper = VecRolloutBufferHelper(config.num_parallel_envs, rollout_buffer, logger,
                                         no_done_on_out_of_time=config.no_done_on_out_of_time)

    target_agent = agent.create_target_agent()

    if isinstance(agent, AC_Agent):
      target_agent = agent.create_target_agent(tau=config.tau)

    current_states = torch.as_tensor(env.reset(), device=config.device)
    current_aut_states = torch.tensor([automaton.default_state] * config.num_parallel_envs,
                                      device=config.device, dtype=torch.long)
    
    if isinstance(agent, AC_Agent):
        actor_lr = config.actor_lr
        critic_lr = config.critic_lr

        print(f"Actor Learning Rate: {actor_lr}\nCritic Learning Rate: {critic_lr}")

        actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr = actor_lr)
        critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr = critic_lr)

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

    training_iterations = []
    losses = []
    rewards_list = []
    rewards_per_ep = [0]*config.num_parallel_envs
    rewards_per_ep_list = []
    steps_to_terminal = [0]*config.num_parallel_envs
    steps_to_terminal_total = []
    
    loss_mav = []
    reward_mav = []
    steps_mav = []

    rewards_iterations = []

    for i in range(start_iter_num, config.max_training_steps):
        # print(f"\nStep {i}")
    
        # (1) Choose action based off current state
        if isinstance(agent, AC_Agent):
            actions = agent.choose_action(torch.as_tensor(current_states, device=config.device, dtype=torch.float),
                                             current_aut_states) # DDPG doesnt use current aut states in action selection
        else:
            q_values = agent.calc_q_values_batch(torch.as_tensor(current_states, device=config.device, dtype=torch.float),
                                             current_aut_states)
            actions = take_eps_greedy_action_from_q_values(q_values, config.epsilon)


        obs, rewards, dones, infos = env.step(actions)

        # Graphing operations
        steps_to_terminal = [x+1 for x in steps_to_terminal]
        rewards_per_ep =    [x+r for x, r in zip(rewards_per_ep, rewards)]
        for index, done in enumerate(dones):
            if done:
                steps_to_terminal_total.append(steps_to_terminal[index])
                steps_to_terminal[index] = 0

                rewards_per_ep_list.append(rewards_per_ep[index])
                rewards_per_ep[index] = 0

        infos_discrete = copy.deepcopy(infos)
        for info in infos_discrete:
            info['position'] = (int(info['position'][0]), int(info['position'][1]))

        rewards_iterations.append(i)

        obs = torch.as_tensor(obs, device=config.device)
        rewards = torch.as_tensor(rewards, device=config.device)
        rewards_list.append(float(rewards.float().mean()))
        dones = torch.as_tensor(dones, device=config.device)

        # obs = next_states
        states_after_current, next_states = vec_env_distinct_episodes(obs, infos)
        logger.add_scalar("reward", float(rewards.float().mean()), global_step=i)

        # print(f"Observations: \n{obs[0]}")
        # print(f"Infos Continuous: \n{infos[0]}") 
        # print(f"\nInfos Discrete: {infos_discrete}")
        # print(f"Rewards: {rewards[0]}")
        # print(f"Rewards List: {rewards_list}")
        # print(f"Dones: {dones}")
        # print(f"States after current: {states_after_current}")
        # print(f"next states: \n{next_states[0]}")
        # print(f"Action: {((actions[0] + 1) / 2) * 6.28}")

        # assert False

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

        next_aut_states = reset_done_aut_states(aut_states_after_current, dones, automaton)

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

        # logger.add_scalar("experience_generation/extrinsic_reward", float(rewards.float().mean()), global_step=i)
        # logger.add_scalar("experience_generation/intrinsic_reward", float(intr_rewards.float().mean()), global_step=i)

        # DONT KEEP THIS FOR DEBUG
        # config.rollout_buffer_config.min_size_before_training = 1

        # print(f"Steps before training: {config.rollout_buffer_config.min_size_before_training}")
        # print(f"Filled rollout buffer entries: {rollout_buffer.num_filled_approx()}")
        # assert False
        
        if rollout_buffer.num_filled_approx() >= config.rollout_buffer_config.min_size_before_training:
            # print("entered training block")
            # Train off-policy
            if isinstance(agent, AC_Agent):
                if agent.name == "DDPG Agent":
                    loss = DDPG_learn(config=config, actor_optim=actor_optimizer, critic_optim=critic_optimizer, agent=agent, target_agent=target_agent, rollout_buffer=rollout_buffer,
                    automaton=automaton, logger=logger, iter_num=i, reward_machine=reward_machine)
                else:
                    loss = TD3_learn(config=config, actor_optim=actor_optimizer, critic_optim=critic_optimizer, agent=agent, target_agent=target_agent, rollout_buffer=rollout_buffer,
                    automaton=automaton, logger=logger, iter_num=i, reward_machine=reward_machine)
                
                # print(f"DDPG Critic Loss: {loss}")
                losses.append(loss)
                training_iterations.append(i)
            else:
                loss = learn(config=config, optim=optimizer, agent=agent, target_agent=target_agent, rollout_buffer=rollout_buffer,
                  automaton=automaton, logger=logger, iter_num=i, reward_machine=reward_machine)
                # print(f"DQN Critic Loss: {loss}")
                losses.append(loss)
                training_iterations.append(i)
            
            if agent.name != "TD3 Agent":
                target_agent_updater.update_every(config.target_agent_update_every_steps)
        checkpoint_updater.update_every(config.checkpoint_every_steps)

        if i % 1000 == 0 and i != 0:
        
            loss_mav = moving_average(losses)
            reward_mav = moving_average(rewards_list)
            reward_ep_mav = moving_average(rewards_per_ep_list)
            steps_mav = moving_average(steps_to_terminal_total)

            print(f"Completed Steps: {i:8} || Avg Steps: {int(steps_mav[-1]):4} || Avg Rew: {reward_ep_mav[-1]:.3f}")

    # print("Top Losses")
    # for i in range(len(top_loss)):
    #     print(f"Cumulative Loss: \n{top_loss[i][0]}\n")
    #     print(f"Critic Loss: \n{top_loss[i][1]}\n")
    #     print(f"Actor Loss: \n{top_loss[i][2]}\n")
    #     print(f"States: \n{top_loss[i][3]}\n")
    #     print(f"Actions: \n{top_loss[i][4]}\n")
    #     print(f"Nest States: \n{top_loss[i][5]}\n")
    #     print(f"Rewards: \n{top_loss[i][6]}\n")
    #     print(f"=============================================================================")

    start_time = time.time()
    end_time = 0

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
    print(f"\n\nExporting Plots to:\n{path_to_out}\n\n")

    loss_mav = moving_average(losses)
    reward_mav = moving_average(rewards_list)
    reward_ep_mav = moving_average(rewards_per_ep_list)
    steps_mav = moving_average(steps_to_terminal_total)

    plt.plot(training_iterations, losses,   color='blue', label='Raw Losses')
    plt.plot(training_iterations, loss_mav, color='red' , label='Moving Average Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

    # os.mkdir(path_to_out)

    # plt.ylim([0,2])

    if isinstance(agent, AC_Agent):
        plt.savefig(f'{path_to_out}/Student_Losses.png')
    else:
        plt.savefig(f'{path_to_out}/Teacher_Losses.png')

    plt.clf()

    # print("saved, now moving on...")

    # print(f"reward iters, list")
    # print(rewards_iterations)
    # print(rewards_list)
    plt.plot([i for i in range(len(rewards_per_ep_list))], rewards_per_ep_list, color='blue', label='Raw Rewards')
    plt.plot([i for i in range(len(rewards_per_ep_list))], reward_ep_mav,   color='red',  label='Moving Average Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards per Episode')
    plt.legend(loc="upper right")
    
    if isinstance(agent, AC_Agent):
        plt.savefig(f'{path_to_out}/Student_Rewards.png')
    else:
        plt.savefig(f'{path_to_out}/Teacher_Rewards.png')

    plt.clf()

    steps_iterations = [i+1 for i in range(len(steps_to_terminal_total))]

    plt.plot(steps_iterations, steps_to_terminal_total, color='blue', label='Raw Steps to Terminal State')
    plt.plot(steps_iterations, steps_mav, color='red', label = 'Moving Average Steps to Terminal State')
    plt.xlabel('Episodes')
    plt.ylabel('Steps to Terminal State')
    plt.legend(loc="upper right")
    
    if isinstance(agent, AC_Agent):
        plt.savefig(f'{path_to_out}/Student_Steps.png')
    else:
        plt.savefig(f'{path_to_out}/Teacher_Steps.png')

    return agent

