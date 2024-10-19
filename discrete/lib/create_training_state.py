from typing import Tuple

import torch
import gym

from discrete.lib.agent.agent import Agent
from discrete.lib.automaton.ap_extractor import APExtractor
from discrete.lib.automaton.automaton import Automaton
from discrete.lib.checkpoint import checkpoint_exists, load_checkpoint
from discrete.lib.config import Configuration
from discrete.lib.env.util import make_env
from discrete.lib.rollout_buffer import RolloutBuffer


def create_training_state(config: Configuration) -> Tuple[Agent, RolloutBuffer, APExtractor, Automaton, int]:
    """
    Loads training state from a checkpoint, or creates a default training state if no checkpoint exists
    """
    print(f"Env Config: {config.env_config}")
    sample_env = make_env(config.env_config)
    print(f"Env Name: {config.env_config.env_name}")
    print(f"Observation Space: {sample_env.observation_space}")
    print(f"Automaton Num States: {config.automaton.num_states}")
    print(f"Action Space: {sample_env.action_space}")
    num_actions = 0
    num_options = 0

    # ADEN WAS HERE
    # agent = config.agent_cls.create_agent(sample_env.observation_space.shape, config.automaton.num_states,
    #                                       sample_env.action_space.n).to(config.device)
    if isinstance(sample_env.action_space, gym.spaces.discrete.Discrete):
        num_options = sample_env.action_space.n # This represents the number of options, not the simension of the action space. Discrete environments in Gym are always one dimension
        num_actions = 1
    else:
        num_options = sample_env.action_space.shape[0]
        num_actions = sample_env.action_space.shape[0]

    agent = config.agent_cls.create_agent(sample_env.observation_space.shape, config.automaton.num_states,
                                            num_options).to(config.device)
    agent.to(config.device)

    print(f"Agent Name: {agent.name}")

    rollout_buffer = config.rollout_buffer_config.rollout_buffer_cls.create_empty(
        capacity=config.rollout_buffer_config.capacity,
        input_shape=sample_env.observation_space.shape,
        num_actions=num_actions,
        state_dtype=getattr(torch, sample_env.observation_space.dtype.name),  # Convert np dtype to torch dtype
        device=config.device
    )
    ap_extractor = config.ap_extractor
    automaton = config.automaton

    start_iter = 0

    # if checkpoint_exists(config):
    #     print("Loading from checkpoint")
    #     checkpoint = load_checkpoint(config)
    #     start_iter = checkpoint.iter_num + 1
    #     ap_extractor.load_state_dict(checkpoint.ap_extractor_state)
    #     automaton.load_state_dict(checkpoint.automaton_state)
    #     rollout_buffer.load_state_dict(checkpoint.rollout_buffer_state)
    #     agent.load_state_dict(checkpoint.agent_state)
    # else:
    #     print("NOT Loading from checkpoint")

    return agent, rollout_buffer, ap_extractor, automaton, start_iter
