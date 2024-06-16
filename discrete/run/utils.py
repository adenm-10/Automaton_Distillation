import json
from typing import List, Type

import torch

from discrete.lib.agent.normal_agent import DuelingQNetworkAgent
from discrete.lib.agent.DDPG_Agent import DDPG_Agent
from discrete.lib.agent.one_hot_automaton_agent import OneHotAutomatonAfterFeatureExtractorAgent
from discrete.lib.automaton.ltl_automaton import LTLAutomaton
from discrete.lib.automaton.mine_env_ap_extractor import AP, MineEnvApExtractor
from discrete.lib.automaton.reward_machine import RewardMachine
from discrete.lib.automaton.target_automaton import AnnealTargetAutomaton
from discrete.lib.config import Configuration, RolloutBufferConfig
from discrete.lib.config import EnvConfig
from discrete.lib.intrinsic_reward import DummyIntrinsicRewardCalculator
from discrete.lib.rollout_buffer import CircularRolloutBuffer
from discrete.run.env.blind_craftsman import blind_craftsman_aps, blind_craftsman_ltlf


def construct_ap_extractor_automaton(aps: List[AP], ltlf: str, device: torch.device):
    automaton = LTLAutomaton.from_ltlf(ap_names=[ap.name for ap in aps], ltlf=ltlf, device=device)
    ap_extractor = MineEnvApExtractor(ap_funcs=[ap.func for ap in aps], device=device)

    return automaton, ap_extractor


dummy_aps = []
dummy_ltlf = "True"


def teacher_config_v1(env_config: EnvConfig, run_name: str, device: torch.device, agent_cls=DuelingQNetworkAgent,
                      no_done_on_out_of_time: bool = False, aps: List = dummy_aps, ltlf: str = dummy_ltlf,
                      max_training_steps=int(1e6), online_distill: bool = False, gamma: float = 0.99, alr: float = 0.001, clr: float = 0.001):
    automaton, ap_extractor = construct_ap_extractor_automaton(aps, ltlf, device)


    return Configuration(
        env_config=env_config,
        num_parallel_envs=8,
        rollout_buffer_config=RolloutBufferConfig(
            rollout_buffer_cls=CircularRolloutBuffer,
            capacity=150000,
            priority_scale=0.7,
            min_size_before_training=1000
        ),
        agent_cls=agent_cls,
        automaton=automaton,
        reward_machine=None,
        epsilon=0.1,
        agent_train_batch_size=128, # was 32 # changing to 128 from 100
        target_agent_update_every_steps=1000,
        max_training_steps=max_training_steps,
        checkpoint_every_steps=int(1e4),
        gamma=gamma,
        intrinsic_reward_calculator=DummyIntrinsicRewardCalculator(),
        distill=online_distill,
        temperature=0.01,
        no_done_on_out_of_time=no_done_on_out_of_time,
        ap_extractor=ap_extractor,
        device=device,
        run_name=run_name,
        actor_lr=alr,
        critic_lr=clr
    )


def student_config_v1(env_config: EnvConfig, teacher_run_name: str, student_run_name: str,
                      device: torch.device, anneal_target_aut_class: Type[AnnealTargetAutomaton],
                      anneal_target_aut_kwargs, new_gamma: float = 0.99,
                      agent_cls=DDPG_Agent, max_training_steps=int(1e6),
                      no_done_on_out_of_time=False, aps: List = dummy_aps, ltlf: str = dummy_ltlf,
                      reward_machine: bool = False):
    teacher_config = teacher_config_v1(env_config, teacher_run_name, device, agent_cls, aps=aps, ltlf=ltlf)
    
    with open(f"automaton_q/{teacher_run_name}.json", "r") as f:
        teacher_aut_info = json.load(f)

    print(teacher_aut_info)
    assert False
    
    if reward_machine:
        reward_machine = RewardMachine.from_json(teacher_config, device),
    else:
        reward_machine = None
    
    student_config = teacher_config._replace(
        env_config=env_config,
        automaton=anneal_target_aut_class(
            inner_automaton=teacher_config.automaton,
            teacher_aut_info=teacher_aut_info,
            device=device,
            min_source_q_count=1,
            **anneal_target_aut_kwargs
        ),
        reward_machine=reward_machine,
        run_name=student_run_name,
        gamma=new_gamma,
        distill=False,
        max_training_steps=max_training_steps,
        no_done_on_out_of_time=no_done_on_out_of_time
    )

    return student_config

def student_config_reward_machine(env_config: EnvConfig, teacher_run_name: str, student_run_name: str,
                      device: torch.device, new_gamma: float = 0.99, agent_cls=DuelingQNetworkAgent,
                      max_training_steps=int(1e6), no_done_on_out_of_time=False, aps: List = dummy_aps,
                      ltlf: str = dummy_ltlf, automaton: LTLAutomaton = None):
    teacher_config = teacher_config_v1(env_config, teacher_run_name, device, agent_cls, aps=aps, ltlf=ltlf)
    
    if automaton != None:
        teacher_config = teacher_config._replace(automaton=automaton)
    
    reward_machine = RewardMachine.from_json(teacher_config, device)
    
    student_config = teacher_config._replace(
        env_config=env_config,
        automaton=reward_machine,
        reward_machine=reward_machine,
        run_name=student_run_name,
        gamma=new_gamma,
        distill=False,
        max_training_steps=max_training_steps,
        no_done_on_out_of_time=no_done_on_out_of_time
    )

    return student_config

def teacher_config_productMDP(env_config: EnvConfig, run_name: str, device: torch.device, agent_cls=OneHotAutomatonAfterFeatureExtractorAgent,
                      no_done_on_out_of_time: bool = False, online_distill: bool = False):
    automaton, ap_extractor = construct_ap_extractor_automaton(blind_craftsman_aps, blind_craftsman_ltlf, device)

    return Configuration(
        env_config=env_config,
        num_parallel_envs=8,
        rollout_buffer_config=RolloutBufferConfig(
            rollout_buffer_cls=CircularRolloutBuffer,
            capacity=150000,
            priority_scale=0.7,
            min_size_before_training=1000
        ),
        agent_cls=agent_cls,
        automaton=automaton,
        epsilon=0.1,
        agent_train_batch_size=100, # was 32
        target_agent_update_every_steps=1000,
        max_training_steps=int(1e6),
        checkpoint_every_steps=int(1e4),
        gamma=0.99,
        intrinsic_reward_calculator=DummyIntrinsicRewardCalculator(),
        distill=online_distill,
        temperature=0.01,
        no_done_on_out_of_time=no_done_on_out_of_time,
        ap_extractor=ap_extractor,
        device=device,
        run_name=run_name
    )
