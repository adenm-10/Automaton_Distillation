import copy
import sys

print(sys.version)

from torch.utils.tensorboard import SummaryWriter

from discrete.lib.config import Configuration
from discrete.lib.construct_q_automaton import construct_q_automaton
from discrete.lib.create_training_state import create_training_state
from discrete.lib.training import train_agent, distill_agent
from discrete.lib.agent.AC_Agent import AC_Agent

from datetime import datetime

def run_distill(config: Configuration):
    teacher, teacher_buffer, _, _, _ = create_training_state(config)
    
    student_config = config._replace(
        run_name=f"{config.run_name}_distill"
        # rollout_buffer_config=config.rollout_buffer_config._replace(capacity=1000)
    )
    
    student, student_buffer, ap_extractor, automaton, start_iter = create_training_state(student_config)

    logger = SummaryWriter(f"logs/{student_config.run_name}", purge_step=start_iter)

    distill_agent(student_config, teacher, student, automaton, ap_extractor, teacher_buffer, student_buffer, logger, start_iter)

def run_training(config: Configuration,
                 run_name=None):
                 
    if run_name is None:
        run_name = config.run_name

    print("Starting training at: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    agent, rollout_buffer, ap_extractor, automaton, start_iter = create_training_state(config)

    # ADEN WAS HERE
    # start_iter = 0

    logger = SummaryWriter(f"logs/{run_name}", purge_step=start_iter)

    train_agent(config, agent, automaton, ap_extractor, rollout_buffer, logger, start_iter, run_name=run_name)

    # For teacher uncomment
    if not isinstance(agent, AC_Agent):
        construct_q_automaton(agent=agent, rollout_buffer=rollout_buffer, ap_extractor=ap_extractor, automaton=automaton,
                            device=config.device, run_name=run_name)

    print("Finished training at: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
