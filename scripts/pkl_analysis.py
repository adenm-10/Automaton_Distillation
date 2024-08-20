import copy
import time
from typing import Tuple, Dict, List
from datetime import datetime
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":

    file_distill_name = "rew_and_steps_lists.pkl"
    path_distill_to_out = "td3_10k_eps_distill"
    file_distill_path = f"{path_distill_to_out}/{file_distill_name}"
    with open(file_distill_path, 'rb') as file_distill_:
        data = pickle.load(file_distill_)

    reward_ep_mav_distill = data['reward_ep_mav'][0:10000]
    steps_mav_distill = data['steps_mav'][0:10000]

    file_independent_name = "rew_and_steps_lists.pkl"
    path_independent_to_out = "362855/dragon-r_10_bound-persist_True_bounding-dist_3_seq-level_2_tau_"
    file_independent_path = f"{path_independent_to_out}/{file_independent_name}"
    with open(file_independent_path, 'rb') as file:
        data = pickle.load(file)

    reward_ep_mav_independent = data['reward_ep_mav'][0:10000]
    steps_mav_independent = data['steps_mav'][0:10000]

    # plt.plot([i for i in range(10000)], rewards_per_ep_list, color='blue', label='Raw Rewards')
    plt.plot([i for i in range(10000)], reward_ep_mav_distill,   color='blue',  label='Distilled Agent')
    plt.plot([i for i in range(10000)], reward_ep_mav_independent,   color='red',  label='Independent Agent')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards per Episode')
    plt.legend(loc="upper right")

    plt.savefig(f'{path_distill_to_out}/Student_Rewards_Comp.png')

    plt.clf()

    steps_iterations = [i+1 for i in range(10000)]

    # plt.plot(steps_iterations, steps_to_terminal_total, color='blue', label='Raw Steps to Terminal State')
    plt.plot(steps_iterations, steps_mav_distill, color='blue', label = 'Distilled Agent')
    plt.plot(steps_iterations, steps_mav_independent, color='red', label = 'Independent Agent')
    plt.xlabel('Episodes')
    plt.ylabel('Steps to Terminal State')
    plt.legend(loc="upper right")

    plt.savefig(f'{path_distill_to_out}/Student_Steps_Comp.png')
    
    plt.clf()