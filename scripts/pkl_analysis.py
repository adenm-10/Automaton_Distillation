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

    x_length = 3000
    x_axis_iterable = [i for i in range(x_length)]


    # Distillation data dictionary read
    file_distill_name = "rew_and_steps_lists.pkl"
    path_distill_to_out = "past_runs/363227"
    file_distill_path = f"{path_distill_to_out}/{file_distill_name}"
    with open(file_distill_path, 'rb') as file_distill_:
        data = pickle.load(file_distill_)

    data_distill = data
    for key, value in data_distill.items():
        data_distill[key] = moving_average(value)[0:x_length]
        
    # Independent agent data dictionary read
    file_independent_name = "rew_and_steps_lists.pkl"
    path_independent_to_out = "past_runs/363228/dragon-r_10_bound-persist_True_bounding-dist_3_seq-level__index_1"
    file_independent_path = f"{path_independent_to_out}/{file_independent_name}"
    with open(file_independent_path, 'rb') as file:
        data = pickle.load(file)

    data_independent = data
    for key, value in data_independent.items():
        data_independent[key] = moving_average(value)[0:x_length]

    # Rewards per episode
    plt.plot(x_axis_iterable, data_distill['rewards_per_episode'],   color='blue',  label='Distilled Agent')
    plt.plot(x_axis_iterable, data_independent['rewards_per_episode'],   color='red',  label='Independent Agent')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards per Episode')
    plt.legend(loc="upper right")

    plt.savefig(f'{path_distill_to_out}/Student_Rewards_Comp_{x_length}_Episodes.png')

    plt.clf()


    # Steps to Terminal State per episode
    plt.plot(x_axis_iterable, data_distill['steps_to_term_per_episode'], color='blue', label = 'Distilled Agent')
    plt.plot(x_axis_iterable, data_independent['steps_to_term_per_episode'], color='red', label = 'Independent Agent')
    plt.xlabel('Episodes')
    plt.ylabel('Steps to Terminal State')
    plt.legend(loc="upper right")

    plt.savefig(f'{path_distill_to_out}/Student_Steps_Comp_{x_length}_Episodes.png')
    
    plt.clf()