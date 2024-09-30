#!/bin/bash

username=$(whoami)
SLURM_JOB_ID=2

#Select File to run
export PYTHONPATH=${PYTHONPATH}:/lustre/fs1/home/${username}/Local_Neuro_Symbolic/

## Run job ## 
cd ~/ddpg_envs/Local_Neuro_Symbolic/Local_Neuro_Symbolic
source ~/conda_init.sh
source ~/ddpg_envs/Local_Neuro_Symbolic/env_act.sh

rm -rf ./automaton_q/*
rm -rf ./logs/*
rm -rf ./checkpoints/*

lists=(
    "1 1 10"
    # "1 1 1"
    # "10 10 10"
    # "1 10 100"
)

# Iterate through each list in the list of lists
for seq_level in 2 #3
do
    for list in "${lists[@]}"
    do
        # Convert the list into an array
        elements=($list)
        
        # Assign elements to variables
        key_r=${elements[0]}
        sword_n_shield_r=${elements[1]}
        dragon_r=${elements[2]}

        dir_name="./dragon-r_${dragon_r}_key_r_${key_r}_sword_i_${sword_n_shield_r}_shield_i_${sword_n_shield_r}_seq-level_${seq_level}_2"
        mkdir "$dir_name"
        source ./scripts/local_disc_single.slurm $dir_name $SLURM_JOB_ID $dragon_r $key_r $sword_n_shield_r $sword_n_shield_r $seq_level #> ./"$dir_name"/out.txt

    done
done

unset dragon_r
unset bounding_persist
unset bounding_dist
unset seq_level