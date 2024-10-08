#!/bin/bash

#SBATCH --nodes=1 #try this (distributed - more than 1 computing node)
#SBATCH --constraint=v100 # try this again - if this does not work eliminate it
#SBATCH --time=20:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=1
#SBATCH --job-name=Automaton
#SBATCH --account=cenyioha
#SBATCH --output=out.job.%J.out
#SBATCH --error=err.job.%J.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pr556776@ucf.edu

rm -rf ./automaton_q/*
rm -rf ./logs/*
rm -rf ./checkpoints/*

# Hyperparameters
alr=0.0003
clr=0.0003
gamma=0.99
tau=0.005
total_steps=1000000
export POLICY_FREQ=2
export NOISE_CLIP=0.5
export NOISE_STDDEV=0.2

python -u ./discrete/run/teacher/gold_mine_teacher_rew_per_step_7_productMDP.py --total-steps="$total_steps" --path-to-out="$dir_name"
python -u ./discrete/run/reward_machine/dungeon_quest_machine_rew_per_step.py
export seq_level=3
python -u ./discrete/run/target/gold_mine_7_target_rew_per_step_productMDP_TD3.py --alr="$alr" --clr="$clr" --gamma="$gamma" --tau="$tau" --total-steps="$total_steps" --path-to-out="$dir_name"
echo

mv  $dir_name/

unset dragon_r
unset bounding_persist
unset bounding_dist
unset seq_level