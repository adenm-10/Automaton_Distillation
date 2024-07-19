#!/bin/bash

#SBATCH --nodes=1
#SBATCH --constraint=v100 # try this again - if this does not work eliminate it
#SBATCH --time=72:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=1
#SBATCH --job-name=Automaton
#SBATCH --account=cenyioha
#SBATCH --output=out.job.%J.out
#SBATCH --error=err.job.%J.err

dir_name=$1
orig_job_id=$2

# Hyperparameters
alr=0.0001
clr=0.001
gamma=0.99
tau=0.001
total_steps=2000

python discrete/run/teacher/dungeon_quest_teacher_ddpg.py --alr="$alr" --clr="$clr" --gamma="$gamma" --tau="$tau" --total-steps="$total_steps"

mv ./test_output/test_output* ./out.job.%J.out ./err.job.%J.err $dirname
mv -f $dirname $orig_job_id/