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
#SBATCH --mail-type=END
#SBATCH --mail-user=pr556776@ucf.edu

export PYTHONPATH=${PYTHONPATH}:/lustre/fs1/home/pnwaorgu/Local_Neuro/

module load anaconda
source activate autd

dir_name=$1
orig_job_id=$2

export dragon_r=$3
export key_r=$4
export shield_r=$5
export sword_r=$6
export seq_level=$7

export bounding_persist=$8
export bounding_dist=$9
export tau=$10


out_file=out.job.${SLURM_JOB_ID}.out
err_file=err.job.${SLURM_JOB_ID}.err

# Hyperparameters
alr=0.001
clr=0.001
gamma=0.99
tau=0.005
total_steps=500000
export POLICY_FREQ=2
export NOISE_CLIP=0.5
export NOISE_STDDEV=0.1

python discrete/run/teacher/dungeon_quest_teacher_td3.py --alr="$alr" --clr="$clr" --gamma="$gamma" --tau="$tau" --total-steps="$total_steps" --path-to-out="$dir_name"
mv $out_file $err_file $dir_name/
mv $dir_name $orig_job_id/
# 
unset POLICY_FREQ
unset NOISE_CLIP
unset NOISE_STDDEV

