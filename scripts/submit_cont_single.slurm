#!/bin/bash

#SBATCH --nodes=1
#SBATCH --constraint=v100 # try this again - if this does not work eliminate it
#SBATCH --time=10:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=1
#SBATCH --job-name=Automaton
#SBATCH --account=cenyioha
#SBATCH --output=out.job.%J.out
#SBATCH --error=err.job.%J.err
#SBATCH --mail-type=END
#SBATCH --mail-user=aden.mckinney@ucf.edu

dir_name=$1
orig_job_id=$2

export dragon_r=$3
export bounding_persist=$4
export bounding_dist=$5
export seq_level=$6
export tau=$7
export PATH_TO_OUT="${dir_name}"

out_file=out.job.${SLURM_JOB_ID}.out
err_file=err.job.${SLURM_JOB_ID}.err

# Hyperparameters
alr=0.0003
clr=0.0003
gamma=0.99
tau=0.005
total_steps=2000000
export POLICY_FREQ=2
export NOISE_CLIP=0.5
export NOISE_STDDEV=0.2

python discrete/run/teacher/dungeon_quest_teacher_td3.py --alr="$alr" --clr="$clr" --gamma="$gamma" --tau="$tau" --total-steps="$total_steps"
mv test_output/test_output* $out_file $err_file $dir_name/
mv $dir_name $orig_job_id/

unset POLICY_FREQ
unset NOISE_CLIP
unset NOISE_STDDEV

