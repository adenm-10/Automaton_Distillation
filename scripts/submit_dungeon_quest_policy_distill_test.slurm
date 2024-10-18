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

username=$(whoami)

#Select File to run
export PYTHONPATH=${PYTHONPATH}:/lustre/fs1/home/${username}/Local_Neuro/

#Select how logs get stored
mkdir $SLURM_JOB_ID
export debug_logs="$SLURM_JOB_ID/job_$SLURM_JOB_ID.log"
export benchmark_logs="$SLURM_JOB_ID/job_$SLURM_JOB_ID.log"

out_file=out.job.${SLURM_JOB_ID}.out
err_file=err.job.${SLURM_JOB_ID}.err

##Load Modules
module purge
module load anaconda
module load cuda

## Enter Working Directory ##
cd $SLURM_SUBMIT_DIR

## Create Log File ##
echo $SLURM_SUBMIT_DIR
echo "JobID: $SLURM_JOB_ID" >> $debug_logs
##echo "Running on $SLURM_NODELIST" >> $debug_logs
##echo "Running on $SLURM_NNODES nodes." >> $debug_logs
##echo "Running on $SLURM_NPROCS processors." >> $debug_logs
echo  "Current working directory is `pwd`" >> $debug_logs
mkdir "tmp"
mkdir "tmp/policy"
mkdir "learning_curves"

## Module debugging ##
module list >> $debug_logs
nvidia-smi topo -m
##which mpirun >> $debug_logs

date >> $benchmark_logs
echo "ulimit -l: " >> $benchmark_logs
ulimit -l >> $benchmark_logs

## Run job ## 
conda activate autd

rm -rf ./automaton_q/*
rm -rf ./logs/*
rm -rf ./checkpoints/*

# Hyperparameters
alr=0.0003
clr=0.0003
gamma=0.99
tau=0.005
total_steps=2000000
export POLICY_FREQ=2
export NOISE_CLIP=0.5
export NOISE_STDDEV=0.2
# export seq_level=2

dir_name=$SLURM_JOB_ID
mkdir "$dir_name"

python -u discrete/run/distill/dungeon_quest_policy_distillation_test.py --total-steps="$total_steps" --path-to-out="$dir_name"

mv ./test_output/test_output* $SLURM_JOB_ID/
echo

mv $out_file $err_file $dir_name/

unset dragon_r
unset bounding_persist
unset bounding_dist
unset seq_level

date >> $benchmark_logs
echo "ulimit -l" >> $benchmark_logs
ulimit -l >> $benchmark_logs

## Directory Cleanup ##
mv err.job.$SLURM_JOB_ID.err $SLURM_JOB_ID/
mv out.job.$SLURM_JOB_ID.out $SLURM_JOB_ID/
mv learning_curves $SLURM_JOB_ID/
mv tmp $SLURM_JOB_ID/