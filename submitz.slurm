#!/bin/bash

##SBATCH -n 2 #try this (distributed - more than 1 computing node)
##SBATCH --constraint=h100 # try this again - if this does not work eliminate it
#SBATCH --time=72:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=3
#SBATCH --job-name=Automaton
#SBATCH --account=cenyioha
#SBATCH --output=job.%J.out
#SBATCH --error=job.%J.err

## --nodes=4
## --ntasks-per-node=10
## --mem-per-cpu=1990

#Select File to run
export PYTHONPATH=${PYTHONPATH}:/lustre/fs1/home/pnwaorgu/Local_Neuro_Symbolic/
export file="./discrete/run/teacher/dungeon_quest_teacher_ddpg.py"
export args=""

#Select how logs get stored
mkdir $SLURM_JOB_ID
export debug_logs="$SLURM_JOB_ID/job_$SLURM_JOB_ID.log"
export benchmark_logs="$SLURM_JOB_ID/job_$SLURM_JOB_ID.log"

##Load Modules
module purge
module load anaconda
module load cuda/cuda-11.4

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
source activate autd
###mpirun -np $SLURM_NTASKS python $file
##python $file $args
echo
time python -u $file $args
echo
sleep 3

date >> $benchmark_logs
echo "ulimit -l" >> $benchmark_logs
ulimit -l >> $benchmark_logs

## Directory Cleanup ##
mv job.$SLURM_JOB_ID.err $SLURM_JOB_ID/
mv job.$SLURM_JOB_ID.out $SLURM_JOB_ID/
mv learning_curves $SLURM_JOB_ID/
mv tmp $SLURM_JOB_ID/