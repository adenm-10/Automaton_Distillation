#!/bin/bash

#SBATCH --nodes=2 #try this (distributed - more than 1 computing node)
#SBATCH --constraint=v100 # try this again - if this does not work eliminate it
##SBATCH --nodelist=evc1
##SBATCH --ntasks=8
##SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=4
#SBATCH --job-name=Automaton
#SBATCH --account=cenyioha
#SBATCH --output=out.job.%J.out
#SBATCH --error=err.job.%J.err

## --nodes=4
## --ntasks-per-node=10
## --mem-per-cpu=1990

#Select File to run
export PYTHONPATH=${PYTHONPATH}:/lustre/fs1/home/pnwaorgu/Local_Neuro_Symbolic/
#export PYTHONPATH=${PYTHONPATH}:/lustre/fs1/home/amckinney/Local_Neuro_Symbolic/
export file="./discrete/run/teacher/dungeon_quest_teacher_rew_per_step_7_productMDP.py"
export args=""

#Select how logs get stored
mkdir $SLURM_JOB_ID
export debug_logs="$SLURM_JOB_ID/job_$SLURM_JOB_ID.log"
export benchmark_logs="$SLURM_JOB_ID/job_$SLURM_JOB_ID.log"

##Load Modules
module purge
module load anaconda/anaconda3
module load cuda/cuda-11.3

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

#echo "Made needed log directories"

## Module debugging ##
module list >> $debug_logs
nvidia-smi topo -m
##which mpirun >> $debug_logs

date >> $benchmark_logs
echo "ulimit -l: " >> $benchmark_logs
ulimit -l >> $benchmark_logs

#echo "Activating conda env"

conda deactivate

## Run job ## 
source activate autd

rm -rf ./automaton_q/*
rm -rf ./logs/*
rm -rf ./checkpoints/*

###mpirun -np $SLURM_NTASKS python $file
echo
time python -u $file $args
echo
# sleep 3

date >> $benchmark_logs
echo "ulimit -l" >> $benchmark_logs
ulimit -l >> $benchmark_logs

## Directory Cleanup ##
mv ./test_output/test_output* $SLURM_JOB_ID/
mv err.job.$SLURM_JOB_ID.err $SLURM_JOB_ID/
mv out.job.$SLURM_JOB_ID.out $SLURM_JOB_ID/
mv learning_curves $SLURM_JOB_ID/
mv tmp $SLURM_JOB_ID/
