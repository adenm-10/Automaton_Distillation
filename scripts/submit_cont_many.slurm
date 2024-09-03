#!/bin/bash

#SBATCH --nodes=1 #try this (distributed - more than 1 computing node)
#SBATCH --constraint=v100 # try this again - if this does not work eliminate it
#SBATCH --time=10:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=1
#SBATCH --job-name=Automaton
#SBATCH --account=cenyioha
#SBATCH --output=out.job.%J.out
#SBATCH --error=err.job.%J.err

echo "checkpoint"

username=$(whoami)

#Select File to run
export PYTHONPATH=${PYTHONPATH}:/lustre/fs1/home/${username}/Local_Neuro_Symbolic/

#Select how logs get stored
mkdir $SLURM_JOB_ID
export debug_logs="$SLURM_JOB_ID/job_$SLURM_JOB_ID.log"
export benchmark_logs="$SLURM_JOB_ID/job_$SLURM_JOB_ID.log"

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

###mpirun -np $SLURM_NTASKS python $file

# baseline
# for dragon_r_i in 10 100
# do
#     for key_r_i in 2 3
#     do
#         for bounding_dist_i in 3 #6
#         do
#             for index in 1 2
#             do
#                 dir_name="./dragon-r_${dragon_r_i}_key_r_${bounding_persist_i}_sword_i_${bounding_dist_i}_shield_i_${shield_}_seq-level_${seq_level_i}"
#                 mkdir "$dir_name"
#                 sbatch ./scripts/submit_cont_single.slurm $dir_name $SLURM_JOB_ID $dragon_r_i $bounding_persist_i $bounding_dist_i $seq_level_i $tau_i
#             done
#         done
#     done
# done

lists=(
    "1 1 10"
    "1 1 1"
    "10 10 10"
    # "1 10 100"
)

# Iterate through each list in the list of lists
for seq_level in 3
do
    for list in "${lists[@]}"
    do
        # Convert the list into an array
        elements=($list)
        
        # Assign elements to variables
        key_r=${elements[0]}
        sword_n_shield_r=${elements[1]}
        dragon_r=${elements[2]}

        dir_name="./dragon-r_${dragon_r}_key_r_${key_r}_sword_i_${sword_n_shield_r}_shield_i_${sword_n_shield_r}_seq-level_${seq_level}"
        mkdir "$dir_name"
        sbatch ./scripts/submit_cont_single.slurm $dir_name $SLURM_JOB_ID $dragon_r $key_r $sword_n_shield_r $sword_n_shield_r $seq_level

    done
done

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
