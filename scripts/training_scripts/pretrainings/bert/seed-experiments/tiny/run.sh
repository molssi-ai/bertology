#!/bin/bash
##################################################################
# Author: Mohammad Mostafanejad                                  #
# Date: October 2025                                             #
# Description:                                                   #
# This script is used within a Slurm sbatch job script.          #
##################################################################

#####################################
# Copy the files to TMPFS
#####################################
cd $main_folder
bash copy_to_tmpfs.sh

#####################################
#  Run the training script on each node
#####################################
rm -rf ./hub \
    ./outputs \
    ./modules \
    ./wandb \
    ./mlruns \
    ./tmp \
    ./evaluate

H=`hostname`
RANK=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`

cd $TMPFS/$MODEL_NAME
conda run -n bertfalcon --no-capture-output "huggingface-cli login --token ${HF_TOKEN} &&
        accelerate launch --config_file config.yaml \
        --machine_rank $RANK \
        --num_machines $NUM_NODES \
        --num_processes $WORLD_SIZE \
        --main_process_port $MASTER_PORT \
        --main_process_ip $MASTER_ADDR \
        train.py \
        +model.model_name=${MODEL_NAME}"


#####################################
# Copy the results back to the scratch folder
#####################################
cd $main_folder
sleep 1m
bash copy_from_tmpfs.sh