#!/bin/bash
##################################################################
# Author: Mohammad Mostafanejad                                  #
# Date: October 2025                                             #
# Description:                                                   #
# This is the driver Slurm sbatch job script that runs the       #
# pretraining job on the specified cluster nodes.                #
# NOTE: this file requires the run.sh script which should be     #
# present in the $main_folder directory.                         #
##################################################################
#SBATCH --account=seamm
#SBATCH --job-name=tiny-ms-3456-ds-1234
#SBATCH --partition=a30_normal_q
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCh --gres-flags=enforce-binding
#SBATCH --qos=fal_a30_normal_base
#SBATCH --time=7-00:00:00
#SBATCH -D /scratch/smostafanejad/PUBCHEM
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --priority=2000
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=200G
#SBATCH -x fal006

# environment variables
export MODEL_NAME="gen-cismi-berttiny-wordpiece"
export TASK_NAME="tiny-ms-3456-ds-1234"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_JOB_NAME=$MODEL_NAME
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export NUM_NODES=$SLURM_JOB_NUM_NODES
export WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

echo "==========================="
echo Model Name: $MODEL_NAME
echo Task Name: $TASK_NAME
echo HOSTNAMES: $HOSTNAMES
echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
echo NUM_NODES: $NUM_NODES
echo WORLD_SIZE: $WORLD_SIZE
echo "==========================="

# directories
export main_folder="/projects/chemai/smostafanejad/PubChem-BERT-Collections/PAPER_RESULTS/staged_jobs/${MODEL_NAME}"
export data_path="/projects/chemai/smostafanejad/PubChem-BERT-Collections/data/pubchem04182025_isostereomers_tokenized"
export scratch_folder="/scratch/smostafanejad/PUBCHEM"

# run the script
module load openmpi4/gcc
mpirun --bind-to none -n $NUM_NODES --map-by ppr:1:node $main_folder/run.sh

