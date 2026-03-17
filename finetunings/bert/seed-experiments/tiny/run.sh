#!/bin/bash
##################################################################
# Author: Mohammad Mostafanejad                                  #
# Date: November 2025                                            #
# Description:                                                   #
# This script is used as a driver for fine-tuning pre-trained    #
# BERT models.                                                   #
##################################################################
export MODEL_NAME="org-cismi-bertbase-gdb-f1"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MODEL_NAME="gen-cismi-berttiny-wordpiece-finetuned-regression"

conda run -n bertchemai --no-capture-output "python lightning_finetune_wandb.py \
        +model.model_name=${MODEL_NAME}"
