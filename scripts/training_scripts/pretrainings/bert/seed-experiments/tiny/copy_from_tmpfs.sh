#!/bin/bash
###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: March 2025                                                            #
# Description:                                                                #
# Bash script responsible for copying the results of the training from the    #
# temporary file system (TMPFS) on compute nodes back to the scratch folder   #
# on the main storage. The script is called from the main Slurm job script.   #
###############################################################################

#####################################
# Copy the results back to the scratch folder
#####################################
echo "Copying results back to scratch folder...\n"
mkdir -p ${scratch_folder}/$MODEL_NAME/$TASK_NAME
rsync -rzP $TMPFS/$MODEL_NAME/ ${scratch_folder}/$MODEL_NAME/$TASK_NAME/
echo "Data copied back successfully."
