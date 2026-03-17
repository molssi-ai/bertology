#!/bin/bash
###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: March 2025                                                            #
# Description:                                                                #
# Bash script responsible for copying the training files from the main        #
# storage to the temporary file system (TMPFS) on the compute nodes. The      #
# script is called from the main Slurm job script.                            #
###############################################################################

#####################################
# Copy the data to each node's TMPFS
#####################################
echo "Copying files to TMPFS...\n"
rsync -rzP $main_folder $TMPFS/
rsync -rzP $data_path $TMPFS/
echo "Files copied to TMPFS successfully.\n"