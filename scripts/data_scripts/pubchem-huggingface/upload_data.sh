#!/bin/bash
##################################################################
# Author: Mohammad Mostafanejad                                  #
# Date: May 2025                                                 #
# Description:                                                   #
# This script uploads the dataset card to the Hugging Face Hub.  #
##################################################################

huggingface-cli upload molssiai-hub/pubchem-04-18-2025 \
                ./README.md ./README.md \
                --repo-type=dataset \
                --commit-message="adds the dataset card" \
                --token=$HF_TOKEN
                # --private \
