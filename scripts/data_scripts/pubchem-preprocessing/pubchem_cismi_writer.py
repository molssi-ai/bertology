###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: May 2025                                                              #
# Description:                                                                #
# This script extracts the canonical isomeric SMILES (PUBCHEM_SMILES entry)   #
# from the PubChem data (version 04-18-2025) on huggingface and writes them   #
# into separate lines in a text file (pubchem_isostereosmiles.txt).           #
###############################################################################

import os
from datasets import load_dataset
import numpy as np
from tqdm import tqdm


# define the global variables for the script
output_file = "pubchem_isostereosmiles.txt"
huggingface_repo = "molssiai-hub/pubchem-04-18-2025"

# delete the tmp directory if it exists
if os.path.exists("./tmp"):
    os.rmdir("./tmp")

# load the dataset
dataset = load_dataset(
    path=huggingface_repo,
    split="train",
    cache_dir="./tmp",
    streaming=True,
    token=os.environ["HF_TOKEN"],
    trust_remote_code=True,
)


# create a batched dataset
batched_ds = dataset.select_columns(["PUBCHEM_SMILES"]).iter(batch_size=1_000_000)

# write the SMILES strings to a text file with a progress bar
with tqdm(total=183, desc="Writing SMILES") as pbar:
    # loop over the dataset
    for batch in batched_ds:
        # write the SMILES strings to a text file with numpy
        with open(output_file, "a") as file:
            np.savetxt(file, batch["PUBCHEM_SMILES"], fmt="%s", delimiter="\n")
        pbar.update(1)
