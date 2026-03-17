###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: August 2025                                                           #
# Description:                                                                #
# Script for cleaning up the PubChem (version 04-18-2024) dataset by removing #
# a known problematic entry that causes issues in the ChEMBL standardization  #
# pipeline. The problematic entry has the PubChem CID 61899573.               #
###############################################################################

# import the necessary modules
from datasets import load_from_disk

# set the main variables
input_data_path = "/D2/sina/data/pubchem04182025_isostereomers_tokenized"
output_data_path = "/D3/sina/data/pubchem04182025_isostereomers_tokenized_cleaned"

# load the dirty dataset
ds = load_from_disk(input_data_path)

# make sure the culprit is still in the dataset
culprit_id = 61899573
culprit_smi = "COC1=CC=C(C=C1)C2C3=C(C=CC4=CC=CC=C43)OC5=CC6=C(C=C5)C7=NC8=C9C=CC1=CC9=C(N8)N=C3C4=C5C=CC(=C4)OC4=C(C(C8=C(C=CC9=CC=CC=C98)OC8=CC9=C(C=C8)C8=NC9=NC9=C%10C=C(C=CC%10=C(N9)N=C9C%10=C(C=C(C=C%10)OC%10=C2C2=CC=CC=C2C=C%10)C(=N9)NC2=NC(=N8)C8=C2C=C(C=C8)OC2=C(C(C8=C(C=CC9=CC=CC=C98)OC8=CC9=C(C=C8)C(=NC5=N3)N=C9NC6=N7)C3=CC=C(C=C3)OC)C3=CC=CC=C3C=C2)OC2=C(C(C3=C(O1)C=CC1=CC=CC=C13)C1=CC=C(C=C1)OC)C1=CC=CC=C1C=C2)C1=CC=C(C=C1)OC)C1=CC=CC=C1C=C4"
assert ds[culprit_id]["smiles"] == culprit_smi

# create the ids without the culprit id
ids = [i for i in range(ds.num_rows) if i != culprit_id]

# filter out the culprit
ds_clean = ds.select(ids)

# save the cleaned dataset
ds_clean.save_to_disk(output_data_path)
