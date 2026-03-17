###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: August 2025                                                           #
# Description:                                                                #
# Script for standardization of the SMILES using ChEMBL standardization       #
# pipeline: [Bento et al. J Cheminform 12 (2020)].                            #
# SMILES are taken from the PubChem (version 04-18-2024) dataset.             #
###############################################################################

# import the necessary modules
import os
import functools
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast


# define the main processing function
def main():
    # set the main variables
    input_data_path = (
        "/D3/sina/data/pubchem04182025_isostereomers_tokenized_and_chemblized"
    )
    output_data_path = (
        "/D3/sina/data/pubchem04182025_chembl_std_isostereomers_tokenized"
    )
    tokenizer_path = "molssiai-hub/gen-mlm-cismi-bert-wordpiece-pubchem04182025"
    cache_dir = "./tmp"

    # load the dataset
    ds = load_from_disk(input_data_path)

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        revision="main",
        cache_dir=cache_dir,
        token=os.getenv("HF_TOKEN"),
        trust_remote_code=True,
    )

    # map the dataset to tokenize the SMILES column and remove unnecessary columns
    ds_new = ds.map(
        functools.partial(
            tokenizer, truncation=True, padding="max_length", max_length=512
        ),
        batched=False,
        input_columns=["SMILES"],
        num_proc=64,
        remove_columns=[
            "SMILES",
            "original_SMILES",
            "original_InChI",
            "original_InChIKey",
            "InChI",
            "InChIKey",
            "problematic_SMILES",
        ],
    )

    # save the new dataset
    ds_new.save_to_disk(output_data_path)


if __name__ == "__main__":
    main()
