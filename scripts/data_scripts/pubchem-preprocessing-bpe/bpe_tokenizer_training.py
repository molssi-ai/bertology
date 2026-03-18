###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: November 2025                                                         #
# Description:                                                                #
# This script trains a byte-level BPE tokenizer, used in RoBERTa, on the      #
# entire set of PUBCHEM_SMILES entry from the PubChem data (version           #
# 04-18-2025). The vocabulary size is changed from 50265 to 30522 to match    #
# that of the original BERT wordpiece tokenizer. The data consumed by this    #
# script can be generated using the pubchem_cismi_writer.py script located    #
# in the pubchem-preprocessing directory.                                     #
###############################################################################

# import necessary libraries
import os
from transformers import AutoTokenizer
from datasets import load_dataset


# set the input paths and main variables
input_file_path = "./pubchem_isostereosmiles.txt"
output_tokenizer_name = "gen-mlm-cismi-roberta-bpe-vocab30522-pubchem04182025"
output_dir = "./tmp/" + output_tokenizer_name
repo_id = "molssiai-hub/" + output_tokenizer_name
vocab_size = 30522
batch_size = 2048

# instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

# load the dataset from the text file and rename the column
ds = load_dataset(
    "text",
    data_files=input_file_path,
    split="train",
).rename_column("text", "smiles")

# create a batched iterator over the dataset
iterator = ds.iter(batch_size=batch_size)

# train the tokenizer
new_tokenizer = tokenizer.train_new_from_iterator(
    iterator,
    vocab_size=vocab_size,
    length=ds.num_rows,
    new_special_tokens=None,
    special_tokens_map={
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "cls_token": "<s>",
        "sep_token": "</s>",
        "mask_token": "<mask>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    },
)

# save the trained tokenizer to disk
new_tokenizer.save_pretrained(output_dir)

# push the tokenizer to the Hugging Face Hub
new_tokenizer.push_to_hub(
    repo_id=repo_id,
    use_temp_dir=True,
    commit_message="Add PubChem canonical SMILES RoBERTa Byte-level BPE Tokenizer (Vocab Size: 30522)",
    private=True,
    token=os.environ["HF_TOKEN"],
)
