###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: May 2025                                                              #
# Description:                                                                #
# This script trains a wordpiece tokenizer on the entire set of               #
# PUBCHEM_SMILES entry from the PubChem data (version 04-18-2025).            #
# The data consumed by this script can be generated using the                 #
# pubchem_cismi_writer.py script located in the same directory.               #
###############################################################################

# import necessary libraries
import os
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


# set the input paths
input_file_path = "./pubchem_isostereosmiles.txt"

# instantiate the tokenizer
# this step takes care of setting up the
# normalizer, pre-tokenizer, post-processor, and decoder
tokenizer = BertWordPieceTokenizer(
    clean_text=False,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False,
    wordpieces_prefix="__",
)

# train the tokenizer on the PubChem SMILES
tokenizer.train(
    files=[input_file_path],
    vocab_size=30522,
    min_frequency=2,
    limit_alphabet=1000,
    initial_alphabet=[],
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    show_progress=True,
    wordpieces_prefix="__",
)

# Define the post-processor template
post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# Set the post-processor in the tokenizer
tokenizer.post_processor = post_processor

# load the tokenizer object into a fast tokenizer
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

# set the fast tokenizer's attributes
fast_tokenizer.model_max_length = 512
fast_tokenizer.name_or_path = "gen-mlm-cismi-bert-wordpiece-pubchem04182025"

# set special tokens
fast_tokenizer.pad_token = "[PAD]"
fast_tokenizer.unk_token = "[UNK]"
fast_tokenizer.cls_token = "[CLS]"
fast_tokenizer.sep_token = "[SEP]"
fast_tokenizer.mask_token = "[MASK]"

# save the fast tokenizer on disk
fast_tokenizer.save_pretrained("./tmp/" + fast_tokenizer.name_or_path)

# push the tokenizer to the Hugging Face Hub
# fast_tokenizer.push_to_hub(repo_id="smostafanejad/gen-mlm-cismi-bert-wordpiece",
#                            use_temp_dir=True,
#                            commit_message="Add PubChem canonical SMILES BERT WordPiece Tokenizer",
#                            private=True,
#                            token=os.environ['HF_TOKEN'],
#                            )
