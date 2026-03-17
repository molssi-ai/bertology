# PubChem Dataset Preprocessing Pipeline

This directory contains scripts for preprocessing the PubChem (version 04-18-2025) dataset through a three-step pipeline: SMILES extraction, tokenizer training, and data tokenization.

## Overview

The pipeline processes the PubChem dataset through the following stages:

1. **SMILES Extraction**: Extract canonical isomeric SMILES from PubChem and write to text file
2. **Tokenizer Training**: Train a WordPiece tokenizer on the extracted SMILES strings from scratch
3. **Data Tokenization**: Tokenize the SMILES data using the trained WordPiece tokenizer

## Prerequisites

- Python 3.x
- Required packages:
  - `datasets`
  - `transformers`
  - `tokenizers`
  - `numpy`
  - `tqdm`

## Pipeline Steps

### Step 1: SMILES Extraction

**Script**: `pubchem_cismi_writer.py`

This step extracts the canonical isomeric SMILES (PUBCHEM_SMILES entry) from the
PubChem dataset on Hugging Face and writes them to a text file. We write the
SMILES into a text file because training a tokenizer on a simple text file
proves to be more efficient.

**Input**: Hugging Face dataset `molssiai-hub/pubchem-04-18-2025`

**Output**: `pubchem_isostereosmiles.txt`

**Usage**:
```bash
export HF_TOKEN="your_huggingface_token"
python pubchem_cismi_writer.py
```

**What it does**:
- Loads the PubChem dataset from Hugging Face
- Extracts the PUBCHEM_SMILES column
- Processes data in batches of 1,000,000 entries
- Writes SMILES strings to a text file (one per line)
- Uses progress bar to track processing

### Step 2: Tokenizer Training

**Script**: `pubchem_tokenizer_training.py`

This step trains a WordPiece tokenizer on the entire set of canonical isomeric SMILES from the PubChem dataset.

**Input**: `pubchem_isostereosmiles.txt`

**Output**: `./tmp/gen-mlm-cismi-bert-wordpiece-pubchem04182025/`

**Usage**:
```bash
python pubchem_tokenizer_training.py
```

**What it does**:
- Instantiates a BERT WordPiece tokenizer with custom settings:
  - Vocabulary size: 30,522
  - Minimum frequency: 2
  - Wordpiece prefix: `__`
  - Special tokens: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`
- Trains the tokenizer on the SMILES text file
- Configures post-processor for BERT-style templates
- Converts to PreTrainedTokenizerFast for compatibility
- Saves the tokenizer to disk
- (Optional) Can push tokenizer to Hugging Face Hub

**Tokenizer Configuration**:
- Clean text: False
- Handle Chinese chars: False
- Strip accents: False
- Lowercase: False
- Max length: 512

### Step 3: Data Tokenization

**Script**: `pubchem_data_tokenizer.py`

This step tokenizes the PubChem canonical isomeric SMILES using the pre-trained WordPiece tokenizer.

**Input**: `pubchem_isostereosmiles.txt`

**Output**: `./tmp/pubchem04182025_isostereomers_tokenized/`

**Usage**:
```bash
python pubchem_data_tokenizer.py
```

**What it does**:
- Loads the pre-trained tokenizer from Step 2
- Loads the SMILES data from the text file
- Renames the column from `text` to `smiles`
- Tokenizes the SMILES strings with:
  - Maximum sequence length: 512
  - Padding to max length
  - Truncation enabled
- Processes data in batches of 2,048 using 4 processes
- Removes `token_type_ids` column (not needed as the chemical tasks do not involve sentence pairs)
- Saves the tokenized dataset to disk in Arrow format

## Running the Complete Pipeline

To process the dataset from start to finish, run the scripts in order:

```bash
# Step 1: Extract canonical isomeric SMILES
export HF_TOKEN="your_huggingface_token"
python pubchem_cismi_writer.py

# Step 2: Train the WordPiece tokenizer
python pubchem_tokenizer_training.py

# Step 3: Tokenize the data
python pubchem_data_tokenizer.py
```

## Data Flow

```
PubChem HuggingFace Dataset
    ↓
[pubchem_cismi_writer.py]
    ↓
Text File with SMILES (one per line)
    ↓
[pubchem_tokenizer_training.py]
    ↓
Trained WordPiece Tokenizer
    ↓
[pubchem_data_tokenizer.py]
    ↓
Tokenized Dataset (ready for BERT pre-training)
```

## Output Dataset Structure

The final tokenized dataset contains:
- `smiles`: Original SMILES strings
- `input_ids`: Token IDs for the SMILES
- `attention_mask`: Attention mask for the tokenized sequences

## Notes

- In Step 1, `streaming=False` should be disabled if enough disk space is available to speed up processing
- The tokenizer is trained with BERT-specific settings and special tokens
- All intermediate outputs are saved in the `./tmp` directory
- The tokenized dataset is saved in Arrow format for efficient loading with Hugging Face datasets
- The trained tokenizer can be pushed to Hugging Face Hub by uncommenting the relevant code in Step 2

## Author

Mohammad Mostafanejad  
Date: May 2025
