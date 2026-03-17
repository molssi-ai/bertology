# PubChem Dataset Processing Pipeline

This directory contains scripts for processing the PubChem (version 04-18-2025) dataset through a three-step pipeline: cleaning, standardization, and tokenization.

## Overview

The pipeline processes SMILES strings from PubChem through the following stages:

1. **Data Cleaning**: Remove problematic entries that cause issues in downstream processing
2. **ChEMBL Standardization**: Standardize molecular structures using the ChEMBL pipeline
3. **WordPiece Tokenization**: Tokenize the standardized SMILES for machine learning models

## Prerequisites

- Python 3.x
- Required packages:
  - `datasets`
  - `transformers`
  - `rdkit`
  - `chembl_structure_pipeline`

## Pipeline Steps

### Step 1: Data Cleaning

**Script**: `pubchem04182025_cleaner.py`

This step removes a known problematic entry (PubChem CID 61899573) from the dataset that causes issues in the ChEMBL standardization pipeline.

**Input**: `/D2/sina/data/pubchem04182025_isostereomers_tokenized`

**Output**: `/D3/sina/data/pubchem04182025_isostereomers_tokenized_cleaned`

**Usage**:
```bash
python pubchem04182025_cleaner.py
```

**What it does**:
- Loads the original PubChem dataset
- Identifies and removes the problematic entry (CID 61899573)
- Saves the cleaned dataset for further processing

### Step 2: ChEMBL Standardization

**Script**: `chembl_standardizer.py`

This step standardizes all SMILES strings using the ChEMBL Structure Pipeline [Bento et al. J Cheminform 12 (2020)], which ensures consistent molecular representations.

**Input**: `/D3/sina/data/pubchem04182025_isostereomers_tokenized_cleaned`

**Output**: `/D3/sina/data/pubchem04182025_isostereomers_tokenized_and_chemblized`

**Usage**:
```bash
python chembl_standardizer.py
```

**What it does**:
- Loads the cleaned PubChem dataset
- For each SMILES string:
  - Converts to RDKit molecule object
  - Generates original InChI and InChIKey
  - Applies ChEMBL standardization pipeline
  - Generates standardized SMILES, InChI, and InChIKey
  - Handles errors gracefully for problematic structures
- Processes data in parallel using multiple processes
- Removes old tokenization columns (input_ids, attention_mask) as they need to be regenerated
- Saves both original and standardized molecular representations

### Step 3: WordPiece Tokenization

**Script**: `chembl_standardized_tokenizer.py`

This step tokenizes the standardized SMILES strings using a pre-trained WordPiece tokenizer designed for molecular representations.

**Input**: `/D3/sina/data/pubchem04182025_isostereomers_tokenized_and_chemblized`

**Output**: `/D3/sina/data/pubchem04182025_chembl_std_isostereomers_tokenized`

**Usage**:
```bash
export HF_TOKEN="your_huggingface_token"
python chembl_standardized_tokenizer.py
```

**What it does**:
- Loads the ChEMBL-standardized dataset
- Loads the WordPiece tokenizer from HuggingFace Hub (`molssiai-hub/gen-mlm-cismi-bert-wordpiece-pubchem04182025`)
- Tokenizes the standardized SMILES column with:
  - Maximum sequence length: 512
  - Padding to max length
  - Truncation enabled
- Processes data in parallel using multiple processes
- Removes intermediate columns, keeping only tokenized representations
- Saves the final tokenized dataset

## Running the Complete Pipeline

To process the dataset from start to finish, run the scripts in order:

```bash
# Step 1: Clean the data
python pubchem04182025_cleaner.py

# Step 2: Standardize with ChEMBL pipeline
python chembl_standardizer.py

# Step 3: Tokenize the standardized SMILES
export HF_TOKEN="your_huggingface_token"
python chembl_standardized_tokenizer.py
```

## Data Flow

```
Original Dataset
    ↓
[pubchem04182025_cleaner.py]
    ↓
Cleaned Dataset (problematic entries removed)
    ↓
[chembl_standardizer.py]
    ↓
ChEMBL-Standardized Dataset (with original & standardized representations)
    ↓
[chembl_standardized_tokenizer.py]
    ↓
Final Tokenized Dataset (ready for ML models)
```

## Output Dataset Structure

The final dataset contains:
- `input_ids`: Token IDs for the standardized SMILES
- `attention_mask`: Attention mask for the tokenized sequences

All intermediate columns (SMILES, InChI, InChIKey, etc.) are removed in the final step to save space.

## Notes

- All scripts use parallel processing with 64 processes for efficiency
- The ChEMBL standardization ensures consistent molecular representations across different input formats
- The WordPiece tokenizer is specifically trained on molecular SMILES data
- Make sure you have sufficient disk space as intermediate datasets are saved at each step

## References

- Bento et al. (2020). "An open source chemical structure curation pipeline using RDKit." *Journal of Cheminformatics* 12(1): 51.

## Author

Mohammad Mostafanejad  
Date: August 2025
