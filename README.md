# BERTology of Molecular Property Prediction

**(Under Construction)**

This repository provides complete access to all codebase and artifacts developed
for the BERTology project (<https://arxiv.org/abs/2603.13627>), amounting to
approximately 8TB. As such, we have organized the codebase and artifacts into a
structured format, with links to external storage locations for the large
datasets and models.

## Overview

The BERTology project focuses on understanding the impacts of various factors on
the pre-training and fine-tuning performance of BERT-based chemical language
models for molecular property prediction. These factors include training data
size, model size, tokenization algorithms, standardization noise, and randomness
in model initialization and data sampling.

## Repository Structure

The repository is organized into several main components, including data
processing scripts, visualization scripts for analyzing experimental results,
training scripts for tokenizers, workflow diagrams, and links to external
artifacts such as datasets, training logs, model artifacts, and evaluation
results.

```
bertology/
├── scripts/                   # Main scripts directory
│   ├── data_scripts/          # Data processing and preparation
│   └── plot_scripts/          # Visualization and analysis scripts
├── drawio/                    # Workflow diagrams
└── links/                     # Links to external artifacts and datasets
```

## Main Components

### 1. Data Scripts (`scripts/data_scripts/`)

Contains scripts for processing molecular data from various sources:

#### ChEMBL Standardizer (`chembl-standardizer/`)

- **Purpose**: Standardize SMILES strings using the ChEMBL Structure Pipeline
- **Pipeline**:
  1. Data cleaning (`pubchem04182025_cleaner.py`)
  2. ChEMBL standardization (`chembl_standardizer.py`)
  3. WordPiece tokenization (`wordpiece_tokenization_on_chembl_smiles.py`)

#### PubChem Generation (`pubchem-generation/`)

- **Purpose**: Download and convert PubChem compound data from FTP to JSON format
- **Scripts**:
  - `ftp_sdf_downloader.py`: Downloads SDF files from PubChem FTP server
  - `dask_runner_json.py`: Converts SDF files to JSON using Dask for parallel processing

#### PubChem Hugging Face (`pubchem-huggingface/`)

- **Purpose**: Prepare and upload PubChem dataset to Hugging Face Hub
- **Dataset**: `molssiai-hub/pubchem-04-18-2025`
- **Scripts**:
  - `pubchem-04-18-2025.py`: Dataset loader script for Hugging Face
  - `upload_data.sh`: Upload script for Hugging Face Hub

#### PubChem Preprocessing - WordPiece (`pubchem-preprocessing/`)

- **Purpose**: Extract SMILES, train WordPiece tokenizers, and tokenize molecular data
- **Pipeline**:
  1. SMILES extraction (`pubchem_cismi_writer.py`)
  2. Tokenizer training (`pubchem_tokenizer_training.py`)
  3. Data tokenization (`pubchem_data_tokenizer.py`)

#### PubChem Preprocessing - BPE (`pubchem-preprocessing-bpe/`)

- **Purpose**: Train Byte Pair Encoding (BPE) tokenizer for molecular data
- **Scripts**:
  - `bpe_tokenizer_training.py`: Trains BPE tokenizer on SMILES data

### 2. Plot Scripts (`scripts/plot_scripts/`)

Visualization and analysis scripts:

#### Pre-training Plots (`pretraining/`)

- **Dataset Size Effect** (`dataset_size_effect/`):
  - Analyzes impact of training data and model sizes on model performance
  - Generates loss and performance plots
  - Scripts:`perf_plotter.py`
  - Output: PDF plots

- **Standardization Noise Effect** (`std_effect/`):
  - Investigates the impact of standardization on model training
  - Generates heatmaps for different metrics (accuracy, perplexity, loss, F1)
  - Analyzes data corruption effects across Tiny, Small, and Base BERT models
  - Scripts: `perf_plotter.py`
  - Output: Multiple heatmap PDFs and performance plots

#### Fine-tuning Plots (`finetuning/`)

- Performance visualization for fine-tuned models
- ADME property prediction analysis
- Scripts: `perf_plotter.py`
- Output: Performance plots for validation and test sets

### 3. Workflow Diagrams (`drawio/`)

Visual representations of data processing and standardization workflows:

- `chembl_std.drawio`: ChEMBL standardization pipeline
- `pubchem_std.drawio`: PubChem standardization workflow
- `data_corruption.drawio`: Data corruption and noise analysis

### 4. Links (`links/`)

- `randomness_experiments.md`: Links to external artifacts on Zenodo for randomness studies
- `data_and_model_size_experiments.md`: Links to datasets, models, and evaluation results for dataset and model size effects
- `standardization_experiments.md`: Links to artifacts related to standardization noise effect experiments
- `tokenization_experiments.md`: Links to tokenization comparison experiments (WordPiece vs BPE)
- `finetuning_experiments.md`: Links to artifacts related to supervised
  finetuning experiments for ADME property prediction, including classical ML
  baselines and BERT models.

## Key Experiments

### 1. Randomness Experiments

- Multiple training runs with different random seeds for model initialization and data sampling
- Model sizes: Tiny-BERT, Small-BERT, Base-BERT

### 2. Dataset and Model Size Effects

- Investigates performance as a function of pre-training data size
- Both pre-training and fine-tuning performance metrics

### 3. Standardization Noise Effect

- Studies the impact of standardization on model training
- Analyzes data corruption effects on model performance
- Generates comprehensive heatmaps for various metrics

### 4. Tokenization Comparison

- Compares WordPiece vs Byte Pair Encoding (BPE) tokenization
- Impact on model performance and training efficiency

### 5. ADME Property Prediction

- Fine-tuning experiments for practical molecular property prediction
- 3-fold cross-validation and hyperparameter search for optimal performance
- Properties: HLM (Human Liver Microsomes), RLM (rat Liver Microsomes), rPPB
  (rat Plasma Protein Binding), hPPB (human Plasma Protein Binding), SOL (solubility at pH 6.8) and MDR1-ER (MDR1-MDCK efflux ratio)
- Classical ML baselines for comparison

## Data Sources

- **PubChem**: Large-scale molecular database (~118M compounds)
  - Version: 04-18-2025
  - Format: Canonical isomeric SMILES
  - https://huggingface.co/datasets/molssiai-hub/pubchem-04-18-2025
- **ADME datasets**: Publicly available property prediction datasets
  - https://github.com/molecularinformatics/Computational-ADME
  - https://polarishub.io/datasets/biogen/adme-fang-v1

## Prerequisites

- Python 3.x
- PyTorch
- Hugging Face Transformers and Tokenizers
- RDKit
- OpenEye toolkit (for SDF processing)
- Dask (for parallel processing)
- ChEMBL Structure Pipeline
- Hydra (for configuration management)
- Draw.io (optional, for viewing workflow diagrams)

## Citation

If you use this code or data in your research, please cite the following preprint:

https://arxiv.org/abs/2603.13627

```
@misc{mostafanejad:2026:bertology,
      title={BERTology of Molecular Property Prediction},
      author={Mohammad Mostafanejad and Paul Saxe and T. Daniel Crawford},
      year={2026},
      eprint={2603.13627},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.13627},
}
```

## License

Please refer to individual dataset and software licenses for usage terms.

## Contributing

Before opening an issue, please check the existing issues to see if your
question has already been addressed. Otherwise, we recommend reaching out via
opening a discussion in the repository or contacting the author directly. 

Pull requests are welcome, but please ensure that they are minimalist, clear and
tested.

## Contact

For inquires or comments about the codebase, models or datasets, please contact
the author at `smostafanejad[at]vt.edu`.

## Author

Mohammad Mostafanejad  
Date: March 2026