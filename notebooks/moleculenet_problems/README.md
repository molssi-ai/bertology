# MoleculeNet BBBP Curation Analysis

This directory contains a notebook for analyzing curation issues in the MoleculeNet BBBP dataset, with a focus on duplicate records, conflicting labels, and invalid SMILES strings.

## Overview

The notebook reproduces and checks claims about data quality issues in the BBBP benchmark dataset from MoleculeNet:

1. **Duplicate Entries**: identify canonical SMILES that appear more than once
2. **Conflicting Labels**: find duplicated molecules that are assigned both labels
3. **Invalid SMILES**: detect rows that cannot be parsed by RDKit

The analysis reads the BBBP CSV file, canonicalizes the SMILES strings, groups rows by canonical structure, and writes summary tables plus molecule images to disk.

## Prerequisites

- Python 3.x
- Required packages:
  - `pandas`
  - `rdkit`
  - `deepchem` (for optional dataset download)
  - `matplotlib` or Jupyter-compatible image support for notebook display

## Notebook Workflow

### Step 1: Load the BBBP Dataset

**Notebook**: `moleculenet_bbbp.ipynb`

This step loads the BBBP dataset from `./data/BBBP.csv` and performs basic sanity checks on the expected columns.

**Input**: `./data/BBBP.csv`

**Alternative Input**: download the dataset directly from DeepChem.

**What it does**:
- Loads the CSV file with pandas
- Checks that the SMILES and label columns exist
- Adds an original row index for traceability

### Step 2: Canonicalize SMILES

The notebook converts each SMILES string to a canonical RDKit SMILES representation.

**What it does**:
- Parses each molecule with RDKit
- Converts valid entries to canonical SMILES
- Separates invalid rows from valid rows

### Step 3: Detect Duplicates and Conflicts

The notebook groups rows by canonical SMILES and compares labels across duplicate structures.

**What it does**:
- Counts repeated canonical SMILES
- Builds a duplicate summary table
- Identifies molecules with inconsistent labels
- Builds a conflict summary table

### Step 4: Save Outputs

The notebook writes CSV summaries and molecule images into the `outputs/` directory.

**What it does**:
- Saves invalid SMILES rows
- Saves duplicate rows and duplicate summaries
- Saves conflicting rows and conflicting summaries
- Saves duplicate count tables
- Renders duplicate and conflicting structures as PNG grids
- Saves individual PNGs for conflicting molecules in a subfolder

## Running the Notebook

Open `moleculenet_bbbp.ipynb` and run the cells in order. If you want to refresh the dataset manually, you can either use DeepChem or download the CSV directly:

```bash
# Option 1: download with DeepChem inside the notebook
from deepchem.molnet import load_bbbp
bbbp = load_bbbp(data_dir="./data")

# Option 2: download directly
wget -P ./data https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv
```

## Data Flow

```
BBBP.csv
    ↓
[moleculenet_bbbp.ipynb]
    ↓
Canonical SMILES + label comparison
    ↓
CSV summaries and molecule images in outputs/
```

## Output Structure

The notebook creates the following files in `outputs/`:

- `bbbp_invalid_smiles.csv`
- `bbbp_duplicate_rows.csv`
- `bbbp_duplicate_summary.csv`
- `bbbp_duplicate_counts.csv`
- `bbbp_conflict_rows.csv`
- `bbbp_conflict_summary.csv`
- `bbbp_duplicate_structures_page_*.png`
- `bbbp_conflict_structures_page_*.png`
- `conflict_singletons/`

## Notes

- The notebook assumes the BBBP file is available as `./data/BBBP.csv`.
- RDKit is required for canonicalization and molecule rendering.
- The output directory is created automatically if it does not already exist.
- The analysis is intended to support manual inspection of MoleculeNet curation problems rather than to train a model.

## Author

Mohammad Mostafanejad  
Date: June 2026