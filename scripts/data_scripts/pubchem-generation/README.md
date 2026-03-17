# PubChem SDF to JSON Conversion Pipeline

This directory contains scripts for downloading and processing PubChem compound data from the FTP server into JSON format suitable for uploading to Hugging Face Hub.

## Overview

The pipeline downloads SDF files from PubChem's FTP server and converts them to JSON format through the following stages:

1. **FTP Download**: Download compressed SDF files from PubChem's FTP server
2. **JSON Conversion**: Convert SDF files to JSON format using Dask for parallel processing

## Prerequisites

- Python 3.x
- Required packages:
  - `wget`
  - `tqdm`
  - `openeye` (OEChem toolkit)
  - `dask`
  - `distributed`
  - `pandas`
  - `numpy`

## Pipeline Steps

### Step 1: Download SDF Files from PubChem FTP

**Script**: `ftp_sdf_downloader.py`

This step downloads compressed SDF files from the PubChem FTP server. PubChem organizes compounds in batches of 500,000 structures per file.

**FTP Source**: `ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/`

**Output Directory**: `./ftp_pubchem_compound_data`

**Usage**:
```bash
python ftp_sdf_downloader.py
```

**What it does**:
- Creates a local directory for storing downloaded files
- Downloads SDF files in batches of 500,000 compounds each
- Shows progress with a progress bar
- Automatically creates output directory if it doesn't exist

**Download Details**:
- Total compounds: ~118 million
- Batch size: 500,000 compounds per file
- File format: Gzipped SDF (.sdf.gz)

### Step 2: Convert SDF to JSON Format

**Script**: `dask_runner_json.py`

This step converts the downloaded SDF files to JSON format using parallel processing with Dask. The script uses OpenEye's OEChem toolkit to parse molecular structures and extract properties.

**Input Directory**: `./ftp_pubchem_compound_data/*.sdf.gz`

**Output Directory**: `./outputs`

**Dask Dashboard**: `http://localhost:8889`

**Usage**:
```bash
python dask_runner_json.py
```

**What it does**:
- Sets up an adaptive Dask cluster for parallel processing
  - Starts with 1 worker, scales up to 64 workers as needed
  - Each worker uses 1 thread with 1GB memory limit
- For each SDF file:
  - Reads molecular structures using OEChem
  - Extracts PubChem properties and metadata
  - Extracts 3D coordinates
  - Extracts atomic symbols and atomic numbers
  - Processes special tags (bond annotations, counts, masses, etc.)
- Converts data types appropriately:
  - Integer fields: counts, charges, TPSA, rotatable bonds, etc.
  - Float fields: masses, weights, surface areas
  - Arrays: coordinates, bond annotations
- Saves each batch as a JSON file with proper indentation
- Handles errors gracefully with dedicated error logging
- Processes files in parallel batches of 64

**Dask Cluster Configuration**:
- **Cluster Name**: PubChem-OEChem
- **Initial Workers**: 1
- **Maximum Workers**: 64 (adaptive scaling)
- **Threads per Worker**: 1
- **Memory Limit**: 1GB per worker
- **Scratch Directory**: `./scratch`
- **Dashboard Port**: 8889

## Running the Complete Pipeline

To process the PubChem data from start to finish, run the scripts in order:

```bash
# Step 1: Download SDF files from PubChem FTP
python ftp_sdf_downloader.py

# Step 2: Convert SDF files to JSON format
python dask_runner_json.py
```

## Data Flow

```
PubChem FTP Server
    ↓
[ftp_sdf_downloader.py]
    ↓
Local SDF Files (./ftp_pubchem_compound_data/*.sdf.gz)
    ↓
[dask_runner_json.py]
    ↓
JSON Files (./outputs/*.json)
    ↓
Ready for Hugging Face Hub Upload
```

## Directory Structure

```
pubchem-generation/
├── ftp_sdf_downloader.py      # Step 1: Download SDF files
├── dask_runner_json.py         # Step 2: Convert to JSON
├── ftp_pubchem_compound_data/  # Downloaded SDF files
├── outputs/                    # Generated JSON files
└── scratch/                    # Dask temporary files
```

### Conversion Progress
The Dask runner provides:
- Progress bar for file processing
- Real-time dashboard at `http://localhost:8889`
- Error logging in `.err` files alongside JSON outputs

## Notes

- The script automatically skips files that have already been converted
- JSON files are formatted with indentation for readability (can be modified for space efficiency)
- The Dask cluster uses local storage for temporary files in `./scratch`
- Each worker is limited to 1GB memory to prevent system overload
- Bond annotations and coordinate types are parsed as integer arrays
- All coordinate data is stored as float64 arrays

## References

- PubChem FTP Documentation: https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/README-Compound-SDF
- OpenEye OEChem Toolkit: https://docs.eyesopen.com/toolkits/python/oechemtk/
- Dask Documentation: https://docs.dask.org/

## Author

Mohammad Mostafanejad  
Date: April 2025