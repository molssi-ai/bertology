---
license:
  - pddl
license_link: https://opendatacommons.org/licenses/pddl
tags:
  - pubchem
  - small-molecules
  - InChI
  - SMILES
  - molecular-geometry
  - molecular-properties
  - chemical-properties
  - cheminformatics
annotations_creators:
  - crowdsourced
pretty_name: pubchem-04-18-2025
size_categories:
  - 100M<n<200M
source_datasets:
  - pubchem-compound
  - pubchem-04-18-2025
task_categories:
  - tabular-regression
  - other
task_ids:
  - tabular-single-column-regression
viewer: false
configs:
  - config_name: pubchem-04-18-2025
    data_files:
      - split: train
        path: "data/train/*.json"
    default: true
---

# PubChem Dataset (version 04-18-2025)

* **Important Note:** The current version of the PubChem dataset includes a new
  entry, `PUBCHEM_SMILES`, which is equivalent to isomeric SMILES and contains
  both stereochemical and isotopic information. This entry is set to replace
  both `PUBCHEM_OPENEYE_CAN_SMILES` and `PUBCHEM_OPENEYE_ISO_SMILES` in the future.
  For further details, please refer to the [PubChem documentation](https://pubchem.ncbi.nlm.nih.gov/docs/glossary#section=SMILES).

## Table of Contents

- [PubChem Dataset (version 04-18-2025)](#pubchem-dataset-version-04-18-2025)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits and Configurations](#data-splits-and-configurations)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://pubchem.ncbi.nlm.nih.gov
- **Paper:** https://doi.org/10.1093/nar/gkac956
- **Point of Contact:** [Sunghwan Kim](kimsungh@ncbi.nlm.nih.gov)
- **Point of Contact:** [Mohammad Mostafanejad](smostafanejad@vt.edu)
- **Point of Contact:** [MolSSI-AI Hub](hub@molssi.org)

### Dataset Summary

[PubChem](https://pubchem.ncbi.nlm.nih.gov) is a popular chemical information
resource that serves a wide range of use cases. It is an open chemistry
database at the National Institutes of Health (NIH). "Open" means that you can
put your scientific data in PubChem and that others may use it. Since the launch
in 2004, PubChem has become a key chemical information resource for scientists,
students, and the general public. Each month our website and programmatic
services provide data to several million users worldwide.

PubChem mostly contains small molecules, but also larger molecules such as
nucleotides, carbohydrates, lipids, peptides, and chemically-modified
macromolecules. PubChem collects information on chemical structures,
identifiers, chemical and physical properties, biological activities, patents,
health, safety, toxicity data, and many others.

## Dataset Structure

### Data Instances

An example of a data instance is as follows:

```json
{
'PUBCHEM_COMPOUND_CID': '12',
'PUBCHEM_COMPOUND_CANONICALIZED': '1',
'PUBCHEM_CACTVS_COMPLEXITY': 104.0,
'PUBCHEM_CACTVS_HBOND_ACCEPTOR': 4,
'PUBCHEM_CACTVS_HBOND_DONOR': 4,
'PUBCHEM_CACTVS_ROTATABLE_BOND': 0,
'PUBCHEM_CACTVS_SUBSKEYS': 'AAADcYBgOAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAAAABAAAAGgAACAAACASAkAAwBoAAAgCAACBCAAACAAAgIAAAiAAGiIgJJyKCERKAcAElwBUJmAfAYAQAAQAACAAAQAACAAAQAACAAAAAAAAAAA==',
'PUBCHEM_IUPAC_OPENEYE_NAME': 'benzene-1,2,3,5-tetrol',
'PUBCHEM_IUPAC_CAS_NAME': 'benzene-1,2,3,5-tetrol',
'PUBCHEM_IUPAC_NAME_MARKUP': 'benzene-1,2,3,5-tetrol',
'PUBCHEM_IUPAC_NAME': 'benzene-1,2,3,5-tetrol',
'PUBCHEM_IUPAC_SYSTEMATIC_NAME': 'benzene-1,2,3,5-tetrol',
'PUBCHEM_IUPAC_TRADITIONAL_NAME': 'benzene-1,2,3,5-tetrol',
'PUBCHEM_IUPAC_INCHI': 'InChI=1S/C6H6O4/c7-3-1-4(8)6(10)5(9)2-3/h1-2,7-10H',
'PUBCHEM_IUPAC_INCHIKEY': 'RDJUHLUBPADHNP-UHFFFAOYSA-N',
'PUBCHEM_XLOGP3_AA': None,
'PUBCHEM_EXACT_MASS': 142.02660867,
'PUBCHEM_MOLECULAR_FORMULA': 'C6H6O4',
'PUBCHEM_MOLECULAR_WEIGHT': 142.11,
'PUBCHEM_OPENEYE_CAN_SMILES': 'C1=C(C=C(C(=C1O)O)O)O',
'PUBCHEM_OPENEYE_ISO_SMILES': 'C1=C(C=C(C(=C1O)O)O)O',
'PUBCHEM_CACTVS_TPSA': 80.9,
'PUBCHEM_MONOISOTOPIC_WEIGHT': 142.02660867,
'PUBCHEM_TOTAL_CHARGE': 0,
'PUBCHEM_HEAVY_ATOM_COUNT': 10,
'PUBCHEM_ATOM_DEF_STEREO_COUNT': 0,
'PUBCHEM_ATOM_UDEF_STEREO_COUNT': 0,
'PUBCHEM_BOND_DEF_STEREO_COUNT': 0,
'PUBCHEM_BOND_UDEF_STEREO_COUNT': 0,
'PUBCHEM_ISOTOPIC_ATOM_COUNT': 0,
'PUBCHEM_COMPONENT_COUNT': 1,
'PUBCHEM_CACTVS_TAUTO_COUNT': 15,
'PUBCHEM_COORDINATE_TYPE': [1, 5, 255],
'PUBCHEM_BONDANNOTATIONS': [5,
6,
8,
...,
8],
'COORDS': [4.269000053405762,
2.0,
0.0,
...,
0.0],
'ATOMIC_INDICES': [1, 2, 3, ..., 16],
'ATOMIC_SYMBOLS': ['O',
'O',
'O',
...,
'H'],
'ATOMIC_NUMBERS': [8, 8, 8, ..., 1],
'ATOMIC_FORMAL_CHARGES': [0, 0, 0, ..., 0],
'BOND_ORDERS': [1,
5,
1,
...,
1],
'PUBCHEM_XLOGP3': '0.8',
'PUBCHEM_NONSTANDARDBOND': None,
'PUBCHEM_REFERENCE_STANDARDIZATION': None
}
```

### Data Fields

| Field                             | Description                                                            |
| --------------------------------- | ---------------------------------------------------------------------- |
| PUBCHEM_COMPOUND_CID              | PubChem Compound ID                                                    |
| PUBCHEM_COMPOUND_CANONICALIZED    | Canonicalized form of the compound computed by OEChem 2.3.0            |
| PUBCHEM_CACTVS_COMPLEXITY         | Complexity of the compound computed by Cactvs 3.4.8.18                 |
| PUBCHEM_CACTVS_HBOND_ACCEPTOR     | Number of hydrogen bond acceptors computed by Cactvs 3.4.8.18          |
| PUBCHEM_CACTVS_HBOND_DONOR        | Number of hydrogen bond donors computed by Cactvs 3.4.8.18             |
| PUBCHEM_CACTVS_ROTATABLE_BOND     | Number of rotatable bonds computed by Cactvs 3.4.8.18                  |
| PUBCHEM_CACTVS_SUBSKEYS           | Substructure keys computed by Cactvs 3.4.8.18                          |
| PUBCHEM_IUPAC_OPENEYE_NAME        | IUPAC name of the compound computed by OEChem 2.3.0                    |
| PUBCHEM_IUPAC_CAS_NAME            | CAS name of the compound                                               |
| PUBCHEM_IUPAC_NAME_MARKUP         | IUPAC name markup                                                      |
| PUBCHEM_IUPAC_NAME                | IUPAC name computed by Lexichem TK 2.7.0                               |
| PUBCHEM_IUPAC_SYSTEMATIC_NAME     | IUPAC systematic name                                                  |
| PUBCHEM_IUPAC_TRADITIONAL_NAME    | IUPAC traditional name                                                 |
| PUBCHEM_IUPAC_INCHI               | InChI of the compound computed by InChI 1.0.6                          |
| PUBCHEM_IUPAC_INCHIKEY            | InChI key of the compound computed by InChI 1.0.6                      |
| PUBCHEM_XLOGP3_AA                 | XLogP3 with atom additive model computed by XLogP3 3.0                 |
| PUBCHEM_EXACT_MASS                | Exact mass of the compound computed by PubChem 2.2                     |
| PUBCHEM_MOLECULAR_FORMULA         | Molecular formula of the compound computed by PubChem 2.2              |
| PUBCHEM_MOLECULAR_WEIGHT          | Molecular weight of the compound computed by PubChem 2.2               |
| PUBCHEM_SMILES                    | Isomeric SMILES (deposited or) computed by OEChem 2.3.0                |
| PUBCHEM_OPENEYE_CAN_SMILES        | Canonical SMILES of the compound computed by OEChem 2.3.0              |
| PUBCHEM_OPENEYE_ISO_SMILES        | Isomeric SMILES of the compound computed by OEChem 2.3.0               |
| PUBCHEM_CACTVS_TPSA               | Topological polar surface area computed by Cactvs 3.4.8.18             |
| PUBCHEM_MONOISOTOPIC_WEIGHT       | Monoisotopic weight of the compound computed by PubChem 2.2            |
| PUBCHEM_TOTAL_CHARGE              | Total charge of the compound computed by PubChem                       |
| PUBCHEM_HEAVY_ATOM_COUNT          | Number of heavy atoms in the compound computed by PubChem              |
| PUBCHEM_ATOM_DEF_STEREO_COUNT     | Number of defined stereo centers in the compound computed by PubChem   |
| PUBCHEM_ATOM_UDEF_STEREO_COUNT    | Number of undefined stereo centers in the compound computed by PubChem |
| PUBCHEM_BOND_DEF_STEREO_COUNT     | Number of defined stereo bonds in the compound computed by PubChem     |
| PUBCHEM_BOND_UDEF_STEREO_COUNT    | Number of undefined stereo bonds in the compound computed by PubChem   |
| PUBCHEM_ISOTOPIC_ATOM_COUNT       | Number of isotopic atoms in the compound computed by PubChem           |
| PUBCHEM_COMPONENT_COUNT           | Number of components in the compound computed by PubChem               |
| PUBCHEM_CACTVS_TAUTO_COUNT        | Number of tautomers of the compound computed by Cactvs 3.4.8.18        |
| PUBCHEM_COORDINATE_TYPE           | Coordinate type                                                        |
| PUBCHEM_BONDANNOTATIONS           | Bond annotations                                                       |
| COORDS                            | Cartesian coordinates of the molecular geometry                        |
| ATOMIC_INDICES                    | Atomic indices                                                         |
| ATOMIC_SYMBOLS                    | Atomic symbols                                                         |
| ATOMIC_NUMBERS                    | Atomic numbers                                                         |
| ATOMIC_FORMAL_CHARGES             | Atomic formal charges                                                  |
| BOND_ORDERS                       | Bond orders                                                            |
| PUBCHEM_XLOGP3                    | XLogP3 computed by XLogP3 3.0                                          |
| PUBCHEM_NONSTANDARDBOND           | Non-standard bond                                                      |
| PUBCHEM_REFERENCE_STANDARDIZATION | Reference standardization                                              |

### Data Splits and Configurations

The dataset has only one `train` split and one configuration/subset:

- `pubchem-04-18-2025` (default)

## Dataset Creation

### Curation Rationale

The present version of PubChem dataset has been extracted from its original
ftp repository, transformed into a dictionary and stored in the `.json`
format.

### Source Data

The link to the original PubChem dataset FTP repository can be found
[here](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/)

#### Initial Data Collection and Normalization

Other than the changes detailed in Sec. [Curation Rationale](#curation-rationale),
no data modification has been performed on the PubChem dataset.

### Personal and Sensitive Information

The PubChem dataset does not involve any personal or sensitive information.

## Considerations for Using the Data

### Social Impact of Dataset

The PubChem dataset paves the way for applications in drug discovery and materials science, among others.

## Additional Information

### Dataset Curators

- **Sunghwan Kim**, National Center for Biotechnology Information, National
  Library of Medicine, National Institutes of Health, Department of Health and
  Human Services, Bethesda, MD, 20894 USA
- **Jie Chen**, National Center for Biotechnology Information, National Library
  of Medicine, National Institutes of Health, Department of Health and Human
  Services, Bethesda, MD, 20894 USA
- **Tiejun Cheng**, National Center for Biotechnology Information, National
  Library of Medicine, National Institutes of Health, Department of Health and
  Human Services, Bethesda, MD, 20894 USA
- **Asta Gindulyte**, National Center for Biotechnology Information, National
  Library of Medicine, National Institutes of Health, Department of Health and
  Human Services, Bethesda, MD, 20894 USA
- **Jia He**, National Center for Biotechnology Information, National Library of
  Medicine, National Institutes of Health, Department of Health and Human
  Services, Bethesda, MD, 20894 USA
- **Siqian He**, National Center for Biotechnology Information, National Library
  of Medicine, National Institutes of Health, Department of Health and Human
  Services, Bethesda, MD, 20894 USA
- **Qingliang Li**, National Center for Biotechnology Information, National
  Library of Medicine, National Institutes of Health, Department of Health and
  Human Services, Bethesda, MD, 20894 USA
- **Benjamin A Shoemaker**, National Center for Biotechnology Information, National
  Library of Medicine, National Institutes of Health, Department of Health and
  Human Services, Bethesda, MD, 20894 USA
- **Paul A Thiessen**, National Center for Biotechnology Information, National
  Library of Medicine, National Institutes of Health, Department of Health and
  Human Services, Bethesda, MD, 20894 USA
- **Bo Yu**, National Center for Biotechnology Information, National Library of
  Medicine, National Institutes of Health, Department of Health and Human
  Services, Bethesda, MD, 20894 USA
- **Leonid Zaslavsky**, National Center for Biotechnology Information, National
  Library of Medicine, National Institutes of Health, Department of Health and
  Human Services, Bethesda, MD, 20894 USA
- **Jian Zhang**, National Center for Biotechnology Information, National
  Library of Medicine, National Institutes of Health, Department of Health and
  Human Services, Bethesda, MD, 20894 USA
- **Evan E Bolton**, National Center for Biotechnology Information, National
  Library of Medicine, National Institutes of Health, Department of Health and
  Human Services, Bethesda, MD, 20894 USA

### Licensing Information

[Free Public Domain License](https://www.ncbi.nlm.nih.gov/home/about/policies/#data)

### Citation Information

```tex
@article{Kim:2022:D1373,
    author = {Kim, Sunghwan and Chen, Jie and Cheng, Tiejun and Gindulyte, Asta and He, Jia and He, Siqian and Li, Qingliang and Shoemaker, Benjamin A and Thiessen, Paul A and Yu, Bo and Zaslavsky, Leonid and Zhang, Jian and Bolton, Evan E},
    title = "{PubChem 2023 update}",
    journal = {Nucleic Acids Research},
    volume = {51},
    pages = {D1373-D1380},
    year = {2022},
    doi = {10.1093/nar/gkac956}
}
```

### Contributions

- **Mohammad Mostafanejad**, The Molecular Sciences Software Institute (MolSSI)
