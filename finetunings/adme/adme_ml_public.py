###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: December 2025                                                         #
# Description:                                                                #
# This script is a modified version of Biogen's manuscript code that adds the #
# standardized PubChem SMILES to the generated dataset from SDF files using   #
# PubChem API. See the following link for the original code and source SDF    #
# files:                                                                      #
# https://github.com/molecularinformatics/Computational-ADME/tree/main/ML     #
###############################################################################

import os
import re
import logging
from tqdm import tqdm
import requests
from time import perf_counter_ns, sleep
from requests.exceptions import ConnectionError, Timeout
import pandas as pd
from datasets import DatasetDict, load_dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# quash the rdkit warnings
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

# set the PubChem PUG REST URL
pug_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def PC_standardize(structures):
    """Use the PubChem standardization service with SMILES.

    Parameters
    ----------
    structures : [str]
        The SMILES of the structures to standardize

    Returns
    -------
    [{str: str}]
        A list of dicts of the results.

    Note:
        The response contains information to throttle the requests, like this:

        X-Throttling-Control: Request Count status: Green (0%),
                              Request Time status: Green (0%),
                              Service status: Green (20%)
    """
    results = []
    problematic_smiles = []
    max_count = 0
    max_time = 0
    max_service = 0
    n_sleep = 0
    with requests.Session() as session:
        t0 = perf_counter_ns()
        for SMILES in structures:
            while True:
                try:
                    response = session.post(
                        f"{pug_url}/standardize/SMILES/JSON",
                        data={"smiles": SMILES, "include_components": False},
                    )
                except (ConnectionError, Timeout) as e:
                    print(f"Connection error: {e}. Retrying...")
                    sleep(30)
                    continue

                # Check the throttling request
                if "X-Throttling-Control" in response.headers:
                    throttling_header = response.headers["X-Throttling-Control"]
                    match = re.search(
                        r"Request Count status: \w+ \((\d+)%\), "
                        r"Request Time status: \w+ \((\d+)%\), "
                        r"Service status: \w+ \((\d+)%\)",
                        throttling_header,
                    )
                    if match:
                        count = int(match.group(1))
                        time = int(match.group(2))
                        service = int(match.group(3))

                        max_count = max(count, max_count)
                        max_time = max(time, max_time)
                        max_service = max(service, max_service)

                        if count > 75 or time > 75 or service > 75:
                            n_sleep += 1
                            sleep(1)

                if response.status_code == 503:
                    print("status code = 503")
                    continue

                if response.status_code == 400:
                    print(f"status code = 400 for SMILES {SMILES}")
                    problematic_smiles.append(SMILES)
                    break

                response.raise_for_status()

                break

            data = response.json()

            if not data.get("PC_Compounds"):
                print(f"No compounds found for SMILES {SMILES}")
                continue

            # Pull out some info from the data make sure to catch key errors
            cmpds = data["PC_Compounds"]
            if len(cmpds) > 1:
                raise ValueError(
                    f"There are {len(cmpds)} in the PubChem standardization!"
                )

            cmpd = cmpds[0]
            result = {"original SMILES": SMILES}
            if "id" in cmpd and "id" in cmpd["id"] and "cid" in cmpd["id"]["id"]:
                result["cid"] = cmpd["id"]["id"]["cid"]

            for props in cmpd["props"]:
                if "urn" in props and "value" in props and "label" in props["urn"]:
                    key = props["urn"]["label"]
                    value = props["value"]
                    _type = [*value.keys()][0]
                    if _type == "sval":
                        result[key] = value[_type]
                    else:
                        raise TypeError(f"Unknown type {_type} in compound properties.")

            results.append(result)
        t1 = perf_counter_ns()
        t = (t1 - t0) / 1000000000
        n = len(structures)
        per = t / n

    return {
        "data": results,
        "n_structures": len(structures),
        "throttling": {
            "maximum count": max_count,
            "maximum time": max_time,
            "maximum service": max_service,
            "number of slowdowns": n_sleep,
        },
        "time": {
            "total": t,
            "per structure": per,
        },
    }


# A batch version of the PubChem standardizer
def pubchem_standardizer_batch(smi: str):
    """
    Standardize SMILES using PubChem's standardization routine and collect selected identifiers.

    This function is a thin wrapper around PC_standardize(smi). It expects the result
    from PC_standardize to be a mapping with a "data" key that contains an iterable
    of mapping-like records. For each record in results["data"] the function extracts
    the original SMILES, the PubChem CID (converted to int when present), the
    PubChem-standardized SMILES, and the PubChem InChI. Missing fields are returned
    as None.

    Parameters
    ----------
    smi : str or Sequence[str]
        Input SMILES or an object accepted by PC_standardize. Typically this is a single
        SMILES string or a sequence/list of SMILES that PC_standardize can process.

    Returns
    -------
    dict
        Dictionary with the following keys and values (all lists of the same length,
        corresponding to the entries in results["data"]):
        - "original_SMILES" : list[str | None]
            The original SMILES strings as returned by PC_standardize under the key
            "original SMILES", or None when that key is missing for an entry.
        - "pubchem_cid" : list[int | None]
            PubChem CIDs converted to int where present (res["cid"]), otherwise None.
        - "pubchem_SMILES" : list[str | None]
            The standardized SMILES strings returned under the key "SMILES", or None.
        - "pubchem_InChI" : list[str | None]
            The InChI strings returned under the key "InChI", or None.

    Raises
    ------
    KeyError
        If the top-level "data" key is missing from the mapping returned by PC_standardize.
    TypeError
        If the value of results["data"] is not iterable or its elements are not
        mapping-like (i.e., do not support .get or key-based access).
    """
    results = PC_standardize(smi)
    return {
        "original_SMILES": [
            res.get("original SMILES", None) for res in results["data"]
        ],
        "pubchem_cid": [
            int(res["cid"]) if res.get("cid", None) is not None else None
            for res in results["data"]
        ],
        "pubchem_SMILES": [res.get("SMILES", None) for res in results["data"]],
        "pubchem_InChI": [res.get("InChI", None) for res in results["data"]],
    }


def standardize(mol):
    """
    Custom standardization function for molecular structures using RDKit's standardization tools.

    This function performs a series of molecular standardization steps including
    cleanup, fragment removal, neutralization, and tautomer canonicalization to
    produce a standardized representation of the input molecule.
    For details see:
    https://github.com/molecularinformatics/Computational-ADME/blob/main/ML/ADME_ML_public.py#L63-L88

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input RDKit molecule object to be standardized.

    Returns
    -------
    rdkit.Chem.Mol
        Standardized molecule object. If standardization fails, attempts to
        reconstruct the molecule from its SMILES representation. If that also
        fails, returns the original molecule.

    Notes
    -----
    The standardization process includes the following steps:
    1. Cleanup: Remove hydrogens, disconnect metal atoms, normalize and reionize
    2. Fragment removal: Extract the parent fragment (largest component)
    3. Neutralization: Remove charges where chemically appropriate
    4. Tautomer canonicalization: Convert to canonical tautomer form

    If any step in the standardization process fails, the function falls back to
    converting the molecule to SMILES and back. If that also fails, the original
    molecule is returned unchanged.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CC(=O)[O-].[Na+]')
    >>> std_mol = standardize(mol)
    >>> Chem.MolToSmiles(std_mol)
    'CC(=O)O'
    """
    try:
        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol)

        # if many fragments, get the "parent" (the actual mol we are interested in)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = (
            rdMolStandardize.Uncharger()
        )  # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        # try to Canonicalize tautomers
        te = rdMolStandardize.TautomerEnumerator()
        mol_final = te.Canonicalize(uncharged_parent_clean_mol)
    except:
        try:
            mol_final = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        except:
            mol_final = mol

    return mol_final


if __name__ == "__main__":
    # Create both Morgan Fingerprints (FCFP4) and RDkit Molecular Descriptors
    descType = "FCFP4_rdMolDes"

    # create a mapping from ADME tag to target property in the SDF file
    prop_to_tag = {
        "ADME_HLM": "LOG HLM_CLint (mL/min/kg)",
        "ADME_hPPB": "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)",
        "ADME_rPPB": "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)",
        "ADME_RLM": "LOG RLM_CLint (mL/min/kg)",
        "ADME_Sol": "LOG SOLUBILITY PH 6.8 (ug/mL)",
        "ADME_MDR1_ER": "LOG MDR1-MDCK ER (B-A/A-B)",
    }

    for sdf_file, ADME_tag in tqdm(prop_to_tag.items(), total=len(prop_to_tag)):
        # get the ADME property values & descriptors
        act = {}
        maccs = {}
        fcfp4_bit = {}
        rdMD = {}
        rdkit_smiles = {}
        name_list = []

        # specify the sdf file and corresponding ADME tag
        sdFile = Chem.SDMolSupplier(
            os.path.join(os.getcwd(), "data", "%s.sdf" % sdf_file)
        )

        # count the total number of molecules
        mols = [mol for mol in sdFile if mol is not None]
        total_mols = len(mols)

        i = 1
        for mol in tqdm(sdFile, total=total_mols):
            if mol is not None:
                mol = standardize(mol)
                try:
                    molName = mol.GetProp("_Name")
                except:
                    try:
                        molName = mol.GetProp("Vendor ID")
                    except:
                        molName = "Molecule_%s" % i
                name_list.append(molName)
                try:
                    activity = mol.GetProp("%s" % ADME_tag)
                except KeyError:
                    activity = "0.0000"
                act[molName] = float(activity)
                # get the smiles here
                rdkit_smiles[molName] = Chem.MolToSmiles(
                    mol,
                    isomericSmiles=True,
                    kekuleSmiles=False,
                    rootedAtAtom=-1,
                    canonical=True,
                    allBondsExplicit=False,
                    allHsExplicit=False,
                )
                MDlist = []
                try:
                    MDlist.append(rdMolDescriptors.CalcTPSA(mol))
                    MDlist.append(rdMolDescriptors.CalcFractionCSP3(mol))
                    MDlist.append(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))
                    MDlist.append(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
                    MDlist.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
                    MDlist.append(rdMolDescriptors.CalcNumAmideBonds(mol))
                    MDlist.append(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
                    MDlist.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
                    MDlist.append(rdMolDescriptors.CalcNumAromaticRings(mol))
                    MDlist.append(rdMolDescriptors.CalcNumLipinskiHBA(mol))
                    MDlist.append(rdMolDescriptors.CalcNumLipinskiHBD(mol))
                    MDlist.append(rdMolDescriptors.CalcNumHeteroatoms(mol))
                    MDlist.append(rdMolDescriptors.CalcNumRings(mol))
                    MDlist.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
                    MDlist.append(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol))
                    MDlist.append(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))
                    MDlist.append(rdMolDescriptors.CalcNumSaturatedRings(mol))
                    MDlist.append(rdMolDescriptors.CalcHallKierAlpha(mol))
                    MDlist.append(rdMolDescriptors.CalcKappa1(mol))
                    MDlist.append(rdMolDescriptors.CalcKappa2(mol))
                    MDlist.append(rdMolDescriptors.CalcKappa3(mol))
                    MDlist.append(rdMolDescriptors.CalcChi0n(mol))
                    MDlist.append(rdMolDescriptors.CalcChi0v(mol))
                    MDlist.append(rdMolDescriptors.CalcChi1n(mol))
                    MDlist.append(rdMolDescriptors.CalcChi1v(mol))
                    MDlist.append(rdMolDescriptors.CalcChi2n(mol))
                    MDlist.append(rdMolDescriptors.CalcChi2v(mol))
                    MDlist.append(rdMolDescriptors.CalcChi3n(mol))
                    MDlist.append(rdMolDescriptors.CalcChi3v(mol))
                    MDlist.append(rdMolDescriptors.CalcChi4n(mol))
                    MDlist.append(rdMolDescriptors.CalcChi4v(mol))
                    MDlist.append(rdMolDescriptors.CalcAsphericity(mol))
                    MDlist.append(rdMolDescriptors.CalcEccentricity(mol))
                    MDlist.append(rdMolDescriptors.CalcInertialShapeFactor(mol))
                    MDlist.append(rdMolDescriptors.CalcExactMolWt(mol))
                    MDlist.append(rdMolDescriptors.CalcPBF(mol))
                    MDlist.append(rdMolDescriptors.CalcPMI1(mol))
                    MDlist.append(rdMolDescriptors.CalcPMI2(mol))
                    MDlist.append(rdMolDescriptors.CalcPMI3(mol))
                    MDlist.append(rdMolDescriptors.CalcRadiusOfGyration(mol))
                    MDlist.append(rdMolDescriptors.CalcSpherocityIndex(mol))
                    MDlist.append(rdMolDescriptors.CalcLabuteASA(mol))
                    MDlist.append(rdMolDescriptors.CalcNPR1(mol))
                    MDlist.append(rdMolDescriptors.CalcNPR2(mol))
                    for d in rdMolDescriptors.PEOE_VSA_(mol):
                        MDlist.append(d)
                    for d in rdMolDescriptors.SMR_VSA_(mol):
                        MDlist.append(d)
                    for d in rdMolDescriptors.SlogP_VSA_(mol):
                        MDlist.append(d)
                    for d in rdMolDescriptors.MQNs_(mol):
                        MDlist.append(d)
                    for d in rdMolDescriptors.CalcCrippenDescriptors(mol):
                        MDlist.append(d)
                    for d in rdMolDescriptors.CalcAUTOCORR2D(mol):
                        MDlist.append(d)
                except:
                    print("The RDdescritpor calculation failed!")

                rdMD[molName] = MDlist

                # Morgan (Circular) Fingerprints (FCFP4) BitVector
                try:
                    fcfp4_bit_fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, 2, useFeatures=True, nBits=1024
                    )
                    fcfp4_bit[molName] = fcfp4_bit_fp.ToBitString()
                except:
                    fcfp4_bit[molName] = ""
                    print("The FCFP4 calculation failed!")

            i = i + 1

        ####################
        # Merge descriptors#
        ####################
        dlist = descType.split("_")
        combinedheader = []
        dtable = {}
        fcfp4Test = 1
        rdMDTest = 1

        # Take the common set of keys among all the descriptors blocks
        fcfp4Set = set(fcfp4_bit.keys())
        rdMDSet = set(rdMD.keys())
        actSet = set(act.keys())

        # MODIFIED THE LOOP ENUMERATOR HERE
        for idx, key in enumerate(name_list):
            name = key
            if act[key] != "":
                tmpTable = []
                activity = act[key]

                # MODIFIED HERE
                rdkit_smi = rdkit_smiles[key]
                tmpTable.append(rdkit_smi)

                if idx == 0:
                    combinedheader.append("molecule_ID")
                    combinedheader.append("rdkit_SMILES")

                if "FCFP4" in dlist:
                    fcfp4D = fcfp4_bit[key]
                    z = fcfp4D.replace("0", "0,")
                    o = z.replace("1", "1,")
                    f = o[:-1]
                    fcfp4D = f.split(",")
                    k = 1
                    for i in fcfp4D:
                        tmpTable.append(i)
                        if fcfp4Test:
                            varname = "fcfp4_%d" % k
                            combinedheader.append(varname)
                            k += 1
                    fcfp4Test = 0

                if "rdMolDes" in dlist:
                    rdMD_des = rdMD[key]
                    k = 1
                    for i in rdMD_des:
                        tmpTable.append(str(i))
                        if rdMDTest:
                            varname = "rdMD_%d" % k
                            combinedheader.append(varname)
                            k += 1
                    rdMDTest = 0

                tmpTable.append(activity)
                dtable[key] = tmpTable

        combinedheader.append("activity")

        # Save out the descriptor file
        rawData = open(os.path.join(os.getcwd(), "data", "rawData.csv"), "w")
        for h in combinedheader[:-1]:
            rawData.write("%s," % h)
        rawData.write("%s\n" % combinedheader[-1])
        for cmpd in dtable.keys():
            comboD = dtable[cmpd]
            rawData.write("%s," % cmpd)
            for d in comboD[:-1]:
                rawData.write("%s," % d)
            rawData.write("%s\n" % comboD[-1])
        rawData.close()

        # read the raw data
        df = pd.read_csv(
            os.path.join(os.getcwd(), "data", "rawData.csv"), low_memory=False
        )

        # standardize the rdkit smiles using pubchem API
        smi_list = df["rdkit_SMILES"].tolist()
        pubchem_result = pubchem_standardizer_batch(smi_list)
        # insert the new columns into the dataframe
        df.insert(loc=2, column="pubchem_cid", value=pubchem_result["pubchem_cid"])
        df.insert(
            loc=3, column="pubchem_SMILES", value=pubchem_result["pubchem_SMILES"]
        )
        # drop rows with missing values
        df.dropna(inplace=True)
        # make sure pubchem_cid is integer
        df["pubchem_cid"] = df["pubchem_cid"].astype(int)
        # save the final dataframe to csv
        df.to_csv(
            os.path.join(os.getcwd(), "data", f"rawData_{sdf_file}.csv"),
            index=False,
            header=True,
        )

    # load the HLM dataset to get the features
    ds_hlm = load_dataset(
        "csv",
        data_files=os.path.join(os.getcwd(), "data", "rawData_ADME_HLM.csv"),
    )["train"]
    ds_hlm_features = ds_hlm.features

    # read all generated csv files and merge them into a single HuggingFace dataset dictionary
    ds = DatasetDict(
        {
            "HLM": ds_hlm,
            "RLM": load_dataset(
                "csv",
                data_files=os.path.join(os.getcwd(), "data", "rawData_ADME_RLM.csv"),
                features=ds_hlm_features,
            )["train"],
            "MDR1_ER": load_dataset(
                "csv",
                data_files=os.path.join(
                    os.getcwd(), "data", "rawData_ADME_MDR1_ER.csv"
                ),
                features=ds_hlm_features,
            )["train"],
            "SOL": load_dataset(
                "csv",
                data_files=os.path.join(os.getcwd(), "data", "rawData_ADME_Sol.csv"),
                features=ds_hlm_features,
            )["train"],
            "hPPB": load_dataset(
                "csv",
                data_files=os.path.join(os.getcwd(), "data", "rawData_ADME_hPPB.csv"),
                features=ds_hlm_features,
            )["train"],
            "rPPB": load_dataset(
                "csv",
                data_files=os.path.join(os.getcwd(), "data", "rawData_ADME_rPPB.csv"),
                features=ds_hlm_features,
            )["train"],
        }
    )

    # save the dataset dictionary to disk
    ds.save_to_disk(os.path.join(os.getcwd(), "data", "adme_ml_public"))
