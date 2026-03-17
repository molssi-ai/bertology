###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: April 2025                                                            #
# Description:                                                                #
# This module will use the sdf files downloaded from the PubChem ftp server   #
# (by the ftp_sdf_downloader.py script) and processes them into json files    #
# before uploading them to hugging face hub. The script uses an adaptive      #
# local Dask cluster for parallel processing. The dashboard will run on       #
# the localhost:8889.                                                         #
###############################################################################

import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import json
import numpy as np

# set the global variables
scratch_dir = "./scratch"
sdf_dir = "./ftp_pubchem_compound_data/*.sdf.gz"
output_dir = "./outputs"

# create the output directory
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# get the list of files
files_list = glob(sdf_dir)


# define the function to process the files
def oechem_processor(ifile: str, ofile: str) -> None:
    from openeye import oechem

    fname = ofile.replace(".csv", ".err")
    errfs = oechem.oeofstream()
    if not errfs:
        oechem.OEThrow.Fatal("Unable to create %s" % fname)
    oechem.OEThrow.SetOutputStream(errfs)

    ifs = oechem.oemolistream(file)
    temp = {}
    entries = []
    # for mol in tqdm(ifs.GetOEGraphMols(), total=500_000, desc="molecules"):
    for mol in ifs.GetOEGraphMols():
        for dp in oechem.OEGetSDDataPairs(mol):
            tag = str(dp.GetTag())
            if tag not in temp.keys():
                if tag == "PUBCHEM_BONDANNOTATIONS" or tag == "PUBCHEM_COORDINATE_TYPE":
                    value = oechem.OEGetSDData(mol, tag).split()
                    value = [int(x) for x in value if x != "\n"]
                    temp[tag] = value
                elif (
                    tag.endswith("_COUNT")
                    or tag.endswith("_TPSA")
                    or tag.endswith("_CHARGE")
                    or tag == "PUBCHEM_CACTVS_ROTATABLE_BOND"
                    or tag == "PUBCHEM_CACTVS_HBOND_ACCEPTOR"
                    or tag == "PUBCHEM_CACTVS_HBOND_DONOR"
                    or tag == "PUBCHEM_CACTVS_COMPLEXITY"
                ):
                    temp[tag] = int(oechem.OEGetSDData(mol, tag))
                elif (
                    tag.endswith("_MASS")
                    or tag.endswith("_WEIGHT")
                    or tag.endswith("_AA")
                ):
                    temp[tag] = float(oechem.OEGetSDData(mol, tag))
                else:
                    temp[tag] = oechem.OEGetSDData(mol, tag)
        coords = oechem.OEFloatArray(mol.GetMaxAtomIdx() * 3)
        mol.GetCoords(coords)
        temp["coords"] = np.array(coords, dtype=np.float64).tolist()
        atomic_symbols = []
        atomic_numbers = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            atomic_numbers.append(atomic_num)
            atomic_symbols.append(oechem.OEGetAtomicSymbol(atomic_num))
        temp["atomic_symbols"] = atomic_symbols
        temp["atomic_numbers"] = atomic_numbers
        entries.append(temp)
        temp = {}

    with open(file.replace(".sdf.gz", ".json"), "w") as f:
        json.dump(entries, indent=4, fp=f)

    ifs.close()
    errfs.close()


# Run the function over all files
if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster
    import dask.distributed as dd

    # Setup the Dask cluster
    cluster = LocalCluster(
        name="PubChem-OEChem",
        n_workers=1,
        threads_per_worker=1,
        memory_limit="1GB",
        dashboard_address=":8889",
        local_directory=scratch_dir,
    )

    # Create an adaptive cluster
    cluster.adapt(minimum=1, maximum=64)

    # Connect to the cluster
    client = Client(cluster)

    # Submit the tasks
    all_results = []
    for idx, file in enumerate(tqdm(files_list, total=len(files_list))):
        ofile = os.path.basename(file).replace(".sdf.gz", ".csv")
        if ofile in os.listdir(output_dir):
            continue
        args = {"ifile": file, "ofile": os.path.join(output_dir, ofile)}
        future = client.submit(
            oechem_processor,
            **args,
            retries=0,
            key=os.path.basename(file).strip(".sdf.gz")
        )
        all_results.append(future)

        if len(all_results) % 64 == 0:
            dd.wait(all_results)
            all_results = []

    # Close the client
    client.close()

    # Close the cluster
    cluster.close()
