###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: April 2025                                                            #
# Description:                                                                #
# This module will download sdf files from PubChem ftp server                 #
# and save them to a local directory                                          #
###############################################################################

import os
import wget
from tqdm import tqdm

# set the global variables
ftp_url = "ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/Compound_{start}_{end}.sdf.gz"
local_dir = "ftp_pubchem_compound_data"

# check if the local directory exists
if not os.path.exists(local_dir):
    os.makedirs(local_dir, exist_ok=True)

# loop over the range of indices and download the sdf files
# currently, the step sizes is set to 500,000 by PubChem ftp server
# check their readme file for more details
# https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/README-Compound-SDF
for i in tqdm(range(1, 173_000_001, 500_000), total=346):
    # create start and end indices
    idx_start = i
    idx_end = i + 500_000
    str_idx_start = str(idx_start).zfill(9)
    str_idx_end = str(idx_end - 1).zfill(9)
    # # create the ftp url
    formatted_ftp_url = ftp_url.format(start=str_idx_start, end=str_idx_end)
    # download the sdf files
    wget.download(formatted_ftp_url, local_dir)

# print a message when the download is complete
print("Download complete!")
