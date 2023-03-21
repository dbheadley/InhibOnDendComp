""" Creates a list of Ca2+ spike times from a dendritic ca CSV file
Author: Drew B. Headley

"""

import pandas as pd
import numpy as np


def load_caspks_csv(fpath):
    """
    Loads a CSV file that contains calcium spike times from the nexus segment
    converts them to a numpy array of integer simulation time steps

    Parameters
    ----------
    fpath : string
        full file path to the ca CSV file

    Returns
    ----------
    caspk_t : numpy array of integers
        simulation time steps where a ca spike was emitted at the nexus

    Examples
    ----------

    """

    caspk_df = pd.read_csv(fpath)

    # find the segment with the least attenuation from the nexus
    min_atten = caspk_df["Elec_distance_nexus"].max()
    seg_ser = caspk_df[caspk_df["Elec_distance_nexus"] == min_atten].loc[:, "segmentID"]

    # if multiple segments are returned, only keep the most common one
    seg_id = seg_ser.value_counts().index[0]

    # get start times of calcium spikes for the nexus segment
    caspk_t = caspk_df[caspk_df["segmentID"] == seg_id].loc[:, "ca_lower_bound"]

    # Convert Ca spike times to integers and sort
    caspk_t = np.sort(caspk_t).astype(int)

    return caspk_t


if __name__ == "__main__":
    import pdb

    print("Testing load_caspks_csv.py")
    fpath_test = "Z:\\DendOscSub\\output_allpoisson\\output_allpoisson_ca_nex.csv"
    caspk_test = load_caspks_csv(fpath_test)
    print(caspk_test)
