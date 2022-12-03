""" Creates numpy array from an H5 file containing membrane voltage
Author: Drew B. Headley

"""


import numpy as np
import h5py


def load_v_h5(fpath, seg_ind):
    """
    Loads an H5 file that contains voltages and selects a specific segment
    converts it to a numpy array

    Parameters
    ----------
    fpath : string
        full file path to the H5 file
    seg_ind : integer
        segment (column of voltage matrix) to return

    Returns
    ----------
    mem_v : numpy array
        membrane voltages

    Examples
    ----------

    """
    spikes = h5py.File(fpath, "r")
    mem_v = np.array(spikes["report"]["biophysical"]["data"][:, seg_ind])

    return mem_v


"""
if __name__ == "__main__":
    print("Testing load_v_h5.py")
    fpath_test = "../Data/spikes.h5"
    spk_test = load_spike_h5(fpath_test)
    print(spk_test)
"""
