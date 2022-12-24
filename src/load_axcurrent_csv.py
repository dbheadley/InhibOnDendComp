""" Creates numpy array from a CSV file containing axial currents
Author: Drew B. Headley

"""


import numpy as np
import pandas as pd


def load_axcurrent_csv(fpath, dend_pairs, apic_pair):
    """
    Loads a CSV file that contains axial currents from dendritic segments
    entering the soma.

    Parameters
    ----------
    fpath : string
        full file path to the CSV file
    dend_pairs : list of strings
        segment pairs corresponding to basal dendritic compartments
    apic_pair : string
        segment pair corresponding to apical dendritic compartment

    Returns
    ----------
    dend_i : numpy array of numbers
        the axial currents entering the soma from the basal dendrites
        Dimensions are [time, dendrite]
    apic_i : numpy array of numbers
        the axial currents entering the soma from the apical dendrite
        Dimensions are [time, dendrite]

    Examples
    ----------

    """
    ax_df = pd.read_csv(fpath)
    dend_i = ax_df[dend_pairs].values
    apic_i = ax_df[apic_pair].values

    return dend_i, apic_i


if __name__ == "__main__":
    print("Axial current files too large for test data")
