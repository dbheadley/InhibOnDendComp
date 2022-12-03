""" Bin a series by the values in another series
Author: Drew B. Headley

"""


"""


MAKE BINTEMP A NUMPY ARRAY OF NANS
CHANGE RANGE(EDGES.SIZE) TO RANGE(1,EDGES.SIZE)



"""
import numpy as np


def bin_serser(val_ser, bin_ser, edges, func=np.mean):
    """
    A series of values are binned together based on shared corresponding
    values in another series and a function is applied.

    Parameters
    ----------
    val_ser : numpy array, N-length
        array of values to be binned
    bin_ser : numpy array, N-length
        array of values to use for binning
    edges : array-like
        bin edges
    func : function (default is numpy's mean)
        binned values are passed as a list to function

    Returns
    ----------
    bin_dict : a dictionary of numpy arrays
        'values' is the results of binning
        'edges' is the edges of the bins

    Examples
    ----------

    """

    # ensure arrays are same length
    if val_ser.size != bin_ser.size:
        raise ValueError("Value and bin arrays are not equally sized")

    # format input data for consistency
    val_ser = val_ser.reshape(-1)
    bin_ser = bin_ser.reshape(-1)

    # bin values in the bin array
    bin_inds = np.digitize(bin_ser, edges)

    # apply function to each set of values with the same bin
    bin_temp = []
    for curr_ind in range(edges.size):
        bin_temp.append(func(val_ser[bin_inds == curr_ind]))

    bin_dict = {"values": np.array(bin_temp), "edges": edges}
    return bin_dict


# Debug test


if __name__ == "__main__":
    from numpy.random import randn, randint
    from prettytable import PrettyTable
    import pdb

    print("Testing bin_serser.py")
    ser_len = 1000
    test_vals = randn(ser_len)
    test_bins = np.arange(ser_len) % 10
    test_vals[test_bins == 4] += 1

    test_bin_mean = bin_serser(test_vals, test_bins, np.arange(-1, 12))
    test_bin_count = bin_serser(test_vals, test_bins, np.arange(-1, 12), np.size)

    bin_tbl = PrettyTable()
    bin_tbl.add_column("BinStart", test_bin_mean["edges"][0:-1])
    bin_tbl.add_column("BinFinish", test_bin_mean["edges"][1:])
    bin_tbl.add_column("Mean", np.round(test_bin_mean["values"][1:], 2))
    bin_tbl.add_column("Count", test_bin_count["values"][1:])
    print(bin_tbl)
