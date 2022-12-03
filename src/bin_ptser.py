""" Bin a point process by the values in a series
Author: Drew B. Headley

"""

import numpy as np


def bin_ptser(val_pt, bin_ser, edges, func=np.mean):
    """
    A points are binned together based on shared corresponding
    values in another series and a function is applied.

    Parameters
    ----------
    val_pt : numpy array of integers
        array of values to be binned. Should correspond to indices of bin_ser
    bin_ser : numpy array, N-length
        array of values to use for binning
    edges : array-like
        bin edges
    func : function (default is a numpy array)
        binned values are passed as a list to a function. Should return a scalar.

    Returns
    ----------
    bin_dict : a dictionary of numpy arrays
        'values' is the results of binning
        'edges' is the edges of the bins

    Examples
    ----------

    """
    # pdb.set_trace()
    # format input data for consistency
    val_pt = val_pt.reshape(-1)
    bin_ser = bin_ser.reshape(-1)

    # bin values in the bin array
    bin_inds = np.digitize(bin_ser, edges)

    # apply function to each set of values with the same bin
    bin_vals = np.full(edges.size - 1, np.nan)
    for curr_ind in range(1, edges.size):
        bin_vals[curr_ind - 1] = func(
            np.intersect1d(val_pt, np.where(bin_inds == curr_ind))
        )

    bin_dict = {"values": bin_vals, "edges": edges}
    return bin_dict


# Debug test


if __name__ == "__main__":
    from prettytable import PrettyTable
    import pdb

    print("Testing bin_ptser.py")
    test_pts = np.array([1, 3, 5, 9])
    test_ser = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    test_edges = np.array([0, 1, 2])

    test_bin_mean = bin_ptser(test_pts, test_ser, test_edges)
    test_bin_count = bin_ptser(test_pts, test_ser, test_edges, np.size)

    print(test_bin_mean)
    bin_tbl = PrettyTable()
    bin_tbl.add_column("BinStart", test_bin_mean["edges"][0:-1])
    bin_tbl.add_column("BinFinish", test_bin_mean["edges"][1:])
    bin_tbl.add_column("Mean", test_bin_mean["values"])
    bin_tbl.add_column("Count", test_bin_count["values"])
    print(bin_tbl)
