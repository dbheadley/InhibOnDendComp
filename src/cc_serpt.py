""" Cross-correlation between a time series and point process
Author: Drew B. Headley

"""

import numpy as np


def cc_serpt(oth_ser, ref_pt, bin=1, win=[-10, 10]):
    """
    Calculates the cross-correlation a time series and point process. This
    is effectively the average of oth_ser triggered on the occurrence of
    ref_pt.

    Parameters
    ----------
    oth_ser : array-like of numbers
        the time series
    ref_pt : array-like of ints
        the reference point process as indices for oth_ser
    bin : int (default is 1)
        the indices in the time series to pool together when calculating
        the mean value at each time lag from the reference point process
    win : [int, int] (default is [-10, 10])
        a 2 element list-like specifying the number of bins for either edge
        of the cross correlation function

    Returns
    ----------
    cc_dict : a dictionary of numpy arrays
        'values' is the the mean values from the time series at different lags
        from the point process
        'edges' is the edges of the bins

    Examples
    ----------

    """

    # ensure proper formatting of the point process
    ref_pt = ref_pt.reshape((-1, 1)).astype(int)

    # create list of relative indices
    b_edge = win[0] * bin
    t_edge = win[1] * bin
    rel_inds = np.arange(b_edge, t_edge).reshape((1, -1))

    # nan pad the edges of the time series to account for point process
    # times near the start and end
    oth_ser = np.pad(
        oth_ser, (np.abs(b_edge), t_edge), "constant", constant_values=np.nan
    )

    # offset the times of the point process to account for padding
    ref_pt = ref_pt + np.abs(b_edge)

    # extract windowed portions of the time series surrounding the point
    # process
    samp_inds = rel_inds + ref_pt
    samp_ser = oth_ser[samp_inds]

    # collapse across windows
    cc = np.nanmean(samp_ser, 0)

    # bin accumulated cc points by the bin size
    if bin != 1:
        cc = np.reshape(cc, [-1, bin])
        cc = np.nanmean(cc, axis=1)
        bin_edges = np.arange(b_edge, t_edge + bin, bin)
    else:
        bin_edges = np.arange(b_edge, t_edge + 1)

    cc_dict = {"values": cc, "edges": bin_edges}
    return cc_dict


# Debug test


if __name__ == "__main__":
    from numpy.random import randn, randint
    from prettytable import PrettyTable
    import time
    import pdb

    ser_len = 100000
    test_ref_pts = randint(0, ser_len, 100)
    test_oth_ser = randn(ser_len)
    tic = time.perf_counter()
    cc_test = cc_serpt(test_oth_ser, test_ref_pts, 1, [-10, 10])
    toc = time.perf_counter()
    print(
        "A crosscorrelation between a {} length series and 100 reference spikes took {} seconds".format(
            ser_len, (toc - tic)
        )
    )

    cc_tbl = PrettyTable()
    cc_tbl.add_column("BinStarts", cc_test["edges"][:-1])
    cc_tbl.add_column("BinEnds", cc_test["edges"][1:])
    cc_tbl.add_column("Values", cc_test["values"])
    print(cc_tbl)

    # Set the time series to 1 whenever the point process occurs
    test_oth_ser[test_ref_pts] = 1
    cc_test = cc_serpt(test_oth_ser, test_ref_pts, 1, [-10, 10])
    print("Set the time series to 1 whenever the point process occurred")
    cc_tbl = PrettyTable()
    cc_tbl.add_column("BinStarts", cc_test["edges"][:-1])
    cc_tbl.add_column("BinEnds", cc_test["edges"][1:])
    cc_tbl.add_column("Values", cc_test["values"])
    print(cc_tbl)

    # now bin by 2
    cc_test = cc_serpt(test_oth_ser, test_ref_pts, 2, [-5, 5])
    print("Changed the binning to 2")
    cc_tbl = PrettyTable()
    cc_tbl.add_column("BinStarts", cc_test["edges"][:-1])
    cc_tbl.add_column("BinEnds", cc_test["edges"][1:])
    cc_tbl.add_column("Values", cc_test["values"])
    print(cc_tbl)
