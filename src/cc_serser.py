""" Cross-covariance between two series
Author: Drew B. Headley

"""

import sys

sys.path.append(".")  # have to do this for relative imports to work consistently
import numpy as np


def cc_serser(oth_ser, ref_ser, win=[-10, 10]):
    """
    Calculates the cross-covariance between two time series.

    Parameters
    ----------
    oth_ser : array-like of numbers
        the other time series
    ref_ser : array-like of numbers
        the reference time series
    win : [int, int] (default is [-10, 10])
        a 2 element list-like specifying the number of bins for either edge
        of the cross correlation function

    Returns
    ----------
    cc_dict : a dictionary of numpy arrays
        'values' is the the covariance between the time series at different lags
        'lags' is the lags between time series
        'means' is the means of the time series, [oth_mean, ref_mean]

    Examples
    ----------

    """

    # time series must have same length
    if ref_ser.size != oth_ser.size:
        raise ValueError("Sizes of reference and other time series do not agree")

    # ensure proper formatting of the time series
    ref_ser = ref_ser.reshape(-1)
    oth_ser = oth_ser.reshape(-1)

    # create list of relative indices
    b_edge = win[0]
    t_edge = win[1]
    rel_inds = np.arange(b_edge, t_edge).reshape(-1).astype(int)

    # de-mean each time series
    ref_mean = np.nanmean(ref_ser)
    oth_mean = np.nanmean(oth_ser)
    ref_ser -= ref_mean
    oth_ser -= oth_mean

    cc = np.full(rel_inds.size, np.nan)
    for lag in rel_inds:
        if lag < 0:
            cc[lag - b_edge] = np.nanmean(ref_ser[-lag:] * oth_ser[:lag])
        elif lag > 0:
            cc[lag - b_edge] = np.nanmean(ref_ser[:-lag] * oth_ser[lag:])
        else:
            cc[lag - b_edge] = np.nanmean(ref_ser * oth_ser)

    cc_dict = {"values": cc, "lags": rel_inds, "means": [oth_mean, ref_mean]}
    return cc_dict


# Debug test


if __name__ == "__main__":
    from numpy.random import randn
    from prettytable import PrettyTable

    # import time
    import pdb

    ser_len = 100000

    test_ref_ser = randn(ser_len)
    test_oth_ser = np.hstack((test_ref_ser[5:], np.ones(5)))
    test_cc = cc_serser(test_oth_ser, test_ref_ser, [-10, 10])
    """tic = time.perf_counter()
    cc_test = cc_serpt(test_oth_ser, test_ref_pts, 1, [-10, 10])
    toc = time.perf_counter()
    print(
        "A crosscorrelation between a {} length series and 100 reference spikes took {} seconds".format(
            ser_len, (toc - tic)
        )
    )"""

    cc_tbl = PrettyTable()
    cc_tbl.align = "r"
    cc_tbl.add_column("Lags", test_cc["lags"])
    cc_tbl.add_column("CCov", np.round(test_cc["values"], 2))
    print(cc_tbl)
