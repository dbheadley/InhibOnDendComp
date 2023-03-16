""" Cross-correlation between two point processes
Author: Drew B. Headley

"""

import sys

sys.path.append(".")  # have to do this for relative imports to work consistently
import numpy as np
from .ser_pt import ser_pt


def cc_ptpt(oth_pts, ref_pts, bin_size, win=[-10, 10], sm_win=1, notch_freq=None):
    """
    Calculates the cross-correlation between two point processes.
    Not optimized for speed.

    Parameters
    ----------
    oth_pt : array-like of numbers
        the times of the other point process
    ref_pt : array-like of numbers
        the times of the reference point process
    bin_size : numeric
        the bin size for the CC in units of seconds
    win : [int, int] (default is [-10, 10])
        a 2 element list-like specifying the number of bins for either edge
        of the cross correlation function
    sm_win : int (default is 1)
        the number of bins to smooth the cross-correlation function by
    notch_freq : numeric (default is None)
        the frequency to notch out of the corrected cross-correlation function

    Returns
    ----------
    cc_dict : a dictionary of numpy arrays
        'values' is the the co-occurrence counts between the point processes at
        different time lags
        'lags' is the lags between the point processes

    Examples
    ----------

    """

    # ensure proper formatting of the point processes
    ref_pts = np.array(ref_pts).reshape(-1)
    oth_pts = np.array(oth_pts).reshape(-1)

    # create list of relative indices
    b_edge = win[0]
    t_edge = win[1]
    rel_inds = np.arange(b_edge, t_edge).reshape(-1).astype(int)

    # get last point
    last_pt = np.max([ref_pts[-1], oth_pts[-1]])

    # bin the point processes
    ref_binned = ser_pt(ref_pts, bin_size, [0, last_pt]).values
    oth_binned = ser_pt(oth_pts, bin_size, [0, last_pt]).values

    # calculate the raw cross-correlation
    cc = lagged_dot_prod(ref_binned, oth_binned, rel_inds)

    # calculate the autocorrelations
    ac_oth = lagged_dot_prod(oth_binned, oth_binned, rel_inds)
    ac_ref = lagged_dot_prod(ref_binned, ref_binned, rel_inds)

    # correct for the autocorrelations

    sm_kern = np.ones(sm_win) / sm_win
    # add repeated elements to edges of cc to avoid boundary effects
    cc_b = np.pad(cc, (sm_win, sm_win), mode="reflect")
    ac_oth_b = np.pad(ac_oth, (sm_win, sm_win), mode="reflect")
    ac_ref_b = np.pad(ac_ref, (sm_win, sm_win), mode="reflect")

    cc_b = np.convolve(cc_b, sm_kern, mode="same")
    ac_oth_b = np.convolve(ac_oth_b, sm_kern, mode="same")
    ac_ref_b = np.convolve(ac_ref_b, sm_kern, mode="same")

    # remove padding from ccs
    cc_b = cc_b[sm_win:-sm_win]
    ac_oth_b = ac_oth_b[sm_win:-sm_win]
    ac_ref_b = ac_ref_b[sm_win:-sm_win]

    # correct cc for autocorrelations
    cc_fft = np.fft.rfft(cc_b)
    ac_oth_fft = np.fft.rfft(ac_oth_b)
    ac_ref_fft = np.fft.rfft(ac_ref_b)
    cc_fft /= np.sqrt(ac_oth_fft * ac_ref_fft)

    # remove frequencies around notch
    freqs = np.fft.rfftfreq(cc_b.size, d=0.001)
    if notch_freq is not None:
        notch_ind = np.argmin(np.abs(freqs - notch_freq))
        cc_fft[notch_ind - 1 : notch_ind + 2] = 0

    cc_corr = np.fft.irfft(cc_fft, len(cc_b))

    cc_dict = {
        "values": cc,
        "lags": rel_inds,
        "ref_norm": 1 / np.sum(ref_binned),
        "values_corr": cc_corr,
    }
    return cc_dict


def lagged_dot_prod(ser1, ser2, lags):
    """Calculates the lagged dot product between two series

    Parameters
    ----------
    ser1 : array-like
        the first series
    ser2 : array-like
        the second series
    lags : array-like
        the lags to calculate the dot product at

    Returns
    ----------
    ldp : array-like
        the lagged dot product between the two series
    """
    ldp = np.full(lags.size, np.nan)
    for ind, lag in enumerate(lags):
        if lag < 0:
            ldp[ind] = np.dot(ser1[-lag:], ser2[:lag])
        elif lag > 0:
            ldp[ind] = np.dot(ser1[:-lag], ser2[lag:])
        else:
            ldp[ind] = np.dot(ser1, ser2)
    return ldp


def ac_scaled(ser, lags):
    """Calculates an autocorrelation scaled to a sum of 1

    Parameters
    ----------
    ser : array-like
        a binned time series
    lags : array-like
        the lags of the autocorrelation

    Returns
    ----------
    ac_scaled : array-like
        the scaled autocorrelation
    """

    ac = lagged_dot_prod(ser, ser, lags)
    z_lag = np.argwhere(lags == 0)
    nz_lags = np.argwhere(lags != 0)
    ac[z_lag] = 0
    ac_scaled = (ac - np.nanmean(ac[nz_lags])) / np.sum(ser)
    ac_scaled[z_lag] = 1
    return ac_scaled


# Debug test
if __name__ == "__main__":
    from numpy.random import randint
    from prettytable import PrettyTable

    import time
    import pdb

    ser_len = 100000
    num_pts = 5000
    fs = 1000

    test_ref_pts = np.sort(randint(0, ser_len, num_pts)) / fs
    test_oth_pts = np.sort(randint(0, ser_len, num_pts)) / fs

    print(test_ref_pts[:5])
    print(test_oth_pts[:5])
    tic = time.perf_counter()
    pdb.set_trace()
    test_cc = cc_ptpt(test_oth_pts, test_ref_pts, 0.01, [-10, 10])
    toc = time.perf_counter()
    print(
        "A crosscorrelation between {} length point processes took {} seconds".format(
            num_pts, (toc - tic)
        )
    )

    cc_tbl = PrettyTable()
    cc_tbl.align = "r"
    cc_tbl.add_column("Lags", test_cc["lags"])
    cc_tbl.add_column("CC", test_cc["values"])
    print(cc_tbl)
