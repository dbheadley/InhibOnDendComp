"""Calculates the STA from files
Author: Drew B. Headley

"""
import numpy as np
import pandas as pd
from .load_dendevt_csv import load_dendevt_csv
from .load_spike_h5 import load_spike_h5
from .seg_dendevt import seg_dendevt
from .ser_seg_dendevt import ser_seg_dendevt
from .sta_ap_dendevt import sta_ap_dendevt


def sta_files(dend_fname, spk_fname, step_ser, win_ser, bin_sta, win_sta):
    """
    Uses the files for the dendritic events and action potentials to calculate
    a spike triggered average (STA) stratified by a dendritic segment's
    electrotonic distance.

    Parameters
    ----------
    dend_fname : string
        file path for dendritic spike events csv
    spk_fname : string
        file path for somatic action potentials csv
    step_len : numeric
        the length of each step in the series. Generally, it should be less
        than the shortest time between start and stop points. By default,
        it is set to that time
    win_ser : list, 2 elements
        the beginning and end of the series. When inf is used, the series
        will start at the first start point and end at the last stop point.
        By default, the window starts at the first start point and ends at
        the last stop point
    bin_sta : int
        the indices in the time series to pool together when calculating
        the mean value at each time lag from the reference point process
    win_sta : [int, int]
        a 2 element list-like specifying the number of bins for either edge
        of the cross correlation function

    Returns
    ----------
    dend_seg : dataframe
        sta for each dendritic event. Column name is 'sta'

    Examples
    ----------

    """
    dend_t = load_dendevt_csv(dend_fname)
    spk_t = load_spike_h5(spk_fname)

    dend_seg = seg_dendevt(dend_t)
    dend_seg = ser_seg_dendevt(dend_seg, step_len=step_ser, win_lim=win_ser)
    dend_seg = sta_ap_dendevt(
        dend_seg, np.round(spk_t / step_ser), bin=bin_sta, win=win_sta
    )

    dend_seg = dend_seg.groupby(("Type")).aggregate({"sta": np.vstack})
    return dend_seg


if __name__ == "__main__":

    print("Testing calcium spikes with sta_files.py")
    ca_fpath_test = (
        "Y:\\DendCompOsc\\16Hzapical_exc_mod\\"
        "output_16Hz_dend_inh_0deg_exc_10p_ca.csv"
    )
    ap_fpath_test = (
        "Y:\\DendCompOsc\\16Hzapical_exc_mod\\"
        "output_16Hz_dend_inh_0deg_exc_10p\\spikes.h5"
    )

    sta_test = sta_files(ca_fpath_test, ap_fpath_test, 20, [0, 2000000], 1, [-100, 100])
    print(sta_test)
