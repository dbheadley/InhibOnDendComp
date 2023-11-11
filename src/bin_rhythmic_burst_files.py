""" Binned dendritic events with respect to rhythmic bursts
Author: Drew B. Headley

"""
import numpy as np
import pandas as pd
from .load_dendevt_csv import load_dendevt_csv
from .load_spike_h5 import load_spike_h5
from .seg_dendevt import seg_dendevt
from .ser_seg_dendevt import ser_seg_dendevt
from .bin_rhym_dendevt import bin_rhym_dendevt


def bin_rhythmic_burst_files(dend_fname, rhym_ser, step_ser, win_ser, edges):
    """
    Uses the files for the dendritic events and phase of an afferent bursting
    rhythm to calculate a phase binned ocurrence of dendritic spikes by a dendritic
    segment's electrotonic distance.

    Parameters
    ----------
    dend_fname : string
        file path for dendritic spike events csv
    rhym_ser : numpy array
        bursty rhythm time series
    step_len : numeric
        the length of each step in the series. Generally, it should be less
        than the shortest time between start and stop points. By default,
        it is set to that time
    win_ser : list, 2 elements
        the beginning and end of the series. When inf is used, the series
        will start at the first start point and end at the last stop point.
        By default, the window starts at the first start point and ends at
        the last stop point
    edges : array-like
        bin edges for the rhythmic series phase values
    
    Returns
    ----------
    dend_seg : dataframe
        phase binned rate for each dendritic event. Column name is 'ph_bin'

    Examples
    ----------

    """
    dend_t = load_dendevt_csv(dend_fname)

    dend_seg = seg_dendevt(dend_t)
    dend_seg = ser_seg_dendevt(dend_seg, step_len=step_ser, win_lim=win_ser)
    dend_seg = bin_rhym_dendevt(dend_seg, rhym_ser, None, edges=edges)

    dend_seg = dend_seg.groupby(("Type")).aggregate({"ph_bin": np.vstack})
    return dend_seg


if __name__ == "__main__":
    """
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
    """
