""" Corrected CC between APs and dendritic events
Author: Drew B. Headley

"""
import sys

sys.path.append(".")  # have to do this for relative imports to work consistently
import numpy as np
import pandas as pd
import pdb
from .load_spike_h5 import load_spike_h5
from .load_dendevt_csv import load_dendevt_csv
from .seg_dendevt import seg_dendevt
from .ser_seg_dendevt import ser_seg_dendevt
from .mean_dendevt import mean_dendevt
from .cc_serser import cc_serser
from .ser_pt import ser_pt


def cc_phdep_files(dend_fname, ap_fname, rhym_ser, step_ser, win_ser, win):
    """
    Uses the files for the dendritic events and phase of an afferent rhythm,
    calculate a phase binned ocurrence of dendritic spikes by a dendritic
    segment's electrotonic distance.

    Parameters
    ----------
    dend_fname : string
        file path for dendritic spike events csv
    ap_fname : numpy array
        file path for action potentials
    rhym_ser : numpy array
        inhibitory rhythm
    step_len : numeric
        the length of each step in the series. Generally, it should be less
        than the shortest time between start and stop points. By default,
        it is set to that time
    win_ser : list, 2 elements
        the beginning and end of the series. When inf is used, the series
        will start at the first start point and end at the last stop point.
        By default, the window starts at the first start point and ends at
        the last stop point
    win : [int, int] (default is [-10, 10])
        a 2 element list-like specifying the number of bins for either edge
        of the cross correlation function

    Returns
    ----------
    dend_seg : dataframe
        corrected crosscov for each dendritic events. Column name is 'ph_cc_<t or p>'

    Examples
    ----------

    """
    ph_mask = rhym_ser >= 0

    # process action potential events
    spk_t = load_spike_h5(ap_fname)
    spk_ser = ser_pt(spk_t, step_ser, win_ser).to_numpy()

    # create spike series and keep only those with the right phase
    spk_ser_t = spk_ser.copy().astype(float)
    spk_ser_p = spk_ser.copy().astype(float)
    spk_ser_t[np.where(ph_mask)[0]] = 0
    spk_ser_p[np.where(~ph_mask)[0]] = 0

    # turn dendritic events into series
    dend_t = load_dendevt_csv(dend_fname)
    dend_seg = seg_dendevt(dend_t)
    dend_seg = ser_seg_dendevt(dend_seg, step_len=step_ser, win_lim=win_ser)
    dend_seg = mean_dendevt(dend_seg)

    # calculate cross-covariance
    cc_t = []
    cc_p = []
    for ind, row in dend_seg.iterrows():
        cc_t.append(cc_serser(row["evt_mean"], spk_ser_t, win=win)["values"])
        cc_p.append(cc_serser(row["evt_mean"], spk_ser_p, win=win)["values"])

    dend_seg["cc_t"] = cc_t
    dend_seg["cc_p"] = cc_p
    """# calculate corrected cross-covariance
    dend_seg_t = sta_corr_ap_dendevt(dend_seg, spk_ser_t, win=win)
    dend_seg_p = sta_corr_ap_dendevt(dend_seg, spk_ser_p, win=win)

    # format output dataframe
    dend_seg = pd.merge(
        dend_seg_t, dend_seg_p, how="inner", on=["Elec_distanceQ", "Type"]
    )
    """
    dend_seg.drop("evt_mean", axis=1)
    dend_seg.sort_index()
    dend_seg = dend_seg.groupby(("Type")).aggregate(
        {"cc_t": np.vstack, "cc_p": np.vstack}
    )
    return dend_seg


if __name__ == "__main__":
    import pdb

    print("Testing calcium spikes with cc_phdep_files.py")

    samps_per_ms = 10
    sim_win = [0, 2000000]  # beginning and start points of simulation in samples
    sta_win = [-100, 100]  # multiply by step to get window in milliseconds
    sta_step = 1  # binning step for each point in the STA
    step = 2 * samps_per_ms  # number of simulation steps for creating the dendritic
    # event occurrence series
    t_ser = np.arange(sim_win[0], sim_win[1], step) / (10000)  # seconds
    sin_ser = np.sin(t_ser * 16 * 2 * np.pi)

    ca_fpath_test = (
        "Y:\\DendCompOsc\\output_16Hz_no_exc_mod\\output_16Hz_no_exc_mod_ca.csv"
    )
    ap_fpath_test = "Y:\\DendCompOsc\\output_16Hz_no_exc_mod\\spikes.h5"

    cc_test = cc_phdep_files(
        ca_fpath_test, ap_fpath_test, sin_ser, step, sim_win, sta_win
    )
    print(cc_test)
