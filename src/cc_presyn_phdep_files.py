""" Corrected CC between APs and presynaptic events
Author: Drew B. Headley

"""
import sys

sys.path.append(".")  # have to do this for relative imports to work consistently
import numpy as np
import pandas as pd
import pdb
from .load_spike_h5 import load_spike_h5
from .load_spike_aux_h5 import load_spike_aux_h5
from .cc_ptpt import cc_ptpt


def cc_presyn_phdep_files(presyn_fname, ap_fname, rhym_ser, step_len, win):
    """
    Uses the files for the preynaptic spikes and phase of an afferent rhythm,
    calculate a phase stratified cross-correlation between presynaptic spikes
    and action potentials.

    Parameters
    ----------
    presyn_fname : string
        file path for presynaptic spike events h5
    ap_fname : numpy array
        file path for action potentials
    rhym_ser : numpy array
        inhibitory rhythm
    step_len : numeric
        the bin size for the CC in units of seconds
    win : [int, int] (default is [-10, 10])
        a 2 element list-like specifying the number of bins for either edge
        of the cross correlation function

    Returns
    ----------
    cc : dictionary
        cross correlation for each phase. Keys are 't' for trough and 'p' for peak

    Examples
    ----------

    """
    ph_mask = rhym_ser >= 0

    # load action potential events
    spk_pts = load_spike_h5(ap_fname)

    # load presynaptic events into
    pre_pts = load_spike_aux_h5(presyn_fname)

    # create spike series and keep only those with the right phase
    p_inds = np.where(ph_mask)[0]
    t_inds = np.where(~ph_mask)[0]
    # find set of presynaptic spikes in troughs and peaks
    pre_pts_t = pre_pts[np.isin(pre_pts, t_inds)]
    pre_pts_p = pre_pts[np.isin(pre_pts, p_inds)]

    # calculate cross-correlation
    cc = {}
    cc["t"] = cc_ptpt(spk_pts, pre_pts_t, step_len, win)
    cc["p"] = cc_ptpt(spk_pts, pre_pts_p, step_len, win)

    return cc


if __name__ == "__main__":
    from prettytable import PrettyTable
    import pdb

    print("Testing presynaptic spikes with cc_presyn_phdep_files.py")

    samps_per_ms = 10
    sim_win = [0, 200]
    cc_win = [-15, 15]  # multiply by step to get window in milliseconds
    step = 2 * samps_per_ms  # number of simulation steps for creating the dendritic
    # event occurrence series
    t_ser = np.linspace(
        sim_win[0], sim_win[1], samps_per_ms * 1000 * sim_win[1]
    )  # seconds
    sin_ser = np.sin(t_ser * 16 * 2 * np.pi)

    pre_fpath_test = "Z:\\DendOscSub\\output_clust_16Hz_conc\\exc_stim_aux_spikes2.h5"
    ap_fpath_test = "Z:\\DendOscSub\\output_clust_16Hz_conc\\spikes.h5"

    # pdb.set_trace()
    cc_test = cc_presyn_phdep_files(
        pre_fpath_test, ap_fpath_test, sin_ser, step, cc_win
    )

    # use prettytable to print the results with each row a cc lag
    tbl = PrettyTable()
    tbl.field_names = ["Lag (ms)", "Trough", "Peak"]
    for lags, t_cc, p_cc in zip(
        cc_test["t"]["lags"], cc_test["t"]["values"], cc_test["p"]["values"]
    ):
        tbl.add_row([lags, t_cc, p_cc])
    print(tbl)
