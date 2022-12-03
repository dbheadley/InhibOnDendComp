"""Corrected cross-covariance between action potentials and dendritic events
Author: Drew B. Headley

"""
import sys

sys.path.append(".")  # have to do this for relative imports to work consistently
import pdb
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
from .cc_serser import cc_serser


def sta_corr_ap_dendevt(seg_df, ap_ser, **kwargs):
    """
    For each dendritic event time series the corrected cross-covariance
    with the action potential is calculated.

    Parameters
    ----------
    seg_df : dataframe
        dendritic spike events series by segment
    ap_ser : numpy array
        time series of action potential events
    **kwargs : arguments to pass to cc_serser

    Returns
    ----------
    sta_df : dataframe
        sta for each dendritic event. Column names are:
        '<event_type>_cc' is the uncorrected crosscovariance between aps and dend events
        '<event_type>_ac_dend' is the autocovariance of dend events
        '<event_type>_ac_ap' is the autocovariance of aps
        '<event_type>_cc_corr' is the corrected crosscovariance between aps and dend events

    Examples
    ----------

    """

    # determine event names
    # ser_colname = [x for x in seg_df.columns if x.endswith("_ser")][0]

    # get spike triggered average, expressed as percent change from mean
    cc_temp = []
    ac_dend_temp = []
    ac_ap_temp = []
    cc_corr_temp = []
    for _, x in seg_df.iterrows():
        cc_temp.append(
            cc_serser(x["evt_mean"].astype(float), ap_ser.astype(float), **kwargs)[
                "values"
            ],
        )
        ac_dend_temp.append(
            cc_serser(
                x["evt_mean"].astype(float), x["evt_mean"].astype(float), **kwargs
            )["values"],
        )
        ac_ap_temp.append(
            cc_serser(ap_ser.astype(float), ap_ser.astype(float), **kwargs)["values"],
        )
        cc_corr_temp.append(
            irfft(
                rfft(cc_temp[-1]) ** 2 / (rfft(ac_dend_temp[-1]) * rfft(ac_ap_temp[-1]))
            )
        )

    sta_df = seg_df
    pdb.set_trace()
    sta_df["cc"] = cc_temp
    sta_df["ac_dend"] = ac_dend_temp
    sta_df["ac_ap"] = ac_ap_temp
    sta_df["cc_corr"] = cc_corr_temp

    fig, ax = plt.subplots()
    ax.plot(sta_df["cc"].to_numpy())
    plt.show()

    return sta_df


"""
if __name__ == "__main__":
    from load_spike_h5 import load_spike_h5
    from load_dendevt_csv import load_dendevt_csv
    from seg_dendevt import seg_dendevt
    from ser_seg_dendevt import ser_seg_dendevt

    print("Testing calcium spikes with sta_ap_dendevt.py")
    ca_fpath_test = (
        "Y:\\DendCompOsc\\16Hzapical_exc_mod\\"
        "output_16Hz_dend_inh_0deg_exc_10p_ca.csv"
    )
    ap_fpath_test = (
        "Y:\\DendCompOsc\\16Hzapical_exc_mod\\"
        "output_16Hz_dend_inh_0deg_exc_10p\\spikes.h5"
    )

    dspk_test = load_dendevt_csv(ca_fpath_test)
    ap_test = load_spike_h5(ap_fpath_test)
    seg_test = seg_dendevt(dspk_test)
    seg_test = ser_seg_dendevt(seg_test, step_len=20, win_lim=[0, 2000000])
    sta_test = sta_ap_dendevt(seg_test, np.round(ap_test / 20), bin=1, win=[-100, 100])
    print(sta_test)
"""
