"""Cross-correlation between action potentials and dendritic events
Author: Drew B. Headley

"""
import numpy as np
from scipy.stats import ttest_1samp
from .cc_serpt import cc_serpt


def sta_ap_dendevt(seg_df, ap_pts, **kwargs):
    """
    For each segment's dendritic event time series segment the cross-correlation
    with the action potential is calculated. The mean of these are then calculated
    across segments with the same electrotonic distance. STA is expressed as
    percent change from median. Also returns one-sample t-stats/p-values for each
    time lag.

    Parameters
    ----------
    seg_df : dataframe
        dendritic spike events series by segment
    ap_pts : numpy array of integers
        the indices in the dendritic spike event series where an action potential
        occurred
    **kwargs : arguments to pass to cc_serpt

    Returns
    ----------
    sta_df : dataframe
        sta for each dendritic event. Column name is '<event_type>_sta'

    Examples
    ----------

    """

    # determine event names
    ser_colname = [x for x in seg_df.columns if x.endswith("_ser")][0]

    # get spike triggered average, expressed as percent change from mean
    prc_func = lambda sta, m: ((sta - m) / m) * 100
    sta_temp = []
    for _, x in seg_df.iterrows():
        sta_temp.append(
            prc_func(
                cc_serpt(x[ser_colname].astype(float), ap_pts.astype(int), **kwargs)[
                    "values"
                ],
                np.nanmean(x[ser_colname].astype(float)),
            )
        )
    seg_df["sta"] = sta_temp

    # merge segments by electronic distance quantile
    agg_func = {"sta": lambda x: np.vstack(x)}
    sta_df = seg_df.groupby(["Elec_distanceQ", "Type"]).aggregate(agg_func)

    # mean percent change and t-stats by electrotonic distance
    sta_m = []
    sta_t = []
    sta_p = []
    for _, x in sta_df.iterrows():
        sta_m.append(np.nanmedian(x["sta"], 0))
        t, p = ttest_1samp(x["sta"], 0)
        sta_t.append(t)
        sta_p.append(p)

    sta_df["sta"] = sta_m
    sta_df["t"] = sta_t
    sta_df["p"] = sta_p

    return sta_df


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
