"""Cross-correlation between action potentials and dendritic events
Author: Drew B. Headley

"""
import numpy as np
from .cc_serpt import cc_serpt


def sta_ap_dendevt(seg_df, ap_pts, **kwargs):
    """
    For each segment's dendritic event time series segment the cross-correlation
    with the action potential is calculated. The mean of these are then calculated
    across segments with the same electrotonic distance. STA is expressed as
    percent change from median.

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
    **kwargs : named arguments to pass to ser_ss
    Examples
    ----------

    """

    # determine event names
    low_colname = [x for x in seg_df.columns if x.endswith("lower_bound")]

    # determine event type
    evt_type = low_colname[0].split("_")[0]
    ser_colname = evt_type + "_ser"
    sta_colname = evt_type + "_sta"

    # get spike triggered average
    sta_temp = []
    for _, x in na_seg.iterrows():
        sta_temp.append(
            cc_serpt(x["na_ser"].astype(float), ap_pts.astype(int), **kwargs)["values"]
        )

    # percent change
    prc_func = lambda x: ((x - np.nanmedian(x)) / np.nanmedian(x)) * 100
    na_seg["na_sta"] = sta_temp
    na_seg["na_sta_prc"] = na_seg["na_sta"].apply(prc_func)

    # merge segments by electronic distance quantile
    agg_func = {"na_sta_prc": lambda x: np.nanmean(np.vstack(x), 0)}
    na_seg = na_seg.groupby("Elec_distanceQ").aggregate(agg_func)

    return seg_df


if __name__ == "__main__":
    from load_dendevt_csv import load_dendevt_csv
    from seg_dendevt import seg_dendevt

    print("Testing calcium spikes with ser_seg_dendevt.py")
    fpath_test = (
        "Y:\\DendCompOsc\\16Hzapical_exc_mod\\"
        "output_16Hz_dend_inh_0deg_exc_10p_ca.csv"
    )
    dspk_test = load_dendevt_csv(fpath_test)
    seg_test = seg_dendevt(dspk_test)
    seg_test = ser_seg_dendevt(seg_test, step_len=20, win_lim=[0, 2000000])
    print(seg_test)

    print("Testing sodium spikes with ser_seg_dendevt.py")
    fpath_test = (
        "Y:\\DendCompOsc\\16Hzapical_exc_mod\\"
        "output_16Hz_dend_inh_0deg_exc_10p_na.csv"
    )
    dspk_test = load_dendevt_csv(fpath_test)
    seg_test = seg_dendevt(dspk_test)
    seg_test = ser_seg_dendevt(seg_test, step_len=20, win_lim=[0, 2000000])
    print(seg_test)
