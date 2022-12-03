"""Average dendritic events within electrotonic quantiles
Author: Drew B. Headley

"""
import numpy as np


def mean_dendevt(seg_df):
    """
    For each segment's dendritic event time series it is averaged across segments
    with the same electrotonic quantile.

    Parameters
    ----------
    seg_df : dataframe
        dendritic spike events series by segment

    Returns
    ----------
    mean_df : dataframe
        mean dendritic event series across segments with same electrotonic distance.
        Column name is 'evt_mean'

    Examples
    ----------

    """

    # determine event names
    ser_colname = [x for x in seg_df.columns if x.endswith("_ser")][0]

    # get spike triggered average, expressed as percent change from mean
    mean_temp = []
    for _, x in seg_df.iterrows():
        mean_temp.append(x[ser_colname].astype(float))
    seg_df["evt_mean"] = mean_temp

    # merge segments by electronic distance quantile
    prc_func = lambda bin: ((bin - np.mean(bin)) / np.mean(bin)) * 100
    agg_func = {"evt_mean": lambda x: prc_func(np.nanmean(np.vstack(x), 0))}
    mean_df = seg_df.groupby(["Elec_distanceQ", "Type"]).aggregate(agg_func)

    return mean_df


if __name__ == "__main__":
    """from load_spike_h5 import load_spike_h5
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
    print(sta_test)"""
