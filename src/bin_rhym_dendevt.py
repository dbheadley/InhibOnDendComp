"""Bin dendritic events by rhythmic signal
Author: Drew B. Headley

"""
import numpy as np
from .bin_serser import bin_serser


def bin_rhym_dendevt(seg_df, rhym_ser, bin_num):
    """
    For each segment's dendritic event time series it is binned with
    respect to a rhythmic series.

    Parameters
    ----------
    seg_df : dataframe
        dendritic spike events series by segment
    rhym_ser : numpy array
        the rhythmic signal to be binned with
    bin_num : integer
        number of bins to use for the rhym_ser

    Returns
    ----------
    bin_df : dataframe
        binned for each dendritic event. Column name is 'ph_bin'

    Examples
    ----------

    """

    edges = np.linspace(-np.pi, np.pi, bin_num)

    # determine event names
    ser_colname = [x for x in seg_df.columns if x.endswith("_ser")][0]

    # get spike triggered average, expressed as percent change from mean
    prc_func = lambda bin, m: ((bin - m) / m) * 100
    bin_temp = []
    for _, x in seg_df.iterrows():
        bin_temp.append(
            prc_func(
                bin_serser(
                    x[ser_colname].astype(float), rhym_ser, edges, func=np.nanmean
                )["values"],
                np.nanmean(x[ser_colname].astype(float)),
            )
        )
    seg_df["ph_bin"] = bin_temp

    # merge segments by electronic distance quantile
    agg_func = {"ph_bin": lambda x: np.vstack(x)}
    bin_df = seg_df.groupby(["Elec_distanceQ", "Type"]).aggregate(agg_func)

    # mean percent change and t-stats by electrotonic distance
    bin_m = []
    for _, x in bin_df.iterrows():
        bin_m.append(np.nanmedian(x["ph_bin"], 0))

    bin_df["ph_bin"] = bin_m

    return bin_df


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
