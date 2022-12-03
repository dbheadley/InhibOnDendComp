""" Creates binary time series of dendritic event occurrences
Author: Drew B. Headley

"""
import numpy as np
import sys

sys.path.append(".")  # have to do this for relative imports to work consistently
from .ser_ss import ser_ss
from .ser_pt import ser_pt


def ser_seg_dendevt(seg_df, **kwargs):
    """
    Takes a dendritic events dataframe grouped by segment and creates a binary
    time series of event periods.

    Parameters
    ----------
    seg_df : dataframe
        dendritic spike events grouped by segment

    Returns
    ----------
    seg_df : dataframe
        dendritic spike events grouped by their segment with added column
        containing binary time series. Column name is '<event_type>_ser'
    **kwargs : named arguments to pass to ser_ss
    Examples
    ----------

    """

    # determine event names
    low_colname = [x for x in seg_df.columns if x.endswith("lower_bound")]
    up_colname = [x for x in seg_df.columns if x.endswith("upper_bound")]

    # determine event type
    evt_type = low_colname[0].split("_")[0]

    # if NMDA or Ca spikes, create start/stop series
    if (evt_type == "nmda") or (evt_type == "ca"):
        ser_func = lambda x: np.array(
            ser_ss(x[low_colname[0]], x[up_colname[0]], **kwargs)
        )
    else:
        ser_func = lambda x: np.array(ser_pt(x[low_colname[0]], **kwargs))

    # convert to binary series
    ser_temp = []
    for _, row in seg_df.iterrows():
        ser_temp.append(ser_func(row))
    seg_df[evt_type + "_ser"] = ser_temp

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
