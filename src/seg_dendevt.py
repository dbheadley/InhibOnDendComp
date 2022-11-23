""" Aggregates dendritic events within segments
Author: Drew B. Headley

"""


def seg_dendevt(dspk_df):
    """
    Takes a dendritic events dataframe and aggregates event times by segment.

    Parameters
    ----------
    dspk_df : dataframe
        dendritic spike events and their corresponding segments and
        segment properties

    Returns
    ----------
    seg_df : dataframe
        dendritic spike events grouped by their segment

    Examples
    ----------

    """

    # determine event names
    low_colname = [x for x in dspk_df.columns if x.endswith("lower_bound")]
    up_colname = [x for x in dspk_df.columns if x.endswith("upper_bound")]

    # group times within segments
    agg_funcs = {"Elec_distanceQ": lambda x: x.iloc[0]}
    if len(low_colname) == 1:
        agg_funcs[low_colname[0]] = list
    if len(up_colname) == 1:
        agg_funcs[up_colname[0]] = list

    # group events within the same segment
    seg_df = dspk_df.groupby("segmentID").aggregate(agg_funcs)
    seg_df = seg_df.sort_values(["Elec_distanceQ", "segmentID"])

    return seg_df


if __name__ == "__main__":
    from load_dendevt_csv import load_dendevt_csv

    print("Testing seg_dendevt.py")
    fpath_test = (
        "Y:\\DendCompOsc\\16Hzapical_exc_mod\\"
        "output_16Hz_dend_inh_0deg_exc_10p_ca.csv"
    )
    dspk_test = load_dendevt_csv(fpath_test)
    seg_test = seg_dendevt(dspk_test)
    print(seg_test)
