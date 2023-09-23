"""Counts the number of dendritic spike events from files
Author: Drew B. Headley

"""
import numpy as np
import pandas as pd
from .load_dendevt_csv import load_dendevt_csv
from .seg_dendevt import seg_dendevt
from .ser_seg_dendevt import ser_seg_dendevt

def count_dendevt_files(
    dend_fname,
    dt,
    win_ser,
    agg_colname="Elec_distanceQ",
):
    """
    Calculates the number of dendritic spikes stratified by a dendritic segment's
    electrotonic distance.

    Parameters
    ----------
    dend_fname : string
        file path for dendritic spike events csv
    dt : numeric
        the time step of the simulation
    win_ser : list, 2 elements
        the beginning and end of the series. When inf is used, the series
        will start at the first start point and end at the last stop point.
    agg_colname : str (default: "Elec_distanceQ")
        column name to aggregate dendritic compartments by

    Returns
    ----------
    dend_seg : dataframe
        dendritic spike rate for each dendritic segment. Column name is 'rate'

    Examples
    ----------

    """

    # load dendritic events
    dend_t = load_dendevt_csv(dend_fname)
    dend_seg = seg_dendevt(dend_t, agg_colname=agg_colname)

    # determine event names
    colname = [x for x in dend_seg.columns if x.endswith("lower_bound")][0]

    # calculate mean time between events
    dend_seg["rate"] = dend_seg[colname].apply(lambda x: len(x)/(float(win_ser[1]-win_ser[0])*dt))

    dend_seg = dend_seg.groupby(["Type", agg_colname]).aggregate({"rate": np.mean})
    return dend_seg


if __name__ == "__main__":
    import sys
    sys.path.append('..') # have to do this for relative imports in jupyter
    import pdb
    print("Testing count_dendevt_files.py")
    
    test_dend_fname = "Y:\\DendCompOsc\\output_EI_prox_4_dist_10\\output_EI_prox_4_dist_10_nmda.csv"
    dt = 0.0001
    win_ser = [0, 1500000]


    count_test = count_dendevt_files(test_dend_fname, dt, win_ser)
    print(count_test)
