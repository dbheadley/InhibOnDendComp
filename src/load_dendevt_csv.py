""" Creates a data frame from a csv file containing dendritic spike events
Author: Drew B. Headley

"""

import pandas as pd
import numpy as np


def load_dendevt_csv(fpath):
    """
    Loads a CSV file that contains dendritic spike times and the
    converts them to a numpy array of integer simulation time steps

    Parameters
    ----------
    fpath : string
        full file path to the CSV file

    Returns
    ----------
    dspk_df : dataframe
        dendritic spike events and their corresponding segments and
        segment properties

    Examples
    ----------

    """

    # prespecify datatypes for certain problematic
    dspk_df = pd.read_csv(fpath)

    # determine type of dendritic event
    col_names = dspk_df.columns
    col_name_lb = [x for x in col_names if x.endswith("_lower_bound")][0]

    # remove segments with no events (i.e. nan event times)
    dspk_df = dspk_df.dropna(axis=0, subset=[col_name_lb])

    # make sure columns that should be numeric are numeric
    dspk_df["segmentID"] = dspk_df["segmentID"].apply(int)
    dspk_df["Elec_distanceQ"] = dspk_df["Elec_distanceQ"].apply(int)

    # remove unneeded columns
    dspk_df = dspk_df.drop(
        [
            "Unnamed: 0",
            "BMTK ID",
            "X",
            "Sec ID",
            "Distance",
            "Coord X",
            "Coord Y",
            "Coord Z",
            "Elec_distance",
        ],
        axis=1,
    )

    dspk_df = dspk_df.reset_index()
    return dspk_df


if __name__ == "__main__":
    print("Testing load_dendevt_csv.py")
    fpath_test = (
        "Y:\\DendCompOsc\\16Hzapical_exc_mod\\"
        "output_16Hz_dend_inh_0deg_exc_10p_ca.csv"
    )
    dspk_test = load_dendevt_csv(fpath_test)
    print(dspk_test)


""" 

# nmda_evts = pd.read_csv("../../data/DendEventTimes/nmda_spk_times.csv")
na_evts = pd.read_csv("../data/DendEventTimes/na_spk_times.csv")

# create dataframe of node ID and electrotonic percentile

# first combine all dendritic spike dataframes so no node ID is left out
node_elec = pd.concat(
    [
        nmda_evts[["segmentID", "Elec_distance"]],
        na_evts[["segmentID", "Elec_distance"]],
    ],
    axis=0,
)
node_elec = node_elec.drop_duplicates("segmentID")

# calculate percentile rank and bin into 10th percentile steps
node_elec["Elec_dist_prc"] = round(node_elec.Elec_distance.rank(pct=True), 1)

# group dendritic events by electrotonic percentile
nmda_evts = nmda_evts.merge(
    node_elec[["segmentID", "Elec_dist_prc"]], how="inner", on="segmentID"
)
na_evts = na_evts.merge(
    node_elec[["segmentID", "Elec_dist_prc"]], how="inner", on="segmentID"
)

nmda_evts = nmda_evts.groupby("Elec_dist_prc")["nmda_lower_bound_ms"].apply(np.array)
na_evts = na_evts.groupby("Elec_dist_prc")["na_lower_bound_ms"].apply(np.array)

# Convert spike times to integers and sort
intfunc = lambda x: np.sort(np.round(x * 10).astype(int))  # assuming 10 kHz sample rate
spk_t = intfunc(spk_t)

nmda_evts = nmda_evts.apply(intfunc)
nmda_evts

ccfunc = lambda x: cc.ccptpt(x, spk_t, 1, [-10, 10])
# breakpoint()
nmda_evts["cc"] = nmda_evts.apply(ccfunc)

print(nmda_evts)
 """
