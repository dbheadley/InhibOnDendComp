import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src/"))

import cc
from numpy.random import randint
import h5py
import matplotlib.pyplot as plt

print(__file__)
spikes = h5py.File("../data/spikes.h5", "r")
spk_t = np.array(spikes["spikes"]["biophysical"]["timestamps"])
nmda_evts = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "../data/DendEventTimes/nmda_spk_times.csv")
)
os.path.join(os.path.dirname(__file__), "..", "src/")
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
