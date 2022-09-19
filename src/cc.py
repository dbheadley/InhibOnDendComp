""" Cross-correlation between point processes
Author: Drew B. Headley

"""

import numpy as np
from numpy.random import randint


def ccptpt(oth, ref, bin=1, win=[-10, 10]):
    """
    Calculates the cross-correlation of two point processes
    
    Parameters
    ----------
    oth : array-like of ints 
        the times for the other point process. Should be sorted in ascending order and integers.
    ref : array-like of ints
        the times for the reference point process. Should be sorted in 
        ascending order and integers.
    bin : int (default is 1)
        the number of integer steps to count as a bin.
    win : [int, int] (default is [-10, 10])
        a 2 element list-like specifying the number of bins for either edge 
        of the cross correlation function.
    
    Returns
    ----------
    cc : numpy array
        the cross-correlation counts.
    relBins : numpy array
        the edges of the bins for cc. 
    
    Examples
    ----------
    Calculate cross correlation of two poisson processes with 1 ms 
    bins and a window of -20 to 20 ms.
    
    """
    b_edge = win[0] * bin
    t_edge = win[1] * bin

    cc_temp = np.zeros([t_edge - b_edge, 1])

    b_ind = 0
    t_ind = 0
    r_ind = 0
    l_ref = len(ref)
    l_oth = len(oth)
    win_edges = np.array(range(b_edge, t_edge))

    while r_ind < l_ref:  # iterate over all reference points
        # find index of first point in oth that is after bottom edge of CC
        # window for current reference point
        while ((b_ind < l_oth)) and (oth[b_ind] <= (ref[r_ind] + b_edge)):
            b_ind += 1

        # if you run out of reference points stop building cc
        if b_ind == l_oth:
            break

        # if the first other point after the bottom edge is also past the
        # edge of the CC window then move to the next reference point
        if oth[b_ind] > (ref[r_ind] + t_edge):
            r_ind += 1
            continue

        # find index of last point in other that is before the top
        # edge of the CC window for current reference point
        while (t_ind < l_oth) and (oth[t_ind] < (ref[r_ind] + t_edge)):
            t_ind += 1

        # accumulate the number of points within the window for other at
        # different delays from the current reference point
        for sub_ind in range(b_ind, t_ind):
            cc_ind = (oth[sub_ind] - ref[r_ind]) - b_edge
            cc_temp[cc_ind] += 1

        r_ind += 1

    # bin accumulated cc coints by the bin size
    if bin == 1:
        return cc_temp, win_edges
    else:
        cc_temp = np.reshape(cc_temp, [-1, bin])
        cc = np.sum(cc_temp, axis=1)
        bin_edges = np.array(range(b_edge, t_edge, bin))
        return cc, bin_edges


# Debug test
if __name__ == "__main__":
    ref_pts = np.sort(randint(0, 100000, 10000))
    oth_pts = np.sort(randint(0, 100000, 10000))
    cc, edges = ccptpt(oth_pts, ref_pts, 1, [-10, 10])
    print(cc)
    print(edges)

    cc, edges = ccptpt(ref_pts, ref_pts, 2, [-10, 10])
    print(cc)
    print(edges)
