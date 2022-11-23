""" Permute the times of a point processes
Author: Drew B. Headley

"""

import numpy as np
from numpy import random


def permute_pt(pt):
    """
    Permutes a point process by randomly reordering their intervals.
    Returned process has the same number of points and interval distribution.

    Parameters
    ----------
    pt : array-like of ints
        the times for the other point process. Should be sorted in ascending
        order and integers.

    Returns
    ----------
    perm_pt : numpy array of ints

    Examples
    ----------


    """

    # initialize random number generator
    rng = random.default_rng()

    # ensure proper formatting of input point process
    pt_orig_shape = pt.shape
    pt = pt.flatten()  # reshape((-1, 1))

    # calculate intervals between events, treats it as poisson
    # process so first point time is defined as an interval
    itv = np.insert(np.diff(pt), 0, pt[0])

    # randomly permutes intervals and cumulatively sums the
    # intervals
    perm_pt = np.cumsum(rng.choice(itv, size=itv.size, replace=False))

    return perm_pt.reshape(pt_orig_shape)


# Debug test
if __name__ == "__main__":
    from numpy.random import randint
    from prettytable import PrettyTable

    print("Permute random point process")
    rand_pts = np.sort(randint(0, 100, 10))
    rand_perm_pts = permute_pt(rand_pts)
    perm_tbl = PrettyTable()
    perm_tbl.add_column("OriginalPoints", rand_pts)
    perm_tbl.add_column("PermutedPoints", rand_perm_pts)
    print(perm_tbl)

    print("Permute random point process")
    fixed_pts = np.arange(0, 100, 10)
    fixed_perm_pts = permute_pt(fixed_pts)
    perm_tbl = PrettyTable()
    perm_tbl.add_column("OriginalPoints", fixed_pts)
    perm_tbl.add_column("PermutedPoints", fixed_perm_pts)
    print(perm_tbl)
