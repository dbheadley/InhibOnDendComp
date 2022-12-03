""" Generate binary series using point process times
Author: Drew B. Headley

"""

import numpy as np
import pandas as pd


def ser_pt(pts, step_len, win_lim):
    """
    Produces a times series that is binned counts of a point process

    Parameters
    ----------
    pts : numeric array-like, N length
        the times when an event occurred
    step_len : numeric, optional
        the length of each step in the series. Generally, it should be less
        than the shortest time between points times. By default,
        it is set to that time
    win_lim : list, 2 elements, optional
        the beginning and end of the series. When inf is used, the series
        will start at the first time point and end at the last time point.
        By default the window starts at the first time point and ends at
        the last time point

    Returns
    ----------
    ser : pandas Series
        Series with point process counts

    Examples
    ----------

    """

    # ensure that time points are Nx1 numpy arrays
    pts = np.array(pts).reshape(-1, 1)

    # if step_len is not specified, set it to the shortest time between points
    if step_len is None:
        step_len = np.min(np.diff(np.sort(pts)))

    # if win_lim is not specified, set it to begin at the first start point
    # and end at the last stop point
    if win_lim is None:
        win_lim = [pts[0], pts[-1]]

    # create start/sto
    ser_t = np.arange(win_lim[0], win_lim[1] + step_len, step_len)
    ser_v, _ = np.histogram(pts, ser_t)

    return pd.Series(ser_v, ser_t[:-1])


# Debug test


if __name__ == "__main__":
    print("Testing ser_pt.py")
    test_pts = [0, 1, 13, 14, 15, 20]
    test_ser = ser_pt(test_pts, 2, [0, 20])
    print(test_ser)
