""" Generate count series using start and stop times
Author: Drew B. Headley

"""

import numpy as np
import pandas as pd


def ser_ss(start_pts, stop_pts, step_len, win_lim):
    """
    Produces a times series that counts between corresponding start and stop
    points and zero everywhere else

    Parameters
    ----------
    start_pts : numeric array-like, N length
        the times when an event started
    stop_pts : numeric array-like, N length
        the times when an event stopped
    step_len : numeric, optional
        the length of each step in the series. Generally, it should be less
        than the shortest time between start and stop points. By default,
        it is set to that time
    win_lim : list, 2 elements, optional
        the beginning and end of the series. When inf is used, the series
        will start at the first start point and end at the last stop point.
        By default, the window starts at the first start point and ends at
        the last stop point

    Returns
    ----------
    ser : pandas Series
        Series where the periods between start and stop points are count
        instances of event occurrences, and all other periods are 0

    Examples
    ----------

    """

    # check that start and stop points are the same size
    if len(start_pts) != len(stop_pts):
        raise ValueError("Start and stop points must be the same length")

    # ensure that start and stop points are Nx1 numpy arrays
    start_pts = np.array(start_pts).reshape(-1, 1)
    stop_pts = np.array(stop_pts).reshape(-1, 1)

    # ensure that each start point is before its corresponding stop point
    if np.any((stop_pts - start_pts) < 0):
        raise ValueError("Start and stop points are ill-matched")

    # if step_len is not specified, set it to the shortest time between start
    # and stop
    if step_len is None:
        step_len = np.min(
            np.diff(np.concatenate([start_pts, stop_pts], axis=1), axis=1)
        )

    # if win_lim is not specified, set it to begin at the first start point
    # and end at the last stop point
    if win_lim is None:
        win_lim = [start_pts[0], stop_pts[-1]]

    # create start/stop series
    ss_t = np.arange(win_lim[0], win_lim[1], step_len)
    ss_v = np.zeros(ss_t.shape)

    # discretize start and stop
    start_pts = np.floor(start_pts / step_len).astype(int)
    stop_pts = np.ceil(stop_pts / step_len).astype(int)

    # remove start and stop points outside the window
    # pdb.set_trace()
    early_pts = (start_pts < win_lim[0]) & (stop_pts < win_lim[0])
    late_pts = start_pts >= (win_lim[1] - step_len)
    bad_pts = early_pts | late_pts
    start_pts = np.delete(start_pts, bad_pts.reshape((-1)))
    stop_pts = np.delete(stop_pts, bad_pts.reshape((-1)))
    start_pts = np.clip(start_pts, win_lim[0], win_lim[1])
    stop_pts = np.clip(stop_pts, win_lim[0], win_lim[1])

    # start is r, stop is p
    for (r, p) in zip(start_pts, stop_pts):
        ss_v[r:p] += 1

    return pd.Series(ss_v, ss_t)


# Debug test


if __name__ == "__main__":
    import pdb

    test_starts = [1, 10, 20, 23, 31, 31]
    test_stops = [2, 12, 23, 28, 31, 38]
    test_ser = ser_ss(test_starts, test_stops, 2, [0, 33])
    print(test_ser)
