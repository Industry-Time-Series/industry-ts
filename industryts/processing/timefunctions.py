"""Functions to process time series data that aren't easily found."""

import pandas as pd
import numpy as np


def get_continuous_batches(data: pd.DataFrame, t_s: pd.Timedelta, min_time: str,
                           cut_head: int = 0, cut_tail: int = 0,
                           return_index: bool = False) -> list:
    """
    Get the start and end timestamps for each active batch present in the
    dataset. Used in the dataset after removing the shutdown time, so we only
    have discontinuity when the batch changes, but all the data is when the
    process is active. Furthermore, since the time for each batch is different,
    get to know the start and end of each one is extremely important. Also we
    have the min_time parameter, which is the minimum time that a batch has to
    be active to be considered as a batch. This is useful when we have a
    dataset with a lot of batches, but we only want to consider the ones that
    are active for a certain amount of time. If return_index is True, the
    function will return the index of the start and end of each batch, instead
    of the timestamp.


    Args:

        data (pd.DataFrame): The data to obtain the start and end timestamps
        t_s (pd.Timedelta): The sample time
        min_time(str): The minimum time to consider a batch as active. Ex:
            "1h" for 1 hour, "1d" for 1 day, etc.
        cut_head (int, optional): Number of samples to cut from the head of
            the batch. Defaults to 0.
        cut_tail (int, optional): Number of samples to cut from the tail of
            the batch. Defaults to 0.
        return_index (bool, optional): If True, the function will return the
            index of the start and end of each batch, instead of the timestamp.
    Returns:

        batches (list): The list of dicts containing the start and end
            of each batch.
    """
    dataset = data.copy()

    # All time instants where events are happening, with an indicator of
    # whether that instant is the start (first sample) of an event window.

    # ! Since we're using the diff from numpy, we have to prepend the first
    # ! value because the function does not return NaN for the first sample
    # ! as the diff from pandas.

    t = dataset.index.to_series().diff().gt(t_s).astype(int)

    # Index of timeline array "t" where each event window begins
    event_starts = np.r_[0, np.where(t != 0)[0]]
    # Index of timeline array "t" where each event window ends
    # Subtract one because the last event ends on the last sample before the
    # next event starts. Add the last sample "t.shape[0]-1" as the end of the
    # last event.

    event_ends = np.r_[np.where(t != 0)[0] - 1, t.shape[0] - 1]
    batches_dicts_time = []
    batches_dicts_index = []

    # Dict containing start and end of all events
    # Remove samples in the head and tail of the series
    event_starts += cut_head
    event_ends -= cut_tail
    for start, end in zip(event_starts, event_ends):
        if end >= start:
            start_time = dataset.index[start]
            end_time = dataset.index[end]
            if end_time - start_time >= pd.to_timedelta(min_time):
                batches_dicts_time.append(
                    {"start": start_time, "end": end_time})
                batches_dicts_index.append({"start": start, "end": end})
    if return_index:
        return batches_dicts_index
    else:
        return batches_dicts_time
