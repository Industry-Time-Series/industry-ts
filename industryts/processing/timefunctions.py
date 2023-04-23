"""Functions to process time series data that aren't easily found."""

import pandas as pd
import numpy as np


def get_continuous_patches(data: pd.DataFrame, t_s: pd.Timedelta,
                           min_time: str = '0S', drop_head: int = 0,
                           drop_tail: int = 0,
                           return_num_index: bool = False) -> list:
    """
    Function to extract the time limits for all continuous patches of data.

    In the context of industrial processes, each continuous patch may be
    associated with a batch, or simply a period of time where the process is
    continuously active between shutdowns.

    This function is often used in conjunction with a function for removivng
    samples in the data that were collected in periods of shutdown.

    Args:
        data (pd.DataFrame): The data to obtain the start and end timestamps.
        t_s (pd.Timedelta): Sampling time.
        min_time(str): The minimum duration time to consider a patch as valid.
            Defaults to '0S', which means that all patches will be considered.
        drop_head (int, optional): Number of samples to drop from the head of
            each patch of data. Defaults to 0.
        drop_tail (int, optional): Number of samples to drop from the tail of
            each patch of data. Defaults to 0.
        return_num_index (bool, optional): If True, the function will return
            the row number of the start and end of each patch,
            instead of the timestamp index.
    Returns:
        patches (list): The list of dicts containing the start and end
            of each patch.
    """
    # All time instants where events are happening, with an indicator of
    # whether that instant is the start (first sample) of an event window.
    t = data.index.to_series().diff().gt(t_s).astype(int)

    # Index of timeline array "t" where each event window begins
    event_starts = np.r_[0, np.where(t != 0)[0]]
    # Index of timeline array "t" where each event window ends
    # Subtract one because the last event ends on the last sample before the
    # next event starts. Add the last sample "t.shape[0]-1" as the end of the
    # last event.
    event_ends = np.r_[np.where(t != 0)[0] - 1, t.shape[0] - 1]
    patches_dicts = []

    # Dict containing start and end of all events
    # Remove samples in the head and tail of the series
    event_starts += drop_head
    event_ends -= drop_tail
    for start, end in zip(event_starts, event_ends):
        if end >= start:
            start_time = data.index[start]
            end_time = data.index[end]
            if end_time - start_time >= pd.to_timedelta(min_time):
                if return_num_index:
                    patches_dicts.append({"start": start, "end": end})
                else:
                    patches_dicts.append(
                        {"start": start_time, "end": end_time})

    return patches_dicts


def infer_sampling_time(data: pd.DataFrame) -> float:
    """
    Infers sampling time from datetime index.

    If the index is not a datetime index, it will raise an error.

    Returns the sampling time as a pd.Timedelta object.

    Args:
        data (pd.DataFrame): Data frame with datetime index to infer
          the sampling time

    Returns:
        t_s (pd.Timedelta): The inferred sampling time
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a datetime index.")

    freq = pd.infer_freq(data.index[:10])
    sample = 1
    while freq is None:
        freq = pd.infer_freq(data.index[sample: 10 + sample])
        sample += 1
        if sample > data.shape[0] - 10:
            raise ValueError(
                "Could not infer sampling time. Exhauted all samples.")
    # If the frequency is 1S, this function should return 1S instead of S
    if len(str(freq)) == 1:
        freq = f"1{freq}"
    t_s = pd.to_timedelta(freq)

    return t_s
