"""
    Module with functions for filtering samples.
"""

import pandas as pd
import numpy as np

from typing import Union
from processing.timestamps import infer_sampling_time


def get_continuous_patches(
        data: Union[pd.DataFrame, np.ndarray],
        sampling_time: pd.Timedelta = None,
        min_length: Union[str, int] = 0,
        drop_leading: int = 0,
        drop_trailing: int = 0,
        return_num_index: bool = False
) -> list:
    """
    Function to extract the time limits for all continuous patches of data.

    In the context of industrial processes, each continuous patch may be
    associated with a batch, or simply a period of time where the process is
    continuously active between shutdowns.

    This function is often used in conjunction with a function for removivng
    samples in the data that were collected in periods of shutdown.

    Args:
        data (pd.DataFrame): The data to obtain the start and end timestamps.

        sampling_time (pd.Timedelta): Sampling time. If None, it will be
            inferred from the data index. Defaults to None.

        min_length(str, int): The minimum duration to consider a patch as
        valid. If 'str', it must be a string that can be converted to a
        pd.Timedelta. If 'int', it must be an integer representing the number
        of samples. Defaults to 0, which means that all patches will be
        considered.

        drop_leading (int, optional): Number of samples to drop from
            the start of each patch of data. Defaults to 0.

        drop_trailing (int, optional): Number of samples to drop from the end
            of each patch of data. Defaults to 0.

        return_num_index (bool, optional): If True, the function will return
            the row number of the start and end of each patch,
            instead of the timestamp index (if available).
    Returns:
        patches (list): The list of dicts containing the start and end
            of each patch.
    """
    if isinstance(data.index, pd.DatetimeIndex):
        if sampling_time is None:
            sampling_time = infer_sampling_time(data)
        # All time instants where events are happening, with an indicator of
        # whether that instant is the start (first sample) of an event window.
        idx_disc = data.index.to_series().diff().gt(sampling_time).astype(int)
        if isinstance(min_length, str):
            min_length = pd.to_timedelta(min_length)
        elif min_length == 0:
            min_length = pd.to_timedelta('0S')
        else:
            raise ValueError("min_length must be a string if data index is "
                                "a DatetimeIndex.")
    else:
        idx_disc = data.index.to_series().diff().gt(1).astype(int)
        if isinstance(min_length, str):
            raise ValueError("min_length must be an integer if data index is "
                             "not a DatetimeIndex.")

    # Index of timeline array "t" where each event window begins
    event_starts = np.r_[0, np.where(idx_disc != 0)[0]]
    # Index of timeline array "t" where each event window ends
    # Subtract one because the last event ends on the last sample before the
    # next event starts. Add the last sample "t.shape[0]-1" as the end of the
    # last event.
    event_ends = np.r_[np.where(idx_disc != 0)[0] - 1, idx_disc.shape[0] - 1]
    patches_dicts = []

    # Dict containing start and end of all events
    # Remove samples in the head and tail of the series
    event_starts += drop_leading
    event_ends -= drop_trailing
    if isinstance(data.index, pd.DatetimeIndex):
        for start, end in zip(event_starts, event_ends):
            if end >= start:
                start_time = data.index[start]
                end_time = data.index[end]
                if end_time - start_time >= min_length:
                    if return_num_index:
                        patches_dicts.append({"start": start, "end": end})
                    else:
                        patches_dicts.append(
                            {"start": start_time, "end": end_time})
    else:
        for start, end in zip(event_starts, event_ends):
            if end >= start:
                if end - start >= min_length:
                    patches_dicts.append(
                        {"start": start, "end": end})

    return patches_dicts


def rm_stopped_operation(data: pd.DataFrame, rm_events_mask: np.ndarray,
                         rm_interval_start: Union[str, pd.Timedelta] = "0S",
                         rm_interval_stop: Union[str, pd.Timedelta] = "0S",
                         minimum_interval: Union[str, pd.Timedelta] = "0S",
                         return_shutdown_dict: bool = False,
                         ) -> pd.DataFrame:
    """
    Remove all samples in rm_events_mask, plus/minus stop_interval.

    Args:
        data (pd.DataFrame): data to be processed, must have datetime indexing

        rm_events_mask (ndarray): boolean ndarray with length equal to number
                             of rows in data, where rows to be removed are True

        rm_interval_start (Union[str, pd.Timedelta]): time interval to be
                                removed before the events in the mask

        rm_interval_stop (Union[str, pd.Timedelta]): time interval to be
                                removed after the events in the mask

        minimum_interval (string): mininum duration of a stop

        return_shutdown_dict (bool): if True, returns a dictionary with start
                                    and end indices of all events

    Returns:
        pd.DataFrame: data with the rows of rm_events_mask and samples around
            stop_interval removed, in addition to (optionally) shutdown dict.
    """
    # First of all, we convert the time intervals to pd.Timedelta objects
    if type(rm_interval_start) == str:
        rm_interval_start = pd.to_timedelta(rm_interval_start)
    if type(rm_interval_stop) == str:
        rm_interval_stop = pd.to_timedelta(rm_interval_stop)
    if type(minimum_interval) == str:
        minimum_interval = pd.to_timedelta(minimum_interval)

    dataset = data.copy()
    # All time instants where events are happening
    rm_events_idx = dataset[rm_events_mask].index
    freq = pd.infer_freq(dataset.index[:10])
    if len(str(freq)) == 1:
        freq = f"1{freq}"
    t_s = pd.to_timedelta(freq)

    # All time instants where events are happening (in the index), with an
    # indicator (series value equal to 1) of whether that instant is the start
    # (first sample) of an event window.
    t = rm_events_idx.to_series().diff().gt(t_s).astype(int)
    # Index of TIMELINE ARRAY "t" where each event window begins
    event_starts = np.r_[0, np.where(t == 1)[0]]
    # Index of TIMELINE ARRAY "t" where each event window ends
    # Subtract one because the last event ends on the last sample before the
    # next event starts. Add the last sample "t.shape[0]-1" as the end of the
    # last event.
    event_ends = np.r_[np.where(t == 1)[0]-1, t.shape[0]-1]

    # List of dicts containing start and end of all events
    shutdown_dicts = []
    for start, end in zip(event_starts, event_ends):
        start_time = t.index[start]
        end_time = t.index[end]
        shutdown_dicts.append({"start": start_time, "end": end_time})
    # All indexes to be removed
    stop_idx = np.empty((0), dtype='datetime64[ns]')

    for event in shutdown_dicts:
        stop_interval = event['end'] - event['start']
        if stop_interval >= minimum_interval:
            start = event['start'] - rm_interval_start
            # Ending moment of the window post event
            end = event['end'] + rm_interval_stop
            interval = pd.date_range(start, end, freq=freq)
            stop_idx = np.r_[stop_idx, np.array(interval)]
            stop_idx = np.unique(stop_idx)

    stop_idx = np.intersect1d(stop_idx, dataset.index)
    dataset = dataset.drop(stop_idx)
    if return_shutdown_dict:
        return dataset, shutdown_dicts
    else:
        return dataset
