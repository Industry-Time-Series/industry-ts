"""Functions to process time series data that aren't easily found."""

import pandas as pd
import numpy as np
from typing import Union


def get_continuous_patches(data: pd.DataFrame, t_s: pd.Timedelta = None,
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

        t_s (pd.Timedelta): Sampling time. If None, it will be inferred from
            the data index. Defaults to None.

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
    if t_s is None:
        t_s = infer_sampling_time(data)

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


def counts_ratio_per_patch(timeseries: pd.Series, patches_dicts: list,
                           column: str,
                           return_timestamps: bool = False) -> pd.DataFrame:
    """
    Calculates the ratio between counts of all values in a batch and the
        samples in the batch. Used for series with discrete inputs as a way of
        aggregating a batch.

    Args:
        timeseries (pd.Series): The time series to be processed

        batches_dicts (list): The list containing the start and end of
            every batch

        column (str): The column name to be used in the ratio calculation

        return_timestamps (bool): If True, the function will return the
            timestamps of the start and end of every batch. Defaults to False.

    Returns:
        df_column_values (pd.DataFrame): A dataframe contaning the ratio of
            column in every data window.
    """

    # Possible values for that column in any batch
    values = list(timeseries.unique())

    # The final list with every ratio value for every batch
    column_values = []

    for event in patches_dicts:
        # Starting moment of the window
        start = event['start']
        # Ending moment of the window
        end = event['end']

        # Create a key for every possible value in that batch
        batch_values = {value: 0 for value in values}

        # Count of each value in that batch
        values_event = timeseries.loc[start:end].value_counts().to_dict()
        size_event = timeseries.loc[start:end].shape[0]

        # * Since the batch does not necessarily has all the values, we use
        # * it's value counts to update the batches values
        for key, _ in values_event.items():
            batch_values[key] = values_event[key]/size_event

        if return_timestamps:
            batch_values["start"] = start
            batch_values["end"] = end

        column_values.append(batch_values)

    df_column_values = pd.DataFrame(column_values)

    for df_column in df_column_values.columns:
        df_column_values = df_column_values.rename(
            {
                df_column: f'{column}_{df_column}'
            },
            axis='columns')

    return df_column_values


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
