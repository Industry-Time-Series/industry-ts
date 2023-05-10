"""
    Module with functions for filtering samples.
"""

import pandas as pd
import numpy as np

from typing import Union
from processing.timestamps import infer_sampling_time


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


def filter_static_windows(data: pd.DataFrame, columns: list = None,
                          threshold: int = 10,
                          remove_window: bool = False,
                          return_n_removed: bool = True) -> pd.DataFrame:
    """
    Filter windows where there is no variation for 'threshold' consecutive
        samples. In other words, replace static windows for null values or
        remove windows, where a window is considered static if the value of the
        series remains unchanged for at least 'threshold' samples.

    Args:
        data (pd.DataFrame): Dataframe with the values to be analyzed
        columns (list, optional): Columns to check if the windows are static.
            Defaults to [].
        threshold (int, optional): Minimum length of sequence of samples with
            the same value to remove. Defaults to 10.
        remove_window (bool, optional): If True, removes the static windows.
            If False, replace static windows with NA. Defaults to False.
        return_n_removed (bool, optional): If True, returns the amount of
            removed samples for every column. Defaults to True.

    Returns:
        pd.DataFrame: Dataframe without the static windows (or NAs inplace).
        pd.DataFrame: Columns with amount of static windows removed (or NAs
        inplace).
    """
    if not columns:
        columns = data.columns
    column_time_static = {}
    for column in columns:
        time_static = 0
        series = data[column]
        series_diff = series.diff()
        series_diff_0 = np.where(series_diff == 0)[0]
        if list(series_diff_0):
            dis_points_start = [series_diff_0[0]] + list(series_diff_0[
                np.where(np.diff(series_diff_0,
                                 prepend=series_diff_0[0]) > 1)[0]])
            dis_points_end = (list(series_diff_0[np.where(np.diff(
                series_diff_0, prepend=series_diff_0[0]) > 1)[0] - 1]) +
                [series_diff_0[-1]])

            # Reverse the lists to drop correctly the samples (if we drop the
            # first sample, the index of the second sample is not coherent
            # anymore)
            dis_points_start.reverse()
            dis_points_end.reverse()

            for start, end in zip(dis_points_start, dis_points_end):
                if (end - start) >= threshold:
                    # If remove is True, remove the window
                    if remove_window:
                        # * Drop including the 'end' sample
                        data = data.drop(data[start:end+1].index)
                    else:
                        # * Replace the window with null values
                        data.loc[start:end, column] = np.nan
                    time_static += (end - start)

        column_time_static[column] = time_static
        time_static_pd = pd.DataFrame.from_dict(column_time_static,
                                                orient='index',
                                                columns=['time_static'])
        time_static_pd.sort_values(by='time_static', ascending=False,
                                   inplace=True)

    if return_n_removed:
        return data, time_static_pd
    else:
        return data


def remove_static_columns(df: pd.DataFrame, min_std_cv: float = 0.01,
                          columns: list = None,
                          return_removed: bool = True):
    """Checks if the values of the column don't change a minimum value,
        that would result in bad feature

    Args:
        df (pd.DataFrame): Process' dataframe
        min_std_cv (float, optional): Minimum variation coefficient.
            Defaults to 0.01.
        columns (list, optional): List of columns to check. If None, all
            columns are checked. Defaults to None.
        return_removed (bool, optional): If True, returns the removed columns
        and its coefficient of variation. Defaults to True.

    Returns:
        pd.DataFrame: The dataframe without the columns that don't change
        pd.DataFrame: Dataframe with the columns that don't change and their
            coefficient of variation
    """
    if columns is None:
        columns = df.columns
    column_coef_var = {}
    for col in columns:
        col_mean = df[col].mean()
        col_std = df[col].std()
        if col_mean != 0:
            col_cv = col_std / col_mean
            # ! If the column's standard deviation is 0, it is removed. It's
            # ! not possible to calculate the coefficient of variation in this
            # ! case, so we have to drop the column.
            if np.abs(col_cv) <= min_std_cv:
                df = df.drop(col, axis=1)
                column_coef_var[col] = col_cv
        else:
            if col_std == 0:
                df = df.drop(col, axis=1)
                column_coef_var[col] = 0
    coef_var_pd = pd.DataFrame.from_dict(column_coef_var, orient='index',
                                         columns=['coef_var'])
    if return_removed:
        return df, coef_var_pd
    else:
        return df
