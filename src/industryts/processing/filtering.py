"""
    Module with functions for filtering samples.
"""
from typing import Union

import pandas as pd
import numpy as np

from processing.timestamps import infer_sampling_time

from pandas.tseries.frequencies import to_offset


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
    event_starts = np.r_[0, np.nonzero(idx_disc)[0]]
    # Index of timeline array "t" where each event window ends
    # Subtract one because the last event ends on the last sample before the
    # next event starts. Add the last sample "t.shape[0]-1" as the end of the
    # last event.
    event_ends = np.r_[np.nonzero(idx_disc)[0] - 1, idx_disc.shape[0] - 1]
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
            if (end >= start)  & (end - start >= min_length):
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
    if isinstance(rm_interval_start, str):
        rm_interval_start = pd.to_timedelta(rm_interval_start)
    if isinstance(rm_interval_stop, str):
        rm_interval_stop = pd.to_timedelta(rm_interval_stop)
    if isinstance(minimum_interval, str):
        minimum_interval = pd.to_timedelta(minimum_interval)

    dataset = data.copy()
    # All time instants where events are happening
    rm_events_idx = dataset[rm_events_mask].index
    # Sampling time
    t_s = infer_sampling_time(dataset)
    offset = to_offset(t_s)
    freq = offset.freqstr

    # All time instants where events are happening (in the index), with an
    # indicator (series value equal to 1) of whether that instant is the start
    # (first sample) of an event window.
    t = rm_events_idx.to_series().diff().gt(t_s).astype(int)
    # Index of TIMELINE ARRAY "t" where each event window begins
    event_starts = np.r_[0, np.nonzero(t)[0]]
    # Index of TIMELINE ARRAY "t" where each event window ends
    # Subtract one because the last event ends on the last sample before the
    # next event starts. Add the last sample "t.shape[0]-1" as the end of the
    # last event.
    event_ends = np.r_[np.nonzero(t)[0]-1, t.shape[0]-1]

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
        series_diff_0 = np.nonzero(series_diff)[0]
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
    """
    Removes columns for which the coefficient of variation is below a selected
        threshold.

        If the mean of the column is 0, the standard deviation is used instead
        of the coefficient of variation.

    Args:
        df (pd.DataFrame): Process dataframe
        min_std_cv (float, optional): Minimum variation coefficient.
            Defaults to 0.01.
        columns (list, optional): List of columns to check. If None, all
            columns are checked. Defaults to None.
        return_removed (bool, optional): If True, returns the removed columns
            and their coefficient of variation. Defaults to True.

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
            else:
                if np.abs(col_std) <= min_std_cv:
                    df = df.drop(col, axis=1)
                    column_coef_var[col] = col_cv

    coef_var_pd = pd.DataFrame.from_dict(column_coef_var, orient='index',
                                         columns=['coef_var'])
    if return_removed:
        return df, coef_var_pd

    return df


def format_start(df: pd.DataFrame, s: int = 0, m: int = 0,
                 h: int = 0):
    """
    Function to remove (if necessary) the first rows of the data in
        order to have the first row in a specific format.

    If the formatting is not possible, the function returns the original
        dataframe.

    Args:

        df (pd.DataFrame): df whose index is in datetime format.
        s (int): initial second, with all windows starting with this second.
        m (int): initial minute, with all windows starting with this minute.
        h (int): if 0, initial hours are even; if 1, initial hours are odd.

    Returns:

        pd.DataFrame: with the specified format.

    """
    # hour = 0 -> even hours
    # hour = 1 -> odd hours
    matches = np.array([True] * len(df))
    if s != -1:
        matches = matches & (df.index.second == s)
    if m != -1:
        matches = matches & (df.index.minute == m)
    if h != -1:
        matches = matches & (df.index.hour % 2 == h)

    if matches.any():
        first_match = df.index[matches].min()
        df = df.loc[first_match:]
    return df
