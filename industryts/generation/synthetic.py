"""
    Module dedicated for functions used in the generation of synthetic time
    series.
"""
import collections.abc
from typing import Union
import numpy as np
import pandas as pd


def ar_process(coefs: list, samples: int = 100, noise: float = 0
               ) -> np.ndarray:
    """
    Generate synthetic data from an Autoregressive (AR) process of a given
    length and known coefficients, with the possibility of adding noise to
    the measurements.

    Args:
        coefs (list): list with coefficients of lagged measurements of
        the series. The order of the AR process will be defined by the number
        of defined coefficients. For example, if coefs = [0.5, 0.3], the
        generated series will be an AR(2) process, where 0.5 is the coefficient
        of the first lagged measurement and 0.3 is the coefficient of the
        second lagged measurement.
        samples (int): number of data points to be generated. Default is 100.
        noise (float): standard deviation of the noise to be added to the
        measurements. Default is 0, which means no noise.

    Returns:
        series: array with the generated AR process.
    """
    # If coefs is not a list, make it one
    if not isinstance(coefs, collections.abc.Sequence):
        coefs = [coefs]
    # The order of the AR process is the number of coefficients
    order = len(coefs)
    coefs = np.array(coefs)

    y = np.zeros(samples)
    # Initial values y[0, 1, .., order]. These can be thought of as the
    # initial conditions of the AR process.
    y[:order] = [np.random.normal() for _ in range(order)]

    for k in range(order, samples):
        # Get previous values of the series, reversed. This is done to
        # match the order of the coefficients.
        prev_samples = y[(k - order):k][::-1]

        y[k] = np.sum(np.array(prev_samples) * coefs) + np.random.normal()

    # Since the noise is intended to emulate measurement noise, it is
    # added to the measurements after the AR process is generated.
    if noise:
        y += np.random.normal(0, noise, samples)

    return np.array(y)


def ma_process(coefs: list, samples: int = 100, noise: float = 0
               ) -> np.ndarray:
    """
    Generate synthetic data from a Moving Average (MA) process of a given
    length and known coefficients, with the possibility of adding noise to
    the measurements.

    Args:
        coefs (list): list with coefficients of lagged measurements of
        the series. The order of the MA process will be defined by the number
        of defined coefficients. For example, if coefs = [0.5, 0.3], the
        generated series will be an MA(2) process, where 0.5 is the coefficient
        of the first lagged measurement and 0.3 is the coefficient of the
        second lagged measurement.
        samples (int): number of data points to be generated. Default is 100.
        noise (float): standard deviation of the noise to be added to the
        measurements. Default is 0, which means no noise.

    Returns:
        series: array with the generated MA process.
    """
    # If coefs is not a list, make it one
    if not isinstance(coefs, collections.abc.Sequence):
        coefs = [coefs]
    # The order of the MA process is the number of coefficients
    order = len(coefs)
    coefs = np.array(coefs)

    y = np.zeros(samples)
    # White noise series of errors that will be used in the MA process
    nu = [np.random.normal() for _ in range(samples)]
    # Initialize the process for k = 0 where the previous values of nu are
    # zero.
    y[0] = nu[0]
    for k in range(1, samples):
        if k < order:
            # Not all past measurements of nu are available yet, so the
            # previous values of the series are calculated differently.
            prev_samples = nu[0:k][::-1]
            y[k] = np.sum(np.array(prev_samples) * coefs[:k]) + nu[k]
        else:
            # Get previous values of the series, reversed. This is done to
            # match the order of the coefficients.
            prev_samples = nu[(k - order):k][::-1]

            y[k] = np.sum(np.array(prev_samples) * coefs) + nu[k]

    if noise:
        y += np.random.normal(0, noise, samples)

    return np.array(y)


def discontinuous_timeseries(start_date: Union[str, pd.Timestamp],
                             end_date: Union[str, pd.Timestamp],
                             freq: Union[str, pd.Timedelta],
                             num_discontinuities: int,
                             is_categorical: bool = False) -> pd.Series:
    """
    Generate a discontinuous time series with random data.

    Args:
        start_date (str or Timestamp): start date of the time series.
        end_date (str or Timestamp): end date of the time series.
        freq (str or Timedelta): frequency of the time series.
        num_discontinuities (int): number of discontinuity points to be
        generated.

    Returns:
        ts (Series): discontinuous time series with random data.
    """
    # Create a date range with specified frequency
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
    if isinstance(freq, str):
        freq = pd.Timedelta(freq)
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    # Randomly select discontinuity points
    discontinuity_points = np.random.choice(date_range, num_discontinuities,
                                            replace=False)
    # Create the time series with random data
    if is_categorical:
        data = np.random.choice(['A', 'B', 'C', 'D'], len(date_range))
    else:
        data = np.random.rand(len(date_range))
    ts = pd.Series(data, index=date_range)
    # Drop NaN values at the discontinuity points
    for point in discontinuity_points:
        ts.drop(point, inplace=True)
    return ts


def part_static_timeseries(start_date: Union[str, pd.Timestamp],
                           end_date: Union[str, pd.Timestamp],
                           frequency: Union[str, pd.Timedelta],
                           n_samples_static: int,
                           value_static: float = 1.0) -> pd.DataFrame:
    """Generate a time series with a part that is static.

    Args:
        start_date (Union[str, pd.Timestamp]): Start date of the time series.
        end_date (Union[str, pd.Timestamp]): End date of the time series.
        n_samples_static (int): Number of samples that will be static.

    Raises:
        ValueError: Error raised if the number of samples is less than the
        number of static samples.

    Returns:
        pd.DataFrame: Time series with a part that is static.
    """
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
    if isinstance(frequency, str):
        frequency = pd.Timedelta(frequency)
    # Generate a DatetimeIndex with a frequency of 1 minute
    dt_index = pd.date_range(start=start_date, end=end_date, freq=frequency)

    # Create 3 random float columns
    col1 = np.random.randn(len(dt_index))
    col2 = np.random.randn(len(dt_index))
    col3 = np.random.randn(len(dt_index))

    if len(dt_index) < n_samples_static:
        raise ValueError('The number of samples must be greater than the '
                         'number of static samples.')
    # Set one of the columns to a constant value for n_samples_static samples
    col1[:n_samples_static] = value_static

    # Create the DataFrame
    df = pd.DataFrame({'col1': col1, 'col2': col2, 'col3': col3},
                      index=dt_index)

    return df
