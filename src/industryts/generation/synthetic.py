"""
    Module dedicated for functions used in the generation of synthetic time
    series.
"""
import collections.abc
from typing import Union

import numpy as np
import pandas as pd

rng = np.random.default_rng(1525)


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
          generated series will be an AR(2) process, where 0.5 is the
          coefficient of the first lagged measurement and 0.3 is the
          coefficient of the second lagged measurement.
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
    y[:order] = [rng.standard_normal() for _ in range(order)]

    for k in range(order, samples):
        # Get previous values of the series, reversed. This is done to
        # match the order of the coefficients.
        prev_samples = y[(k - order):k][::-1]

        y[k] = np.sum(np.array(prev_samples) * coefs) + rng.standard_normal()

    # Since the noise is intended to emulate measurement noise, it is
    # added to the measurements after the AR process is generated.
    if noise:
        y += rng.standard_normal(size=samples)*noise

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
          generated series will be an MA(2) process, where 0.5 is the
          coefficient of the first lagged measurement and 0.3 is the
          coefficient of the second lagged measurement.
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
    nu = [rng.standard_normal() for _ in range(samples)]
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
        y += rng.standard_normal(size=samples)*noise

    return np.array(y)


def seasonal_component(samples: int = 100, period: int = 10,
                       amplitude: float = 1, noise: float = 0
                       ) -> np.ndarray:
    """
    Generate a seasonal component of a given length, period and amplitude,
    with the possibility of adding noise to the measurements.

    The nature of the seasonal component is a sine wave with the given period
    and amplitude.

    Args:
        samples (int): number of data points to be generated. Default is 100.
        period (int): period of the seasonal component. Default is 10.
        amplitude (float): amplitude of the seasonal component. Default is 1.
        noise (float): standard deviation of the noise to be added to the
        measurements. Default is 0, which means no noise.

    Returns:
        series: array with the generated seasonal component.
    """
    # The seasonal component is a sine wave with the given period and
    # amplitude
    omega = 2 * np.pi / period
    y = amplitude * np.sin(omega * np.arange(samples))

    if noise:
        y += rng.standard_normal(size=samples)*noise

    return np.array(y)


def trend_component(samples: int = 100, slope: float = 0.1,
                    intercept: float = 0, noise: float = 0) -> np.ndarray:
    """
    Generate a trend component of a given length, slope and intercept, with
    the possibility of adding noise to the measurements.

    Args:
        samples (int): number of data points to be generated. Default is 100.
        slope (float): slope of the trend component. Default is 0.
        intercept (float): intercept of the trend component. Default is 0.
        noise (float): standard deviation of the noise to be added to the
        measurements. Default is 0, which means no noise.

    Returns:
        series: array with the generated trend component.
    """
    y = np.arange(samples) * slope + intercept

    if noise:
        y += rng.standard_normal(size=samples)*noise

    return np.array(y)


def discontinuous_timeseries(start_timestamp: Union[str, pd.Timestamp],
                             end_timestamp: Union[str, pd.Timestamp],
                             freq: Union[str, pd.Timedelta],
                             num_discontinuities: int,
                             is_categorical: bool = False) -> pd.Series:
    """
    Generate a random discontinuous time series with given start and end
    timestamps and frequency.

    Args:
        start_timestamp (str or Timestamp): start date of the time series.
        end_timestamp (str or Timestamp): end date of the time series.
        freq (str or Timedelta): frequency of the time series.
        num_discontinuities (int): number of discontinuity points to be
          generated.
        is_categorical (bool): if True, the time series will be categorical
          with levels 'A', 'B', 'C' and 'D'. Default is False.

    Returns:
        ts (Series): discontinuous time series with random data.
    """
    # Create a date range with specified frequency
    if isinstance(start_timestamp, str):
        start_timestamp = pd.Timestamp(start_timestamp)
    if isinstance(end_timestamp, str):
        end_timestamp = pd.Timestamp(end_timestamp)
    if isinstance(freq, str):
        freq = pd.Timedelta(freq)
    date_range = pd.date_range(
        start=start_timestamp, end=end_timestamp, freq=freq)
    # Randomly select discontinuity points
    discontinuity_points = rng.choice(date_range, num_discontinuities,
                                            replace=False)
    # Create the time series with random data
    if is_categorical:
        data = rng.choice(['A', 'B', 'C', 'D'], len(date_range))
    else:
        data = rng.uniform(size=len(date_range))
    ts = pd.Series(data, index=date_range)
    # Drop NaN values at the discontinuity points
    for point in discontinuity_points:
        ts.drop(point, inplace=True)
    return ts


def part_static_timeseries(start_timestamp: Union[str, pd.Timestamp],
                           end_timestamp: Union[str, pd.Timestamp],
                           frequency: Union[str, pd.Timedelta],
                           n_samples_static: int,
                           value_static: float = 1.0) -> pd.DataFrame:
    """Generate a time series with a part that is static.

    Args:
        start_timestamp (Union[str, pd.Timestamp]): Start date of the time
          series.
        end_timestamp (Union[str, pd.Timestamp]): End date of the time series.
        n_samples_static (int): Number of samples that will be static.

    Raises:
        ValueError: Error raised if the number of samples is less than the
          number of static samples.

    Returns:
        pd.DataFrame: Time series with a part that is static.
    """
    if isinstance(start_timestamp, str):
        start_timestamp = pd.Timestamp(start_timestamp)
    if isinstance(end_timestamp, str):
        end_timestamp = pd.Timestamp(end_timestamp)
    if isinstance(frequency, str):
        frequency = pd.Timedelta(frequency)
    # Generate a DatetimeIndex with a frequency of 1 minute
    dt_index = pd.date_range(
        start=start_timestamp, end=end_timestamp, freq=frequency)

    # Create 3 random float columns
    col1 = rng.standard_normal(size=len(dt_index))
    col2 = rng.standard_normal(size=len(dt_index))
    col3 = rng.standard_normal(size=len(dt_index))

    if len(dt_index) < n_samples_static:
        raise ValueError('The number of samples must be greater than the '
                         'number of static samples.')
    # Set one of the columns to a constant value for n_samples_static samples
    col1[:n_samples_static] = value_static

    # Create the DataFrame
    df = pd.DataFrame({'col1': col1, 'col2': col2, 'col3': col3},
                      index=dt_index)

    return df
