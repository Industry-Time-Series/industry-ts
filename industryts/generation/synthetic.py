"""
    Module dedicated for functions used in the generation of synthetic time
    series.
"""
import collections.abc

import numpy as np


def ar_process(coefs: list, samples: int = 100, noise: float = 0):
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
        prev_samples = y[(k-order):k][::-1]

        y[k] = np.sum(np.array(prev_samples) * coefs) + np.random.normal()

    # Since the noise is intended to emulate measurement noise, it is
    # added to the measurements after the AR process is generated.
    if noise:
        y += np.random.normal(0, noise, samples)

    return np.array(y)
