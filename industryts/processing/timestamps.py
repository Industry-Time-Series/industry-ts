"""
    Module with functions related to timestamp indices.
"""

import pandas as pd


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
