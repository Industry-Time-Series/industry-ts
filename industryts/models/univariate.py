"""
    Module with classes for univariate models.
"""
import abc
import numpy as np
import pandas as pd


class UnivariateModel(metaclass=abc.ABCMeta):
    """
    Base class for univariate models.
    """
    def __init__(self):
        super(UnivariateModel, self).__init__()

    def fit(self, data: pd.DataFrame, **kwargs):
        """
        Fit the model to the data.
        """
        pass
