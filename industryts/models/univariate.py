"""
    Module with classes for univariate models.
"""
import abc

import numpy as np
import pandas as pd

from typing import Union


class UnivariateModel(metaclass=abc.ABCMeta):
    """
    Base class for univariate models.
    """

    def __init__(self):
        super(UnivariateModel, self).__init__()

    @abc.abstractmethod
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the model to the data.
        """
        pass

    @abc.abstractmethod
    def predict(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Predict the values of the data.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(
            self,
            data: Union[pd.DataFrame, np.ndarray],
            **kwargs) -> np.ndarray:
        self.predict(data, **kwargs)
