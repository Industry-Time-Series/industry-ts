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

    def __call__(
            self,
            data: Union[pd.DataFrame, np.ndarray],
            **kwargs) -> np.ndarray:
        self.predict(data, **kwargs)


class AutoRegressive(UnivariateModel):
    """
    Class for purely autoregressive models of order p.

    Args:
        p (int): Order of the model. Defaults to 1.
        bias (bool): Whether to include a bias term in the model.
            Defaults to True.

    Attributes:
        p: Order of the model.
        coef: Coefficients of the model.
    """

    def __init__(self, p: int = 1, bias: bool = True):
        super(AutoRegressive, self).__init__()
        self.p = p
        self.coef = None
        self.__bias = bias

    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the model to the data.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        # If data is 1D, make it 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        regressors = np.hstack(
            [data[i:-self.p + i] for i in range(self.p)])
        targets = data[self.p:]
        if self.__bias:
            regressors = np.hstack([np.ones((regressors.shape[0], 1)), regressors])
        self.coef = np.linalg.lstsq(regressors, targets, rcond=None)[0]

    def predict(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Predict the values of the data.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        if self.__bias:
            data = np.hstack([np.ones((data.shape[0], 1)), data])
        return data @ self.coef
