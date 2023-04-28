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

    def predict(self, data: Union[pd.DataFrame, np.ndarray]):
        """
        Predict the values of the data.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        if self.__bias:
            data = np.hstack([np.ones((data.shape[0], 1)), data])
        return data @ self.coef

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
            regressors = np.hstack(
                [np.ones((regressors.shape[0], 1)), regressors])
        self.coef = np.linalg.lstsq(regressors, targets, rcond=None)[0]


class MovingAverage(UnivariateModel):
    """
    Class for purely moving average models of order q.

    Args:
        q (int): Order of the model. Defaults to 1.
        bias (bool): Whether to include a bias term in the model.
            Defaults to True.

    Attributes:
        q: Order of the model.
        coef: Coefficients of the model.
    """

    def __init__(self, q: int = 1, bias: bool = True):
        super(MovingAverage, self).__init__()
        self.q = q
        self.coef = None
        self.__bias = bias

    def fit(self, data: Union[pd.DataFrame, np.ndarray],
            n_iterations: int = 100):
        """
        Fit the model to the data.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        # If data is 1D, make it 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.coef = [0 for _ in range(self.q)]
        # Initialize the residuals
        delayed_residuals = np.hstack(
            [data[i:-self.q + i] for i in range(self.q)])
        targets = data[self.q:]
        for _ in range(n_iterations):
            # Calculate the coefficients
            if self.__bias:
                regressors = np.hstack(
                    [np.ones((delayed_residuals.shape[0], 1)),
                     delayed_residuals])
            else:
                regressors = delayed_residuals
            self.coef = np.linalg.lstsq(regressors, targets, rcond=None)[0]
            # Calculate the residuals
            residuals = targets - regressors @ self.coef
            # Update the delayed residuals
            delayed_residuals = np.zeros((residuals.shape[0], self.q))
            for i in range(self.q):
                delayed_residuals[i + 1:, i] = residuals[:-(i + 1)].flatten()
