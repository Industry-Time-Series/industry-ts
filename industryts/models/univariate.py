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

    Args:
        order (int): Order of the model.
        family (str): Family of the model. Defaults to "AR".

    Attributes:
        order: Order of the model.
        family: Family of the model.
    """

    def __init__(self, order: int, family: str):
        super(UnivariateModel, self).__init__()
        self.__order = order
        self.__family = family

    @abc.abstractmethod
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the model to the data.
        """
        pass

    @abc.abstractmethod
    def forecast(self, initial_conditions: Union[pd.DataFrame, np.ndarray],
                 horizon: int = 1) -> np.ndarray:
        """
        Simulate the model forward in time.

        Args:
            initial_conditions (ArrayLike): Initial conditions for the model.
            horizon (int): Number of steps to forecast. Defaults to 1.

        Returns:
            forecast (ndarray): Forecasted values.
        """
        pass

    def _fix_dim_type(self, data: Union[pd.DataFrame, np.ndarray]
                      ) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            data = data.values
        # If data is 1D, make it 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data

    def _prepare_regressors(self, data: np.ndarray,
                            inference: bool = False) -> np.ndarray:
        """
        Prepare the data for prediction.

        Args:
            data (ArrayLike): Data to be predicted.
            inference (bool): Whether the data is being used for inference.

        Returns:
            data (ndarray): Prepared data.
        """
        if inference:
            regressors = data[-self.__order:].reshape(1, -1)
        else:
            regressors = np.hstack(
                [data[i:-self.__order + i] for i in range(self.__order)])

        # Reverse the regressors to match the order of the coefficients
        regressors = regressors[:, ::-1]

        if self._bias:
            regressors = np.hstack([np.ones((regressors.shape[0], 1)),
                                    regressors])
        # *** The order of the returned columns is as follows
        # [bias, y[k-1], y[k-2], ..., y[k-order]]
        return regressors

    def __call__(
            self,
            data: Union[pd.DataFrame, np.ndarray],
            **kwargs) -> np.ndarray:
        self.predict(data, **kwargs)

    def __str__(self) -> str:
        return self.__family + " model of order " + str(self.__order)


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
        if p < 1:
            raise ValueError("The order of the model must be greater than 0.")
        if not isinstance(p, int):
            raise TypeError("The order of the model must be an integer.")
        if not isinstance(bias, bool):
            raise TypeError("The bias must be a boolean value.")
        super(AutoRegressive, self).__init__(order=p, family='Autoregressive')
        self.p = p
        self.coef = None
        self._bias = bias

    def fit(self, data: Union[pd.DataFrame, np.ndarray]):
        """
        Fit the model to the data.
        """
        if self.p > data.shape[0]:
            raise ValueError("The order of the model must be less than or "
                             "equal to the number of observations.")

        data = self._fix_dim_type(data)

        regressors = self._prepare_regressors(data)
        targets = data[self.p:]

        self.coef = np.linalg.lstsq(regressors, targets, rcond=None)[0]

    def forecast(self, initial_condition: Union[pd.DataFrame, np.ndarray],
                 horizon: int = 1):
        """
        Simulate the model forward in time.

        Args:
            initial_conditions (ArrayLike): Initial conditions for the model.
            horizon (int): Number of steps to forecast. Defaults to 1.

        Returns:
            forecast (ndarray): Forecasted values.
        """
        if self.coef is None:
            raise ValueError("The model must be fitted before forecasting.")
        if not isinstance(horizon, int):
            raise TypeError("The horizon must be an integer.")
        if horizon < 1:
            raise ValueError("The horizon must be greater than 0.")
        if initial_condition.shape[0] < self.p:
            raise ValueError("The initial conditions must have at least as "
                             "many observations as the order of the model.")

        regressors = self._prepare_regressors(initial_condition,
                                              inference=True)
        forecast = np.zeros((self.p + horizon))
        forecast[:self.p] = initial_condition[-self.p:]

        for i in range(self.p, self.p + horizon):
            forecast[i] = regressors @ self.coef
            regressors = self._prepare_regressors(forecast[-self.p:],
                                                  inference=True)
        return forecast[-horizon:]


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
        super(MovingAverage, self).__init__(order=q, family='Moving Average')
        self.q = q
        self.coef = None
        self._bias = bias

    def fit(self, data: Union[pd.DataFrame, np.ndarray],
            n_iterations: int = 100) -> None:
        """
        Fit the model to the data.
        """
        data = self._fix_dim_type(data)
        targets = data[self.q:]

        # Initialize the residuals
        residuals = data
        regressors = self._prepare_regressors(residuals)

        self.coef = [0 for _ in range(self.q)]

        for _ in range(n_iterations):
            # Calculate the coefficients
            self.coef = np.linalg.lstsq(regressors, targets, rcond=None)[0]
            # Calculate the residuals
            residuals = targets - regressors @ self.coef
            # Add q zeros to the beginning of the residuals to match the
            # dimensions of the regressors
            residuals = np.vstack([np.zeros((self.q, residuals.shape[1])),
                                   residuals])
            # Update the regressors
            regressors = self._prepare_regressors(residuals)
