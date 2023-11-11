"""
    Module with functions to train models, such as variations of
    the least squares method.
"""
import numpy as np


class LeastSquaresOptimizer:
    """
    Class with different least squares methods.

    Args:
        method (str): Method to use. Is one of the following:
            'OLS': Ordinary least squares.
            'RLS': Recursive least squares.
            'ELS': Extended least squares. (Not implemented yet)
            'regOLS': Regularized least squares. (Not implemented yet)
          Defaults to "OLS".
    """

    def __init__(self, method: str = "OLS"):
        self.method = method
        self._coefs = None
        self._fitted = False
        # Covariance matrix used in RLS.
        self.__p_mat = None

    @property
    def coefs(self):
        """
        Get the coefficients of the model.

        Returns:
            coef (ndarray): Coefficients of the model.
        """
        if self._fitted:
            return self._coefs
        else:
            raise ValueError("The model must be fitted first.")

    @coefs.setter
    def coefs(self, value):
        """
        Set the coefficients of the model.

        Args:
            value (ndarray): Coefficients of the model.
        """
        self._coefs = value
        self._fitted = True

    def fit(self, regressors: np.ndarray, targets: np.ndarray,
            inplace: bool = False, **kwargs):
        """
        Fit the model to the data.

        Args:
            regressors (ndarray): Matrix with regressors, commonly denominated
                the Phi matrix. Each column is a regressor, and each row is an
                observation.
            targets (ndarray): Vector with targets, commonly denominated the
                y vector. Each row is an observation.
            **kwargs: Additional keyword arguments for the different methods.

        Returns:
            coef (ndarray): Coefficients of the model.
        """
        if self.method == "OLS":
            coefs = self._ols(regressors, targets)
        elif self.method == "RLS":
            coefs = self._rls(regressors, targets, **kwargs)
        elif self.method == "ELS":
            raise ValueError("Method not implemented yet.")
        elif self.method == "regOLS":
            raise ValueError("Method not implemented yet.")
        else:
            raise ValueError("Method not recognized yet.")
        self.coefs = coefs

        if not inplace:
            return coefs


    @staticmethod
    def _ols(regressors: np.ndarray, targets: np.ndarray):
        """
        Ordinary least squares.

        Computes the vector x that approximately solves the equation
          regressors @ x = targets.

        Args:
            regressors (ndarray): Matrix with regressors, commonly denominated
                the Phi matrix. Each column is a regressor, and each row is an
                observation.
            targets (ndarray): Vector with targets, commonly denominated the
                y vector. Each row is an observation.

        Returns (optional):
            coef (ndarray): Coefficients of the model. Will be in shape [p, 1]
        """
        return np.linalg.lstsq(regressors, targets, rcond=None)[0]

    def _rls(self, regressors: np.ndarray,
             targets: np.ndarray,
             forgetting_factor: float = 0.99,
             initial_covariance: float = 1e6,
             return_history: bool = False):
        """
        Recursive least squares.

        Args:
            regressors (ndarray): Matrix with regressors, commonly denominated
                the Phi matrix. Each column is a regressor, and each row is an
                observation. This should be a batch of observations.
            targets (ndarray): Vector with targets, commonly denominated the
                y vector. Each row is an observation.
            forgetting_factor (float): Forgetting factor. Defaults to 0.99.
            initial_covariance (float): Initial covariance used to
                construct the initial covariance matrix P. Defaults to 1e6.
            return_history (bool): Whether to return the history of the
                coefficients. Defaults to False.
        Returns (optional):
            coef (ndarray): Coefficients of the model. Will be in shape [p, 1]
        """

        # Number of observations.
        n = regressors.shape[0]
        # Number of regressors.
        p = regressors.shape[1]

        # Initialize the coefficients and covariance matrix.
        if self._fitted:
            coef = self._coefs
            p_mat = self.__p_mat
        else:
            p_mat = initial_covariance * np.eye(p)
            coef = np.zeros(p)
        if return_history:
            coef_history = np.zeros((n, p))

            # Iterate over the observations.
            for i in range(n):
                # Get regressors and target for the current observation.
                phi = regressors[i, :].reshape(p, -1)
                y = targets[i]

                # Calculate the gain vector.
                k_mat_aux = np.linalg.pinv(
                    forgetting_factor + phi.T @ (p_mat @ phi))
                k_mat = (p_mat @ phi) @ k_mat_aux

                # Update the coefficients.
                coef += k_mat @ (y - phi.T @ coef)

                # Update the covariance matrix.
                p_mat = (
                    ((np.eye(p) - (k_mat @ phi.T)) @ p_mat)/forgetting_factor)

                # Save the coefficients.
                coef_history[i, :] = coef
        else:
            # Iterate over the observations.
            for i in range(n):
                # Get regressors and target for the current observation.
                phi = regressors[i, :].reshape(p, -1)
                y = targets[i]

                # Calculate the gain vector.
                k_mat_aux = np.linalg.pinv(
                    forgetting_factor + phi.T @ (p_mat @ phi))
                k_mat = (p_mat @ phi) @ k_mat_aux

                # Update the coefficients.
                coef += k_mat @ (y - phi.T @ coef)

                # Update the covariance matrix.
                p_mat = (
                    ((np.eye(p) - (k_mat @ phi.T)) @ p_mat)/forgetting_factor)
        self.__p_mat = p_mat
        if return_history:
            return coef_history
        else:
            return coef.reshape(p, 1)
