"""
    Test least squares optimization
"""
import numpy as np
import pytest

from src.industryts.models.optimization import LeastSquaresOptimizer
from src.industryts.generation.synthetic import ar_process, ma_process


np.random.seed(42)
AR_COEFS = [0.4, -0.3]
MA_COEFS = [0.1]
EX_COEFS = [0.5]
SAMPLES = 502
NOISE_MAGNITUDE = 0.3
INPUT_MAGNITUDE = 0.5
N_EXPS = 400


@pytest.fixture(scope="module")
def ar_measurements(experiments: int = N_EXPS):
    """Generate synthetic data from an AR process."""
    exps = []
    for _ in range(experiments):
        measurements = ar_process(AR_COEFS, SAMPLES).reshape(-1, 1)
        order = len(AR_COEFS)
        regressors = np.hstack(
            [measurements[i:-order + i] for i in range(order)])

        # Reverse the regressors to match the order of the coefficients
        regressors = regressors[:, ::-1]
        exps.append((regressors, measurements[order:]))

    return exps


@pytest.fixture(scope="module")
def arma_measurements(experiments: int = N_EXPS):
    """Generate synthetic data from an AR process."""
    exps = []
    for _ in range(experiments):
        measurements = ar_process(AR_COEFS, SAMPLES).reshape(-1, 1)
        measurements += ma_process(MA_COEFS, SAMPLES).reshape(-1, 1) * 0.2

        order = len(AR_COEFS)
        regressors = np.hstack(
            [measurements[i:-order + i] for i in range(order)])

        # Reverse the regressors to match the order of the coefficients
        regressors = regressors[:, ::-1]
        exps.append((regressors, measurements[order:]))

    return exps


@pytest.fixture(scope="module")
def ar_measurements_with_noise(experiments: int = N_EXPS):
    """Generate synthetic data from an AR process with MA noise."""
    exps = []
    for _ in range(experiments):
        white_noise = np.random.normal(
            0, NOISE_MAGNITUDE, SAMPLES).reshape(-1, 1)
        # Generate MA noise
        for sample in range(len(MA_COEFS), SAMPLES):
            for order, ma_coef in enumerate(MA_COEFS, start=1):
                white_noise[sample] += ma_coef * white_noise[sample - order]
        ma_noise = white_noise
        # Generate AR process with MA noise
        measurements = np.zeros(SAMPLES).reshape(-1, 1)
        for sample in range(len(AR_COEFS), SAMPLES):
            for order, ar_coef in enumerate(AR_COEFS, start=1):
                measurements[sample] += ar_coef * measurements[sample - order]
            measurements[sample] += ma_noise[sample]

        order = len(AR_COEFS)
        regressors = np.hstack(
            [measurements[i:-order + i] for i in range(order)])

        # Reverse the regressors to match the order of the coefficients
        regressors = regressors[:, ::-1]
        exps.append((regressors, measurements[order:]))

    return exps


@pytest.fixture(scope="module")
def arx_measurements_with_noise(experiments: int = N_EXPS):
    """Generate synthetic data from an ARX process."""
    exps = []
    for _ in range(experiments):
        white_noise = np.random.normal(
            0, NOISE_MAGNITUDE, SAMPLES).reshape(-1, 1)
        random_exogenous_input = np.random.normal(
            0, INPUT_MAGNITUDE, SAMPLES).reshape(-1, 1)
        # Generate MA noise
        for sample in range(len(MA_COEFS), SAMPLES):
            for order, ma_coef in enumerate(MA_COEFS, start=1):
                white_noise[sample] += ma_coef * white_noise[sample - order]
        ma_noise = white_noise
        # Generate AR process with MA noise
        measurements = np.zeros(SAMPLES).reshape(-1, 1)
        for sample in range(len(AR_COEFS), SAMPLES):
            for order, ar_coef in enumerate(AR_COEFS, start=1):
                measurements[sample] += ar_coef * measurements[sample - order]
            measurements[sample] += (EX_COEFS[0] *
                                     random_exogenous_input[sample - 1])
            measurements[sample] += ma_noise[sample]

        order = len(AR_COEFS)
        regressors = np.hstack(
            [measurements[i:-order + i] for i in range(order)])
        regressors = np.hstack(
            [regressors, random_exogenous_input[(order - 1):-1]])

        # Reverse the regressors to match the order of the coefficients
        regressors = regressors[:, ::-1]
        exps.append((regressors, measurements[order:]))

    return exps


class TestLeastSquaresOptimizer:
    """Test the LeastSquaresOptimizer class."""

    def test_ols_linear(self, ar_measurements):
        """Test the Ordinary Least Squares method."""

        coefs = []
        for regressors, targets in ar_measurements:
            for _ in range(30):
                ols = LeastSquaresOptimizer('OLS')
                ols.fit(regressors, targets, inplace=True)
                coefs.append(ols.coefs)
        coefs = np.array(coefs).reshape(-1, len(AR_COEFS))
        print("OLS Linear Coefs: ", np.mean(coefs, axis=0))
        assert np.mean(coefs, axis=0) == pytest.approx(AR_COEFS, abs=0.01)

    def test_ols_extended_linear(self, arx_measurements_with_noise):
        """Test the Ordinary Least Squares method with noise."""
        coefs = []
        for phi, targets in arx_measurements_with_noise:
            ols = LeastSquaresOptimizer(method="OLS")
            ols.fit(phi, targets, inplace=True)
            coefs.append(ols.coefs)
        coefs = np.array(coefs).reshape(-1, len(AR_COEFS) + len(EX_COEFS))
        print("OLS Extended Linear Coefs: ", np.mean(coefs, axis=0))
        assert np.mean(coefs, axis=0) != pytest.approx(
            EX_COEFS + AR_COEFS, abs=0.01)

    def test_els_extended_linear(self, arx_measurements_with_noise):
        """Test the Extended Least Squares method with noise."""
        coefs_els = []
        for phi, targets in arx_measurements_with_noise:
            els = LeastSquaresOptimizer(method="ELS")
            els.fit(phi, targets, inplace=True, n_it=1000, criterion='theta',
                    ma_order=1)
            coefs_els.append(els.coefs.T)
        coefs_els = np.array(coefs_els).reshape(
            -1, len(AR_COEFS) + len(EX_COEFS))
        print("ELS Extended Linear Coefs: ", np.mean(coefs_els, axis=0))
        assert np.mean(coefs_els, axis=0) == pytest.approx(
            EX_COEFS + AR_COEFS, abs=0.01)

    def test_ols_arma(self, arma_measurements):
        """Test the Ordinary Least Squares method."""
        coefs = []
        for regressors, targets in arma_measurements:
            for _ in range(30):
                ols = LeastSquaresOptimizer('OLS')
                ols.fit(regressors, targets, inplace=True)
                coefs.append(ols.coefs)
        coefs = np.array(coefs).reshape(-1, len(AR_COEFS))
        print("OLS ARMA Coefs: ", np.mean(coefs, axis=0))
        assert np.mean(coefs, axis=0) != pytest.approx(AR_COEFS, abs=0.01)

    def test_els_arma(self, arma_measurements):
        """Test the Extended Least Squares method."""
        coefs_els = []
        for regressors, targets in arma_measurements:
            els = LeastSquaresOptimizer(method="ELS")
            els.fit(regressors, targets, inplace=True, n_it=1000,
                    criterion='theta', ma_order=1)
            coefs_els.append(els.coefs.T)
        coefs_els = np.array(coefs_els).reshape(-1, len(AR_COEFS))
        print("ELS ARMA Coefs: ", np.mean(coefs_els, axis=0))
        assert np.mean(coefs_els, axis=0) == pytest.approx(AR_COEFS, abs=0.01)
