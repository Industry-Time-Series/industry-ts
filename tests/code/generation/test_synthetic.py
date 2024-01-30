"""
Test synthetic data generation functions using pytest.
"""
import numpy as np

from src.industryts.generation.synthetic import ar_process


class TestARProcess():
    """Test the AR process generator."""
    def test_output_length(self):
        """Test that the output length is equal to the number of samples.
        """
        output = ar_process([0.1], samples=500, noise=0)
        assert len(output) == 500

    def test_stable_system(self):
        """Test that all samples are finite if the system is stable.
        """
        output = ar_process([0.2], samples=5000, noise=0)
        print(max(output))
        assert all(np.isfinite(output))

    def test_unstable_system(self):
        """Test that some samples are infinite if the system is unstable.
        """
        output = ar_process([1.2], samples=5000, noise=0)
        assert not all(np.isfinite(output))

    def test_scalar_coef(self):
        """Test that the function can handle a scalar coefficient.
        """
        output = ar_process(0.2, samples=100, noise=0)
        assert len(output) == 100


