import time
import os
import numpy as np


def sample_bivariate_normal(mu, sigma, rho, num_samples=1):
    """
    Given Bivariate normal distribution parameters (mu, sigma, rho) draws samples.

    Args:
        mu: 2D array.
        sigma: 2D array.
        rho: Scalar.

    Returns:
        2D samples.
    """
    s1, s2 = sigma
    cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
    return np.random.multivariate_normal(mu, cov, num_samples)