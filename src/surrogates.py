"""
src/surrogates.py
Three Statistical Null Surrogate Generators for Energy Landscape Calibration.
- N1: i.i.d. Gaussian Null (Preserves static covariance, destroys temporal dynamics)
- N2: Phase-Randomized Surrogate (Preserves linear spectral power, destroys non-linear coupling)
- N3: Stationary Block Bootstrap (Preserves short-term autocorrelation, disrupts regime clustering)
"""

import numpy as np
import pandas as pd

def generate_n1_surrogate(returns: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    N1: i.i.d. Gaussian Null
    """
    rng = np.random.default_rng(seed)
    T, N = returns.shape
    mu = returns.mean().values
    cov = returns.cov().values

    synthetic_rets = rng.multivariate_normal(mu, cov, size=T)
    return pd.DataFrame(synthetic_rets, index=returns.index, columns=returns.columns)

def generate_n2_surrogate(returns: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    N2: Phase-Randomized Surrogate
    """
    rng = np.random.default_rng(seed)
    data = returns.values
    T, N = data.shape

    fft_coeffs = np.fft.rfft(data, axis=0)
    n_freq = fft_coeffs.shape[0]

    phases = rng.uniform(0, 2 * np.pi, size=(n_freq, 1))
    phases[0] = 0
    if T % 2 == 0:
        phases[-1] = 0

    fft_randomized = fft_coeffs * np.exp(1j * phases)
    surrogate_data = np.fft.irfft(fft_randomized, n=T, axis=0)

    return pd.DataFrame(surrogate_data, index=returns.index, columns=returns.columns)

def generate_n3_surrogate(returns: pd.DataFrame, block_length: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    N3: Stationary Block Bootstrap Surrogate
    """
    rng = np.random.default_rng(seed)
    T, N = returns.shape
    num_blocks = int(np.ceil(T / block_length))

    sampled_blocks = []
    for _ in range(num_blocks):
        start_idx = rng.integers(0, T - block_length + 1)
        sampled_blocks.append(returns.iloc[start_idx:start_idx + block_length].values)

    surrogate_data = np.concatenate(sampled_blocks, axis=0)[:T]
    return pd.DataFrame(surrogate_data, index=returns.index, columns=returns.columns)
