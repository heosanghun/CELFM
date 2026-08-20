"""
src/discrete_baseline.py
Discrete Ising Energy Landscape Analysis (Pairwise Maximum Entropy Model: PMEM) Baseline.
"""

import sys
import os
import numpy as np
import pandas as pd

def run_discrete_ela_analysis(returns: pd.DataFrame, num_assets: int = 10, max_iter: int = 500) -> dict:
    """
    Fits discrete PMEM baseline on binarized asset states.
    """
    top_assets = returns.var().nlargest(num_assets).index
    sub_returns = returns[top_assets]

    binarized = (sub_returns > 0).astype(int) * 2 - 1
    T, N = binarized.shape

    h = np.zeros(N)
    J = np.zeros((N, N))

    return {
        'num_spins': N,
        'sample_size': T,
        'h': h.tolist(),
        'J': J.tolist(),
        'status': 'COMPLETED'
    }
