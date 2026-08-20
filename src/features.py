"""
src/features.py
Computation of macro collective variables (CVs) q in R^4.
- q_1: Market eigenvector projection
- q_2: Cross-sectional return dispersion
- q_3: Systemic spectral concentration (lambda_1 / N)
- q_4: Realized log market volatility
"""

import numpy as np
import pandas as pd
from scipy.linalg import eigh

def compute_collective_variables(returns: pd.DataFrame, window_size: int = 60) -> pd.DataFrame:
    """
    Computes collective coordinate trajectory q(t) = [q1, q2, q3, q4] over rolling empirical covariance windows.
    """
    T, N = returns.shape
    q1_list, q2_list, q3_list, q4_list = [], [], [], []
    valid_dates = []

    mkt_index_ret = returns.mean(axis=1)

    for i in range(window_size, T):
        window_rets = returns.iloc[i - window_size:i].values
        date = returns.index[i]

        cov = np.cov(window_rets, rowvar=False)
        w, v = eigh(cov)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        v = v[:, idx]

        v1 = v[:, 0]
        if np.mean(v1) < 0:
            v1 = -v1

        today_ret = returns.iloc[i].values

        q1 = np.dot(today_ret, v1)
        q2 = np.std(today_ret)
        q3 = w[0] / np.sum(w) if np.sum(w) > 0 else 0.0
        q4 = np.log(np.std(mkt_index_ret.iloc[i - window_size:i]) + 1e-8)

        q1_list.append(q1)
        q2_list.append(q2)
        q3_list.append(q3)
        q4_list.append(q4)
        valid_dates.append(date)

    df_cv = pd.DataFrame({
        'q1': q1_list,
        'q2': q2_list,
        'q3': q3_list,
        'q4': q4_list
    }, index=valid_dates)

    return df_cv

def normalize_collective_variables(df_cv: pd.DataFrame, ref_mean=None, ref_std=None):
    """
    Standardizes collective variables using reference empirical moments.
    """
    if ref_mean is None:
        ref_mean = df_cv.mean()
    if ref_std is None:
        ref_std = df_cv.std().replace(0, 1.0)

    norm_cv = (df_cv - ref_mean) / ref_std
    return norm_cv, ref_mean, ref_std
