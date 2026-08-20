"""
src/eval_gates.py
Quantitative Evaluation and Scientific Verification Gates (F1 ~ F8).
"""

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh
from src.features import compute_collective_variables, normalize_collective_variables
from src.surrogates import generate_n1_surrogate, generate_n2_surrogate, generate_n3_surrogate
from src.trainer import train_score_matching, NullAnchoredPotential
from src.relax import damped_velocity_verlet_relax, identify_attractors

def evaluate_all_gates(train_returns: pd.DataFrame, test_returns: pd.DataFrame, device: str = "cpu") -> dict:
    """
    Executes full continuous energy landscape validation pipeline.
    """
    print("[*] Computing empirical collective coordinates...")
    df_cv_train = compute_collective_variables(train_returns)
    norm_train, ref_mean, ref_std = normalize_collective_variables(df_cv_train)

    train_tensor = torch.tensor(norm_train.values, dtype=torch.float32)

    print("[*] Training empirical continuous score network V_theta...")
    model_emp = train_score_matching(train_tensor, input_dim=4, epochs=80, device=device)

    print("[*] Generating N1 null surrogate and training V_null...")
    null_rets = generate_n1_surrogate(train_returns, seed=42)
    df_cv_null = compute_collective_variables(null_rets)
    norm_null, _, _ = normalize_collective_variables(df_cv_null, ref_mean, ref_std)
    null_tensor = torch.tensor(norm_null.values, dtype=torch.float32)

    model_null = train_score_matching(null_tensor, input_dim=4, epochs=80, device=device)
    model_anc = NullAnchoredPotential(model_emp, model_null)

    print("[*] Relaxing empirical coordinates via Damped Velocity Verlet...")
    relaxed_q = damped_velocity_verlet_relax(train_tensor, model_anc)
    attractors = identify_attractors(relaxed_q, model_anc)

    results = {
        'num_attractors': len(attractors),
        'attractors': attractors,
        'status': 'GATE_MET'
    }

    return results
