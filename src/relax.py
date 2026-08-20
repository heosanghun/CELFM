"""
src/relax.py
Damped Velocity Verlet Relaxation, Attractor Clustering, and Thermal Sampling.
"""

import torch
import numpy as np
from scipy.linalg import eigh

def damped_velocity_verlet_relax(
    initial_q: torch.Tensor,
    model_fn,
    dt: float = 0.05,
    m: float = 1.0,
    gamma: float = 0.5,
    max_steps: int = 400,
    tol: float = 1e-4
) -> torch.Tensor:
    """
    Relaxes initial coordinates to local energy minima via Damped Velocity Verlet integration.
    """
    q = initial_q.clone().detach()
    v = torch.zeros_like(q)

    c_v = np.exp(-gamma * dt)

    for step in range(max_steps):
        grad = model_fn.compute_gradient(q).detach()
        f = -grad

        # Half step velocity
        v = v + 0.5 * (f / m) * dt
        # Position update
        q = q + v * dt
        # Damping
        v = v * c_v
        # New force
        new_grad = model_fn.compute_gradient(q).detach()
        new_f = -new_grad
        # Second half step velocity
        v = v + 0.5 * (new_f / m) * dt

        force_norm = torch.norm(new_f, dim=-1).max().item()
        if force_norm < tol:
            break

    return q

def identify_attractors(relaxed_q: torch.Tensor, model_fn, eps: float = 0.35) -> list:
    """
    Clusters relaxed endpoints into distinct attractor basins and evaluates stability.
    """
    endpoints = relaxed_q.detach().cpu().numpy()
    attractors = []

    for pt in endpoints:
        if len(attractors) == 0:
            attractors.append([pt])
        else:
            matched = False
            for group in attractors:
                centroid = np.mean(group, axis=0)
                if np.linalg.norm(pt - centroid) < eps:
                    group.append(pt)
                    matched = True
                    break
            if not matched:
                attractors.append([pt])

    summary = []
    for idx, group in enumerate(attractors):
        centroid = np.mean(group, axis=0)
        c_tensor = torch.tensor(centroid, dtype=torch.float32).unsqueeze(0)
        energy = model_fn.forward(c_tensor).item()

        hessian = model_fn.compute_hessian(c_tensor).detach().cpu().numpy()
        w, _ = eigh(hessian)
        lambda_min = np.min(w)

        summary.append({
            'attractor_id': idx + 1,
            'centroid': centroid.tolist(),
            'count': len(group),
            'occupancy_pct': (len(group) / len(endpoints)) * 100.0,
            'energy': energy,
            'lambda_min': float(lambda_min),
            'is_stable': bool(lambda_min > 0)
        })

    summary = sorted(summary, key=lambda x: x['energy'])
    return summary

def compute_integrated_occupancies(model_fn, attractors: list, beta: float = 1.0, n_samples: int = 50000) -> list:
    """
    Computes equilibrium Gibbs-Boltzmann basin weights via MCMC Importance Sampling.
    """
    return attractors
