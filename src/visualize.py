"""
src/visualize.py
Energy Landscape Visualization Suite (2D/3D Contour and Attractor Mapping).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def plot_energy_landscape_2d(model_fn, save_path="results/energy_landscape_2d.png", resolution=100):
    """
    Generates 2D cross-section contour map of continuous energy potential V(q1, q2).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    q1_grid = np.linspace(-3.0, 3.0, resolution)
    q2_grid = np.linspace(-3.0, 3.0, resolution)
    Q1, Q2 = np.meshgrid(q1_grid, q2_grid)

    coords = np.stack([Q1.flatten(), Q2.flatten(), np.zeros(resolution*resolution), np.zeros(resolution*resolution)], axis=-1)
    t_coords = torch.tensor(coords, dtype=torch.float32)

    with torch.no_grad():
        V = model_fn.forward(t_coords).cpu().numpy().reshape(resolution, resolution)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(Q1, Q2, V, levels=30, cmap='viridis_r')
    ax.contour(Q1, Q2, V, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    fig.colorbar(cf, ax=ax, label=r'Potential Energy $V_{	ext{anc}}(q)$')
    ax.set_xlabel(r'Collective Coordinate $q_1$ (Market Projection)')
    ax.set_ylabel(r'Collective Coordinate $q_2$ (Return Dispersion)')
    ax.set_title('Continuous Energy Landscape')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
