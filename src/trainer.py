"""
src/trainer.py
Denoising Score Matching (DSM) Training and Null-Anchoring Protocol.
- DSM objective for empirical potential V_theta
- DSM objective for null surrogate V_null
- Null-anchored relative potential V_anc = V_theta - V_null
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from src.model import ContinuousPotentialNet

def train_score_matching(
    data_tensor: torch.Tensor,
    input_dim: int = 4,
    sigma: float = 0.25,
    hidden_dim: int = 64,
    num_layers: int = 3,
    lr: float = 1e-3,
    epochs: int = 150,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    device: str = "cpu"
) -> ContinuousPotentialNet:
    """
    Trains ContinuousPotentialNet via Denoising Score Matching.
    """
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = ContinuousPotentialNet(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            q_clean = batch[0].to(device)
            noise = torch.randn_like(q_clean) * sigma
            q_noisy = (q_clean + noise).requires_grad_(True)

            score = -model.compute_gradient(q_noisy)
            target_score = -(q_noisy - q_clean) / (sigma ** 2)

            loss = 0.5 * torch.mean(torch.sum((score - target_score) ** 2, dim=-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * len(q_clean)

        scheduler.step()

    model.eval()
    return model

class NullAnchoredPotential:
    def __init__(self, model_emp: ContinuousPotentialNet, model_null: ContinuousPotentialNet):
        self.model_emp = model_emp
        self.model_null = model_null

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        v_emp = self.model_emp(q)
        v_null = self.model_null(q)
        v_anc = v_emp - v_null
        return v_anc - torch.min(v_anc)

    def compute_gradient(self, q: torch.Tensor) -> torch.Tensor:
        return self.model_emp.compute_gradient(q) - self.model_null.compute_gradient(q)

    def compute_hessian(self, q: torch.Tensor) -> torch.Tensor:
        return self.model_emp.compute_hessian(q) - self.model_null.compute_hessian(q)
