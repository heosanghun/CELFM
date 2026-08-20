"""
src/model.py
Continuous Energy Landscape Neural Network Architecture.
- C^2 smooth scalar potential V_theta(q) with Softplus activations
- Confinement boundary potential V_valve(q)
- Analytical automatic differentiation for score nabla V_total and Hessian nabla^2 V_total
"""

import torch
import torch.nn as nn

class ContinuousPotentialNet(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, num_layers: int = 3,
                 box_min: float = -4.0, box_max: float = 4.0, kappa: float = 2.0):
        super().__init__()
        self.input_dim = input_dim
        self.box_min = box_min
        self.box_max = box_max
        self.kappa = kappa

        layers = []
        in_d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.Softplus(beta=1.0))
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, 1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Computes total continuous potential V_total(q) = V_theta(q) + V_valve(q).
        """
        v_theta = self.net(q)

        # Boundary soft-wall valve potential
        lower_viol = torch.relu(self.box_min - q)
        upper_viol = torch.relu(q - self.box_max)
        v_valve = self.kappa * torch.sum(lower_viol**2 + upper_viol**2, dim=-1, keepdim=True)

        return v_theta + v_valve

    def compute_gradient(self, q: torch.Tensor) -> torch.Tensor:
        """
        Computes analytical score vector nabla_q V_total(q).
        """
        q_clone = q.detach().requires_grad_(True)
        v = self.forward(q_clone)
        grad = torch.autograd.grad(
            outputs=v,
            inputs=q_clone,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True
        )[0]
        return grad

    def compute_hessian(self, q: torch.Tensor) -> torch.Tensor:
        """
        Computes analytical Hessian matrix nabla^2_q V_total(q).
        """
        q_single = q.squeeze(0).detach().requires_grad_(True)
        v = self.forward(q_single.unsqueeze(0))
        grad = torch.autograd.grad(v, q_single, create_graph=True)[0]
        hessian = []
        for i in range(self.input_dim):
            grad_i = grad[i]
            h_row = torch.autograd.grad(grad_i, q_single, retain_graph=True)[0]
            hessian.append(h_row)
        return torch.stack(hessian, dim=0)
