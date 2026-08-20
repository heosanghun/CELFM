"""
src/build_panel.py
End-to-End Orchestrator for Energy Landscape Reconstruction & Analysis Pipeline.
"""

import os
import sys
import json
import torch
import pandas as pd

from src.data_loader import download_or_load_data, compute_log_returns, get_train_test_split
from src.eval_gates import evaluate_all_gates
from src.visualize import plot_energy_landscape_2d

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Starting CELFM Pipeline on device: {device}")

    prices = download_or_load_data()
    returns = compute_log_returns(prices)
    train_rets, test_rets = get_train_test_split(returns)

    print(f"[*] Train panel: {train_rets.shape}, Test panel: {test_rets.shape}")

    results = evaluate_all_gates(train_rets, test_rets, device=device)
    os.makedirs("results", exist_ok=True)
    with open("results/summary_table.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[*] Pipeline completed successfully. Results saved to results/summary_table.json.")

if __name__ == '__main__':
    main()
