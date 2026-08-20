# Continuous Energy Landscape Learning (CELFM)

Official Anonymous Code Repository for ICLR 2027 Submission under Double-Blind Review.

- **Anonymous Repository Link**: [https://anonymous.4open.science/r/CELFM-Energy/](https://anonymous.4open.science/r/CELFM-Energy/)

---

## Overview

This repository provides the reference implementation of **Continuous Energy Landscape Learning (CELFM)**, a framework for non-equilibrium multistable energy landscape reconstruction from dynamical trajectories using Denoising Score Matching, Damped Velocity Verlet relaxation, and statistical Null-Anchoring.

---

## Repository Structure

```
├── main.py                     # Primary pipeline execution entrypoint
├── requirements.txt            # Environment dependencies
├── README.md                   # Repository documentation and anonymous portal
├── .gitignore                  # Git tracking rules
├── src/                        # Core algorithmic modules
│   ├── data_loader.py          # Trajectory and empirical coordinate loaders
│   ├── features.py             # Collective variable (CV) coordinate projections
│   ├── model.py                # C^2 neural potential network architecture
│   ├── surrogates.py           # Statistical null surrogate generators (N1, N2, N3)
│   ├── trainer.py              # Denoising Score Matching & Null-Anchoring engine
│   ├── relax.py                # Damped Velocity Verlet relaxation & attractor clustering
│   ├── discrete_baseline.py    # Pairwise Maximum Entropy Ising baseline
│   ├── eval_gates.py           # Verification gate benchmarks
│   ├── eval_f1_null_control.py # Negative control & falsification test suite
│   ├── visualize.py            # 2D/3D energy surface plotting routines
│   └── build_panel.py          # End-to-end pipeline orchestrator
├── results/                    # Generated figures and evaluation metrics
└── paperA_reconstruction/      # Manuscript sources and reproduction assets
    ├── faithful_copy_EQUIPHASE_ICLR2027.tex
    ├── EQUIPHASE_ICLR2027_WITH_ED_ADDENDUM.tex
    ├── figures/                # High-resolution manuscript figure assets
    └── normalizer.py           # Text normalization and comparison utilities
```

---

## Installation

```bash
git clone https://anonymous.4open.science/r/CELFM-Energy/
cd CELFM-Energy
pip install -r requirements.txt
```

---

## Quickstart & Reproducibility

To run the continuous energy landscape learning pipeline:

```bash
python main.py
```

To run the negative control falsification suite:

```bash
python -m src.eval_f1_null_control
```

---

## License

This project is released under the MIT License for academic research purposes.
