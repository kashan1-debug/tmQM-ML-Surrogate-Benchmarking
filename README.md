# Benchmarking Machine Learning Models for HOMO–LUMO Gap Prediction in the tmQM Complex Dataset

This repository contains the dataset preprocessing scripts, model training pipelines, and figure-generation workflows for benchmarking non-spatial machine learning surrogate models on the tmQM (Transition Metal Quantum Mechanics) dataset (N = 195,206).

## Project Overview

Density Functional Theory (DFT) calculations for transition metal complexes carry substantial computational overhead. Machine learning surrogate architectures offer a rapid alternative for property prediction.

In this benchmark study, we evaluate **Linear Regression**, **XGBoost**, and **Random Forest** on the same leak-free global scalar quantum chemical descriptors to establish the empirical upper bound of non-spatial tabular features for predicting the HOMO–LUMO orbital gap.

## Benchmark Results

| Model Architecture | Test R² | Test RMSE (eV) |
|---|---|---|
| Linear Regression Baseline | 0.085 | 0.032 |
| XGBoost Regressor | 0.294 | 0.028 |
| **Random Forest Regressor** | **0.743** | **0.017** |

All three models were trained on the same feature set: global scalar descriptors from tmQM, with all frontier orbital energies and derived conceptual DFT parameters purged to prevent target leakage. Random Forest used `n_estimators=100`, `random_state=42`. XGBoost used `n_estimators=100`, `learning_rate=0.1`, `random_state=42`.

### Key Findings

- **Dominant descriptors:** Metal partial charge (`Metal_q`, importance ≈ 0.27) and total electronic energy (`Electronic_E`, importance ≈ 0.21).
- **Target leakage prevention:** HOMO and LUMO energies, electronegativity, hardness, and softness were excluded from all training features.
- **Spatial limitations:** Global scalar descriptors lack ligand-field symmetry and coordination-geometry information, establishing an empirical upper bound at R² ≈ 0.74 and motivating 3D graph representations.

## Repository Structure
