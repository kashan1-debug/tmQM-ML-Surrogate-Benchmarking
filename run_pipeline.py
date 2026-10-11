#!/usr/bin/env python3
"""
Benchmarking ML models for HOMO-LUMO gap prediction on the tmQM 2024 dataset.

This pipeline is intentionally strict:
  - It uses only the 2024 tmQM release (108,541 unique complexes).
  - It fails loudly if the tmQM repository cannot be cloned.
  - It never substitutes synthetic or fallback data.
  - It excludes all columns that could leak target information.

Outputs:
  results.csv                    - benchmark R2 and RMSE for each model
  tmqm_correlation_matrix.png    - Pearson correlation heatmap
  feature_importance.png         - Random Forest descriptor ranking (sorted)
  real_parity_plot.png           - Random Forest parity plot (DFT vs. ML)
"""

import os
import sys
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Run: pip install xgboost")


# ======================================================================
# Step 1 — Clone and load the tmQM 2024 release
# ======================================================================
REPO_DIR = "tmqm"
CSV_PATH = os.path.join(REPO_DIR, "tmQM", "tmQM_y.csv")

if not os.path.exists(REPO_DIR):
    print("Cloning tmQM repository ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/uiocompcat/tmqm.git", REPO_DIR],
    )
    if result.returncode != 0:
        sys.exit("ERROR: git clone failed. Refusing to continue without real data.")

if not os.path.exists(CSV_PATH):
    sys.exit(f"ERROR: expected data file not found at {CSV_PATH}.")

df = pd.read_csv(CSV_PATH, sep=";")
print(f"Loaded tmQM 2024 release: {len(df):,} rows, {df.shape[1]} columns")


# ======================================================================
# Step 2 — Explicit feature/target split (no leakage)
# ======================================================================
FEATURES = ["Electronic_E", "Dispersion_E", "Dipole_M", "Metal_q", "Polarizability"]
TARGET   = "HL_Gap"

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    sys.exit(f"ERROR: missing expected columns: {missing}")

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[TARGET], errors="coerce")

mask = X.notna().all(axis=1) & y.notna()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"Target:                {TARGET}")
print(f"Final dataset: X={X.shape}  y={y.shape}")


# ======================================================================
# Step 3 — Train models
# ======================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train rows: {len(X_train):,}   Test rows: {len(X_test):,}")

models = {
    "Linear Regression Baseline": LinearRegression(),
    "Random Forest Regressor":    RandomForestRegressor(
                                      n_estimators=100,
                                      random_state=42,
                                      n_jobs=-1,
                                  ),
}
if HAS_XGB:
    models["XGBoost Regressor"] = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )

results = []
preds = {}

for name, model in models.items():
    print(f"Training: {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    preds[name] = y_pred
    results.append({
        "Model": name,
        "R2":    r2_score(y_test, y_pred),
        "RMSE":  root_mean_squared_error(y_test, y_pred),
    })

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)

print()
print("=" * 60)
print("BENCHMARK PERFORMANCE RESULTS")
print("=" * 60)
print(results_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
print()

results_df.to_csv("results.csv", index=False)
print("Saved: results.csv")


# ======================================================================
# Step 4 — Figures
# ======================================================================
sns.set_theme(style="ticks")
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.weight":       "bold",
    "axes.labelweight":  "bold",
    "axes.titleweight":  "bold",
    "axes.edgecolor":    "#000000",
    "axes.linewidth":    1.0,
})

# ---- 4a. Correlation matrix -----------------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(
    pd.concat([X, y], axis=1).corr(),
    annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
)
plt.title("tmQM Feature Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.savefig("tmqm_correlation_matrix.png", dpi=300)
plt.close()
print("Saved: tmqm_correlation_matrix.png")

# ---- 4b. Feature importance (sorted descending) ----------------------
rf = models["Random Forest Regressor"]
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(8, 4))
plt.barh(importances.index, importances.values, color="teal")
plt.xlabel("Feature Importance Weight")
plt.title("Random Forest Descriptor Ranking")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()
print("Saved: feature_importance.png")

# ---- 4c. Random Forest parity plot ----------------------------------
y_pred_rf = preds["Random Forest Regressor"]
r2_rf = r2_score(y_test, y_pred_rf)

fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=300)
ax.scatter(
    y_test, y_pred_rf,
    color="#1f77b4", alpha=0.50, s=16,
    edgecolors="black", linewidths=0.2, rasterized=True,
)
lo = min(y_test.min(), y_pred_rf.min())
hi = max(y_test.max(), y_pred_rf.max())
ax.plot([lo, hi], [lo, hi], color="#d62728", linestyle="--",
        linewidth=1.8, label=r"$\mathbf{Ideal\ Fit}$", zorder=5)
ax.text(
    0.05, 0.90,
    rf"$\mathbf{{R^2 = {r2_rf:.3f}}}$",
    transform=ax.transAxes, fontsize=9, fontweight="bold",
    verticalalignment="top",
    bbox=dict(boxstyle="square,pad=0.3", facecolor="white",
              edgecolor="#000000", linewidth=0.8, alpha=0.95),
)
ax.set_xlabel("DFT HOMO-LUMO Gap (eV)", fontsize=9, fontweight="bold", labelpad=6)
ax.set_ylabel("RF Predicted Gap (eV)", fontsize=9, fontweight="bold", labelpad=6)
ax.set_title("Random Forest Parity", fontsize=10, fontweight="bold", pad=8)
ax.tick_params(axis="both", which="major", labelsize=8, width=1.0, length=4)
ax.grid(True, linestyle=":", alpha=0.5, color="#888888")
ax.legend(loc="lower right", frameon=True, facecolor="white",
          framealpha=0.9, prop={"weight": "bold", "size": 8})
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("#000000")
plt.tight_layout()
plt.savefig("real_parity_plot.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: real_parity_plot.png")

print()
print("Pipeline execution complete.")
