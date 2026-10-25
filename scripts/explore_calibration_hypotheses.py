"""
Explore calibration hypotheses on validation set without touching test set or frozen artifacts.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from scripts.calibration_audit_and_baseline import load_data, stable_sigmoid, CLASSES
from src.calibration.metrics import binary_nll, brier_score, expected_calibration_error, binary_auroc

val_targets, val_logits = load_data(Path("outputs/evaluation/validation_predictions.csv"))

print("="*80)
print("EXPLORING CALIBRATION METHODS ON VALIDATION DATA (16,828 SAMPLES)")
print("="*80)

# 1. Temperature-only optimization under different objectives
# What is the minimum possible ECE achievable with pure temperature scaling T > 0?
print("\n--- 1. PURE TEMPERATURE SCALING: Best T for ECE vs Best T for NLL ---")
for i, c in enumerate(CLASSES):
    y = val_targets[:, i]
    z = val_logits[:, i]
    
    # Grid search for T that minimizes ECE15 on validation
    t_grid = np.linspace(0.1, 4.0, 391)
    eces = [expected_calibration_error(y, stable_sigmoid(z / t), n_bins=15) for t in t_grid]
    nlls = [binary_nll(y, stable_sigmoid(z / t)) for t in t_grid]
    
    best_t_ece_idx = np.argmin(eces)
    best_t_ece = t_grid[best_t_ece_idx]
    min_ece = eces[best_t_ece_idx]
    
    best_t_nll_idx = np.argmin(nlls)
    best_t_nll = t_grid[best_t_nll_idx]
    min_nll = nlls[best_t_nll_idx]
    
    raw_ece = expected_calibration_error(y, stable_sigmoid(z), n_bins=15)
    raw_nll = binary_nll(y, stable_sigmoid(z))
    
    print(f"{c:<18}: Raw ECE={raw_ece:.4f} | Min T-ECE={min_ece:.4f} at T={best_t_ece:.2f} | T_NLL={best_t_nll:.2f} (ECE={eces[best_t_nll_idx]:.4f}, NLL={min_nll:.4f})")

# 2. PLATT SCALING (a * z + b)
print("\n--- 2. PLATT SCALING: Fitting a * z + b via NLL ---")
platt_params = {}
platt_val_metrics = {"nll": [], "brier": [], "ece10": [], "ece15": [], "ece20": [], "auroc": []}

for i, c in enumerate(CLASSES):
    y = val_targets[:, i]
    z = val_logits[:, i]
    
    # Fit logistic regression on validation logits: p = sigmoid(a * z + b)
    # Using PyTorch LBFGS with a > 0: a = exp(log_a)
    log_a = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    
    z_t = torch.tensor(z, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    
    optimizer = torch.optim.LBFGS([log_a, b], lr=0.1, max_iter=150, line_search_fn="strong_wolfe")
    
    def closure():
        optimizer.zero_grad()
        a_val = torch.exp(log_a)
        scaled_logits = a_val * z_t + b
        loss = F.binary_cross_entropy_with_logits(scaled_logits, y_t)
        loss.backward()
        return loss
        
    optimizer.step(closure)
    
    a_fitted = float(torch.exp(log_a).item())
    b_fitted = float(b.item())
    platt_params[c] = {"a": a_fitted, "b": b_fitted}
    
    cal_logits = a_fitted * z + b_fitted
    p_cal = stable_sigmoid(cal_logits)
    
    nll = binary_nll(y, p_cal)
    brier = brier_score(y, p_cal)
    ece10 = expected_calibration_error(y, p_cal, n_bins=10)
    ece15 = expected_calibration_error(y, p_cal, n_bins=15)
    ece20 = expected_calibration_error(y, p_cal, n_bins=20)
    auc = binary_auroc(y, p_cal)
    
    raw_ece15 = expected_calibration_error(y, stable_sigmoid(z), n_bins=15)
    raw_nll = binary_nll(y, stable_sigmoid(z))
    raw_brier = brier_score(y, stable_sigmoid(z))
    
    platt_val_metrics["nll"].append(nll)
    platt_val_metrics["brier"].append(brier)
    platt_val_metrics["ece10"].append(ece10)
    platt_val_metrics["ece15"].append(ece15)
    platt_val_metrics["ece20"].append(ece20)
    platt_val_metrics["auroc"].append(auc)
    
    print(f"{c:<18}: a={a_fitted:.4f}, b={b_fitted:+.4f} | ECE15: {raw_ece15:.4f} -> {ece15:.4f} ({ece15 - raw_ece15:+.4f}) | NLL: {raw_nll:.4f} -> {nll:.4f} | Brier: {raw_brier:.4f} -> {brier:.4f}")

print("\nPLATT SCALING MACRO VALIDATION:")
print(f"  Macro ECE10: {np.mean(platt_val_metrics['ece10']):.5f}")
print(f"  Macro ECE15: {np.mean(platt_val_metrics['ece15']):.5f}")
print(f"  Macro ECE20: {np.mean(platt_val_metrics['ece20']):.5f}")
print(f"  Macro NLL:   {np.mean(platt_val_metrics['nll']):.5f}")
print(f"  Macro Brier: {np.mean(platt_val_metrics['brier']):.5f}")
print(f"  Macro AUROC: {np.mean(platt_val_metrics['auroc']):.5f}")

