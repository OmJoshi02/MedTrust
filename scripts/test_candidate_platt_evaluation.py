"""
Evaluate Candidate Platt Scaling on untouched Test Set (N = 16,805).
Fitted on full Validation Set (N = 16,828) only.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import torch
import torch.nn.functional as F

from scripts.calibration_audit_and_baseline import load_data, stable_sigmoid, CLASSES
from src.calibration.metrics import binary_nll, brier_score, expected_calibration_error, binary_auroc

# 1. Load validation data (used strictly for fitting)
val_targets, val_logits = load_data(Path("outputs/evaluation/validation_predictions.csv"))

# 2. Load test data (untouched holdout)
test_targets, test_logits = load_data(Path("outputs/evaluation/test_predictions.csv"))

# 3. Fit Platt scaling parameters on validation data ONLY
platt_params = {}
for i, c in enumerate(CLASSES):
    y_tr = val_targets[:, i]
    z_tr = val_logits[:, i]
    
    log_a = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    z_t = torch.tensor(z_tr, dtype=torch.float64)
    y_t = torch.tensor(y_tr, dtype=torch.float64)
    
    optimizer = torch.optim.LBFGS([log_a, b], lr=0.1, max_iter=150, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(torch.exp(log_a) * z_t + b, y_t)
        loss.backward()
        return loss
    optimizer.step(closure)
    
    a_fitted = float(torch.exp(log_a).item())
    b_fitted = float(b.item())
    platt_params[c] = {
        "a": a_fitted,
        "b": b_fitted,
        "temperature": 1.0 / a_fitted,
        "intercept": b_fitted,
    }

# Load current frozen temperature baseline for comparison
with open("outputs/calibration/temperature.json", "r") as f:
    orig_temp_json = json.load(f)
orig_temps = np.array([float(orig_temp_json["temperatures"][c]) for c in CLASSES])

print("="*90)
print("FINAL TEST EVALUATION: UNCALIBRATED vs FROZEN TEMP vs CANDIDATE PLATT (N = 16,805)")
print("="*90)
print(f"{'Disease':<18} | {'N_pos':<5} | {'N_neg':<5} | {'Prev%':<5} | {'Raw ECE15':<9} | {'Temp ECE15':<9} | {'Platt ECE15':<10} | {'Platt NLL':<9} | {'Platt Brier':<10}")
print("-" * 105)

test_metrics = {
    "raw": {"nll": [], "brier": [], "ece10": [], "ece15": [], "ece20": [], "auroc": []},
    "temp": {"nll": [], "brier": [], "ece10": [], "ece15": [], "ece20": [], "auroc": []},
    "platt": {"nll": [], "brier": [], "ece10": [], "ece15": [], "ece20": [], "auroc": []},
}

for i, c in enumerate(CLASSES):
    y = test_targets[:, i]
    z = test_logits[:, i]
    n_pos = int(np.sum(y))
    n_neg = int(len(y) - n_pos)
    prev = (n_pos / len(y)) * 100
    
    # 1. Raw
    p_raw = stable_sigmoid(z)
    test_metrics["raw"]["nll"].append(binary_nll(y, p_raw))
    test_metrics["raw"]["brier"].append(brier_score(y, p_raw))
    test_metrics["raw"]["ece10"].append(expected_calibration_error(y, p_raw, 10))
    test_metrics["raw"]["ece15"].append(expected_calibration_error(y, p_raw, 15))
    test_metrics["raw"]["ece20"].append(expected_calibration_error(y, p_raw, 20))
    test_metrics["raw"]["auroc"].append(binary_auroc(y, p_raw))
    
    # 2. Frozen Temp
    p_temp = stable_sigmoid(z / orig_temps[i])
    test_metrics["temp"]["nll"].append(binary_nll(y, p_temp))
    test_metrics["temp"]["brier"].append(brier_score(y, p_temp))
    test_metrics["temp"]["ece10"].append(expected_calibration_error(y, p_temp, 10))
    test_metrics["temp"]["ece15"].append(expected_calibration_error(y, p_temp, 15))
    test_metrics["temp"]["ece20"].append(expected_calibration_error(y, p_temp, 20))
    test_metrics["temp"]["auroc"].append(binary_auroc(y, p_temp))
    
    # 3. Candidate Platt
    a_c = platt_params[c]["a"]
    b_c = platt_params[c]["b"]
    p_platt = stable_sigmoid(a_c * z + b_c)
    test_metrics["platt"]["nll"].append(binary_nll(y, p_platt))
    test_metrics["platt"]["brier"].append(brier_score(y, p_platt))
    test_metrics["platt"]["ece10"].append(expected_calibration_error(y, p_platt, 10))
    test_metrics["platt"]["ece15"].append(expected_calibration_error(y, p_platt, 15))
    test_metrics["platt"]["ece20"].append(expected_calibration_error(y, p_platt, 20))
    test_metrics["platt"]["auroc"].append(binary_auroc(y, p_platt))
    
    r_ece = test_metrics["raw"]["ece15"][-1]
    t_ece = test_metrics["temp"]["ece15"][-1]
    p_ece = test_metrics["platt"]["ece15"][-1]
    p_nll = test_metrics["platt"]["nll"][-1]
    p_brier = test_metrics["platt"]["brier"][-1]
    
    print(f"{c:<18} | {n_pos:<5} | {n_neg:<5} | {prev:<4.1f}% | {r_ece:<9.4f} | {t_ece:<9.4f} | {p_ece:<10.4f} | {p_nll:<9.4f} | {p_brier:<10.4f}")

print("-" * 105)
print("\nMACRO TEST PERFORMANCE SUMMARY (N = 16,805):")
for k in ["ece15", "ece10", "ece20", "nll", "brier", "auroc"]:
    m_r = np.mean(test_metrics["raw"][k])
    m_t = np.mean(test_metrics["temp"][k])
    m_p = np.mean(test_metrics["platt"][k])
    d_vs_temp = m_p - m_t
    print(f"{k.upper():<8} | Raw: {m_r:.7f} | Temp (Frozen): {m_t:.7f} | Platt (Candidate): {m_p:.7f} | Change vs Temp: {d_vs_temp:+.7f}")

