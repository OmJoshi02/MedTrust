"""
Leakage-safe patient-disjoint cross-validation study on validation data.
Validates generalization of calibration methods out-of-sample across patients.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.model_selection import KFold

from scripts.calibration_audit_and_baseline import load_data, stable_sigmoid, CLASSES, CLASS_TO_CSV
from src.calibration.metrics import binary_nll, brier_score, expected_calibration_error, binary_auroc

# 1. Load validation data and extract patient IDs
val_csv = Path("outputs/evaluation/validation_predictions.csv")
targets_list, logits_list, filenames = [], [], []

with open(val_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        filenames.append(row["filename"])
        targets_list.append([float(row[f"target_{CLASS_TO_CSV[c]}"]) for c in CLASSES])
        logits_list.append([float(row[f"logit_{CLASS_TO_CSV[c]}"]) for c in CLASSES])

targets = np.array(targets_list, dtype=np.float64)
logits = np.array(logits_list, dtype=np.float64)

# Group image indices by patient ID (e.g. "00000001_000.png" -> "00000001")
patient_to_indices = defaultdict(list)
for idx, fn in enumerate(filenames):
    pid = fn.split("_")[0]
    patient_to_indices[pid].append(idx)

unique_patients = np.array(sorted(patient_to_indices.keys()))
n_patients = len(unique_patients)
print(f"Validation cohort: {len(filenames)} images across {n_patients} unique patients.")

# 2. Patient-disjoint 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def fit_temperature(z_train, y_train):
    """Fit T > 0 via NLL minimization."""
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    z_t = torch.tensor(z_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z_t / torch.exp(log_t), y_t)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(torch.exp(log_t).item())

def fit_platt(z_train, y_train):
    """Fit a > 0, b via NLL minimization."""
    log_a = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    z_t = torch.tensor(z_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    optimizer = torch.optim.LBFGS([log_a, b], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(torch.exp(log_a) * z_t + b, y_t)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(torch.exp(log_a).item()), float(b.item())

# Out-of-fold predictions
oof_raw_probs = np.zeros_like(targets)
oof_temp_probs = np.zeros_like(targets)
oof_platt_probs = np.zeros_like(targets)

fold_results = []

for fold, (train_pids_idx, val_pids_idx) in enumerate(kf.split(unique_patients)):
    train_pids = set(unique_patients[train_pids_idx])
    val_pids = set(unique_patients[val_pids_idx])
    
    # Verify patient disjointness
    assert len(train_pids.intersection(val_pids)) == 0, "Patient leakage detected!"
    
    train_idx = [i for pid in train_pids for i in patient_to_indices[pid]]
    val_idx = [i for pid in val_pids for i in patient_to_indices[pid]]
    
    y_tr, z_tr = targets[train_idx], logits[train_idx]
    y_val, z_val = targets[val_idx], logits[val_idx]
    
    for c_idx in range(len(CLASSES)):
        # Raw
        oof_raw_probs[val_idx, c_idx] = stable_sigmoid(z_val[:, c_idx])
        
        # Temp scaling
        t_val = fit_temperature(z_tr[:, c_idx], y_tr[:, c_idx])
        oof_temp_probs[val_idx, c_idx] = stable_sigmoid(z_val[:, c_idx] / t_val)
        
        # Platt scaling
        a_val, b_val = fit_platt(z_tr[:, c_idx], y_tr[:, c_idx])
        oof_platt_probs[val_idx, c_idx] = stable_sigmoid(a_val * z_val[:, c_idx] + b_val)
        
    print(f"Completed Fold {fold+1}/5 (Train: {len(train_idx)} imgs, Val: {len(val_idx)} imgs)")

print("\n" + "="*80)
print("OUT-OF-FOLD EVALUATION ON VALIDATION COHORT (LEAKAGE-SAFE)")
print("="*80)

def compute_all_metrics(probs):
    nll_list, brier_list = [], []
    ece10_list, ece15_list, ece20_list = [], [], []
    auroc_list = []
    
    for c_idx in range(len(CLASSES)):
        y_c = targets[:, c_idx]
        p_c = probs[:, c_idx]
        nll_list.append(binary_nll(y_c, p_c))
        brier_list.append(brier_score(y_c, p_c))
        ece10_list.append(expected_calibration_error(y_c, p_c, n_bins=10))
        ece15_list.append(expected_calibration_error(y_c, p_c, n_bins=15))
        ece20_list.append(expected_calibration_error(y_c, p_c, n_bins=20))
        auroc_list.append(binary_auroc(y_c, p_c))
        
    return {
        "nll": float(np.mean(nll_list)),
        "brier": float(np.mean(brier_list)),
        "ece10": float(np.mean(ece10_list)),
        "ece15": float(np.mean(ece15_list)),
        "ece20": float(np.mean(ece20_list)),
        "auroc": float(np.mean(auroc_list)),
    }

m_raw = compute_all_metrics(oof_raw_probs)
m_temp = compute_all_metrics(oof_temp_probs)
m_platt = compute_all_metrics(oof_platt_probs)

print(f"{'Metric':<12} | {'Uncalibrated (Raw)':<20} | {'Temperature Scaling (OOF)':<25} | {'Platt Scaling (OOF)':<20}")
print("-" * 85)
for k in ["ece15", "ece10", "ece20", "nll", "brier", "auroc"]:
    print(f"{k.upper():<12} | {m_raw[k]:<20.7f} | {m_temp[k]:<25.7f} | {m_platt[k]:<20.7f}")

print("\nPer-Disease ECE15 Comparison (Out-of-Fold):")
print(f"{'Disease':<18} | {'Raw ECE15':<12} | {'Temp ECE15':<12} | {'Platt ECE15':<12} | {'ECE Reduction':<15}")
print("-" * 75)
for c_idx, c in enumerate(CLASSES):
    y_c = targets[:, c_idx]
    r_ece = expected_calibration_error(y_c, oof_raw_probs[:, c_idx], n_bins=15)
    t_ece = expected_calibration_error(y_c, oof_temp_probs[:, c_idx], n_bins=15)
    p_ece = expected_calibration_error(y_c, oof_platt_probs[:, c_idx], n_bins=15)
    reduc = ((r_ece - p_ece) / r_ece) * 100
    print(f"{c:<18} | {r_ece:<12.5f} | {t_ece:<12.5f} | {p_ece:<12.5f} | {reduc:>6.1f}% reduction")

