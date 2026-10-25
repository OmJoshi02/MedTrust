"""
Script to establish safe calibration baseline and perform root-cause audit.
"""
import os
import sys
import shutil
import hashlib
import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.calibration.metrics import (
    binary_nll,
    brier_score,
    expected_calibration_error,
    binary_auroc,
    calculate_binary_calibration_metrics,
)

CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

CLASS_TO_CSV = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "effusion",
    "Infiltration": "infiltration",
    "Mass": "mass",
    "Nodule": "nodule",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Pleural_Thickening": "pleural_thickening",
    "Hernia": "hernia",
}

def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def backup_and_hash():
    backup_root = Path("backup/calibration_baseline")
    backup_src = backup_root / "src_calibration"
    backup_out = backup_root / "outputs_calibration"
    
    if backup_root.exists():
        shutil.rmtree(backup_root)
    
    backup_src.mkdir(parents=True, exist_ok=True)
    backup_out.mkdir(parents=True, exist_ok=True)
    
    hashes = {}
    
    # Backup src/calibration
    src_dir = Path("src/calibration")
    for f in src_dir.rglob("*"):
        if f.is_file() and "__pycache__" not in str(f):
            rel = f.relative_to(src_dir)
            dest = backup_src / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            hashes[f"src/calibration/{rel.as_posix()}"] = hash_file(f)
            
    # Backup outputs/calibration
    out_dir = Path("outputs/calibration")
    for f in out_dir.rglob("*"):
        if f.is_file() and not f.name.endswith(".rar"):
            rel = f.relative_to(out_dir)
            dest = backup_out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            hashes[f"outputs/calibration/{rel.as_posix()}"] = hash_file(f)
            
    hashes_path = backup_root / "original_hashes.json"
    with open(hashes_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)
        
    print(f"Backed up {len(hashes)} files to {backup_root} with SHA256 hashes recorded.")
    return hashes

def load_data(csv_path: Path):
    targets, logits = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_row = [float(row[f"target_{CLASS_TO_CSV[c]}"]) for c in CLASSES]
            l_row = [float(row[f"logit_{CLASS_TO_CSV[c]}"]) for c in CLASSES]
            targets.append(t_row)
            logits.append(l_row)
    return np.array(targets, dtype=np.float64), np.array(logits, dtype=np.float64)

def stable_sigmoid(x):
    pos = x >= 0
    neg = ~pos
    res = np.empty_like(x, dtype=np.float64)
    res[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    res[neg] = exp_x / (1.0 + exp_x)
    return res

def audit_baseline():
    val_targets, val_logits = load_data(Path("outputs/evaluation/validation_predictions.csv"))
    test_targets, test_logits = load_data(Path("outputs/evaluation/test_predictions.csv"))
    
    with open("outputs/calibration/temperature.json", "r", encoding="utf-8") as f:
        temp_data = json.load(f)
    temperatures = np.array([float(temp_data["temperatures"][c]) for c in CLASSES], dtype=np.float64)
    
    val_probs_raw = stable_sigmoid(val_logits)
    val_probs_cal = stable_sigmoid(val_logits / temperatures)
    
    test_probs_raw = stable_sigmoid(test_logits)
    test_probs_cal = stable_sigmoid(test_logits / temperatures)
    
    # Calculate Macro metrics
    metrics = {
        "val_raw": {"nll": [], "brier": [], "ece10": [], "ece15": [], "ece20": [], "auroc": []},
        "val_cal": {"nll": [], "brier": [], "ece10": [], "ece15": [], "ece20": [], "auroc": []},
        "test_raw": {"nll": [], "brier": [], "ece10": [], "ece15": [], "ece20": [], "auroc": []},
        "test_cal": {"nll": [], "brier": [], "ece10": [], "ece15": [], "ece20": [], "auroc": []},
    }
    
    print("\n" + "="*80)
    print("CLASS-BY-CLASS DETAILED AUDIT (TEST SET)")
    print("="*80)
    print(f"{'Class':<18} | {'N_pos':<6} | {'Prev%':<6} | {'T':<6} | {'ECE_raw':<8} -> {'ECE_cal':<8} (Delta) | {'NLL_raw':<7} -> {'NLL_cal':<7} | {'Brier_raw':<9} -> {'Brier_cal':<9}")
    print("-" * 105)
    
    for i, c in enumerate(CLASSES):
        # Val
        metrics["val_raw"]["nll"].append(binary_nll(val_targets[:, i], val_probs_raw[:, i]))
        metrics["val_raw"]["brier"].append(brier_score(val_targets[:, i], val_probs_raw[:, i]))
        metrics["val_raw"]["ece10"].append(expected_calibration_error(val_targets[:, i], val_probs_raw[:, i], n_bins=10))
        metrics["val_raw"]["ece15"].append(expected_calibration_error(val_targets[:, i], val_probs_raw[:, i], n_bins=15))
        metrics["val_raw"]["ece20"].append(expected_calibration_error(val_targets[:, i], val_probs_raw[:, i], n_bins=20))
        metrics["val_raw"]["auroc"].append(binary_auroc(val_targets[:, i], val_probs_raw[:, i]))
        
        metrics["val_cal"]["nll"].append(binary_nll(val_targets[:, i], val_probs_cal[:, i]))
        metrics["val_cal"]["brier"].append(brier_score(val_targets[:, i], val_probs_cal[:, i]))
        metrics["val_cal"]["ece10"].append(expected_calibration_error(val_targets[:, i], val_probs_cal[:, i], n_bins=10))
        metrics["val_cal"]["ece15"].append(expected_calibration_error(val_targets[:, i], val_probs_cal[:, i], n_bins=15))
        metrics["val_cal"]["ece20"].append(expected_calibration_error(val_targets[:, i], val_probs_cal[:, i], n_bins=20))
        metrics["val_cal"]["auroc"].append(binary_auroc(val_targets[:, i], val_probs_cal[:, i]))
        
        # Test
        tr_nll = binary_nll(test_targets[:, i], test_probs_raw[:, i])
        tc_nll = binary_nll(test_targets[:, i], test_probs_cal[:, i])
        tr_brier = brier_score(test_targets[:, i], test_probs_raw[:, i])
        tc_brier = brier_score(test_targets[:, i], test_probs_cal[:, i])
        tr_ece15 = expected_calibration_error(test_targets[:, i], test_probs_raw[:, i], n_bins=15)
        tc_ece15 = expected_calibration_error(test_targets[:, i], test_probs_cal[:, i], n_bins=15)
        
        metrics["test_raw"]["nll"].append(tr_nll)
        metrics["test_raw"]["brier"].append(tr_brier)
        metrics["test_raw"]["ece10"].append(expected_calibration_error(test_targets[:, i], test_probs_raw[:, i], n_bins=10))
        metrics["test_raw"]["ece15"].append(tr_ece15)
        metrics["test_raw"]["ece20"].append(expected_calibration_error(test_targets[:, i], test_probs_raw[:, i], n_bins=20))
        metrics["test_raw"]["auroc"].append(binary_auroc(test_targets[:, i], test_probs_raw[:, i]))
        
        metrics["test_cal"]["nll"].append(tc_nll)
        metrics["test_cal"]["brier"].append(tc_brier)
        metrics["test_cal"]["ece10"].append(expected_calibration_error(test_targets[:, i], test_probs_cal[:, i], n_bins=10))
        metrics["test_cal"]["ece15"].append(tc_ece15)
        metrics["test_cal"]["ece20"].append(expected_calibration_error(test_targets[:, i], test_probs_cal[:, i], n_bins=20))
        metrics["test_cal"]["auroc"].append(binary_auroc(test_targets[:, i], test_probs_cal[:, i]))
        
        n_pos = int(np.sum(test_targets[:, i]))
        prev = n_pos / len(test_targets) * 100
        ece_diff = tc_ece15 - tr_ece15
        
        print(f"{c:<18} | {n_pos:<6} | {prev:<5.2f}% | {temperatures[i]:<6.3f} | {tr_ece15:<8.4f} -> {tc_ece15:<8.4f} ({ece_diff:+7.4f}) | {tr_nll:<7.4f} -> {tc_nll:<7.4f} | {tr_brier:<9.4f} -> {tc_brier:<9.4f}")
        
    print("-" * 105)
    
    # Compute macro
    def macro(key, split):
        return np.mean(metrics[split][key])
        
    print("\nMACRO COMPARISON SUMMARY:")
    print("Validation Set (N = 16,828):")
    print(f"  NLL:   {macro('nll', 'val_raw'):.7f} -> {macro('nll', 'val_cal'):.7f} (d {macro('nll', 'val_cal') - macro('nll', 'val_raw'):+.7f})")
    print(f"  Brier: {macro('brier', 'val_raw'):.7f} -> {macro('brier', 'val_cal'):.7f} (d {macro('brier', 'val_cal') - macro('brier', 'val_raw'):+.7f})")
    print(f"  ECE10: {macro('ece10', 'val_raw'):.7f} -> {macro('ece10', 'val_cal'):.7f} (d {macro('ece10', 'val_cal') - macro('ece10', 'val_raw'):+.7f})")
    print(f"  ECE15: {macro('ece15', 'val_raw'):.7f} -> {macro('ece15', 'val_cal'):.7f} (d {macro('ece15', 'val_cal') - macro('ece15', 'val_raw'):+.7f})")
    print(f"  ECE20: {macro('ece20', 'val_raw'):.7f} -> {macro('ece20', 'val_cal'):.7f} (d {macro('ece20', 'val_cal') - macro('ece20', 'val_raw'):+.7f})")
    print(f"  AUROC: {macro('auroc', 'val_raw'):.7f} -> {macro('auroc', 'val_cal'):.7f} (d {macro('auroc', 'val_cal') - macro('auroc', 'val_raw'):+.7f})")
    
    print("\nTest Set (N = 16,805):")
    print(f"  NLL:   {macro('nll', 'test_raw'):.7f} -> {macro('nll', 'test_cal'):.7f} (d {macro('nll', 'test_cal') - macro('nll', 'test_raw'):+.7f})")
    print(f"  Brier: {macro('brier', 'test_raw'):.7f} -> {macro('brier', 'test_cal'):.7f} (d {macro('brier', 'test_cal') - macro('brier', 'test_raw'):+.7f})")
    print(f"  ECE10: {macro('ece10', 'test_raw'):.7f} -> {macro('ece10', 'test_cal'):.7f} (d {macro('ece10', 'test_cal') - macro('ece10', 'test_raw'):+.7f})")
    print(f"  ECE15: {macro('ece15', 'test_raw'):.7f} -> {macro('ece15', 'test_cal'):.7f} (d {macro('ece15', 'test_cal') - macro('ece15', 'test_raw'):+.7f})")
    print(f"  ECE20: {macro('ece20', 'test_raw'):.7f} -> {macro('ece20', 'test_cal'):.7f} (d {macro('ece20', 'test_cal') - macro('ece20', 'test_raw'):+.7f})")
    print(f"  AUROC: {macro('auroc', 'test_raw'):.7f} -> {macro('auroc', 'test_cal'):.7f} (d {macro('auroc', 'test_cal') - macro('auroc', 'test_raw'):+.7f})")

if __name__ == "__main__":
    backup_and_hash()
    audit_baseline()
