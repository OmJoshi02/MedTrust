import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.OOD.scores import msp_score, energy_score, mahalanobis_score

# 1. Load validation data
val_data = np.load(PROJECT_ROOT / 'outputs/ood/features/id_val.npz')
logits = val_data['logits']
features = val_data['features']
targets = val_data['targets']
filenames = val_data['filenames']
N_val = len(logits)
print(f"Validation sample count: {N_val}")

# 2. Load mahalanobis stats
stats = np.load(PROJECT_ROOT / 'outputs/ood/artifacts/mahalanobis_statistics.npz')
mean = stats['mean']
safe_var = stats['safe_variance']

# 3. Compute raw scores
s_msp = msp_score(logits)
s_eng = energy_score(logits)
s_mah = mahalanobis_score(features, mean, safe_var)

# Check ties in validation scores
print(f"Unique MSP scores: {len(np.unique(s_msp))} / {N_val}")
print(f"Unique Energy scores: {len(np.unique(s_eng))} / {N_val}")
print(f"Unique Maha scores: {len(np.unique(s_mah))} / {N_val}")

# 4. Independent search for uniform quantile q
# In numpy, default percentile method is 'linear'. Let's test methods: 'linear', 'nearest', 'lower', 'higher', 'midpoint'
print("\n--- Quantile Search for Union FPR = 10.0% ---")
target_k = int(round(0.10 * N_val))  # 1683 samples
print(f"Target rejection count: {target_k} / {N_val} ({target_k/N_val*100:.4f}%)")

# Search for quantile q with high precision
q_grid = np.linspace(95.0, 95.3, 3001)
best_q = None
best_diff = 999999
best_counts = {}

for q in q_grid:
    t_m = np.percentile(s_msp, q, method='linear')
    t_e = np.percentile(s_eng, q, method='linear')
    t_h = np.percentile(s_mah, q, method='linear')
    
    # Exact >= comparison as in predict.py
    flags_ge = (s_msp >= t_m) | (s_eng >= t_e) | (s_mah >= t_h)
    cnt_ge = np.sum(flags_ge)
    
    # Strict > comparison
    flags_gt = (s_msp > t_m) | (s_eng > t_e) | (s_mah > t_h)
    cnt_gt = np.sum(flags_gt)
    
    diff = abs(cnt_ge - target_k)
    if diff < best_diff:
        best_diff = diff
        best_q = q
        best_counts = {'cnt_ge': cnt_ge, 'cnt_gt': cnt_gt, 'tm': t_m, 'te': t_e, 'th': t_h}

print(f"Best quantile found: {best_q:.6f}%")
print(f"Threshold MSP:   {best_counts['tm']:.6f}")
print(f"Threshold Energy:{best_counts['te']:.6f}")
print(f"Threshold Maha:  {best_counts['th']:.6f}")
print(f"Rejection count (>=): {best_counts['cnt_ge']} / {N_val} ({best_counts['cnt_ge']/N_val*100:.4f}%)")
print(f"Rejection count (> ): {best_counts['cnt_gt']} / {N_val} ({best_counts['cnt_gt']/N_val*100:.4f}%)")

# 5. Check exact match with proposed values: 0.737333, -0.071788, 975.531329
p_tm = 0.737333
p_te = -0.071788
p_th = 975.531329

p_flags_ge = (s_msp >= p_tm) | (s_eng >= p_te) | (s_mah >= p_th)
p_flags_gt = (s_msp > p_tm) | (s_eng > p_te) | (s_mah > p_th)

print("\n--- Verification of Proposed Floats ---")
print(f"Proposed MSP:   {p_tm}")
print(f"Proposed Energy:{p_te}")
print(f"Proposed Maha:  {p_th}")
print(f"Empirical Rejections with >=: {np.sum(p_flags_ge)} / {N_val} ({np.mean(p_flags_ge)*100:.4f}%)")
print(f"Empirical Rejections with > : {np.sum(p_flags_gt)} / {N_val} ({np.mean(p_flags_gt)*100:.4f}%)")
print(f"Exact match to 1683: {np.sum(p_flags_ge) == 1683}")

# Individual detector rejections with proposed thresholds
msp_rej = np.sum(s_msp >= p_tm)
eng_rej = np.sum(s_eng >= p_te)
mah_rej = np.sum(s_mah >= p_th)
print(f"Individual MSP rejections:   {msp_rej} / {N_val} ({msp_rej/N_val*100:.2f}%)")
print(f"Individual Energy rejections:{eng_rej} / {N_val} ({eng_rej/N_val*100:.2f}%)")
print(f"Individual Maha rejections:  {mah_rej} / {N_val} ({mah_rej/N_val*100:.2f}%)")

# Overlap breakdown
m_only = np.sum((s_msp >= p_tm) & ~(s_eng >= p_te) & ~(s_mah >= p_th))
e_only = np.sum(~(s_msp >= p_tm) & (s_eng >= p_te) & ~(s_mah >= p_th))
h_only = np.sum(~(s_msp >= p_tm) & ~(s_eng >= p_te) & (s_mah >= p_th))
me_only = np.sum((s_msp >= p_tm) & (s_eng >= p_te) & ~(s_mah >= p_th))
mh_only = np.sum((s_msp >= p_tm) & ~(s_eng >= p_te) & (s_mah >= p_th))
eh_only = np.sum(~(s_msp >= p_tm) & (s_eng >= p_te) & (s_mah >= p_th))
meh_all = np.sum((s_msp >= p_tm) & (s_eng >= p_te) & (s_mah >= p_th))

print("\n--- Overlap & Disagreement Breakdown (Validation N = 16,828) ---")
print(f"MSP only       : {m_only} ({m_only/1683*100:.1f}% of rejections)")
print(f"Energy only    : {e_only} ({e_only/1683*100:.1f}% of rejections)")
print(f"Maha only      : {h_only} ({h_only/1683*100:.1f}% of rejections)")
print(f"MSP + Energy   : {me_only} ({me_only/1683*100:.1f}% of rejections)")
print(f"MSP + Maha     : {mh_only} ({mh_only/1683*100:.1f}% of rejections)")
print(f"Energy + Maha  : {eh_only} ({eh_only/1683*100:.1f}% of rejections)")
print(f"All 3 triggered: {meh_all} ({meh_all/1683*100:.1f}% of rejections)")

# 6. Subgroup Analysis: No Finding vs Disease Positive
is_no_finding = (targets.sum(axis=1) == 0)
n_nf = np.sum(is_no_finding)
n_dp = np.sum(~is_no_finding)

# Baseline thresholds (0.668147, -0.436292, 843.067070)
b_tm, b_te, b_th = 0.6681468806846924, -0.4362915412914423, 843.0670704711493
b_flags = (s_msp >= b_tm) | (s_eng >= b_te) | (s_mah >= b_th)

print("\n--- Subgroup Fairness: No Finding vs Disease Positive ---")
print(f"Total No Finding validation cases: {n_nf} ({n_nf/N_val*100:.1f}%)")
print(f"Total Disease-Positive validation cases: {n_dp} ({n_dp/N_val*100:.1f}%)")
print("\n[Baseline Configuration]")
print(f"No Finding Rejection Rate      : {np.mean(b_flags[is_no_finding])*100:.2f}% ({np.sum(b_flags[is_no_finding])}/{n_nf})")
print(f"Disease-Positive Rejection Rate: {np.mean(b_flags[~is_no_finding])*100:.2f}% ({np.sum(b_flags[~is_no_finding])}/{n_dp})")
print(f"Disparity Ratio (NF / DP)      : {np.mean(b_flags[is_no_finding]) / np.mean(b_flags[~is_no_finding]):.2f}x")

print("\n[Candidate 1 Configuration]")
print(f"No Finding Rejection Rate      : {np.mean(p_flags_ge[is_no_finding])*100:.2f}% ({np.sum(p_flags_ge[is_no_finding])}/{n_nf})")
print(f"Disease-Positive Rejection Rate: {np.mean(p_flags_ge[~is_no_finding])*100:.2f}% ({np.sum(p_flags_ge[~is_no_finding])}/{n_dp})")
print(f"Disparity Ratio (NF / DP)      : {np.mean(p_flags_ge[is_no_finding]) / np.mean(p_flags_ge[~is_no_finding]):.2f}x")

