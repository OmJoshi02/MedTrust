import sys
from pathlib import Path
import numpy as np
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.OOD.scores import msp_score, energy_score, mahalanobis_score

# Load validation features
val_data = np.load(PROJECT_ROOT / 'outputs/ood/features/id_val.npz')
logits = val_data['logits']
features = val_data['features']
targets = val_data['targets']
N_val = len(logits)

# Load maha stats
data_stats = np.load(PROJECT_ROOT / 'outputs/ood/artifacts/mahalanobis_statistics.npz')
mean = data_stats['mean']
safe_var = data_stats['safe_variance']

# Raw scores on validation
s_msp = msp_score(logits)
s_eng = energy_score(logits)
s_mah = mahalanobis_score(features, mean, safe_var)

# Baseline thresholds (90th percentile)
th_msp_base = np.percentile(s_msp, 90.0)
th_eng_base = np.percentile(s_eng, 90.0)
th_mah_base = np.percentile(s_mah, 90.0)

base_or = (s_msp >= th_msp_base) | (s_eng >= th_eng_base) | (s_mah >= th_mah_base)
base_vote = ((s_msp >= th_msp_base).astype(int) + (s_eng >= th_eng_base).astype(int) + (s_mah >= th_mah_base).astype(int)) >= 2

print(f"Validation Sample Size: {N_val}")
print(f"Baseline OR (90th pct) Rejection Rate: {np.mean(base_or)*100:.2f}% ({np.sum(base_or)}/{N_val})")
print(f"Baseline Majority Vote (>=2 flags) Rejection Rate: {np.mean(base_vote)*100:.2f}% ({np.sum(base_vote)}/{N_val})")

# Find quantile q for Joint Union Calibration
def find_union_quantile(target_rate):
    # Binary search for quantile q in [80.0, 99.9]
    low, high = 80.0, 99.99
    best_q = None
    best_diff = 1.0
    for _ in range(50):
        mid = (low + high) / 2
        t_m = np.percentile(s_msp, mid)
        t_e = np.percentile(s_eng, mid)
        t_h = np.percentile(s_mah, mid)
        flags = (s_msp >= t_m) | (s_eng >= t_e) | (s_mah >= t_h)
        rate = np.mean(flags)
        if abs(rate - target_rate) < best_diff:
            best_diff = abs(rate - target_rate)
            best_q = mid
        if rate > target_rate:
            low = mid
        else:
            high = mid
    return best_q

q_10 = find_union_quantile(0.10)
q_05 = find_union_quantile(0.05)

t_m_10, t_e_10, t_h_10 = np.percentile(s_msp, q_10), np.percentile(s_eng, q_10), np.percentile(s_mah, q_10)
flags_10 = (s_msp >= t_m_10) | (s_eng >= t_e_10) | (s_mah >= t_h_10)

t_m_05, t_e_05, t_h_05 = np.percentile(s_msp, q_05), np.percentile(s_eng, q_05), np.percentile(s_mah, q_05)
flags_05 = (s_msp >= t_m_05) | (s_eng >= t_e_05) | (s_mah >= t_h_05)

print("\n--- Candidate 1: Joint Union Calibration targeting 10.0% Combined FPR ---")
print(f"Optimal Quantile: {q_10:.3f}% (Individual rejection rate: {100 - q_10:.2f}%)")
print(f"Thresholds: MSP={t_m_10:.6f}, Energy={t_e_10:.6f}, Maha={t_h_10:.6f}")
print(f"Combined Validation Rejection Rate: {np.mean(flags_10)*100:.2f}% ({np.sum(flags_10)}/{N_val})")
print(f"Individual trigger counts: MSP={np.sum(s_msp >= t_m_10)}, Energy={np.sum(s_eng >= t_e_10)}, Maha={np.sum(s_mah >= t_h_10)}")

print("\n--- Candidate 2: Joint Union Calibration targeting 5.0% Combined FPR ---")
print(f"Optimal Quantile: {q_05:.3f}% (Individual rejection rate: {100 - q_05:.2f}%)")
print(f"Thresholds: MSP={t_m_05:.6f}, Energy={t_e_05:.6f}, Maha={t_h_05:.6f}")
print(f"Combined Validation Rejection Rate: {np.mean(flags_05)*100:.2f}% ({np.sum(flags_05)}/{N_val})")
print(f"Individual trigger counts: MSP={np.sum(s_msp >= t_m_05)}, Energy={np.sum(s_eng >= t_e_05)}, Maha={np.sum(s_mah >= t_h_05)}")

# Candidate 3: Standardized Z-Score Summation (Soft Ensemble)
z_msp = (s_msp - np.mean(s_msp)) / np.std(s_msp)
z_eng = (s_eng - np.mean(s_eng)) / np.std(s_eng)
z_mah = (s_mah - np.mean(s_mah)) / np.std(s_mah)
s_ensemble = z_msp + z_eng + z_mah

th_ens_10 = np.percentile(s_ensemble, 90.0)
th_ens_05 = np.percentile(s_ensemble, 95.0)

print("\n--- Candidate 3: Standardized Score Ensemble ---")
print(f"Ensemble Threshold @ 10% FPR: {th_ens_10:.4f} -> Empirical Rate: {np.mean(s_ensemble >= th_ens_10)*100:.2f}%")
print(f"Ensemble Threshold @ 5% FPR:  {th_ens_05:.4f} -> Empirical Rate: {np.mean(s_ensemble >= th_ens_05)*100:.2f}%")

# Test on True OOD samples
from src.OOD.model_adapter import MedTrustModelAdapter
from src.transformer import eval_transform
from PIL import Image

adapter = MedTrustModelAdapter()
ood_paths = ['test_images/foot.png', 'test_images/hand.png', 'test_images/mri.jpg']
print("\n--- True OOD Sensitivity Check across Candidates ---")

for path_str in ood_paths:
    p = PROJECT_ROOT / path_str
    if not p.exists():
        continue
    with Image.open(p) as src:
        img = src.convert('L')
    tensor = eval_transform(img).unsqueeze(0)
    out = adapter.predict(tensor)
    f_val = out['features'].detach().cpu().numpy()
    l_val = out['logits'].detach().cpu().numpy()
    
    val_m = msp_score(l_val)[0]
    val_e = energy_score(l_val)[0]
    val_h = mahalanobis_score(f_val, mean, safe_var)[0]
    
    # Decisions
    d_base = (val_m >= th_msp_base) or (val_e >= th_eng_base) or (val_h >= th_mah_base)
    d_10 = (val_m >= t_m_10) or (val_e >= t_e_10) or (val_h >= t_h_10)
    d_05 = (val_m >= t_m_05) or (val_e >= t_e_05) or (val_h >= t_h_05)
    d_vote = ((val_m >= th_msp_base) + (val_e >= th_eng_base) + (val_h >= th_mah_base)) >= 2
    
    val_z_m = (val_m - np.mean(s_msp)) / np.std(s_msp)
    val_z_e = (val_e - np.mean(s_eng)) / np.std(s_eng)
    val_z_h = (val_h - np.mean(s_mah)) / np.std(s_mah)
    val_ens = val_z_m + val_z_e + val_z_h
    d_ens_10 = val_ens >= th_ens_10
    d_ens_05 = val_ens >= th_ens_05
    
    print(f"{path_str:<22}: Maha={val_h:.1f} | Base_OR={d_base} | Union_10%={d_10} | Union_5%={d_05} | MajorityVote={d_vote} | Ens_10%={d_ens_10} | Ens_5%={d_ens_05}")

