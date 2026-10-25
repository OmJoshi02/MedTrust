import json
import hashlib
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 1. Check SHA-256 hashes of baseline artifact, backup, and candidate exp artifact
base_path = Path("outputs/ood/artifacts/ood_thresholds.json")
bak_path = Path("outputs/ood/artifacts/ood_thresholds.json.bak_baseline")
cand_path = Path("outputs/ood/artifacts/ood_thresholds_candidate1_exp.json")

base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
bak_hash = hashlib.sha256(bak_path.read_bytes()).hexdigest()
cand_hash = hashlib.sha256(cand_path.read_bytes()).hexdigest()

expected_base_hash = "0b0bcaa38eee056a62ce550b8657457570f8eee79e9086329583b73691df8a0c"
print(f"Baseline file SHA-256: {base_hash} (Matches expected: {base_hash == expected_base_hash})")
print(f"Backup file SHA-256  : {bak_hash} (Matches baseline: {bak_hash == base_hash})")
print(f"Candidate file SHA-256: {cand_hash}")

# Check Candidate schema and values
with open(cand_path) as f:
    cand_data = json.load(f)

c_msp = cand_data["thresholds"]["msp"]["threshold"]
c_eng = cand_data["thresholds"]["energy"]["threshold"]
c_mah = cand_data["thresholds"]["mahalanobis"]["threshold"]

print(f"Candidate values: MSP={c_msp}, Energy={c_eng}, Maha={c_mah}")
assert c_msp == 0.737333
assert c_eng == -0.071788
assert c_mah == 975.531329

# 2. Recalculate decisions directly from raw validation arrays
from src.OOD.scores import msp_score, energy_score, mahalanobis_score

val_data = np.load("outputs/ood/features/id_val.npz")
logits = val_data["logits"]
features = val_data["features"]
targets = val_data["targets"]
N_val = len(logits)

stats = np.load("outputs/ood/artifacts/mahalanobis_statistics.npz")
mean = stats["mean"]
safe_var = stats["safe_variance"]

# Direct score calculation
s_m = msp_score(logits)
s_e = energy_score(logits)
s_h = mahalanobis_score(features, mean, safe_var)

flag_m = s_m >= c_msp
flag_e = s_e >= c_eng
flag_h = s_h >= c_mah

# Exact per-detector counts
cnt_m = int(np.sum(flag_m))
cnt_e = int(np.sum(flag_e))
cnt_h = int(np.sum(flag_h))

print(f"\nExact Per-Detector Rejection Counts (N={N_val}):")
print(f"  MSP Rejections (score >= {c_msp}):   {cnt_m} ({cnt_m/N_val*100:.4f}%)")
print(f"  Energy Rejections (score >= {c_eng}):{cnt_e} ({cnt_e/N_val*100:.4f}%)")
print(f"  Maha Rejections (score >= {c_mah}):  {cnt_h} ({cnt_h/N_val*100:.4f}%)")

# Exact disjoint Venn components
m_only = int(np.sum(flag_m & ~flag_e & ~flag_h))
e_only = int(np.sum(~flag_m & flag_e & ~flag_h))
h_only = int(np.sum(~flag_m & ~flag_e & flag_h))
me_only = int(np.sum(flag_m & flag_e & ~flag_h))
mh_only = int(np.sum(flag_m & ~flag_e & flag_h))
eh_only = int(np.sum(~flag_m & flag_e & flag_h))
meh_all = int(np.sum(flag_m & flag_e & flag_h))

sum_disjoint = m_only + e_only + h_only + me_only + mh_only + eh_only + meh_all
union_flag = flag_m | flag_e | flag_h
union_cnt = int(np.sum(union_flag))

print(f"\nVenn Diagram Disjoint Partition Breakdown:")
print(f"  A (MSP only)       : {m_only}")
print(f"  B (Energy only)    : {e_only}")
print(f"  C (Maha only)      : {h_only}")
print(f"  AB (MSP + Energy)  : {me_only}")
print(f"  AC (MSP + Maha)    : {mh_only}")
print(f"  BC (Energy + Maha) : {eh_only}")
print(f"  ABC (All Three)    : {meh_all}")
print(f"  ---------------------------------")
print(f"  Sum of Disjoint Regions : {sum_disjoint}")
print(f"  Direct Union Count (A|B|C): {union_cnt}")
print(f"  Empirical Union Rate     : {union_cnt / N_val * 100:.4f}% ({union_cnt}/{N_val})")

# Mathematical consistency checks
print(f"\nMathematical Identity Checks:")
print(f"  |MSP| == m_only + me_only + mh_only + meh_all: {cnt_m} == {m_only + me_only + mh_only + meh_all} ({cnt_m == m_only + me_only + mh_only + meh_all})")
print(f"  |Energy| == e_only + me_only + eh_only + meh_all: {cnt_e} == {e_only + me_only + eh_only + meh_all} ({cnt_e == e_only + me_only + eh_only + meh_all})")
print(f"  |Maha| == h_only + mh_only + eh_only + meh_all: {cnt_h} == {h_only + mh_only + eh_only + meh_all} ({cnt_h == h_only + mh_only + eh_only + meh_all})")
print(f"  Total individual rejections sum: {cnt_m + cnt_e + cnt_h}")
print(f"  Redundant overlap count: {(cnt_m + cnt_e + cnt_h) - union_cnt}")
print(f"  Overlaps: me_only + mh_only + eh_only + 2*meh_all = {me_only + mh_only + eh_only + 2*meh_all} == {(cnt_m + cnt_e + cnt_h) - union_cnt}")

# 3. Verify the 8/100 candidate test cohort result from saved raw test scores
with open("scratch/ood_100_images_results.json") as f:
    test_100 = json.load(f)["results"]

test_rej = []
for r in test_100:
    m = r["msp_score"] >= c_msp
    e = r["energy_score"] >= c_eng
    h = r["mahalanobis_score"] >= c_mah
    if m or e or h:
        test_rej.append((r["image"], r["ground_truth"], r["top_prediction"], m, e, h))

print(f"\nTest-Cohort Evaluation (N=100):")
print(f"  Rejected Count: {len(test_rej)} / 100 ({len(test_rej)}.0%)")
for img, gt, pred, m, e, h in test_rej:
    flags = [k for k, v in [("MSP", m), ("Energy", e), ("Maha", h)] if v]
    print(f"    {img} ({gt}): Top {pred} | Flags: {'+'.join(flags)}")

# 4. Subgroup breakdown
is_no_finding = (targets.sum(axis=1) == 0)
n_nf = int(np.sum(is_no_finding))
n_dp = int(np.sum(~is_no_finding))
nf_rej = int(np.sum(union_flag[is_no_finding]))
dp_rej = int(np.sum(union_flag[~is_no_finding]))

print(f"\nSubgroup In-Distribution Rejection Breakdown:")
print(f"  No Finding     : {nf_rej} / {n_nf} ({nf_rej/n_nf*100:.2f}%)")
print(f"  Disease-Positive: {dp_rej} / {n_dp} ({dp_rej/n_dp*100:.2f}%)")
print(f"  Disparity Ratio : { (nf_rej/n_nf) / (dp_rej/n_dp):.2f}x")
