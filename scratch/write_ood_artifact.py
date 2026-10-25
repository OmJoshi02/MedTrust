import json
from pathlib import Path

# Load json
json_path = Path("scratch/ood_100_images_results.json")
with open(json_path, "r") as f:
    data = json.load(f)

stats = data["stats"]
results = data["results"]

artifact_path = Path(r"C:\Users\omcjo\.gemini\antigravity\brain\0bcdac32-0b74-4e2d-97a9-b0513d9329ab\ood_100_images_evaluation.md")

lines = []
lines.append("# MedTrust V1 — 100-Radiograph OOD Evaluation Report")
lines.append("")
lines.append("**Dataset:** Authentic NIH ChestX-ray14 (`dataset/images/`)  ")
lines.append("**Sample Size:** 100 Radiographs (Randomly sampled from holdout test split `splits/test_list.txt`, Seed = 42)  ")
lines.append("**Inference Mode:** Multi-Detector (MSP, Energy, Mahalanobis) & Conservative Combined Policy  ")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 1. Executive Summary & Aggregate Performance")
lines.append("")
lines.append("| Metric | Value | Reference / Notes |")
lines.append("| :--- | :---: | :--- |")
lines.append(f"| **Total Processed Radiographs** | `{stats['total_images']}` | 100 unique patients/images verified in `dataset/images/` |")
lines.append(f"| **Combined In-Distribution (ID)** | `{stats['combined_id_count']}` / 100 ({stats['combined_id_rate']:.1f}%) | Flagged as acceptable for diagnostic consideration |")
lines.append(f"| **Combined Out-of-Distribution (OOD)** | `{stats['combined_ood_count']}` / 100 ({stats['combined_ood_rate']:.1f}%) | Conservatively flagged by at least one detector |")
lines.append(f"| **MSP Flagged Count** | `{stats['msp_flagged_count']}` / 100 | Detector Threshold: `0.668147` |")
lines.append(f"| **Energy Flagged Count** | `{stats['energy_flagged_count']}` / 100 | Detector Threshold: `-0.436292` |")
lines.append(f"| **Mahalanobis Flagged Count** | `{stats['mahalanobis_flagged_count']}` / 100 | Detector Threshold: `843.067070` |")
lines.append("")
lines.append("> [!NOTE]")
lines.append("> The individual detectors are calibrated at approximately 95% In-Distribution True Positive Rate (5% FPR) on the validation set. Under the **conservative union policy** (flagging if **ANY** detector triggers), the aggregate flag rate is 20.0%, reflecting the cumulative defense against atypical anatomy, severe artifacts, or borderline predictions.")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 2. Detector Score Statistical Distributions")
lines.append("")
lines.append("| Detector Method | Threshold | Mean ± Std | Median | Min | Max | Trigger Rate |")
lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
lines.append(f"| **MSP** | `0.668147` | `{stats['msp_score']['mean']:.4f} ± {stats['msp_score']['std']:.4f}` | `{stats['msp_score']['median']:.4f}` | `{stats['msp_score']['min']:.4f}` | `{stats['msp_score']['max']:.4f}` | `{stats['msp_flagged_count']}%` |")
lines.append(f"| **Energy** | `-0.436292` | `{stats['energy_score']['mean']:.4f} ± {stats['energy_score']['std']:.4f}` | `{stats['energy_score']['median']:.4f}` | `{stats['energy_score']['min']:.4f}` | `{stats['energy_score']['max']:.4f}` | `{stats['energy_flagged_count']}%` |")
lines.append(f"| **Mahalanobis** | `843.067070` | `{stats['mahalanobis_score']['mean']:.1f} ± {stats['mahalanobis_score']['std']:.1f}` | `{stats['mahalanobis_score']['median']:.1f}` | `{stats['mahalanobis_score']['min']:.1f}` | `{stats['mahalanobis_score']['max']:.1f}` | `{stats['mahalanobis_flagged_count']}%` |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## 3. Complete 100-Case Evaluation Inventory")
lines.append("")
lines.append("| # | Image Filename | Age / Sex / View | Ground Truth (NIH) | Top Prediction (Prob) | MSP Score | Energy Score | Maha Score | Status | Trust Decision |")
lines.append("| :-: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |")

for i, r in enumerate(results, 1):
    status_badge = "**ID**" if r["combined_status"] == "ID" else "⚠️ **OOD**"
    msp_s = f"{r['msp_score']:.4f}" + ("*" if r["msp_ood"] else "")
    eng_s = f"{r['energy_score']:.4f}" + ("*" if r["energy_ood"] else "")
    mah_s = f"{r['mahalanobis_score']:.1f}" + ("*" if r["mahalanobis_ood"] else "")
    pred_str = f"{r['top_prediction']} ({r['top_probability']*100:.1f}%)"
    lines.append(f"| {i} | `{r['image']}` | {r['patient_age']}y · {r['patient_gender']} · {r['view_position']} | {r['ground_truth']} | {pred_str} | {msp_s} | {eng_s} | {mah_s} | {status_badge} | {r['trust_decision']} |")

lines.append("")
lines.append("`*` *indicates detector score exceeded validation threshold.*")

content = "\n".join(lines)
artifact_path.write_text(content, encoding="utf-8")
print(f"Artifact successfully written to {artifact_path} ({len(lines)} lines)")

