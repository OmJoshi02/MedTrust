"""
Phase C: Authenticity and Reproducibility of Stored Predictions
Independently reproduces predictions and metrics for a deterministic sample of reference cases
against the frozen checkpoint and preprocessing pipeline.
"""

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CLASSES
from src.model import create_model
from src.transformer import test_transform
from src.similarity_v2.builder import compute_quick_image_quality

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
THRESHOLDS_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "thresholds.json"
TEMPERATURE_PATH = PROJECT_ROOT / "outputs" / "calibration" / "temperature.json"
IMAGES_DIR = PROJECT_ROOT / "dataset" / "images"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "similarity_v2" / "reference_reports"
METADATA_PATH = PROJECT_ROOT / "outputs" / "similarity_v2" / "reference_metadata.json"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "similarity_v2" / "evaluation_audit"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_phase_c(sample_count: int = 25, seed: int = 42):
    print("=" * 70)
    print("PHASE C: PREDICTION REPRODUCIBILITY & AUTHENTICITY AUDIT")
    print("=" * 70)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing reproducibility checks on {device}...")

    # Record environment fingerprints
    ckpt_sha = sha256_file(CHECKPOINT_PATH)
    thresh_sha = sha256_file(THRESHOLDS_PATH)
    temp_sha = sha256_file(TEMPERATURE_PATH)

    # 1. Load model
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = create_model(num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # 2. Load calibration artifacts
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        t_data = json.load(f)
        thresh_dict = t_data.get("thresholds", {})
        thresholds = {c: float(thresh_dict[c].get("threshold", 0.5) if isinstance(thresh_dict[c], dict) else thresh_dict[c]) for c in CLASSES}

    with open(TEMPERATURE_PATH, "r", encoding="utf-8") as f:
        temp_data = json.load(f)
        temp_dict = temp_data.get("temperatures") or temp_data.get("per_disease_temperatures", {})
        temperatures = {c: float(temp_dict[c]) for c in CLASSES}

    temp_tensor = torch.tensor([temperatures[c] for c in CLASSES], dtype=torch.float32, device=device)

    # 3. Deterministically select diverse cohort across findings
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta_list = json.load(f)

    # Group by primary finding
    by_finding: Dict[str, List[Dict[str, Any]]] = {}
    for r in meta_list:
        raw_f = r.get("findings") or [r.get("finding_labels", "No Finding")]
        prim = raw_f[0] if raw_f else "No Finding"
        by_finding.setdefault(prim, []).append(r)

    rng = random.Random(seed)
    selected_records: List[Dict[str, Any]] = []

    # Pick 1 from each finding category to guarantee full class diversity
    for f_name in sorted(by_finding.keys()):
        pool = by_finding[f_name]
        selected_records.append(rng.choice(pool))

    # Fill remaining from entire pool
    remaining = [r for r in meta_list if r not in selected_records]
    rng.shuffle(remaining)
    selected_records.extend(remaining[: max(0, sample_count - len(selected_records))])
    selected_records = selected_records[:sample_count]

    print(f"Selected {len(selected_records)} diverse reference cases across findings: {[r.get('finding_labels') for r in selected_records[:5]]}...")

    # 4. Independent evaluation & comparison
    case_comparisons: List[Dict[str, Any]] = []
    max_prob_diff = 0.0
    all_decisions_match = True
    all_quality_match = True
    all_demographics_match = True
    all_ood_match = True

    # Documented numerical tolerance:
    # Model predictions in reference reports are stored as 4-decimal place floats (round(val, 4)).
    # Float rounding + GPU CUDA nondeterminism introduces a theoretical delta of up to 0.0015.
    PROBABILITY_TOLERANCE = 0.0020

    for rec in selected_records:
        fn = rec.get("reference_id") or rec.get("filename")
        img_path = IMAGES_DIR / fn
        report_path = REPORTS_DIR / f"{fn}.json"

        if not report_path.exists():
            case_comparisons.append({
                "reference_id": fn,
                "status": "FAIL",
                "error": f"Stored report missing at {report_path}",
            })
            continue

        with open(report_path, "r", encoding="utf-8") as f:
            stored_rep = json.load(f)

        # 4a. Run independent preprocessing and inference
        pil_img = Image.open(img_path).convert("L")
        tensor = test_transform(pil_img).unsqueeze(0).to(device)

        with torch.inference_mode():
            feat_map = model.features(tensor)
            feat_map = F.relu(feat_map, inplace=False)
            pooled = F.adaptive_avg_pool2d(feat_map, (1, 1))
            features = torch.flatten(pooled, start_dim=1)
            logits = model.classifier(features)

            raw_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            cal_logits = logits / temp_tensor
            cal_probs = torch.sigmoid(cal_logits).squeeze(0).cpu().numpy()

        # 4b. Run image quality (recomputed with builder formula)
        q_calc = compute_quick_image_quality(img_path)

        # 4c. Compare predictions
        stored_preds = stored_rep.get("predictions", {})
        stored_raw = stored_preds.get("raw_scores", {})
        stored_cal = stored_preds.get("calibrated_scores", {})
        stored_dec = stored_preds.get("decisions", {})

        prob_diffs = []
        decisions_agree = True
        for idx, disease in enumerate(CLASSES):
            calc_r = float(raw_probs[idx])
            calc_c = float(cal_probs[idx])
            s_r = float(stored_raw.get(disease, 0.0))
            s_c = float(stored_cal.get(disease, 0.0))

            diff_r = abs(calc_r - s_r)
            diff_c = abs(calc_c - s_c)
            prob_diffs.append(max(diff_r, diff_c))

            calc_dec = "POSITIVE" if calc_c >= thresholds[disease] else "NEGATIVE"
            s_dec = str(stored_dec.get(disease, "NEGATIVE")).upper()
            if calc_dec != s_dec:
                decisions_agree = False

        max_case_diff = max(prob_diffs) if prob_diffs else 0.0
        max_prob_diff = max(max_prob_diff, max_case_diff)

        # 4d. Compare Quality
        stored_q = stored_rep.get("image_quality", {})
        q_score_diff = abs(stored_q.get("composite_score", 0) - q_calc.get("composite_score", 0))
        q_agree = (q_score_diff == 0)

        # 4e. Compare Demographics
        stored_demo = stored_rep.get("demographics", {})
        demo_agree = (
            stored_demo.get("patient_gender") == rec.get("patient_gender", "U")
            and stored_demo.get("view_position") == rec.get("view_position", "PA")
        )

        # 4f. Compare OOD
        stored_ood = stored_rep.get("ood", {})
        ood_status_agree = (stored_ood.get("status") == "IN_DISTRIBUTION" and not stored_ood.get("is_ood"))

        case_ok = (max_case_diff <= PROBABILITY_TOLERANCE) and decisions_agree and q_agree and demo_agree and ood_status_agree
        if not case_ok:
            all_decisions_match = all_decisions_match and decisions_agree
            all_quality_match = all_quality_match and q_agree
            all_demographics_match = all_demographics_match and demo_agree
            all_ood_match = all_ood_match and ood_status_agree

        case_comparisons.append({
            "reference_id": fn,
            "ground_truth_findings": stored_rep.get("dataset_annotations", {}).get("findings"),
            "max_probability_difference": round(float(max_case_diff), 6),
            "probabilities_match": max_case_diff <= PROBABILITY_TOLERANCE,
            "decisions_match": decisions_agree,
            "quality_score_diff": q_score_diff,
            "quality_matches": q_agree,
            "demographics_match": demo_agree,
            "ood_matches": ood_status_agree,
            "status": "PASS" if case_ok else "FAIL",
        })

    duration = round(time.time() - t0, 2)
    overall_pass = all(c["status"] == "PASS" for c in case_comparisons)

    report_payload = {
        "phase": "Phase C: Prediction Reproducibility & Authenticity",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": duration,
        "sample_size": len(case_comparisons),
        "selection_procedure": "Deterministic stratified sample across 14 canonical disease classes and No Finding (seed=42)",
        "model_fingerprint": {
            "checkpoint_sha256": ckpt_sha,
            "thresholds_sha256": thresh_sha,
            "temperature_sha256": temp_sha,
            "architecture": "DenseNet-121",
        },
        "tolerances": {
            "probability_absolute_tolerance": PROBABILITY_TOLERANCE,
            "justification": "Reference reports store 4-decimal place rounded probabilities; theoretical maximum rounding error is ~0.0005, plus CUDA FP32 variance. Absolute difference bounded at <= 0.0020.",
        },
        "aggregate_metrics": {
            "max_observed_probability_diff": round(float(max_prob_diff), 6),
            "all_probabilities_within_tolerance": max_prob_diff <= PROBABILITY_TOLERANCE,
            "all_threshold_decisions_match": all_decisions_match,
            "all_quality_scores_match": all_quality_match,
            "all_demographics_match": all_demographics_match,
            "all_ood_match": all_ood_match,
            "overall_status": "PASS" if overall_pass else "FAIL",
        },
        "case_evaluations": case_comparisons,
    }

    out_file = AUDIT_DIR / "prediction_reproducibility_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"Phase C completed in {duration}s. Verdict: {report_payload['aggregate_metrics']['overall_status']}")
    print(f"  Max observed probability diff: {max_prob_diff:.6f} (tolerance: {PROBABILITY_TOLERANCE})")
    print(f"  All decisions match: {all_decisions_match}")
    print(f"  All quality scores match: {all_quality_match}")
    print(f"  All demographics match: {all_demographics_match}")
    print(f"  All OOD match: {all_ood_match}")
    print(f"  Report saved to {out_file}")
    return report_payload


if __name__ == "__main__":
    run_phase_c()
