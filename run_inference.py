"""
MedTrust V1 — Master Prediction & Diagnostic Entry Point (run_inference.py)

Orchestrates all integrated MedTrust V1 modules through a unified inference pipeline
preserving strict separation of concerns:
- Evaluation     : src.evaluation (Frozen Youden J decision thresholds)
- Calibration    : src.calibration (Frozen per-disease temperature scaling & reliability diagnostics)
- Explainability : src.explainability (Disease-specific Grad-CAM overlays & explanation reliability)
- Inference      : src.inference.MedTrustInferencePipeline (Unified workflow orchestrator)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from src.dataset import CLASSES
from src.inference import MedTrustInferencePipeline, DEFAULT_INFERENCE_OUTPUT_DIR
from src.explainability import GRADCAM_OUTPUT_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run trust-aware diagnostic inference on a chest X-ray."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to chest X-ray image (PNG/JPG). If omitted, automatically checks 'test_images/disease.png'.",
    )
    parser.add_argument(
        "--target-disease",
        type=str,
        default=None,
        choices=list(CLASSES),
        help="Specific disease class to generate Grad-CAM and reliability for (default: top predicted disease).",
    )
    parser.add_argument(
        "--skip-gradcam",
        action="store_true",
        help="Skip Grad-CAM heatmap visualization generation.",
    )
    parser.add_argument(
        "--skip-reliability",
        action="store_true",
        help="Skip explanation perturbation stability and occlusion fidelity analysis.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force compute device ('cuda' or 'cpu'). Default: auto-detect CUDA if available.",
    )
    return parser.parse_args()


def resolve_image_path(user_input: Optional[str]) -> Path:
    if user_input is not None:
        path = Path(user_input)
        if not path.exists():
            raise FileNotFoundError(f"Specified image not found: {path}")
        return path

    # Auto-detection fallback
    default_sample = Path("test_images/disease.png")
    if default_sample.exists():
        print(f"[INFO] No image specified. Using default sample: {default_sample}")
        return default_sample

    sample_candidates = list(Path("test_images").glob("*.png")) + list(Path("test_images").glob("*.jpg"))
    if sample_candidates:
        chosen = sample_candidates[0]
        print(f"[INFO] No image specified. Auto-detected sample: {chosen}")
        return chosen

    raise FileNotFoundError(
        "No image argument provided and no default sample found in 'test_images/'. "
        "Usage: python run_inference.py <path_to_xray_image>"
    )


def print_diagnostic_report(result: dict) -> None:
    print("\n" + "=" * 96)
    print("MEDTRUST V1 — TRUST-AWARE DIAGNOSTIC INFERENCE REPORT")
    print("=" * 96)
    print(f"Target Image         : {result['image']}")
    print(f"Model Architecture   : {result['model_architecture']} (Epoch {result['checkpoint_epoch']})")
    print(f"Checkpoint Artifact  : {result['checkpoint']}")
    print(f"Thresholds Source    : {result['threshold_source']}")
    print(f"Calibration Source   : {result['calibration_source']}")
    print("-" * 96)

    # Findings Table
    print(
        f"{'Pathology':<20} "
        f"{'Raw':>8} "
        f"{'Calibrated':>12} "
        f"{'Temp':>7} "
        f"{'Threshold':>10} "
        f"{'Decision':>10} "
        f"{'Cal. Status':>16}"
    )
    print("-" * 96)

    for item in result["findings"]:
        decision_str = "POSITIVE" if item["decision"] == "POSITIVE" else "Negative"
        cal_status = item.get("calibration_status", "N/A")
        print(
            f"{item['disease']:<20} "
            f"{item['raw_probability']:>8.4f} "
            f"{item['calibrated_probability']:>12.4f} "
            f"{item['temperature']:>7.3f} "
            f"{item['threshold']:>10.4f} "
            f"{decision_str:>10} "
            f"{cal_status:>16}"
        )

    # Top Predictions Summary
    print("-" * 96)
    raw_top = result["top_prediction_raw"]
    cal_top = result["top_prediction_calibrated"]
    print(f"Top Pathology (Raw Score)        : {raw_top['disease']} ({raw_top['probability']:.4f})")
    print(f"Top Pathology (Calibrated Prob)  : {cal_top['disease']} ({cal_top['probability']:.4f})")

    positive_findings = [f for f in result["findings"] if f["decision"] == "POSITIVE"]
    if positive_findings:
        pos_names = ", ".join(f"{f['disease']} ({f['calibrated_probability']:.4f})" for f in positive_findings)
        print(f"Pathologies Crossing Threshold   : {pos_names}")
    else:
        print("Pathologies Crossing Threshold   : None (All raw scores below frozen Youden J thresholds)")

    # Explainability Section (Grad-CAM)
    print("\n" + "=" * 96)
    print("EXPLAINABILITY — GRAD-CAM ATTRIBUTION")
    print("=" * 96)
    explanation = result.get("explanation", {})
    target_d = explanation.get("target_disease", "N/A")
    target_layer = explanation.get("target_layer", "N/A")
    gradcam_file = explanation.get("gradcam_artifact")

    print(f"Target Disease for Attribution   : {target_d}")
    print(f"DenseNet Target Feature Layer    : {target_layer}")
    if gradcam_file:
        print(f"Grad-CAM Heatmap Overlay Saved   : {gradcam_file}")
    else:
        print("Grad-CAM Heatmap Overlay Saved   : [SKIPPED]")

    # Explanation Reliability Section
    rel = result.get("explanation_reliability")
    print("\n" + "=" * 96)
    print("EXPLANATION RELIABILITY & FIDELITY DIAGNOSTICS")
    print("=" * 96)
    if rel:
        print(f"Overall Spatial Stability Band   : {rel['overall_stability']}")
        print(f"Mean Perturbation Pearson Corr.  : {rel['mean_pearson_correlation']:.4f}")
        print(f"Mean Perturbation Spearman Corr. : {rel['mean_spearman_correlation']:.4f}")
        print(f"Mean Region IoU Overlap          : {rel['mean_region_iou']:.4f}")
        print(f"Region Occlusion Fidelity Drop   : {rel['fidelity_probability_drop']:.4f} "
              f"({rel['fidelity_probability_before']:.4f} -> {rel['fidelity_probability_after']:.4f})")
        print(f"Attribution Fidelity Verified    : {'YES (Probability dropped upon masking)' if rel['fidelity_verified'] else 'NO'}")
    else:
        print("Explanation Reliability Status   : [SKIPPED]")

    # Machine-Readable Artifacts
    print("\n" + "=" * 96)
    print("SAVED ARTIFACTS")
    print("=" * 96)
    print(f"Structured JSON Diagnostic Record: {result.get('json_artifact')}")
    if gradcam_file:
        print(f"Grad-CAM Visualization PNG       : {gradcam_file}")

    # Clinical Disclaimer
    print("-" * 96)
    print(f"CAUTION / NOTICE: {result.get('clinical_disclaimer')}")
    print("=" * 96 + "\n")


def main():
    args = parse_args()

    try:
        image_path = resolve_image_path(args.image)
    except FileNotFoundError as err:
        print(f"\n[ERROR] {err}\n", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device) if args.device else None

    # Instantiate unified inference pipeline (modularly encapsulates all components)
    print("\nInitializing MedTrust Inference Pipeline...")
    pipeline = MedTrustInferencePipeline(device=device)

    print(f"Running diagnostic pipeline on: {image_path} (Device: {pipeline.device})...")
    result = pipeline.run(
        image_input=image_path,
        target_disease=args.target_disease,
        skip_gradcam=args.skip_gradcam,
        skip_reliability=args.skip_reliability,
    )

    # Render formatted clinical & technical report to console
    print_diagnostic_report(result)


if __name__ == "__main__":
    main()
