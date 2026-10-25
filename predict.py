import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import CLASSES
from src.model import create_model
from src.transformer import test_transform
from src.explainability.gradcam import GradCAM
from src.explainability.reliability import ExplanationReliabilityAnalyzer
from src.calibration.calibration_analyzer import CalibrationAnalyzer
from src.OOD.predict import predict_all_methods
from src.uncertainty import MCDropoutPredictor, UncertaintyAnalyzer
from src.quality import WeightedImageQualityAnalyzer, print_quality_report


CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
THRESHOLDS_PATH = Path("outputs/evaluation/thresholds.json")
TEMPERATURE_PATH = Path("outputs/calibration/temperature.json")
CALIBRATION_BINS_PATH = Path(
    "outputs/calibration/reliability/validation_reliability_bins.csv"
)
GRADCAM_OUTPUT_DIR = Path("outputs/explainability/gradcam")


# =========================================================
# FROZEN ARTIFACT LOADERS
# =========================================================


def load_thresholds():
    """Load and validate the frozen raw-probability thresholds."""
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Threshold artifact not found: {THRESHOLDS_PATH}"
        )

    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("classes") != list(CLASSES):
        raise RuntimeError(
            "Threshold class order does not match model classes."
        )

    if "thresholds" not in data:
        raise RuntimeError(
            "Threshold artifact does not contain a 'thresholds' section."
        )

    thresholds = np.array(
        [
            float(data["thresholds"][disease]["threshold"])
            for disease in CLASSES
        ],
        dtype=np.float32,
    )

    if not np.isfinite(thresholds).all():
        raise RuntimeError(
            "Threshold artifact contains non-finite values."
        )

    if np.any((thresholds < 0.0) | (thresholds > 1.0)):
        raise RuntimeError(
            "Thresholds must be within [0, 1]."
        )

    return thresholds


def load_temperatures():
    """Load and validate frozen validation-only temperature scaling."""
    if not TEMPERATURE_PATH.exists():
        raise FileNotFoundError(
            f"Calibration artifact not found: {TEMPERATURE_PATH}"
        )

    with open(TEMPERATURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("classes") != list(CLASSES):
        raise RuntimeError(
            "Calibration class order does not match model classes."
        )

    if data.get("calibration_split") != "validation":
        raise RuntimeError(
            "Calibration artifact was not fitted on validation data."
        )

    if data.get("test_data_used_for_fitting") is not False:
        raise RuntimeError(
            "Calibration artifact does not confirm test-data exclusion."
        )

    if "temperatures" not in data:
        raise RuntimeError(
            "Calibration artifact does not contain temperatures."
        )

    temperatures = np.array(
        [
            float(data["temperatures"][disease])
            for disease in CLASSES
        ],
        dtype=np.float32,
    )

    if not np.isfinite(temperatures).all():
        raise RuntimeError(
            "Calibration artifact contains non-finite temperatures."
        )

    if np.any(temperatures <= 0):
        raise RuntimeError(
            "Calibration temperatures must all be positive."
        )

    return temperatures


# =========================================================
# MODEL
# =========================================================


def load_model(device):
    """Load the frozen DenseNet-121 checkpoint and validate class order."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if "classes" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain class metadata."
        )

    if list(checkpoint["classes"]) != list(CLASSES):
        raise RuntimeError(
            "Checkpoint class order does not match CLASSES."
        )

    model = create_model(
        num_classes=len(CLASSES),
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


# =========================================================
# INPUT / PREDICTION
# =========================================================


def prepare_image(image_path, device):
    """Load one X-ray and apply the project's authoritative test transform."""
    image = Image.open(image_path).convert("RGB")

    image_tensor = test_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    if image_tensor.shape != (1, 3, 224, 224):
        raise RuntimeError(
            "Unexpected preprocessed input shape: "
            f"{tuple(image_tensor.shape)}; expected (1, 3, 224, 224)."
        )

    return image, image_tensor


def run_prediction(model, image_tensor, temperatures, thresholds):
    """Run raw prediction, frozen temperature calibration and raw-threshold decisions."""
    with torch.inference_mode():
        logits = model(image_tensor)

        if logits.shape != (1, len(CLASSES)):
            raise RuntimeError(
                "Unexpected model output shape: "
                f"{tuple(logits.shape)}; expected (1, {len(CLASSES)})."
            )

        raw_probabilities = torch.sigmoid(logits)

        temperature_tensor = torch.as_tensor(
            temperatures,
            dtype=logits.dtype,
            device=logits.device,
        ).unsqueeze(0)

        calibrated_logits = logits / temperature_tensor
        calibrated_probabilities = torch.sigmoid(calibrated_logits)

    raw_scores = raw_probabilities.squeeze(0).cpu().numpy()
    calibrated_scores = calibrated_probabilities.squeeze(0).cpu().numpy()

    predictions = (raw_scores >= thresholds).astype(np.int32)

    return (
        logits.detach(),
        raw_scores,
        calibrated_scores,
        predictions,
    )


# =========================================================
# CLASS SELECTION
# =========================================================


def resolve_class_index(class_name):
    """Resolve a disease name case-insensitively against the authoritative CLASSES list."""
    lookup = {
        disease.lower(): index
        for index, disease in enumerate(CLASSES)
    }

    requested = class_name.strip().lower()

    if requested not in lookup:
        available = ", ".join(CLASSES)
        raise ValueError(
            f"Unknown disease class: {class_name}\n"
            f"Available classes: {available}"
        )

    return lookup[requested]


def select_explanation_indices(
    raw_scores,
    predictions,
    explicit_class_name=None,
    explain_all_positive=False,
):
    """
    Select the disease(s) for Module 5/6.

    Default behavior matches the existing Grad-CAM utility: explain the
    highest raw-probability class. If --explain-all-positive is requested,
    analyze every class crossing its frozen raw threshold, ordered by raw
    probability. If none are positive, fall back to the highest raw class.
    """
    if explicit_class_name is not None and explain_all_positive:
        raise ValueError(
            "--explain-disease and --explain-all-positive cannot be used together."
        )

    if explicit_class_name is not None:
        return [resolve_class_index(explicit_class_name)]

    if explain_all_positive:
        positive_indices = np.where(predictions == 1)[0].tolist()
        if positive_indices:
            positive_indices.sort(
                key=lambda i: float(raw_scores[i]),
                reverse=True,
            )
            return positive_indices

    return [int(np.argmax(raw_scores))]


# =========================================================
# MODULE 5 — GRAD-CAM
# =========================================================


def generate_gradcam(model, image_tensor, class_index):
    """Run the existing GradCAM implementation without duplicating its algorithm."""
    target_layer = getattr(model.features, "norm5", None)
    if target_layer is None:
        raise RuntimeError(
            "DenseNet target layer model.features.norm5 was not found."
        )

    with GradCAM(
        model=model,
        target_layer=target_layer,
    ) as gradcam:
        cam, gradcam_logits = gradcam.generate(
            input_tensor=image_tensor,
            class_index=class_index,
        )

    return cam, gradcam_logits


def save_gradcam_figure(
    image,
    cam,
    class_name,
    raw_probability,
    output_dir,
    image_stem,
):
    """Save the standard three-panel Grad-CAM visualization."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    display_image = image.resize((224, 224))
    display_array = np.asarray(display_image).astype(np.float32) / 255.0

    safe_class_name = (
        class_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    output_path = (
        output_dir
        / f"{image_stem}__{safe_class_name}__gradcam.png"
    )

    cam_array = cam.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(display_array, cmap="gray")
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    heatmap = axes[1].imshow(
        cam_array,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(f"Grad-CAM\n{class_name}")
    axes[1].axis("off")

    fig.colorbar(
        heatmap,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )

    axes[2].imshow(display_array)
    axes[2].imshow(
        cam_array,
        cmap="jet",
        alpha=0.40,
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].set_title(
        f"{class_name}\nRaw probability: {raw_probability:.4f}"
    )
    axes[2].axis("off")

    fig.suptitle(
        "MedTrust — Disease-Specific Grad-CAM",
        fontsize=16,
    )
    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path


# =========================================================
# MODULE 6 — EXPLANATION RELIABILITY
# =========================================================


def run_reliability_analysis(model, image_path, class_index, device):
    """Run the existing Module 6 analyzer for one selected disease."""
    analyzer = ExplanationReliabilityAnalyzer(
        model=model,
        target_layer=model.features.norm5,
        device=device,
        random_seed=42,
        occlusion_fraction=0.20,
        brightness_factor=1.05,
        contrast_factor=1.05,
        translation_pixels=4,
        noise_std=0.01,
        high_threshold=0.90,
        moderate_threshold=0.75,
    )

    return analyzer.analyze(
        image_input=image_path,
        class_index=class_index,
    )


# =========================================================
# MODULE 9 — CALIBRATION ANALYZER
# =========================================================


def run_calibration_analysis(calibrated_scores):
    """Run the existing Module 9 diagnostic against frozen validation bins."""
    analyzer = CalibrationAnalyzer(
        bins_csv_path=CALIBRATION_BINS_PATH,
        state="after",
        warning_threshold=0.10,
        high_discrepancy_threshold=0.20,
    )

    return analyzer.analyze_all(calibrated_scores)


# =========================================================
# PRINTING
# =========================================================


def print_prediction_results(
    raw_scores,
    calibrated_scores,
    temperatures,
    thresholds,
    predictions,
):
    print()
    print("=" * 112)
    print("PREDICTION RESULTS")
    print("=" * 112)
    print(
        f"{'Disease':<24}"
        f"{'Raw':>10}"
        f"{'Calibrated':>14}"
        f"{'Temp.':>10}"
        f"{'Threshold':>12}"
        f"{'Decision':>14}"
    )
    print("-" * 112)

    for (
        disease,
        raw_score,
        calibrated_score,
        temperature,
        threshold,
        prediction,
    ) in zip(
        CLASSES,
        raw_scores,
        calibrated_scores,
        temperatures,
        thresholds,
        predictions,
    ):
        decision = "POSITIVE" if prediction == 1 else "Negative"
        print(
            f"{disease:<24}"
            f"{raw_score:>10.4f}"
            f"{calibrated_score:>14.4f}"
            f"{temperature:>10.4f}"
            f"{threshold:>12.4f}"
            f"{decision:>14}"
        )


def print_top_prediction(raw_scores, calibrated_scores):
    raw_top_index = int(np.argmax(raw_scores))
    calibrated_top_index = int(np.argmax(calibrated_scores))

    print()
    print("=" * 112)
    print("TOP PREDICTION")
    print("=" * 112)
    print(
        f"Raw        : {CLASSES[raw_top_index]} "
        f"({raw_scores[raw_top_index]:.4f})"
    )
    print(
        f"Calibrated : {CLASSES[calibrated_top_index]} "
        f"({calibrated_scores[calibrated_top_index]:.4f})"
    )


def print_predicted_findings(raw_scores, calibrated_scores, thresholds, predictions):
    positive_indices = np.where(predictions == 1)[0]

    print()
    print("=" * 112)
    print("PREDICTED FINDINGS")
    print("=" * 112)

    if len(positive_indices) == 0:
        print(
            "No disease label crossed its frozen raw-probability decision threshold."
        )
        return

    positive_indices = sorted(
        positive_indices,
        key=lambda i: float(raw_scores[i]),
        reverse=True,
    )

    for index in positive_indices:
        print(
            f"{CLASSES[index]:<24}"
            f"raw={raw_scores[index]:.4f}  "
            f"calibrated={calibrated_scores[index]:.4f}  "
            f"threshold={thresholds[index]:.4f}"
        )


def print_calibration_analysis(calibration_result):
    print()
    print("=" * 112)
    print("MODULE 9 — CALIBRATION ANALYZER")
    print("=" * 112)
    print(
        f"State                  : {calibration_result['calibration_state']}"
    )
    print(
        f"Diseases analyzed      : {calibration_result['total_diseases_analyzed']}"
    )
    print(
        "Mean absolute gap      : "
        f"{calibration_result['mean_absolute_calibration_gap']:.4f}"
        if calibration_result["mean_absolute_calibration_gap"] is not None
        else "Mean absolute gap      : N/A"
    )
    print(
        "Maximum absolute gap   : "
        f"{calibration_result['max_absolute_calibration_gap']:.4f}"
        if calibration_result["max_absolute_calibration_gap"] is not None
        else "Maximum absolute gap   : N/A"
    )

    high_diseases = calibration_result["high_discrepancy_diseases"]
    print(
        "High-discrepancy labels : "
        + (", ".join(high_diseases) if high_diseases else "None")
    )

    print()
    print(
        f"{'Disease':<24}"
        f"{'Prob.':>10}"
        f"{'Bin':>7}"
        f"{'Val. N':>9}"
        f"{'Empirical':>12}"
        f"{'Gap':>10}"
        f"{'Status':>30}"
    )
    print("-" * 112)

    for result in calibration_result["findings"]:
        gap = result["absolute_calibration_gap"]
        gap_text = f"{gap:.4f}" if gap is not None else "N/A"
        print(
            f"{result['disease']:<24}"
            f"{result['probability']:>10.4f}"
            f"{result['bin_index']:>7}"
            f"{result['bin_sample_count']:>9}"
            f"{result['empirical_positive_rate']:>12.4f}"
            f"{gap_text:>10}"
            f"{result['status']:>30}"
        )


def print_ood_results(ood_res):
    """Print Module 7 Out-of-Distribution (OOD) detection results."""
    print()
    print("Individual Detector Results")
    print("-" * 78)
    for method, item in ood_res["method_results"].items():
        print(
            f"{method:<16}"
            f"Score: {item['score']:>12.6f}  "
            f"Threshold: {item['threshold']:>12.6f}  "
            f"Status: {item['status']}"
        )
    print("-" * 78)
    print("Combined policy  : OOD if ANY detector flags OOD")
    print(f"Status          : {ood_res['status']}")
    print(f"Trust Decision  : {ood_res['trust_decision']}")


def print_uncertainty_results(result, passes, dropout_rate):
    """Print Module 6 Prediction Uncertainty (MC-Dropout) analysis."""
    print()
    print("=" * 112)
    print("MODULE 6 — PREDICTION UNCERTAINTY (MC-DROPOUT)")
    print("=" * 112)
    print(f"MC-Dropout passes   : {passes}")
    print(f"Feature dropout rate: {dropout_rate:.2f}")
    print()
    print(
        f"{'Disease':<24}"
        f"{'Mean Prob.':>14}"
        f"{'Variance':>14}"
        f"{'Std Dev':>14}"
        f"{'Entropy':>14}"
    )
    print("-" * 112)

    for item in result["diseases"]:
        print(
            f"{item.disease:<24}"
            f"{item.probability:>14.4f}"
            f"{item.variance:>14.6f}"
            f"{item.standard_deviation:>14.4f}"
            f"{item.entropy:>14.4f}"
        )

    print("-" * 112)
    print(
        f"Overall mean variance  : {result['overall_mean_variance']:.6f}"
    )
    print(
        f"Overall max variance   : {result['overall_max_variance']:.6f}"
    )
    print(
        f"Most uncertain disease : {result['most_uncertain_disease']}"
    )
    print(
        f"Highest uncertainty    : {result['most_uncertain_variance']:.6f}"
    )


def print_gradcam_and_reliability_result(
    class_index,
    cam,
    gradcam_output_path,
    reliability_result,
):
    disease = CLASSES[class_index]

    cam_array = cam.detach().cpu().numpy()

    print()
    print("=" * 112)
    print(f"MODULE 5 — GRAD-CAM: {disease}")
    print("=" * 112)
    print("Target layer : model.features.norm5")
    print(f"Class index  : {class_index}")
    print(f"CAM shape    : {cam_array.shape}")
    print(f"CAM min      : {cam_array.min():.6f}")
    print(f"CAM max      : {cam_array.max():.6f}")
    print(f"CAM mean     : {cam_array.mean():.6f}")
    print(f"Saved        : {gradcam_output_path}")

    print()
    print("=" * 112)
    print(f"MODULE 6 — EXPLANATION RELIABILITY: {disease}")
    print("=" * 112)
    print(
        f"Overall stability       : {reliability_result['overall_stability']}"
    )
    print(
        f"Mean Pearson            : {reliability_result['mean_pearson_correlation']:.4f}"
    )
    print(
        f"Min Pearson             : {reliability_result['min_pearson_correlation']:.4f}"
    )
    print(
        f"Mean Spearman           : {reliability_result['mean_spearman_correlation']:.4f}"
    )
    print(
        f"Min Spearman            : {reliability_result['min_spearman_correlation']:.4f}"
    )
    print(
        f"Mean MAD                : {reliability_result['mean_absolute_difference']:.4f}"
    )
    print(
        f"Max MAD                 : {reliability_result['max_absolute_difference']:.4f}"
    )
    print(
        f"Mean top-20% region IoU : {reliability_result['mean_region_iou']:.4f}"
    )
    print(
        f"Min top-20% region IoU  : {reliability_result['min_region_iou']:.4f}"
    )
    print(
        f"Fidelity before         : {reliability_result['fidelity_probability_before']:.4f}"
    )
    print(
        f"Fidelity after          : {reliability_result['fidelity_probability_after']:.4f}"
    )
    print(
        f"Fidelity probability drop: {reliability_result['fidelity_probability_drop']:.4f}"
    )
    print(
        f"Fidelity verified       : {reliability_result['fidelity_verified']}"
    )

    print()
    print(f"{'Perturbation':<20}{'Pearson':>10}{'Spearman':>12}{'MAD':>10}{'IoU':>10}{'Band':>14}")
    print("-" * 80)
    for detail in reliability_result["perturbation_details"]:
        print(
            f"{detail['perturbation']:<20}"
            f"{detail['pearson_correlation']:>10.4f}"
            f"{detail['spearman_correlation']:>12.4f}"
            f"{detail['mean_absolute_difference']:>10.4f}"
            f"{detail['region_iou']:>10.4f}"
            f"{detail['stability_band']:>14}"
        )


def print_final_notes():
    print()
    print("=" * 112)
    print("MEDTRUST RUNTIME INTERPRETATION")
    print("=" * 112)
    print(
        "Decision thresholds applied to : RAW probabilities"
    )
    print(
        "Reported calibrated probability : sigmoid(logit / temperature)"
    )
    print(
        "Calibration parameters           : FROZEN validation-derived values"
    )
    print(
        "Calibration refitted at runtime : False"
    )
    print(
        "Module 9                       : diagnostic lookup only; no model modification"
    )
    print(
        "Module 6 (Uncertainty)         : MC-Dropout epistemic variability diagnostic only"
    )
    print(
        "Explanation Reliability        : perturbation stability/fidelity diagnostic only"
    )
    print()
    print(
        "Calibrated probabilities are model probability estimates after temperature "
        "scaling; they are not guarantees of disease presence."
    )
    print(
        "HIGH_DISCREPANCY_WARNING is a calibration diagnostic, not a medical diagnosis "
        "or a statement that the model prediction is clinically wrong."
    )
    print(
        "Grad-CAM highlights image regions associated with the selected model output; "
        "it does not establish clinical localization or diagnostic truth."
    )
    print(
        "This research model output must not be interpreted as a clinical diagnosis."
    )
    print("=" * 112)


def print_similarity_results(result: Dict[str, Any]) -> None:
    print()
    print("=" * 112)
    print("MODULE 8: CASE-BASED SIMILARITY ANALYSIS (DenseNet-121 1024-d Cosine Proximity)")
    print("=" * 112)
    print(f"Total reference cases indexed : {result['total_references']}")
    print(f"Top-K retrieved               : {result['top_k']}")
    if "primary_concordance_rate" in result:
        print(f"Primary Concordance (C_d)     : {result['primary_concordance_rate']:.1f}% ({result.get('primary_concordant_count', 0)} of {result['top_k']} cases)")
    if "any_finding_overlap_rate" in result:
        print(f"Any-Finding Overlap (O)       : {result['any_finding_overlap_rate']:.1f}% ({result.get('any_finding_overlap_count', 0)} of {result['top_k']} cases)")
    if "concordance_summary" in result and result["concordance_summary"]:
        print(f"Assessment Summary            : {result['concordance_summary']}")
    print("-" * 112)
    print(f"{'Rank':<6} {'Reference Case':<22} {'Score':<10} {'Dataset Finding(s)':<30} {'Concordance Status'}")
    print("-" * 112)
    for match in result.get("matches", []):
        meta = match.get("metadata", {}) or {}
        findings_str = meta.get("finding_labels", "N/A")
        conc = match.get("concordance", {}) or {}
        conc_label = conc.get("label", conc.get("status", "N/A"))
        print(
            f"#{match['rank']:<5} {match['filename']:<22} {match['similarity_score']:>6.2f}%   "
            f"{findings_str:<30} {conc_label}"
        )
    print("-" * 112)
    # Print Top Retrieved Label Support vs Global Dataset Prevalence
    weighted_support = result.get("weighted_label_support", [])
    if weighted_support:
        active_support = [s for s in weighted_support if s["retrieved_count"] > 0][:5]
        if active_support:
            print("Top-K Retrieved Disease Label Support (vs Global NIH Dataset Prevalence):")
            print(f"{'Disease Class':<24} {'Weighted Support':>18} {'Retrieved Count':>16} {'Dataset Prevalence':>20}")
            print("-" * 112)
            for s in active_support:
                prev_str = f"{s['dataset_prevalence']:.2f}%" if s.get('dataset_prevalence') is not None else "N/A"
                print(f"{s['disease']:<24} {s['weighted_support']:>17.2f}% {s['retrieved_count']:>11} of {s['top_k']} {prev_str:>20}")
            print("-" * 112)
    print(f"Notice: {result.get('disclaimer', '')}")
    print("=" * 112)


# =========================================================
# MAIN
# =========================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "MedTrust V1 single chest X-ray inference pipeline: "
            "prediction + frozen calibration + Grad-CAM + explanation reliability + calibration analysis."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to PNG/JPG chest X-ray",
    )

    parser.add_argument(
        "--explain-disease",
        type=str,
        default=None,
        help=(
            "Disease to explain with Grad-CAM and Module 6 reliability. "
            "If omitted, the highest raw-probability class is used."
        ),
    )

    parser.add_argument(
        "--explain-all-positive",
        action="store_true",
        help=(
            "Run Grad-CAM and Module 6 reliability for every disease "
            "crossing its frozen raw threshold. If none are positive, "
            "fall back to the highest raw-probability class."
        ),
    )

    parser.add_argument(
        "--no-save-gradcam",
        action="store_true",
        help="Run Grad-CAM but do not save visualization PNG files.",
    )

    parser.add_argument(
        "--mc-passes",
        type=int,
        default=30,
        help="Number of stochastic MC-Dropout passes for uncertainty (default: 30).",
    )

    parser.add_argument(
        "--mc-dropout-rate",
        type=float,
        default=0.20,
        help="Feature dropout rate for MC-Dropout (default: 0.20).",
    )

    parser.add_argument(
        "--skip-uncertainty",
        action="store_true",
        help="Skip MC-Dropout prediction uncertainty estimation.",
    )

    parser.add_argument(
        "--gradcam-output-dir",
        type=str,
        default=str(GRADCAM_OUTPUT_DIR),
        help="Directory for Grad-CAM visualization PNG files.",
    )

    parser.add_argument(
        "--top-k-similar",
        type=int,
        default=5,
        help="Number of similar reference cases to retrieve (default: 5).",
    )

    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip Module 8 case-based similarity retrieval.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 112)
    print("MEDTRUST — TRUST-AWARE SINGLE X-RAY INFERENCE")
    print("=" * 112)
    print(f"Image       : {image_path}")
    print(f"Device      : {device}")

    if torch.cuda.is_available():
        print(
            f"GPU         : {torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------------------
    # Load frozen artifacts and model
    # -----------------------------------------------------

    model, checkpoint = load_model(device)
    thresholds = load_thresholds()
    temperatures = load_temperatures()

    print(f"Checkpoint  : {CHECKPOINT_PATH}")
    print(f"Best epoch  : {checkpoint.get('epoch', 'unknown')}")
    print(f"Thresholds  : {THRESHOLDS_PATH}")
    print("Threshold source : validation-only Youden J")
    print(f"Calibration : {TEMPERATURE_PATH}")
    print("Calibration source: validation-only per-disease temperature scaling")
    print("Calibration status: FROZEN")

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    image, image_tensor = prepare_image(
        image_path,
        device,
    )

    print(f"Input shape : {tuple(image_tensor.shape)}")

    # -----------------------------------------------------
    # Module 5: Technical Image Quality Analysis
    # -----------------------------------------------------

    quality_analyzer = WeightedImageQualityAnalyzer()
    quality_result = quality_analyzer.analyze(image)
    print()
    print_quality_report(quality_result, detailed=True)

    # -----------------------------------------------------
    # Modules 3 + 4: prediction + frozen calibration
    # -----------------------------------------------------

    (
        logits,
        raw_scores,
        calibrated_scores,
        predictions,
    ) = run_prediction(
        model=model,
        image_tensor=image_tensor,
        temperatures=temperatures,
        thresholds=thresholds,
    )

    # -----------------------------------------------------
    # Standard prediction output
    # -----------------------------------------------------

    print_prediction_results(
        raw_scores,
        calibrated_scores,
        temperatures,
        thresholds,
        predictions,
    )

    print_top_prediction(
        raw_scores,
        calibrated_scores,
    )

    print_predicted_findings(
        raw_scores,
        calibrated_scores,
        thresholds,
        predictions,
    )

    # -----------------------------------------------------
    # Module 9: calibration diagnostic for ALL 14 diseases
    # -----------------------------------------------------

    calibration_result = run_calibration_analysis(
        calibrated_scores
    )

    print_calibration_analysis(
        calibration_result
    )

    # -----------------------------------------------------
    # Module 7: Out-of-Distribution (OOD) Detection
    # -----------------------------------------------------

    with torch.inference_mode():
        features_map = model.features(image_tensor)
        features_map = torch.nn.functional.relu(features_map, inplace=False)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features_map, (1, 1))
        extracted_features = torch.flatten(pooled, 1)

    ood_res = predict_all_methods(
        logits=logits.detach().cpu().numpy(),
        features=extracted_features.detach().cpu().numpy(),
    )
    print_ood_results(ood_res)

    # -----------------------------------------------------
    # Module 6: Prediction Uncertainty (MC-Dropout)
    # -----------------------------------------------------

    if not args.skip_uncertainty:
        uncertainty_predictor = MCDropoutPredictor(
            model=model,
            passes=args.mc_passes,
            dropout_rate=args.mc_dropout_rate,
            device=device,
        )
        mc_result = uncertainty_predictor.predict(image_tensor)
        mc_samples = (
            mc_result.probabilities[:, 0, :]
            .detach()
            .cpu()
            .numpy()
        )
        uncertainty_analyzer = UncertaintyAnalyzer(CLASSES)
        uncertainty_result = uncertainty_analyzer.analyze(mc_samples)
        print_uncertainty_results(
            uncertainty_result,
            passes=args.mc_passes,
            dropout_rate=args.mc_dropout_rate,
        )

    # -----------------------------------------------------
    # Modules 5 + 6: selected explanation target(s)
    # -----------------------------------------------------

    explanation_indices = select_explanation_indices(
        raw_scores=raw_scores,
        predictions=predictions,
        explicit_class_name=args.explain_disease,
        explain_all_positive=args.explain_all_positive,
    )

    if args.explain_all_positive and not np.any(predictions == 1):
        print()
        print(
            "No raw-threshold-positive class was found; "
            "Module 5/6 falls back to the highest raw-probability class."
        )

    output_dir = Path(args.gradcam_output_dir)

    for class_index in explanation_indices:
        class_name = CLASSES[class_index]

        print()
        print("=" * 112)
        print(f"RUNNING MODULES 5 + 6 FOR: {class_name}")
        print("=" * 112)

        # Module 5 — existing GradCAM implementation
        cam, gradcam_logits = generate_gradcam(
            model=model,
            image_tensor=image_tensor,
            class_index=class_index,
        )

        # Confirm Grad-CAM's target logit corresponds to the raw probability
        # from the main inference pass.
        gradcam_raw_probability = float(
            torch.sigmoid(gradcam_logits[0, class_index]).item()
        )
        main_raw_probability = float(raw_scores[class_index])

        if not np.isclose(
            gradcam_raw_probability,
            main_raw_probability,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Grad-CAM target probability does not match the main inference "
                f"pass for {class_name}: "
                f"main={main_raw_probability:.8f}, "
                f"gradcam={gradcam_raw_probability:.8f}"
            )

        # Module 6 — existing analyzer; internally reuses GradCAM.
        reliability_result = run_reliability_analysis(
            model=model,
            image_path=image_path,
            class_index=class_index,
            device=device,
        )

        # Save standard visualization unless explicitly disabled.
        if args.no_save_gradcam:
            gradcam_output_path = "Not saved (--no-save-gradcam)"
        else:
            gradcam_output_path = save_gradcam_figure(
                image=image,
                cam=cam,
                class_name=class_name,
                raw_probability=main_raw_probability,
                output_dir=output_dir,
                image_stem=image_path.stem,
            )

        print_gradcam_and_reliability_result(
            class_index=class_index,
            cam=cam,
            gradcam_output_path=gradcam_output_path,
            reliability_result=reliability_result,
        )

    # -----------------------------------------------------
    # Module 8: Case-Based Similarity Analysis
    # -----------------------------------------------------

    if not args.skip_similarity:
        try:
            from src.similarity import SimilarityAnalyzer

            primary_idx = int(np.argmax(raw_scores))
            primary_disease = CLASSES[primary_idx]
            detected_findings = [CLASSES[i] for i in range(len(CLASSES)) if predictions[i] == 1]

            similarity_analyzer = SimilarityAnalyzer(model=model, device=device)
            sim_result = similarity_analyzer.find_similar_cases(
                image,
                top_k=args.top_k_similar,
                query_findings=detected_findings,
                primary_finding=primary_disease,
                exclude_filenames=[Path(args.image).name],
            )
            print_similarity_results(sim_result)
        except Exception as e:
            print(f"\n[!] Similarity analysis skipped: {e}")

    # -----------------------------------------------------
    # Final interpretation / audit note
    # -----------------------------------------------------

    print_final_notes()


if __name__ == "__main__":
    main()