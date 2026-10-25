
"""
Standalone OOD inference for a single image.

Supported methods:
    msp
    energy
    mahalanobis
    combined

Combined mode uses a conservative policy:
    If ANY detector flags the image as OOD, the combined
    decision is OOD.

Important:
    Thresholds are calibrated using ID validation data.
    Combined mode must be evaluated separately to measure
    its actual false-positive and OOD detection rates.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .common import load_json
from .config import ARTIFACT_OUTPUT_DIR, NUM_CLASSES
from .model_adapter import MedTrustModelAdapter
from .scores import (
    msp_score,
    energy_score,
    mahalanobis_score,
)
from .threshold import THRESHOLD_OUTPUT_PATH

from src.dataset import CLASSES
from src.transformer import eval_transform


MAHALANOBIS_PATH = (
    ARTIFACT_OUTPUT_DIR / "mahalanobis_statistics.npz"
)

SUPPORTED_METHODS = {
    "msp",
    "energy",
    "mahalanobis",
    "combined",
}


def load_image(image_path):
    """Load and preprocess an image using MedTrust's evaluation transform."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Input path is not a file: {image_path}"
        )

    with Image.open(image_path) as source:
        image = source.convert("L")

    tensor = eval_transform(image)
    return tensor.unsqueeze(0)


def load_threshold(method):
    """Load a validation-calibrated threshold for one detector."""

    if not THRESHOLD_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            "OOD threshold artifact not found:\n"
            f"{THRESHOLD_OUTPUT_PATH}\n"
            "Run the OOD threshold calibration first."
        )

    data = load_json(THRESHOLD_OUTPUT_PATH)
    thresholds = data.get("thresholds", {})

    if method not in thresholds:
        raise KeyError(
            f"No threshold found for method: {method}"
        )

    if "threshold" not in thresholds[method]:
        raise KeyError(
            f"Threshold value missing for method: {method}"
        )

    return float(thresholds[method]["threshold"])


def load_mahalanobis_statistics():
    """Load the frozen Mahalanobis reference statistics."""

    if not MAHALANOBIS_PATH.exists():
        raise FileNotFoundError(
            "Mahalanobis statistics not found:\n"
            f"{MAHALANOBIS_PATH}\n"
            "Run the ID feature-statistics calibration first."
        )

    with np.load(
        MAHALANOBIS_PATH,
        allow_pickle=False,
    ) as data:
        required = {"mean", "safe_variance"}
        missing = required - set(data.files)

        if missing:
            raise RuntimeError(
                "Mahalanobis artifact is missing: "
                f"{sorted(missing)}"
            )

        mean = np.asarray(
            data["mean"],
            dtype=np.float64,
        )

        safe_variance = np.asarray(
            data["safe_variance"],
            dtype=np.float64,
        )

    return mean, safe_variance


def calculate_score(method, logits, features, mahalanobis_stats=None):
    """Calculate one OOD score. Higher scores are treated as more OOD-like."""

    if method == "msp":
        score = msp_score(logits)[0]

    elif method == "energy":
        score = energy_score(logits)[0]

    elif method == "mahalanobis":
        if mahalanobis_stats is None:
            mahalanobis_stats = load_mahalanobis_statistics()

        mean, safe_variance = mahalanobis_stats

        score = mahalanobis_score(
            features=features,
            mean=mean,
            safe_variance=safe_variance,
        )[0]

    else:
        raise ValueError(
            f"Unsupported individual OOD method: {method}"
        )

    score = float(score)

    if not np.isfinite(score):
        raise ValueError(
            f"{method} produced a non-finite score: {score}"
        )

    return score


def predict_all_methods(logits, features):
    """
    Evaluate MSP, Energy, and Mahalanobis together.

    Conservative combined policy:
    flag the image if at least one detector flags it.

    This policy is not a guarantee of improved detection
    accuracy and requires independent evaluation.
    """

    mahalanobis_stats = load_mahalanobis_statistics()
    results = {}

    for method in ("msp", "energy", "mahalanobis"):
        score = calculate_score(
            method=method,
            logits=logits,
            features=features,
            mahalanobis_stats=mahalanobis_stats,
        )

        threshold = load_threshold(method)
        is_ood = score >= threshold

        results[method] = {
            "score": score,
            "threshold": threshold,
            "is_ood": bool(is_ood),
            "status": "OOD" if is_ood else "ID",
        }

    combined_is_ood = any(
        item["is_ood"]
        for item in results.values()
    )

    return {
        "method_results": results,
        "is_ood": bool(combined_is_ood),
        "status": "OOD" if combined_is_ood else "ID",
        "trust_decision": (
            "DO NOT TRUST PREDICTION"
            if combined_is_ood
            else "PREDICTION CAN BE CONSIDERED"
        ),
    }


def extract_prediction(probabilities):
    """Extract multi-label predictions using a 0.5 probability cutoff."""

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "Probabilities must be 2-dimensional."
        )

    if probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(
            "Unexpected number of disease classes."
        )

    if len(CLASSES) != NUM_CLASSES:
        raise ValueError(
            "CLASSES length does not match NUM_CLASSES."
        )

    predictions = []

    for class_index, probability in enumerate(probabilities[0]):
        if probability >= 0.5:
            predictions.append({
                "disease": CLASSES[class_index],
                "probability": float(probability),
            })

    if not predictions:
        best_index = int(np.argmax(probabilities[0]))

        predictions.append({
            "disease": CLASSES[best_index],
            "probability": float(
                probabilities[0, best_index]
            ),
        })

    predictions.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    return predictions


def predict_single_image(image_path, method="mahalanobis"):
    """Perform single-image OOD inference."""

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            "method must be one of: "
            "msp, energy, mahalanobis, combined"
        )

    image_tensor = load_image(image_path)

    model = MedTrustModelAdapter()
    result = model.predict(image_tensor)

    features = (
        result["features"]
        .detach()
        .cpu()
        .numpy()
    )

    logits = (
        result["logits"]
        .detach()
        .cpu()
        .numpy()
    )

    probabilities = (
        result["probabilities"]
        .detach()
        .cpu()
        .numpy()
    )

    predictions = extract_prediction(probabilities)
    top_prediction = predictions[0]

    output = {
        "image": str(Path(image_path)),
        "method": method,
        "top_prediction": top_prediction["disease"],
        "top_probability": top_prediction["probability"],
        "predictions": predictions,
    }

    if method == "combined":
        combined = predict_all_methods(
            logits=logits,
            features=features,
        )

        output.update(combined)
        output["ood_score"] = None
        output["threshold"] = None

    else:
        score = calculate_score(
            method=method,
            logits=logits,
            features=features,
        )

        threshold = load_threshold(method)
        is_ood = score >= threshold

        output.update({
            "ood_score": score,
            "threshold": threshold,
            "is_ood": bool(is_ood),
            "status": "OOD" if is_ood else "ID",
            "trust_decision": (
                "DO NOT TRUST PREDICTION"
                if is_ood
                else "PREDICTION CAN BE CONSIDERED"
            ),
        })

    return output


def print_result(result):
    """Print a human-readable inference report."""

    print()
    print("=" * 78)
    print("MEDTRUST OOD INFERENCE RESULT")
    print("=" * 78)
    print(f"Image           : {result['image']}")
    print(f"Method          : {result['method']}")

    if result["method"] == "combined":
        print()
        print("Individual Detector Results")
        print("-" * 78)

        for method, item in result["method_results"].items():
            print(
                f"{method:<16}"
                f"Score: {item['score']:>12.6f}  "
                f"Threshold: {item['threshold']:>12.6f}  "
                f"Status: {item['status']}"
            )

        print("-" * 78)
        print("Combined policy  : OOD if ANY detector flags OOD")

    else:
        print(f"OOD Score       : {result['ood_score']:.6f}")
        print(f"OOD Threshold   : {result['threshold']:.6f}")

    print(f"Status          : {result['status']}")
    print(f"Trust Decision  : {result['trust_decision']}")
    print()
    print("Disease Predictions")
    print("-" * 78)

    for prediction in result["predictions"]:
        print(
            f"{prediction['disease']:<28}"
            f"{prediction['probability']:.4f}"
        )

    print("=" * 78)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MedTrust standalone OOD inference"
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="mahalanobis",
        choices=sorted(SUPPORTED_METHODS),
        help="OOD detection method or combined decision.",
    )

    args = parser.parse_args()

    result = predict_single_image(
        image_path=args.image,
        method=args.method,
    )

    print_result(result)


if __name__ == "__main__":
    main()
