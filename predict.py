import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import CLASSES
from src.model import create_model
from src.transformer import test_transform


CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
THRESHOLDS_PATH = Path("outputs/evaluation/thresholds.json")
TEMPERATURE_PATH = Path("outputs/calibration/temperature.json")


# =========================================================
# LOAD FROZEN THRESHOLDS
# =========================================================

def load_thresholds():

    with open(
        THRESHOLDS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if data["classes"] != list(CLASSES):
        raise RuntimeError(
            "Threshold class order does not match model classes."
        )

    thresholds = np.array(
        [
            data["thresholds"][disease]["threshold"]
            for disease in CLASSES
        ],
        dtype=np.float32,
    )

    return thresholds


# =========================================================
# LOAD FROZEN CALIBRATION TEMPERATURES
# =========================================================

def load_temperatures():

    if not TEMPERATURE_PATH.exists():
        raise FileNotFoundError(
            f"Calibration artifact not found: {TEMPERATURE_PATH}"
        )

    with open(
        TEMPERATURE_PATH,
        "r",
        encoding="utf-8",
    ) as f:
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
# LOAD MODEL
# =========================================================

def load_model(device):

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
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
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run calibrated MedTrust prediction "
            "on one chest X-ray."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to PNG/JPG chest X-ray",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 96)
    print("MEDTRUST — CALIBRATED SINGLE X-RAY PREDICTION")
    print("=" * 96)

    print(f"Image       : {image_path}")
    print(f"Device      : {device}")

    if torch.cuda.is_available():
        print(
            f"GPU         : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model, checkpoint = load_model(
        device
    )

    print(
        f"Checkpoint  : {CHECKPOINT_PATH}"
    )

    print(
        f"Best epoch  : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    # -----------------------------------------------------
    # Load frozen thresholds
    # -----------------------------------------------------

    thresholds = load_thresholds()

    print(
        f"Thresholds  : {THRESHOLDS_PATH}"
    )

    print(
        "Threshold source : validation-only Youden J"
    )

    # -----------------------------------------------------
    # Load frozen calibration
    # -----------------------------------------------------

    temperatures = load_temperatures()

    print(
        f"Calibration : {TEMPERATURE_PATH}"
    )

    print(
        "Calibration source: validation-only "
        "per-disease temperature scaling"
    )

    print(
        "Calibration status: FROZEN"
    )

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = test_transform(
        image
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    print(
        f"Input shape : "
        f"{tuple(image_tensor.shape)}"
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    with torch.inference_mode():

        logits = model(
            image_tensor
        )

        raw_probabilities = torch.sigmoid(
            logits
        )

        temperature_tensor = torch.tensor(
            temperatures,
            dtype=logits.dtype,
            device=logits.device,
        ).unsqueeze(0)

        calibrated_logits = (
            logits
            / temperature_tensor
        )

        calibrated_probabilities = (
            torch.sigmoid(
                calibrated_logits
            )
        )

    raw_scores = (
        raw_probabilities
        .squeeze(0)
        .cpu()
        .numpy()
    )

    calibrated_scores = (
        calibrated_probabilities
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # IMPORTANT:
    # Existing thresholds were selected using RAW
    # validation probabilities.
    #
    # Therefore decisions remain based on raw scores.
    predictions = (
        raw_scores >= thresholds
    ).astype(np.int32)

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print()
    print("=" * 96)
    print("PREDICTION RESULTS")
    print("=" * 96)

    print(
        f"{'Disease':<24}"
        f"{'Raw':>10}"
        f"{'Calibrated':>14}"
        f"{'Temp.':>10}"
        f"{'Threshold':>12}"
        f"{'Decision':>14}"
    )

    print("-" * 96)

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

        decision = (
            "POSITIVE"
            if prediction == 1
            else "Negative"
        )

        print(
            f"{disease:<24}"
            f"{raw_score:>10.4f}"
            f"{calibrated_score:>14.4f}"
            f"{temperature:>10.4f}"
            f"{threshold:>12.4f}"
            f"{decision:>14}"
        )

    # -----------------------------------------------------
    # Top predictions
    # -----------------------------------------------------

    raw_top_index = int(
        np.argmax(raw_scores)
    )

    calibrated_top_index = int(
        np.argmax(calibrated_scores)
    )

    print()
    print("=" * 96)
    print("TOP PREDICTION")
    print("=" * 96)

    print(
        f"Raw        : "
        f"{CLASSES[raw_top_index]} "
        f"({raw_scores[raw_top_index]:.4f})"
    )

    print(
        f"Calibrated : "
        f"{CLASSES[calibrated_top_index]} "
        f"({calibrated_scores[calibrated_top_index]:.4f})"
    )

    # -----------------------------------------------------
    # Positive findings
    # -----------------------------------------------------

    positive_indices = np.where(
        predictions == 1
    )[0]

    print()
    print("=" * 96)
    print("PREDICTED FINDINGS")
    print("=" * 96)

    if len(positive_indices) == 0:

        print(
            "No disease label crossed its "
            "frozen raw-probability decision threshold."
        )

    else:

        positive_indices = sorted(
            positive_indices,
            key=lambda i: calibrated_scores[i],
            reverse=True,
        )

        for index in positive_indices:

            print(
                f"{CLASSES[index]:<24}"
                f"raw={raw_scores[index]:.4f}  "
                f"calibrated="
                f"{calibrated_scores[index]:.4f}  "
                f"threshold="
                f"{thresholds[index]:.4f}"
            )

    # -----------------------------------------------------
    # Calibration sanity information
    # -----------------------------------------------------

    print()
    print("=" * 96)
    print("CALIBRATION CHECK")
    print("=" * 96)

    print(
        "Decision thresholds applied to : "
        "RAW probabilities"
    )

    print(
        "Reported calibrated probability : "
        "sigmoid(logit / temperature)"
    )

    print(
        "Temperature parameters          : "
        "FROZEN validation-derived values"
    )

    print(
        "Calibration refitted            : False"
    )

    print()
    print("NOTE:")

    print(
        "Calibrated probabilities are model probability "
        "estimates after temperature scaling; they are "
        "not guarantees of disease presence."
    )

    print(
        "POSITIVE/Negative decisions currently use the "
        "original frozen thresholds selected in the raw "
        "probability space."
    )

    print(
        "This research model output must not be interpreted "
        "as a clinical diagnosis."
    )

    print("=" * 96)


if __name__ == "__main__":
    main()