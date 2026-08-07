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


def load_thresholds():
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data["classes"] != list(CLASSES):
        raise RuntimeError("Threshold class order does not match model classes.")

    thresholds = np.array(
        [
            data["thresholds"][disease]["threshold"]
            for disease in CLASSES
        ],
        dtype=np.float32,
    )

    return thresholds


def load_model(device):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if list(checkpoint["classes"]) != list(CLASSES):
        raise RuntimeError("Checkpoint class order does not match CLASSES.")

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


def main():
    parser = argparse.ArgumentParser(
        description="Run MedTrust prediction on one chest X-ray."
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
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 78)
    print("MEDTRUST — SINGLE X-RAY PREDICTION")
    print("=" * 78)

    print(f"Image      : {image_path}")
    print(f"Device     : {device}")

    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, checkpoint = load_model(device)

    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Best epoch : {checkpoint.get('epoch', 'unknown')}")

    # --------------------------------------------------------
    # Load frozen thresholds
    # --------------------------------------------------------

    thresholds = load_thresholds()

    print(f"Thresholds : {THRESHOLDS_PATH}")
    print("Source     : validation-only Youden J")

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    image_tensor = test_transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(device)

    print(f"Input shape: {tuple(image_tensor.shape)}")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)

    scores = probabilities.squeeze(0).cpu().numpy()

    predictions = (
        scores >= thresholds
    ).astype(np.int32)

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("PREDICTION RESULTS")
    print("=" * 78)

    print(
        f"{'Disease':<24}"
        f"{'Score':>10}"
        f"{'Threshold':>12}"
        f"{'Decision':>14}"
    )

    print("-" * 78)

    for disease, score, threshold, prediction in zip(
        CLASSES,
        scores,
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
            f"{score:>10.4f}"
            f"{threshold:>12.4f}"
            f"{decision:>14}"
        )

    # --------------------------------------------------------
    # Positive findings
    # --------------------------------------------------------

    positive_indices = np.where(
        predictions == 1
    )[0]

    print("\n" + "=" * 78)
    print("PREDICTED FINDINGS")
    print("=" * 78)

    if len(positive_indices) == 0:
        print("No disease label crossed its decision threshold.")

    else:
        # Sort positive findings by score descending.
        positive_indices = sorted(
            positive_indices,
            key=lambda i: scores[i],
            reverse=True,
        )

        for index in positive_indices:
            print(
                f"{CLASSES[index]:<24}"
                f"score={scores[index]:.4f}  "
                f"threshold={thresholds[index]:.4f}"
            )

    print("\nNOTE:")
    print(
        "Scores are raw sigmoid model outputs, NOT calibrated "
        "confidence percentages."
    )

    print(
        "This research model output must not be interpreted as "
        "a clinical diagnosis."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()