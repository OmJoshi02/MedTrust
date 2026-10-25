"""
Standalone MedTrust Uncertainty Inference.

This script is intentionally separate from predict.py and run_inference.py.
It does not integrate, modify, or overwrite the existing inference pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import CLASSES
from src.model import create_model
from src.transformer import test_transform
from src.uncertainty import MCDropoutPredictor, UncertaintyAnalyzer


CHECKPOINT_PATH = Path("checkpoints/best_model.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedTrust standalone MC-Dropout uncertainty inference."
    )
    parser.add_argument(
        "image",
        help="Path to one chest X-ray image.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=30,
        help="Number of stochastic MC-Dropout passes. Default: 30.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.20,
        help="Feature dropout probability. Default: 0.20.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Binary decision threshold used only for variation ratio.",
    )
    return parser.parse_args()


def load_model(device: torch.device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("classes") != list(CLASSES):
        raise RuntimeError(
            "Checkpoint class order does not match src.dataset.CLASSES."
        )

    model = create_model(
        num_classes=len(CLASSES),
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def prepare_image(path: str, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("L")
    tensor = test_transform(image).unsqueeze(0).to(device)

    if tensor.shape != (1, 3, 224, 224):
        raise RuntimeError(
            f"Unexpected input shape: {tuple(tensor.shape)}"
        )

    return tensor


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(device)
    image_tensor = prepare_image(str(image_path), device)

    predictor = MCDropoutPredictor(
        model=model,
        passes=args.passes,
        dropout_rate=args.dropout_rate,
        device=device,
    )

    mc_result = predictor.predict(image_tensor)

    # Shape [passes, batch, classes] -> [passes, classes] for batch size 1.
    samples = (
        mc_result.probabilities[:, 0, :]
        .detach()
        .cpu()
        .numpy()
    )

    analyzer = UncertaintyAnalyzer(CLASSES)

    result = analyzer.analyze(
        samples,
        decision_threshold=args.threshold,
    )

    print("=" * 100)
    print("MEDTRUST — STANDALONE UNCERTAINTY ANALYSIS")
    print("=" * 100)
    print(f"Image              : {image_path}")
    print(f"Device             : {device}")
    print(f"MC-Dropout passes  : {args.passes}")
    print(f"Dropout rate       : {args.dropout_rate:.2f}")
    print()
    print(
        f"{'Disease':<22}"
        f"{'Probability':>14}"
        f"{'Variance':>14}"
        f"{'Std Dev':>14}"
        f"{'Entropy':>14}"
    )
    print("-" * 100)

    for disease in result["diseases"]:
        print(
            f"{disease.disease:<22}"
            f"{disease.probability:>14.4f}"
            f"{disease.variance:>14.6f}"
            f"{disease.standard_deviation:>14.4f}"
            f"{disease.entropy:>14.4f}"
        )

    print("-" * 100)
    print(
        f"Overall mean variance : "
        f"{result['overall_mean_variance']:.6f}"
    )
    print(
        f"Overall max variance  : "
        f"{result['overall_max_variance']:.6f}"
    )
    print(
        f"Most uncertain       : "
        f"{result['most_uncertain_disease']}"
    )
    print(
        f"Highest uncertainty  : "
        f"{result['most_uncertain_variance']:.6f}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
