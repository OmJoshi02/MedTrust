import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

from src.dataset import CLASSES
from src.explainability.gradcam import GradCAM
from src.model import create_model
from src.transformer import test_transform


CHECKPOINT_PATH = Path(
    "checkpoints/best_model.pt"
)

OUTPUT_DIR = Path(
    "outputs/explainability/reliability"
)

CSV_PATH = (
    OUTPUT_DIR
    / "explanation_reliability.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "explanation_reliability_summary.json"
)


# =========================================================
# CONFIGURATION
# =========================================================

# Correlation is measured between the original CAM and
# perturbed-image CAM.
#
# These are deliberately descriptive thresholds rather
# than clinical claims.

HIGH_STABILITY = 0.90
MODERATE_STABILITY = 0.75

# Small perturbations only.
BRIGHTNESS_FACTOR = 1.05
CONTRAST_FACTOR = 1.05
TRANSLATION_PIXELS = 4
NOISE_STD = 0.01


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(device):

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if list(
        checkpoint["classes"]
    ) != list(CLASSES):

        raise RuntimeError(
            "Checkpoint class order does not "
            "match CLASSES."
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
# IMAGE TRANSFORMATIONS
# =========================================================

def brightness_perturbation(image):

    return ImageEnhance.Brightness(
        image
    ).enhance(
        BRIGHTNESS_FACTOR
    )


def contrast_perturbation(image):

    return ImageEnhance.Contrast(
        image
    ).enhance(
        CONTRAST_FACTOR
    )


def translation_perturbation(image):

    pixels = TRANSLATION_PIXELS

    return image.transform(
        image.size,
        Image.AFFINE,
        (
            1,
            0,
            pixels,
            0,
            1,
            0,
        ),
    )


def noise_perturbation(
    image,
    device,
):

    tensor = test_transform(
        image
    )

    noise = torch.randn_like(
        tensor
    ) * NOISE_STD

    tensor = tensor + noise

    return tensor.unsqueeze(
        0
    ).to(device)


def normal_tensor(
    image,
    device,
):

    tensor = test_transform(
        image
    )

    return tensor.unsqueeze(
        0
    ).to(device)


# =========================================================
# GRAD-CAM
# =========================================================

def generate_cam(
    model,
    image_tensor,
    class_index,
):

    target_layer = (
        model.features.norm5
    )

    with GradCAM(
        model=model,
        target_layer=target_layer,
    ) as gradcam:

        cam, logits = gradcam.generate(
            input_tensor=image_tensor,
            class_index=class_index,
        )

    return (
        cam.cpu().numpy(),
        logits.cpu().numpy(),
    )


# =========================================================
# CAM CORRELATION
# =========================================================

def cam_correlation(
    cam_a,
    cam_b,
):

    a = cam_a.reshape(-1)
    b = cam_b.reshape(-1)

    if np.std(a) < 1e-12:
        return 0.0

    if np.std(b) < 1e-12:
        return 0.0

    correlation = np.corrcoef(
        a,
        b,
    )[0, 1]

    if not np.isfinite(
        correlation
    ):

        return 0.0

    return float(
        correlation
    )


# =========================================================
# STABILITY CATEGORY
# =========================================================

def stability_category(
    correlation,
):

    if correlation >= HIGH_STABILITY:

        return "HIGH"

    if correlation >= MODERATE_STABILITY:

        return "MODERATE"

    return "LOW"


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Grad-CAM explanation "
            "stability under small input perturbations."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to chest X-ray.",
    )

    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help=(
            "Disease to explain. If omitted, "
            "the highest raw-probability class "
            "is selected."
        ),
    )

    args = parser.parse_args()

    image_path = Path(
        args.image
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print(
        "MEDTRUST — EXPLANATION RELIABILITY"
    )
    print("=" * 80)

    print(
        f"Image      : {image_path}"
    )

    print(
        f"Device     : {device}"
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model, checkpoint = load_model(
        device
    )

    print(
        f"Checkpoint : {CHECKPOINT_PATH}"
    )

    print(
        f"Best epoch : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # -----------------------------------------------------
    # Determine target disease
    # -----------------------------------------------------

    original_tensor = normal_tensor(
        image,
        device,
    )

    with torch.inference_mode():

        original_logits = model(
            original_tensor
        )

        original_probabilities = (
            torch.sigmoid(
                original_logits
            )
        )

    probabilities = (
        original_probabilities
        .squeeze(0)
        .cpu()
        .numpy()
    )

    if args.class_name is None:

        class_index = int(
            np.argmax(
                probabilities
            )
        )

    else:

        lookup = {
            disease.lower(): index
            for index, disease
            in enumerate(CLASSES)
        }

        requested = (
            args.class_name
            .strip()
            .lower()
        )

        if requested not in lookup:

            raise ValueError(
                f"Unknown disease: "
                f"{args.class_name}\n"
                f"Available classes: "
                f"{', '.join(CLASSES)}"
            )

        class_index = lookup[
            requested
        ]

    disease = CLASSES[
        class_index
    ]

    original_probability = float(
        probabilities[
            class_index
        ]
    )

    print()
    print(
        f"Target disease : {disease}"
    )

    print(
        f"Class index    : {class_index}"
    )

    print(
        f"Raw probability: "
        f"{original_probability:.4f}"
    )

    # -----------------------------------------------------
    # Generate original CAM
    # -----------------------------------------------------

    original_cam, _ = generate_cam(
        model=model,
        image_tensor=original_tensor,
        class_index=class_index,
    )

    # -----------------------------------------------------
    # Build perturbations
    # -----------------------------------------------------

    perturbations = {
        "brightness":
            brightness_perturbation(
                image
            ),

        "contrast":
            contrast_perturbation(
                image
            ),

        "translation":
            translation_perturbation(
                image
            ),
    }

    results = []

    # -----------------------------------------------------
    # Evaluate perturbations
    # -----------------------------------------------------

    for name, perturbed_image in (
        perturbations.items()
    ):

        perturbed_tensor = normal_tensor(
            perturbed_image,
            device,
        )

        perturbed_cam, perturbed_logits = (
            generate_cam(
                model=model,
                image_tensor=perturbed_tensor,
                class_index=class_index,
            )
        )

        correlation = cam_correlation(
            original_cam,
            perturbed_cam,
        )

        perturbed_probability = float(
            torch.sigmoid(
                torch.tensor(
                    perturbed_logits[
                        0,
                        class_index,
                    ]
                )
            ).item()
        )

        category = stability_category(
            correlation
        )

        results.append(
            {
                "perturbation":
                    name,

                "cam_correlation":
                    correlation,

                "stability":
                    category,

                "original_probability":
                    original_probability,

                "perturbed_probability":
                    perturbed_probability,

                "probability_change":
                    abs(
                        perturbed_probability
                        - original_probability
                    ),
            }
        )

        print()
        print(
            f"{name:<15}"
            f"CAM correlation: "
            f"{correlation:.4f}  "
            f"Stability: "
            f"{category}"
        )

    # -----------------------------------------------------
    # Gaussian noise
    # -----------------------------------------------------

    noisy_tensor = noise_perturbation(
        image,
        device,
    )

    noisy_cam, noisy_logits = generate_cam(
        model=model,
        image_tensor=noisy_tensor,
        class_index=class_index,
    )

    noise_correlation = cam_correlation(
        original_cam,
        noisy_cam,
    )

    noisy_probability = float(
        torch.sigmoid(
            torch.tensor(
                noisy_logits[
                    0,
                    class_index,
                ]
            )
        ).item()
    )

    noise_category = stability_category(
        noise_correlation
    )

    results.append(
        {
            "perturbation":
                "gaussian_noise",

            "cam_correlation":
                noise_correlation,

            "stability":
                noise_category,

            "original_probability":
                original_probability,

            "perturbed_probability":
                noisy_probability,

            "probability_change":
                abs(
                    noisy_probability
                    - original_probability
                ),
        }
    )

    print()
    print(
        f"{'gaussian_noise':<15}"
        f"CAM correlation: "
        f"{noise_correlation:.4f}  "
        f"Stability: "
        f"{noise_category}"
    )

    # =====================================================
    # OVERALL RELIABILITY
    # =====================================================

    correlations = [
        row[
            "cam_correlation"
        ]
        for row in results
    ]

    mean_correlation = float(
        np.mean(
            correlations
        )
    )

    minimum_correlation = float(
        np.min(
            correlations
        )
    )

    if mean_correlation >= HIGH_STABILITY:

        overall_stability = "HIGH"

    elif mean_correlation >= MODERATE_STABILITY:

        overall_stability = "MODERATE"

    else:

        overall_stability = "LOW"

    # =====================================================
    # SAVE CSV
    # =====================================================

    fieldnames = [
        "perturbation",
        "cam_correlation",
        "stability",
        "original_probability",
        "perturbed_probability",
        "probability_change",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # =====================================================
    # SAVE SUMMARY
    # =====================================================

    summary = {

        "image":
            str(image_path),

        "checkpoint":
            str(CHECKPOINT_PATH),

        "checkpoint_epoch":
            checkpoint.get(
                "epoch",
                None,
            ),

        "target_disease":
            disease,

        "class_index":
            class_index,

        "original_probability":
            original_probability,

        "target_layer":
            "model.features.norm5",

        "perturbations":
            results,

        "mean_cam_correlation":
            mean_correlation,

        "minimum_cam_correlation":
            minimum_correlation,

        "overall_stability":
            overall_stability,

        "thresholds":
            {
                "high":
                    HIGH_STABILITY,

                "moderate":
                    MODERATE_STABILITY,
            },

        "interpretation":
            (
                "Explanation reliability here refers "
                "to stability of Grad-CAM spatial "
                "patterns under small input "
                "perturbations. It does not establish "
                "clinical localization accuracy."
            ),
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # =====================================================
    # FINAL REPORT
    # =====================================================

    print()
    print("=" * 80)
    print(
        "EXPLANATION RELIABILITY SUMMARY"
    )
    print("=" * 80)

    print(
        f"Mean CAM correlation    : "
        f"{mean_correlation:.4f}"
    )

    print(
        f"Minimum CAM correlation: "
        f"{minimum_correlation:.4f}"
    )

    print(
        f"Overall stability      : "
        f"{overall_stability}"
    )

    print()
    print(
        f"CSV     : {CSV_PATH}"
    )

    print(
        f"Summary : {SUMMARY_PATH}"
    )

    print()
    print(
        "NOTE: Stability is an explanation-quality "
        "indicator, not a clinical validation measure."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()