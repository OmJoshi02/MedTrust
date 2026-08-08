import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import CLASSES
from src.explainability.gradcam import GradCAM
from src.model import create_model
from src.transformer import test_transform


CHECKPOINT_PATH = Path(
    "checkpoints/best_model.pt"
)

OUTPUT_DIR = Path(
    "outputs/explainability/validation"
)

CSV_PATH = (
    OUTPUT_DIR
    / "gradcam_validation.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "gradcam_validation_summary.json"
)


# =========================================================
# VALIDATION CONFIGURATION
# =========================================================

REPRODUCIBILITY_TOLERANCE = 1e-6

# If two normalized CAMs have extremely high correlation,
# they may be effectively describing the same spatial
# pattern.
IDENTICAL_CORRELATION_THRESHOLD = 0.999


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
# LOAD IMAGE
# =========================================================

def load_image(
    image_path,
    device,
):

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

    return image_tensor


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

    return float(correlation)


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Validate MedTrust Grad-CAM "
            "implementation."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help=(
            "Path to a chest X-ray used "
            "for Grad-CAM validation."
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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("MEDTRUST — GRAD-CAM VALIDATION")
    print("=" * 78)

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

    image_tensor = load_image(
        image_path=image_path,
        device=device,
    )

    print(
        f"Input shape: "
        f"{tuple(image_tensor.shape)}"
    )

    # -----------------------------------------------------
    # Raw probabilities
    # -----------------------------------------------------

    with torch.inference_mode():

        logits = model(
            image_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

    probabilities = (
        probabilities
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # -----------------------------------------------------
    # Grad-CAM target layer
    # -----------------------------------------------------

    target_layer = (
        model.features.norm5
    )

    print(
        "Target layer: "
        "model.features.norm5"
    )

    print()

    # -----------------------------------------------------
    # Generate CAM for every disease
    # -----------------------------------------------------

    cams = {}

    validation_rows = []

    with GradCAM(
        model=model,
        target_layer=target_layer,
    ) as gradcam:

        for class_index, disease in enumerate(
            CLASSES
        ):

            print(
                f"Validating "
                f"{disease:<22}",
                end="",
            )

            # ---------------------------------------------
            # First CAM
            # ---------------------------------------------

            cam_1, _ = gradcam.generate(
                input_tensor=image_tensor,
                class_index=class_index,
            )

            cam_1 = (
                cam_1
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # ---------------------------------------------
            # Second CAM
            #
            # Used to verify deterministic reproducibility.
            # ---------------------------------------------

            cam_2, _ = gradcam.generate(
                input_tensor=image_tensor,
                class_index=class_index,
            )

            cam_2 = (
                cam_2
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # ---------------------------------------------
            # Basic validity
            # ---------------------------------------------

            finite = bool(
                np.isfinite(
                    cam_1
                ).all()
            )

            correct_shape = (
                cam_1.shape
                == tuple(
                    image_tensor.shape[-2:]
                )
            )

            cam_min = float(
                cam_1.min()
            )

            cam_max = float(
                cam_1.max()
            )

            cam_mean = float(
                cam_1.mean()
            )

            cam_std = float(
                cam_1.std()
            )

            non_degenerate = bool(
                cam_std > 1e-8
                and cam_max > cam_min
            )

            # ---------------------------------------------
            # Reproducibility
            # ---------------------------------------------

            max_abs_difference = float(
                np.max(
                    np.abs(
                        cam_1
                        - cam_2
                    )
                )
            )

            reproducible = bool(
                max_abs_difference
                <= REPRODUCIBILITY_TOLERANCE
            )

            valid = bool(
                finite
                and correct_shape
                and non_degenerate
                and reproducible
            )

            cams[disease] = cam_1

            validation_rows.append(
                {
                    "class_index":
                        class_index,

                    "disease":
                        disease,

                    "raw_probability":
                        float(
                            probabilities[
                                class_index
                            ]
                        ),

                    "cam_min":
                        cam_min,

                    "cam_max":
                        cam_max,

                    "cam_mean":
                        cam_mean,

                    "cam_std":
                        cam_std,

                    "finite":
                        finite,

                    "correct_shape":
                        correct_shape,

                    "non_degenerate":
                        non_degenerate,

                    "max_reproducibility_difference":
                        max_abs_difference,

                    "reproducible":
                        reproducible,

                    "valid":
                        valid,
                }
            )

            status = (
                "PASS"
                if valid
                else "FAIL"
            )

            print(
                f" {status}"
            )

    # =====================================================
    # CLASS-SPECIFICITY ANALYSIS
    # =====================================================

    print()
    print("=" * 78)
    print("CLASS-SPECIFICITY CHECK")
    print("=" * 78)

    correlation_matrix = {}

    suspicious_pairs = []

    diseases = list(CLASSES)

    for disease_a in diseases:

        correlation_matrix[
            disease_a
        ] = {}

        for disease_b in diseases:

            correlation = (
                cam_correlation(
                    cams[disease_a],
                    cams[disease_b],
                )
            )

            correlation_matrix[
                disease_a
            ][
                disease_b
            ] = correlation

    for i in range(
        len(diseases)
    ):

        for j in range(
            i + 1,
            len(diseases)
        ):

            disease_a = diseases[i]
            disease_b = diseases[j]

            correlation = (
                correlation_matrix[
                    disease_a
                ][
                    disease_b
                ]
            )

            if (
                correlation
                >= IDENTICAL_CORRELATION_THRESHOLD
            ):

                suspicious_pairs.append(
                    {
                        "class_a":
                            disease_a,

                        "class_b":
                            disease_b,

                        "correlation":
                            correlation,
                    }
                )

    class_specificity_pass = (
        len(suspicious_pairs) == 0
    )

    print(
        "Near-identical CAM threshold : "
        f"{IDENTICAL_CORRELATION_THRESHOLD:.3f}"
    )

    print(
        "Suspicious class pairs       : "
        f"{len(suspicious_pairs)}"
    )

    print(
        "Class-specificity status     : "
        f"{'PASS' if class_specificity_pass else 'REVIEW'}"
    )

    # =====================================================
    # SAVE CSV
    # =====================================================

    fieldnames = [
        "class_index",
        "disease",
        "raw_probability",
        "cam_min",
        "cam_max",
        "cam_mean",
        "cam_std",
        "finite",
        "correct_shape",
        "non_degenerate",
        "max_reproducibility_difference",
        "reproducible",
        "valid",
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
            validation_rows
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    valid_classes = sum(
        row["valid"]
        for row in validation_rows
    )

    invalid_classes = (
        len(validation_rows)
        - valid_classes
    )

    overall_pass = bool(
        invalid_classes == 0
        and class_specificity_pass
    )

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

        "target_layer":
            "model.features.norm5",

        "num_classes":
            len(CLASSES),

        "valid_classes":
            int(valid_classes),

        "invalid_classes":
            int(invalid_classes),

        "reproducibility_tolerance":
            REPRODUCIBILITY_TOLERANCE,

        "near_identical_correlation_threshold":
            IDENTICAL_CORRELATION_THRESHOLD,

        "suspicious_class_pairs":
            suspicious_pairs,

        "class_specificity_pass":
            class_specificity_pass,

        "overall_validation_pass":
            overall_pass,

        "interpretation": (
            "This validation checks numerical validity, "
            "deterministic reproducibility, and whether "
            "different disease outputs produce "
            "non-identical Grad-CAM patterns. It does not "
            "validate clinical or anatomical localization "
            "accuracy."
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
    print("=" * 78)
    print("GRAD-CAM VALIDATION SUMMARY")
    print("=" * 78)

    print(
        f"Classes tested       : "
        f"{len(CLASSES)}"
    )

    print(
        f"Valid CAMs           : "
        f"{valid_classes}"
    )

    print(
        f"Invalid CAMs         : "
        f"{invalid_classes}"
    )

    print(
        f"Class specificity    : "
        f"{'PASS' if class_specificity_pass else 'REVIEW'}"
    )

    print(
        f"Overall validation   : "
        f"{'PASS' if overall_pass else 'REVIEW'}"
    )

    print()

    print(
        f"CSV                   : "
        f"{CSV_PATH}"
    )

    print(
        f"Summary               : "
        f"{SUMMARY_PATH}"
    )

    print()

    print(
        "NOTE: Passing this validation confirms "
        "implementation-level Grad-CAM behavior only. "
        "It does not establish clinical localization "
        "accuracy."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()