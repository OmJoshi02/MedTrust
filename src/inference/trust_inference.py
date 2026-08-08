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


CHECKPOINT_PATH = Path(
    "checkpoints/best_model.pt"
)

THRESHOLDS_PATH = Path(
    "outputs/evaluation/thresholds.json"
)

TEMPERATURE_PATH = Path(
    "outputs/calibration/temperature.json"
)

OUTPUT_DIR = Path(
    "outputs/inference"
)


# =========================================================
# LOAD THRESHOLDS
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
            "Threshold class order does not match CLASSES."
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
# LOAD TEMPERATURES
# =========================================================

def load_temperatures():

    with open(
        TEMPERATURE_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if data.get("classes") != list(CLASSES):

        raise RuntimeError(
            "Calibration class order does not match CLASSES."
        )

    if data.get("calibration_split") != "validation":

        raise RuntimeError(
            "Calibration was not fitted on validation data."
        )

    if data.get(
        "test_data_used_for_fitting"
    ) is not False:

        raise RuntimeError(
            "Calibration artifact does not confirm "
            "test-data exclusion."
        )

    temperatures = np.array(
        [
            float(
                data["temperatures"][disease]
            )
            for disease in CLASSES
        ],
        dtype=np.float32,
    )

    if not np.isfinite(
        temperatures
    ).all():

        raise RuntimeError(
            "Calibration temperatures contain "
            "non-finite values."
        )

    if np.any(
        temperatures <= 0
    ):

        raise RuntimeError(
            "Calibration temperatures must be positive."
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

    if list(
        checkpoint["classes"]
    ) != list(CLASSES):

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
# GRAD-CAM
# =========================================================

def generate_gradcam(
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

        cam, _ = gradcam.generate(
            input_tensor=image_tensor,
            class_index=class_index,
        )

    return cam.cpu().numpy()


# =========================================================
# SAVE GRAD-CAM
# =========================================================

def save_gradcam(
    cam,
    image,
    disease,
    raw_probability,
    output_path,
):

    import matplotlib.pyplot as plt

    display_image = image.resize(
        (224, 224)
    )

    display_image = (
        np.asarray(
            display_image
        ).astype(
            np.float32
        ) / 255.0
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    # Original
    axes[0].imshow(
        display_image
    )

    axes[0].set_title(
        "Original X-ray"
    )

    axes[0].axis("off")

    # Heatmap
    heatmap = axes[1].imshow(
        cam,
        cmap="jet",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        f"Grad-CAM\n{disease}"
    )

    axes[1].axis("off")

    fig.colorbar(
        heatmap,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )

    # Overlay
    axes[2].imshow(
        display_image
    )

    axes[2].imshow(
        cam,
        cmap="jet",
        alpha=0.40,
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        (
            f"{disease}\n"
            f"Raw probability: "
            f"{raw_probability:.4f}"
        )
    )

    axes[2].axis("off")

    fig.suptitle(
        "MedTrust — Trust-Aware Explanation"
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "MedTrust trust-aware inference "
            "for one chest X-ray."
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
            "Disease to explain with Grad-CAM. "
            "If omitted, the highest raw-probability "
            "class is used."
        ),
    )

    args = parser.parse_args()

    image_path = Path(
        args.image
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
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

    print("=" * 96)
    print(
        "MEDTRUST — TRUST-AWARE INFERENCE"
    )
    print("=" * 96)

    print(
        f"Image      : {image_path}"
    )

    print(
        f"Device     : {device}"
    )

    # -----------------------------------------------------
    # Load frozen artifacts
    # -----------------------------------------------------

    model, checkpoint = load_model(
        device
    )

    thresholds = load_thresholds()

    temperatures = load_temperatures()

    print(
        f"Checkpoint : {CHECKPOINT_PATH}"
    )

    print(
        f"Best epoch : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Thresholds : {THRESHOLDS_PATH}"
    )

    print(
        f"Calibration: {TEMPERATURE_PATH}"
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
            device=device,
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
    # Existing thresholds were selected in RAW
    # probability space.
    predictions = (
        raw_scores >= thresholds
    ).astype(np.int32)

    # -----------------------------------------------------
    # Select explanation class
    # -----------------------------------------------------

    if args.class_name is None:

        class_index = int(
            np.argmax(
                raw_scores
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
                f"Unknown class: "
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

    # -----------------------------------------------------
    # Grad-CAM
    # -----------------------------------------------------

    cam = generate_gradcam(
        model=model,
        image_tensor=image_tensor,
        class_index=class_index,
    )

    safe_disease = (
        disease
        .lower()
        .replace(" ", "_")
    )

    gradcam_path = (
        OUTPUT_DIR
        / (
            f"{image_path.stem}"
            f"__{safe_disease}"
            f"__gradcam.png"
        )
    )

    save_gradcam(
        cam=cam,
        image=image,
        disease=disease,
        raw_probability=float(
            raw_scores[class_index]
        ),
        output_path=gradcam_path,
    )

    # -----------------------------------------------------
    # Build structured result
    # -----------------------------------------------------

    findings = []

    for index, disease_name in enumerate(
        CLASSES
    ):

        findings.append(
            {
                "disease": disease_name,
                "raw_probability":
                    float(
                        raw_scores[index]
                    ),
                "calibrated_probability":
                    float(
                        calibrated_scores[index]
                    ),
                "threshold":
                    float(
                        thresholds[index]
                    ),
                "decision":
                    (
                        "POSITIVE"
                        if predictions[index] == 1
                        else "Negative"
                    ),
                "temperature":
                    float(
                        temperatures[index]
                    ),
            }
        )

    result = {

        "image":
            str(image_path),

        "checkpoint":
            str(CHECKPOINT_PATH),

        "checkpoint_epoch":
            checkpoint.get(
                "epoch",
                None,
            ),

        "calibration":
            {
                "source":
                    "validation-only",
                "temperatures_frozen":
                    True,
            },

        "thresholds":
            {
                "source":
                    "validation-only Youden J",
                "decision_space":
                    "raw_probability",
                "thresholds_frozen":
                    True,
            },

        "findings":
            findings,

        "explanation":
            {
                "target_disease":
                    disease,
                "class_index":
                    class_index,
                "target_layer":
                    "model.features.norm5",
                "gradcam_path":
                    str(
                        gradcam_path
                    ),
            },

        "interpretation":
            (
                "The raw probability is used for the "
                "existing frozen disease decision "
                "thresholds. The calibrated probability "
                "is reported separately after "
                "validation-derived temperature scaling. "
                "Grad-CAM identifies regions contributing "
                "to the selected disease logit."
            ),

        "clinical_disclaimer":
            (
                "This research model output is not a "
                "clinical diagnosis."
            ),
    }

    json_path = (
        OUTPUT_DIR
        / (
            f"{image_path.stem}"
            "__trust_result.json"
        )
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
        )

    # -----------------------------------------------------
    # Console report
    # -----------------------------------------------------

    print()
    print("=" * 96)
    print("TRUST-AWARE RESULT")
    print("=" * 96)

    print(
        f"Explanation target : "
        f"{disease}"
    )

    print(
        f"Raw probability    : "
        f"{raw_scores[class_index]:.4f}"
    )

    print(
        f"Calibrated         : "
        f"{calibrated_scores[class_index]:.4f}"
    )

    print(
        f"Threshold          : "
        f"{thresholds[class_index]:.4f}"
    )

    print(
        f"Decision           : "
        f"{'POSITIVE' if predictions[class_index] else 'Negative'}"
    )

    print()
    print(
        f"Grad-CAM           : "
        f"{gradcam_path}"
    )

    print(
        f"JSON result        : "
        f"{json_path}"
    )

    print()
    print(
        "NOTE: This is a research model output "
        "and must not be interpreted as a clinical diagnosis."
    )

    print("=" * 96)


if __name__ == "__main__":
    main()