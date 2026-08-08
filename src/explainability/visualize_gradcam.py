import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.dataset import CLASSES
from src.explainability.gradcam import GradCAM
from src.model import create_model
from src.transformer import test_transform


CHECKPOINT_PATH = Path("checkpoints/best_model.pt")

OUTPUT_DIR = Path(
    "outputs/explainability/gradcam"
)


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
# PREPARE DISPLAY IMAGE
# =========================================================

def prepare_display_image(image):

    image = image.convert("RGB")

    display_image = image.resize(
        (224, 224)
    )

    return np.asarray(
        display_image
    ).astype(np.float32) / 255.0


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate disease-specific Grad-CAM "
            "for one chest X-ray."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to PNG/JPG chest X-ray",
    )

    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help=(
            "Disease class to explain. "
            "If omitted, the highest raw-probability "
            "class is used."
        ),
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

    print("=" * 72)
    print("MEDTRUST — GRAD-CAM")
    print("=" * 72)

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

    image_tensor = test_transform(
        image
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    print(
        f"Input shape: "
        f"{tuple(image_tensor.shape)}"
    )

    # -----------------------------------------------------
    # Initial prediction
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
    # Select disease
    # -----------------------------------------------------

    if args.class_name is None:

        class_index = int(
            np.argmax(probabilities)
        )

        class_name = CLASSES[
            class_index
        ]

    else:

        class_lookup = {
            disease.lower(): index
            for index, disease
            in enumerate(CLASSES)
        }

        requested_class = (
            args.class_name
            .strip()
            .lower()
        )

        if requested_class not in class_lookup:

            available = ", ".join(
                CLASSES
            )

            raise ValueError(
                f"Unknown disease class: "
                f"{args.class_name}\n"
                f"Available classes: {available}"
            )

        class_index = class_lookup[
            requested_class
        ]

        class_name = CLASSES[
            class_index
        ]

    raw_probability = float(
        probabilities[class_index]
    )

    print()
    print(
        f"Explaining : {class_name}"
    )

    print(
        f"Class index: {class_index}"
    )

    print(
        f"Raw prob.  : {raw_probability:.4f}"
    )

    # -----------------------------------------------------
    # Grad-CAM
    # -----------------------------------------------------

    target_layer = (
        model.features.norm5
    )

    print(
        "Target layer: model.features.norm5"
    )

    with GradCAM(
        model=model,
        target_layer=target_layer,
    ) as gradcam:

        cam, gradcam_logits = (
            gradcam.generate(
                input_tensor=image_tensor,
                class_index=class_index,
            )
        )

    cam = (
        cam
        .cpu()
        .numpy()
    )

    print(
        f"CAM shape  : {cam.shape}"
    )

    print(
        f"CAM min    : {cam.min():.6f}"
    )

    print(
        f"CAM max    : {cam.max():.6f}"
    )

    # -----------------------------------------------------
    # Prepare visualization
    # -----------------------------------------------------

    display_image = (
        prepare_display_image(
            image
        )
    )

    # -----------------------------------------------------
    # Save figure
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_class_name = (
        class_name
        .lower()
        .replace(" ", "_")
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"{image_path.stem}"
            f"__{safe_class_name}"
            f"__gradcam.png"
        )
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    # Original image
    axes[0].imshow(
        display_image,
        cmap="gray",
    )

    axes[0].set_title(
        "Original X-ray"
    )

    axes[0].axis("off")

    # Heatmap
    heatmap = axes[1].imshow(
        cam,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )

    axes[1].set_title(
        f"Grad-CAM\n{class_name}"
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
        display_image,
    )

    axes[2].imshow(
        cam,
        cmap="jet",
        alpha=0.40,
        vmin=0.0,
        vmax=1.0,
    )

    axes[2].set_title(
        (
            f"{class_name}\n"
            f"Raw probability: "
            f"{raw_probability:.4f}"
        )
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

    print()
    print("=" * 72)
    print("GRAD-CAM RESULT")
    print("=" * 72)

    print(
        f"Disease : {class_name}"
    )

    print(
        f"Raw prob: {raw_probability:.4f}"
    )

    print(
        f"Saved   : {output_path}"
    )

    print()
    print(
        "NOTE: Grad-CAM highlights image regions that "
        "influenced the selected model output. It does "
        "not establish clinical localization or diagnosis."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()