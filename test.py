from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

from src.dataset import CLASSES
from src.model import create_model
from src.transformer import test_transform


# ============================================================
# MEDTRUST — TEST + VISUALIZE_GRADCAM
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "best_model.pt"
)

TEST_LIST_PATH = (
    PROJECT_ROOT / "splits" / "test_list.txt"
)

IMAGE_DIR = (
    PROJECT_ROOT / "dataset" / "images"
)

VISUALIZE_GRADCAM = (
    PROJECT_ROOT
    / "src"
    / "explainability"
    / "visualize_gradcam.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "explainability"
    / "test_gradcam"
)

NUMBER_OF_PATIENTS = 10
TOP_K = 5


# ============================================================
# DEVICE
# ============================================================

def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(device):

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    checkpoint_classes = checkpoint.get(
        "classes"
    )

    if checkpoint_classes is not None:

        if list(checkpoint_classes) != list(CLASSES):

            raise RuntimeError(
                "Checkpoint class order does not match "
                "src.dataset.CLASSES."
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


# ============================================================
# READ TEST LIST
# ============================================================

def read_test_images():

    if not TEST_LIST_PATH.exists():

        raise FileNotFoundError(
            f"Test list not found:\n{TEST_LIST_PATH}"
        )

    with open(
        TEST_LIST_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        image_names = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(image_names) == 0:

        raise RuntimeError(
            "splits/test_list.txt is empty."
        )

    selected = image_names[
        :NUMBER_OF_PATIENTS
    ]

    return selected


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

def predict_image(
    model,
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
        .detach()
        .cpu()
        .numpy()
    )

    # Highest probability first.
    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    top_indices = sorted_indices[
        :TOP_K
    ]

    predictions = []

    for rank, class_index in enumerate(
        top_indices,
        start=1,
    ):

        class_index = int(
            class_index
        )

        probability = float(
            probabilities[class_index]
        )

        predictions.append(
            {
                "rank": rank,
                "class_index": class_index,
                "class_name": CLASSES[
                    class_index
                ],
                "probability": probability,
            }
        )

    return predictions


# ============================================================
# RUN EXISTING VISUALIZE_GRADCAM.PY
# ============================================================

def generate_gradcam(
    image_path,
    class_name,
    patient_number,
):

    safe_class_name = (
        class_name
        .lower()
        .replace(" ", "_")
    )

    # --------------------------------------------------------
    # Expected output generated directly by
    # visualize_gradcam.py
    # --------------------------------------------------------

    expected_output = (
        PROJECT_ROOT
        / "outputs"
        / "explainability"
        / "gradcam"
        / f"patient_{patient_number}"
        / (
            f"{image_path.stem}"
            f"__{safe_class_name}"
            f"__gradcam.png"
        )
    )

    # --------------------------------------------------------
    # Remove old output if it exists.
    # --------------------------------------------------------

    if expected_output.exists():

        expected_output.unlink()

    # --------------------------------------------------------
    # Call visualize_gradcam.py
    #
    # test.py does NOT implement Grad-CAM.
    # --------------------------------------------------------

    command = [
        sys.executable,
        "-m",
        "src.explainability.visualize_gradcam",
        str(image_path),
        "--class-name",
        class_name,
        "--patient-folder",
        f"patient_{patient_number}",
    ]

    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "visualize_gradcam.py failed for "
            f"{image_path.name} / {class_name}"
        )

    # --------------------------------------------------------
    # Verify generated image.
    # --------------------------------------------------------

    if not expected_output.exists():

        raise FileNotFoundError(
            "visualize_gradcam.py completed, but "
            f"expected output was not found:\n"
            f"{expected_output}"
        )

    return expected_output
# ============================================================
# SAVE PATIENT SUMMARY
# ============================================================

def save_patient_summary(
    patient_dir,
    image_name,
    predictions,
):

    summary = {
        "image": image_name,
        "top_k": TOP_K,
        "predictions": predictions,
    }

    summary_path = (
        patient_dir
        / "predictions.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    return summary_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "MEDTRUST — HELD-OUT TEST + "
        "VISUALIZE_GRADCAM"
    )
    print("=" * 78)

    print(
        f"Checkpoint   : {CHECKPOINT_PATH}"
    )

    print(
        f"Test list    : {TEST_LIST_PATH}"
    )

    print(
        f"Image dir    : {IMAGE_DIR}"
    )

    print(
        f"Grad-CAM     : {VISUALIZE_GRADCAM}"
    )

    print(
        f"Output       : {OUTPUT_DIR}"
    )

    print(
        f"Images       : {NUMBER_OF_PATIENTS}"
    )

    print(
        f"Top diseases : {TOP_K}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "10 test images are selected directly "
        "from test_list.txt."
    )

    print(
        "The complete 16,805-image test set is NOT scanned."
    )

    print(
        "test.py does NOT implement Grad-CAM."
    )

    print(
        "visualize_gradcam.py generates every Grad-CAM figure."
    )

    print(
        "No model weights are changed."
    )

    print(
        "No calibration parameters are changed."
    )

    print()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(
        f"Device       : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        CHECKPOINT_PATH,
        TEST_LIST_PATH,
        IMAGE_DIR,
        VISUALIZE_GRADCAM,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, checkpoint = load_model(
        device
    )

    print(
        f"Best epoch   : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    # --------------------------------------------------------
    # Select exactly 10 test images.
    #
    # IMPORTANT:
    # This reads the list but does NOT run inference
    # over the entire list.
    # --------------------------------------------------------

    selected_images = read_test_images()

    print()
    print(
        f"Selected test images : "
        f"{len(selected_images)}"
    )

    print()

    # --------------------------------------------------------
    # Clean/create output directory.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    all_results = []

    total_gradcams = 0

    # ========================================================
    # PROCESS EACH PATIENT
    # ========================================================

    for patient_number, image_name in enumerate(
        selected_images,
        start=1,
    ):

        print()
        print(
            "-" * 78
        )

        print(
            f"PATIENT {patient_number}/{len(selected_images)}"
        )

        print(
            f"Image : {image_name}"
        )

        image_path = (
            IMAGE_DIR
            / image_name
        )

        if not image_path.exists():

            print(
                f"WARNING: image not found: "
                f"{image_path}"
            )

            continue

        # ----------------------------------------------------
        # Patient directory
        # ----------------------------------------------------

        patient_dir = (
            OUTPUT_DIR
            / f"patient_{patient_number}"
        )

        patient_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Predict all 14 diseases for THIS image only.
        # ----------------------------------------------------

        predictions = predict_image(
            model=model,
            image_path=image_path,
            device=device,
        )

        print()
        print(
            "TOP 5 PREDICTIONS"
        )

        for prediction in predictions:

            print(
                f"  "
                f"{prediction['rank']}. "
                f"{prediction['class_name']:<22}"
                f" "
                f"{prediction['probability']:.4f}"
            )

        # ----------------------------------------------------
        # Save prediction information.
        # ----------------------------------------------------

        save_patient_summary(
            patient_dir=patient_dir,
            image_name=image_name,
            predictions=predictions,
        )

        # ----------------------------------------------------
        # Generate Grad-CAM for each top-5 disease.
        #
        # visualize_gradcam.py is called separately
        # for each class.
        # ----------------------------------------------------

        generated_files = []

        print()
        print(
            "GENERATING GRAD-CAM"
        )

        for prediction in predictions:

            rank = prediction[
                "rank"
            ]

            class_name = prediction[
                "class_name"
            ]

            probability = prediction[
                "probability"
            ]

            print()
            print(
                f"[{rank}/5] "
                f"{class_name} "
                f"({probability:.4f})"
            )

            output_path = generate_gradcam(
                image_path=image_path,
                class_name=class_name,
                patient_number=patient_number,
            )

            generated_files.append(
                str(output_path)
            )

            total_gradcams += 1

            print(
                f"Saved: {output_path}"
            )

        # ----------------------------------------------------
        # Patient result
        # ----------------------------------------------------

        patient_result = {
            "patient": patient_number,
            "image": image_name,
            "predictions": predictions,
            "gradcam_files": generated_files,
        }

        all_results.append(
            patient_result
        )

    # ========================================================
    # SAVE OVERALL SUMMARY
    # ========================================================

    summary = {
        "description": (
            "Held-out test inference using 10 directly "
            "selected test images. For each image, the "
            "model predicts all 14 classes and the top 5 "
            "classes are explained using visualize_gradcam.py."
        ),
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "test_list": str(
            TEST_LIST_PATH
        ),
        "number_of_test_images_processed": len(
            all_results
        ),
        "classes": list(CLASSES),
        "top_k": TOP_K,
        "total_gradcams_generated": total_gradcams,
        "results": all_results,
    }

    summary_path = (
        OUTPUT_DIR
        / "test_gradcam_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 78)
    print(
        "TEST + VISUALIZE_GRADCAM COMPLETE"
    )
    print("=" * 78)

    print(
        f"Images processed : "
        f"{len(all_results)}"
    )

    print(
        f"Top diseases/image : "
        f"{TOP_K}"
    )

    print(
        f"Grad-CAMs created : "
        f"{total_gradcams}"
    )

    print(
        f"Output directory : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Summary          : "
        f"{summary_path}"
    )

    print()
    print(
        "Structure:"
    )

    print(
        "  patient_1/"
    )

    print(
        "      predictions.json"
    )

    print(
        "      <image>__<disease>__gradcam.png"
    )

    print(
        "      ... 5 Grad-CAM images"
    )

    print(
        "  patient_2/"
    )

    print(
        "      ..."
    )

    print(
        "  ..."
    )

    print(
        "  patient_10/"
    )

    print(
        "      ..."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The 16,805-image test set was NOT scanned."
    )

    print(
        "Only 10 test images were used."
    )

    print(
        "Each image was evaluated independently."
    )

    print(
        "Grad-CAM was generated by "
        "visualize_gradcam.py."
    )

    print(
        "This is a held-out test-set demonstration "
        "and does not establish clinical localization accuracy."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()