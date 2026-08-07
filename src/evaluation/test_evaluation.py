from pathlib import Path
import csv
import json

import numpy as np
import torch

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader

from src.dataset import (
    CLASSES,
    NIHChestXrayDataset,
)

from src.model import create_model
from src.transformer import test_transform


# =========================================================
# CONFIGURATION
# =========================================================

CHECKPOINT_PATH = Path("checkpoints/best_model.pt")

THRESHOLDS_PATH = Path(
    "outputs/evaluation/thresholds.json"
)

OUTPUT_DIR = Path("outputs/evaluation")

TEST_METRICS_PATH = OUTPUT_DIR / "test_metrics.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

IMAGE_DIR = "dataset/images"
METADATA_CSV = "dataset/Data_Entry_2017_v2020.csv"
TEST_SPLIT = "splits/test_list.txt"

EXPECTED_TEST_SAMPLES = 16805

BATCH_SIZE = 16
NUM_WORKERS = 4


# =========================================================
# LOAD FROZEN THRESHOLDS
# =========================================================

def load_frozen_thresholds():

    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Frozen thresholds not found: "
            f"{THRESHOLDS_PATH}"
        )

    with open(
        THRESHOLDS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        artifact = json.load(f)

    # -----------------------------------------------------
    # Verify provenance
    # -----------------------------------------------------

    if artifact.get(
        "threshold_selection_split"
    ) != "validation":
        raise ValueError(
            "Thresholds were not selected from validation."
        )

    if artifact.get(
        "test_data_used_for_selection"
    ) is not False:
        raise ValueError(
            "Threshold artifact does not confirm "
            "test independence."
        )

    artifact_classes = artifact.get("classes")

    if artifact_classes != list(CLASSES):
        raise ValueError(
            "Threshold class order does not match "
            "src.dataset.CLASSES."
        )

    threshold_dict = artifact.get(
        "thresholds"
    )

    if not isinstance(threshold_dict, dict):
        raise ValueError(
            "Threshold dictionary missing."
        )

    if set(threshold_dict.keys()) != set(CLASSES):
        raise ValueError(
            "Threshold diseases do not match CLASSES."
        )

    thresholds = np.array(
        [
            float(threshold_dict[disease])
            for disease in CLASSES
        ],
        dtype=np.float64,
    )

    if not np.isfinite(thresholds).all():
        raise ValueError(
            "Non-finite frozen threshold detected."
        )

    return thresholds, artifact


# =========================================================
# LOAD FROZEN MODEL
# =========================================================

def load_best_model(
    device,
    threshold_artifact,
):

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if list(checkpoint["classes"]) != list(CLASSES):
        raise ValueError(
            "Checkpoint class order mismatch."
        )

    # Make sure the threshold artifact belongs to the same
    # checkpoint epoch that is being evaluated.
    if int(checkpoint["epoch"]) != int(
        threshold_artifact["checkpoint_epoch"]
    ):
        raise ValueError(
            "Checkpoint epoch does not match "
            "threshold artifact."
        )

    model = create_model(
        num_classes=len(CLASSES),
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


# =========================================================
# TEST LOADER
# =========================================================

def create_test_loader():

    dataset = NIHChestXrayDataset(
        image_dir=IMAGE_DIR,
        metadata_csv=METADATA_CSV,
        split_file=TEST_SPLIT,
        transform=test_transform,
    )

    if len(dataset) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLES} test images, "
            f"found {len(dataset)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    return loader


# =========================================================
# TEST INFERENCE
# =========================================================

@torch.no_grad()
def run_test_inference(
    model,
    loader,
    device,
):

    model.eval()

    all_filenames = []
    all_targets = []
    all_logits = []
    all_probabilities = []

    for batch_idx, (
        images,
        targets,
        filenames,
    ) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True,
        )

        # Match validation inference behavior.
        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)

        probabilities = torch.sigmoid(
            logits
        )

        all_filenames.extend(
            filenames
        )

        all_targets.append(
            targets.numpy()
        )

        all_logits.append(
            logits.float().cpu().numpy()
        )

        all_probabilities.append(
            probabilities.float().cpu().numpy()
        )

        if (batch_idx + 1) % 200 == 0:

            processed = min(
                (batch_idx + 1)
                * loader.batch_size,
                len(loader.dataset),
            )

            print(
                f"Processed "
                f"{processed:5d}/"
                f"{len(loader.dataset)} "
                f"test images"
            )

    targets = np.concatenate(
        all_targets,
        axis=0,
    )

    logits = np.concatenate(
        all_logits,
        axis=0,
    )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    return (
        all_filenames,
        targets,
        logits,
        probabilities,
    )


# =========================================================
# SANITY CHECKS
# =========================================================

def verify_test_outputs(
    filenames,
    targets,
    logits,
    probabilities,
):

    n = EXPECTED_TEST_SAMPLES
    c = len(CLASSES)

    if len(filenames) != n:
        raise ValueError(
            "Test filename count mismatch."
        )

    if len(set(filenames)) != n:
        raise ValueError(
            "Duplicate test filenames detected."
        )

    expected_shape = (n, c)

    if targets.shape != expected_shape:
        raise ValueError(
            f"Unexpected target shape: "
            f"{targets.shape}"
        )

    if logits.shape != expected_shape:
        raise ValueError(
            f"Unexpected logits shape: "
            f"{logits.shape}"
        )

    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Unexpected probability shape: "
            f"{probabilities.shape}"
        )

    if not np.isfinite(logits).all():
        raise ValueError(
            "Non-finite test logits detected."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite test scores detected."
        )


# =========================================================
# FINAL TEST METRICS
# =========================================================

def calculate_test_metrics(
    targets,
    probabilities,
    thresholds,
):

    results = []

    for class_idx, disease in enumerate(
        CLASSES
    ):

        y_true = targets[
            :, class_idx
        ].astype(int)

        y_score = probabilities[
            :, class_idx
        ]

        threshold = thresholds[
            class_idx
        ]

        # IMPORTANT:
        # Threshold came from validation and is NOT
        # optimized using test data.
        y_pred = (
            y_score >= threshold
        ).astype(int)

        # ---------------------------------------------
        # AUROC
        # ---------------------------------------------

        if len(np.unique(y_true)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(
                y_true,
                y_score,
            )

        # ---------------------------------------------
        # Confusion matrix
        # ---------------------------------------------

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        sensitivity = recall

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        results.append({
            "disease": disease,
            "auroc": float(auc),
            "threshold": float(threshold),

            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),

            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(
                sensitivity
            ),
            "specificity": float(
                specificity
            ),
            "f1": float(f1),
        })

    return results


# =========================================================
# MACRO SUMMARIES
# =========================================================

def calculate_macro_summary(
    results,
):

    metric_names = [
        "auroc",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
    ]

    summary = {}

    for metric in metric_names:

        values = np.array(
            [
                result[metric]
                for result in results
            ],
            dtype=np.float64,
        )

        summary[
            f"macro_{metric}"
        ] = float(
            np.nanmean(values)
        )

    return summary


# =========================================================
# SAVE TEST METRICS
# =========================================================

def save_test_metrics(
    results,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "disease",
        "auroc",
        "threshold",
        "tp",
        "tn",
        "fp",
        "fn",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
    ]

    with open(
        TEST_METRICS_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

# =========================================================
# SAVE TEST PREDICTIONS
# =========================================================

def save_test_predictions(
    filenames,
    targets,
    logits,
    probabilities,
    thresholds,
):
    """
    Save image-level final test outputs.

    Each row corresponds to one test X-ray.

    For every disease we store:
    - ground-truth label
    - raw model logit
    - sigmoid score
    - binary prediction using the frozen validation threshold

    IMPORTANT:
    Thresholds are NOT selected or modified here.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = ["filename"]

    for disease in CLASSES:
        fieldnames.extend([
            f"{disease}_target",
            f"{disease}_logit",
            f"{disease}_score",
            f"{disease}_prediction",
        ])

    with open(
        TEST_PREDICTIONS_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for image_idx, filename in enumerate(
            filenames
        ):

            row = {
                "filename": filename
            }

            for class_idx, disease in enumerate(
                CLASSES
            ):

                target = int(
                    targets[
                        image_idx,
                        class_idx,
                    ]
                )

                logit = float(
                    logits[
                        image_idx,
                        class_idx,
                    ]
                )

                score = float(
                    probabilities[
                        image_idx,
                        class_idx,
                    ]
                )

                threshold = float(
                    thresholds[class_idx]
                )

                prediction = int(
                    score >= threshold
                )

                row[
                    f"{disease}_target"
                ] = target

                row[
                    f"{disease}_logit"
                ] = logit

                row[
                    f"{disease}_score"
                ] = score

                row[
                    f"{disease}_prediction"
                ] = prediction

            writer.writerow(row)

# =========================================================
# MAIN
# =========================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print("MEDTRUST — FINAL TEST EVALUATION")
    print("=" * 78)

    print("Device :", device)

    if device.type == "cuda":
        print(
            "GPU    :",
            torch.cuda.get_device_name(0),
        )

    # -----------------------------------------------------
    # Frozen thresholds
    # -----------------------------------------------------

    (
        thresholds,
        threshold_artifact,
    ) = load_frozen_thresholds()

    print()
    print("Frozen threshold verification")
    print("-" * 78)

    print(
        "Threshold source :",
        threshold_artifact[
            "threshold_selection_split"
        ],
    )

    print(
        "Method           :",
        threshold_artifact[
            "selection_method"
        ],
    )

    print(
        "Threshold count  :",
        len(thresholds),
    )

    print(
        "Test used to tune:",
        threshold_artifact[
            "test_data_used_for_selection"
        ],
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model, checkpoint = load_best_model(
        device=device,
        threshold_artifact=threshold_artifact,
    )

    print()
    print("Checkpoint verification")
    print("-" * 78)

    print(
        "Checkpoint :",
        CHECKPOINT_PATH,
    )

    print(
        "Epoch      :",
        checkpoint["epoch"],
    )

    print(
        "Class order: VERIFIED"
    )

    # -----------------------------------------------------
    # Test data
    # -----------------------------------------------------

    test_loader = create_test_loader()

    print()
    print("Test data")
    print("-" * 78)

    print(
        "Images     :",
        len(test_loader.dataset),
    )

    print(
        "Batches    :",
        len(test_loader),
    )

    print(
        "Batch size :",
        BATCH_SIZE,
    )

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    print()
    print(
        "Running FINAL test inference..."
    )

    print()

    (
        filenames,
        targets,
        logits,
        probabilities,
    ) = run_test_inference(
        model=model,
        loader=test_loader,
        device=device,
    )

    verify_test_outputs(
        filenames=filenames,
        targets=targets,
        logits=logits,
        probabilities=probabilities,
    )

    print()
    print("Test inference completed.")
    print(
        "Probability range:",
        f"{probabilities.min():.6f}",
        "to",
        f"{probabilities.max():.6f}",
    )

    # -----------------------------------------------------
    # Final metrics
    # -----------------------------------------------------

    results = calculate_test_metrics(
        targets=targets,
        probabilities=probabilities,
        thresholds=thresholds,
    )

    summary = calculate_macro_summary(
        results
    )

    # -----------------------------------------------------
    # Print disease metrics
    # -----------------------------------------------------

    print()
    print("=" * 120)
    print("FINAL TEST METRICS")
    print("=" * 120)

    print(
        f"{'Disease':20}"
        f"{'AUROC':>9}"
        f"{'Thresh':>9}"
        f"{'TP':>7}"
        f"{'TN':>8}"
        f"{'FP':>7}"
        f"{'FN':>7}"
        f"{'Prec':>9}"
        f"{'Sens':>9}"
        f"{'Spec':>9}"
        f"{'F1':>9}"
    )

    print("-" * 120)

    for r in results:

        print(
            f"{r['disease']:20}"
            f"{r['auroc']:9.4f}"
            f"{r['threshold']:9.4f}"
            f"{r['tp']:7d}"
            f"{r['tn']:8d}"
            f"{r['fp']:7d}"
            f"{r['fn']:7d}"
            f"{r['precision']:9.4f}"
            f"{r['sensitivity']:9.4f}"
            f"{r['specificity']:9.4f}"
            f"{r['f1']:9.4f}"
        )

    print("-" * 120)

    # -----------------------------------------------------
    # Macro results
    # -----------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL TEST MACRO SUMMARY")
    print("=" * 78)

    print(
        "Macro AUROC       :",
        f"{summary['macro_auroc']:.6f}",
    )

    print(
        "Macro Precision   :",
        f"{summary['macro_precision']:.6f}",
    )

    print(
        "Macro Recall      :",
        f"{summary['macro_recall']:.6f}",
    )

    print(
        "Macro Sensitivity :",
        f"{summary['macro_sensitivity']:.6f}",
    )

    print(
        "Macro Specificity :",
        f"{summary['macro_specificity']:.6f}",
    )

    print(
        "Macro F1          :",
        f"{summary['macro_f1']:.6f}",
    )

    print("=" * 78)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_test_metrics(
        results
    )

    save_test_predictions(
        filenames=filenames,
        targets=targets,
        logits=logits,
        probabilities=probabilities,
        thresholds=thresholds,
    )

    print()
    print(
        "Saved final disease metrics to:",
        TEST_METRICS_PATH,
    )

    print(
        "Saved final test predictions to:",
        TEST_PREDICTIONS_PATH,
    )

    print()
    print(
        "IMPORTANT: Test metrics were calculated "
        "using frozen validation thresholds."
    )

    print(
        "No threshold optimization was performed "
        "on test data."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()