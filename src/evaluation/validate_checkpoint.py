from pathlib import Path
import csv
import json
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from torch.utils.data import DataLoader

from src.dataset import CLASSES, NIHChestXrayDataset
from src.model import create_model
from src.transformer import val_transform


# =========================================================
# CONFIGURATION
# =========================================================

CHECKPOINT_PATH = Path("checkpoints/best_model.pt")

IMAGE_DIR = "dataset/images"
METADATA_CSV = "dataset/Data_Entry_2017_v2020.csv"
VAL_SPLIT = "splits/val_list.txt"

BATCH_SIZE = 16
NUM_WORKERS = 4

EXPECTED_VAL_SAMPLES = 16828
EXPECTED_MACRO_AUROC = 0.82777

# This is only a verification tolerance.
# We expect reproduction to be much closer than this.
AUROC_TOLERANCE = 1e-4

OUTPUT_DIR = Path("outputs/evaluation")
THRESHOLDS_PATH = OUTPUT_DIR / "thresholds.json"
OUTPUT_DIR = Path("outputs/evaluation")
THRESHOLDS_PATH = OUTPUT_DIR / "thresholds.json"
VALIDATION_METRICS_PATH = (
    OUTPUT_DIR / "validation_metrics.csv"
)

VALIDATION_PREDICTIONS_PATH = (
    OUTPUT_DIR / "validation_predictions.csv"
)
# =========================================================
# LOAD CHECKPOINT + MODEL
# =========================================================

def load_best_model(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    required_keys = {
        "epoch",
        "model_state_dict",
        "best_macro_auroc",
        "classes",
    }

    missing_keys = required_keys - set(checkpoint.keys())

    if missing_keys:
        raise KeyError(
            f"Checkpoint missing required keys: {missing_keys}"
        )

    checkpoint_classes = checkpoint["classes"]

    if list(checkpoint_classes) != list(CLASSES):
        raise ValueError(
            "Checkpoint class order does not match "
            "src.dataset.CLASSES."
        )

    model = create_model(
        num_classes=len(CLASSES),

        # We immediately load every parameter from the
        # checkpoint, so downloading ImageNet weights here
        # is unnecessary.
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
# VALIDATION LOADER
# =========================================================

def create_validation_loader():
    dataset = NIHChestXrayDataset(
        image_dir=IMAGE_DIR,
        metadata_csv=METADATA_CSV,
        split_file=VAL_SPLIT,
        transform=val_transform,
    )

    if len(dataset) != EXPECTED_VAL_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_VAL_SAMPLES} validation images, "
            f"but found {len(dataset)}."
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
# VALIDATION INFERENCE
# =========================================================

@torch.no_grad()
def run_validation_inference(
    model,
    loader,
    device,
):
    model.eval()

    all_filenames = []
    all_targets = []
    all_logits = []
    all_probabilities = []

    for batch_idx, (images, targets, filenames) in enumerate(
        loader
    ):
        images = images.to(
            device,
            non_blocking=True,
        )

        # Match the validation inference behavior used
        # during training.
        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)

        probabilities = torch.sigmoid(logits)

        all_filenames.extend(filenames)

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
                (batch_idx + 1) * loader.batch_size,
                len(loader.dataset),
            )

            print(
                f"Processed "
                f"{processed:5d}/{len(loader.dataset)} "
                f"validation images"
            )

    targets_np = np.concatenate(
        all_targets,
        axis=0,
    )

    logits_np = np.concatenate(
        all_logits,
        axis=0,
    )

    probabilities_np = np.concatenate(
        all_probabilities,
        axis=0,
    )

    return (
        all_filenames,
        targets_np,
        logits_np,
        probabilities_np,
    )


# =========================================================
# AUROC
# =========================================================

def calculate_aurocs(
    targets,
    probabilities,
):
    class_aurocs = []

    for class_idx, disease in enumerate(CLASSES):
        y_true = targets[:, class_idx]
        y_score = probabilities[:, class_idx]

        unique_labels = np.unique(y_true)

        if len(unique_labels) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(
                y_true,
                y_score,
            )

        class_aurocs.append(auc)

    macro_auroc = float(
        np.nanmean(class_aurocs)
    )

    return class_aurocs, macro_auroc

# =========================================================
# VALIDATION THRESHOLD SELECTION
# =========================================================

def select_youden_thresholds(
    targets,
    probabilities,
):
    """
    Select one threshold per disease using validation data only.

    Youden's J:
        J = sensitivity + specificity - 1
          = TPR - FPR

    The threshold maximizing J is selected independently
    for each disease.

    IMPORTANT:
        These thresholds are derived from VALIDATION ONLY.
        The test set must not influence them.
    """

    results = []

    for class_idx, disease in enumerate(CLASSES):

        y_true = targets[:, class_idx]
        y_score = probabilities[:, class_idx]

        if len(np.unique(y_true)) < 2:
            raise ValueError(
                f"Cannot select threshold for {disease}: "
                "validation data does not contain both "
                "positive and negative samples."
            )

        fpr, tpr, thresholds = roc_curve(
            y_true,
            y_score,
        )

        # Youden's J = sensitivity + specificity - 1
        # Since specificity = 1 - FPR:
        #
        # J = TPR - FPR
        youden_j = tpr - fpr

        # roc_curve may include an infinite first threshold.
        # Exclude non-finite thresholds from selection.
        finite_mask = np.isfinite(thresholds)

        if not finite_mask.any():
            raise ValueError(
                f"No finite ROC thresholds available "
                f"for {disease}."
            )

        valid_indices = np.where(finite_mask)[0]

        best_relative_idx = np.argmax(
            youden_j[finite_mask]
        )

        best_idx = valid_indices[
            best_relative_idx
        ]

        best_threshold = float(
            thresholds[best_idx]
        )

        best_sensitivity = float(
            tpr[best_idx]
        )

        best_specificity = float(
            1.0 - fpr[best_idx]
        )

        best_j = float(
            youden_j[best_idx]
        )

        results.append({
            "disease": disease,
            "threshold": best_threshold,
            "youden_j": best_j,
            "sensitivity": best_sensitivity,
            "specificity": best_specificity,
        })

    return results

# =========================================================
# YOUDEN THRESHOLD TIE CHECK
# =========================================================

def check_youden_ties(
    targets,
    probabilities,
    atol=1e-12,
):
    """
    Check whether multiple finite ROC thresholds achieve
    the same maximum Youden J for any disease.

    This is performed using VALIDATION DATA ONLY.
    """

    tie_results = []

    for class_idx, disease in enumerate(CLASSES):

        y_true = targets[:, class_idx]
        y_score = probabilities[:, class_idx]

        fpr, tpr, thresholds = roc_curve(
            y_true,
            y_score,
        )

        youden_j = tpr - fpr

        finite_mask = np.isfinite(thresholds)

        finite_thresholds = thresholds[finite_mask]
        finite_j = youden_j[finite_mask]

        if len(finite_thresholds) == 0:
            raise ValueError(
                f"No finite ROC thresholds for {disease}."
            )

        max_j = np.max(finite_j)

        # Find every threshold whose J is numerically
        # equal to the maximum J.
        tied_mask = np.isclose(
            finite_j,
            max_j,
            rtol=0.0,
            atol=atol,
        )

        tied_thresholds = finite_thresholds[
            tied_mask
        ]

        tie_results.append({
            "disease": disease,
            "max_j": float(max_j),
            "num_maxima": int(len(tied_thresholds)),
            "thresholds": [
                float(x)
                for x in tied_thresholds
            ],
        })

    return tie_results


# =========================================================
# FREEZE VALIDATION THRESHOLDS
# =========================================================

def save_frozen_thresholds(
    threshold_results,
    checkpoint,
    macro_auroc,
):
    """
    Save the final disease-specific decision thresholds.

    These thresholds were selected using VALIDATION DATA ONLY
    and must remain unchanged during final test evaluation.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_map = {
        result["disease"]: result["threshold"]
        for result in threshold_results
    }

    selection_details = {
        result["disease"]: {
            "threshold": result["threshold"],
            "youden_j": result["youden_j"],
            "sensitivity": result["sensitivity"],
            "specificity": result["specificity"],
        }
        for result in threshold_results
    }

    artifact = {
        "project": "MedTrust",
        "module": "Final Model Evaluation and Decision Threshold Selection",

        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": int(checkpoint["epoch"]),

        "threshold_selection_split": "validation",
        "validation_split_file": VAL_SPLIT,
        "validation_samples": EXPECTED_VAL_SAMPLES,

        "selection_method": "Youden's J",
        "selection_formula": "sensitivity + specificity - 1",

        "validation_macro_auroc": float(macro_auroc),

        "classes": list(CLASSES),

        "thresholds": threshold_map,

        "validation_selection_details": selection_details,

        "test_data_used_for_selection": False,
    }

    # Safety: do not silently overwrite an already frozen
    # threshold artifact.
    if THRESHOLDS_PATH.exists():
        raise FileExistsError(
            f"Frozen threshold file already exists: "
            f"{THRESHOLDS_PATH}\n"
            "Refusing to overwrite it automatically."
        )

    with open(
        THRESHOLDS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            artifact,
            f,
            indent=4,
        )

    return artifact

# =========================================================
# SANITY CHECKS
# =========================================================

def verify_collected_data(
    filenames,
    targets,
    logits,
    probabilities,
):
    n = EXPECTED_VAL_SAMPLES
    c = len(CLASSES)

    if len(filenames) != n:
        raise ValueError(
            f"Filename count mismatch: {len(filenames)} != {n}"
        )

    if targets.shape != (n, c):
        raise ValueError(
            f"Unexpected target shape: {targets.shape}"
        )

    if logits.shape != (n, c):
        raise ValueError(
            f"Unexpected logits shape: {logits.shape}"
        )

    if probabilities.shape != (n, c):
        raise ValueError(
            "Unexpected probability shape: "
            f"{probabilities.shape}"
        )

    if not np.isfinite(logits).all():
        raise ValueError(
            "Non-finite logits detected."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite probabilities detected."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(
            "Sigmoid outputs outside [0, 1]."
        )

    if len(set(filenames)) != len(filenames):
        raise ValueError(
            "Duplicate filenames detected in validation inference."
        )

# =========================================================
# VALIDATION METRICS AT FROZEN THRESHOLDS
# =========================================================

def calculate_validation_metrics(
    targets,
    probabilities,
    threshold_results,
    class_aurocs,
):
    """
    Calculate validation metrics using the already-selected
    validation thresholds.

    This does NOT re-select or modify thresholds.
    """

    results = []

    for class_idx, disease in enumerate(CLASSES):

        y_true = targets[:, class_idx].astype(int)
        y_score = probabilities[:, class_idx]

        threshold = float(
            threshold_results[class_idx]["threshold"]
        )

        y_pred = (
            y_score >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

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
            "auroc": float(class_aurocs[class_idx]),
            "threshold": threshold,

            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),

            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
        })

    return results


# =========================================================
# SAVE VALIDATION METRICS
# =========================================================

def save_validation_metrics(
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
        VALIDATION_METRICS_PATH,
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
# SAVE VALIDATION PREDICTIONS
# =========================================================

def save_validation_predictions(
    filenames,
    targets,
    logits,
    probabilities,
    threshold_results,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    thresholds = np.array(
        [
            result["threshold"]
            for result in threshold_results
        ],
        dtype=np.float64,
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
        VALIDATION_PREDICTIONS_PATH,
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

                score = float(
                    probabilities[
                        image_idx,
                        class_idx,
                    ]
                )

                row[
                    f"{disease}_target"
                ] = int(
                    targets[
                        image_idx,
                        class_idx,
                    ]
                )

                row[
                    f"{disease}_logit"
                ] = float(
                    logits[
                        image_idx,
                        class_idx,
                    ]
                )

                row[
                    f"{disease}_score"
                ] = score

                row[
                    f"{disease}_prediction"
                ] = int(
                    score >= thresholds[class_idx]
                )

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
    print("MEDTRUST — VALIDATION CHECKPOINT EVALUATION")
    print("=" * 78)

    print("Device     :", device)

    if device.type == "cuda":
        print(
            "GPU        :",
            torch.cuda.get_device_name(0),
        )

    print("Checkpoint :", CHECKPOINT_PATH)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model, checkpoint = load_best_model(device)

    print()
    print("Checkpoint verification")
    print("-" * 78)

    print(
        "Best epoch              :",
        checkpoint["epoch"],
    )

    print(
        "Stored best Macro AUROC :",
        f"{checkpoint['best_macro_auroc']:.8f}",
    )

    print(
        "Class order             : VERIFIED"
    )

    print(
        "Model state             : LOADED STRICTLY"
    )

    # -----------------------------------------------------
    # Validation data
    # -----------------------------------------------------

    val_loader = create_validation_loader()

    print()
    print("Validation data")
    print("-" * 78)

    print(
        "Images     :",
        len(val_loader.dataset),
    )

    print(
        "Batches    :",
        len(val_loader),
    )

    print(
        "Batch size :",
        BATCH_SIZE,
    )

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    print()
    print("Running validation inference...")
    print()

    (
        filenames,
        targets,
        logits,
        probabilities,
    ) = run_validation_inference(
        model=model,
        loader=val_loader,
        device=device,
    )

    verify_collected_data(
        filenames=filenames,
        targets=targets,
        logits=logits,
        probabilities=probabilities,
    )

    print()
    print("Collected arrays")
    print("-" * 78)

    print(
        "Filenames     :",
        len(filenames),
    )

    print(
        "Targets       :",
        targets.shape,
    )

    print(
        "Logits        :",
        logits.shape,
    )

    print(
        "Probabilities :",
        probabilities.shape,
    )

    print(
        "Probability range:",
        f"{probabilities.min():.6f}",
        "to",
        f"{probabilities.max():.6f}",
    )

    # -----------------------------------------------------
    # AUROC
    # -----------------------------------------------------

    (
        class_aurocs,
        macro_auroc,
    ) = calculate_aurocs(
        targets=targets,
        probabilities=probabilities,
    )

    print()
    print("=" * 78)
    print("VALIDATION AUROC")
    print("=" * 78)

    for disease, auc in zip(
        CLASSES,
        class_aurocs,
    ):
        if np.isnan(auc):
            auc_text = "NaN"
        else:
            auc_text = f"{auc:.6f}"

        print(
            f"{disease:20}: {auc_text}"
        )

    stored_auroc = float(
        checkpoint["best_macro_auroc"]
    )

    difference = abs(
        macro_auroc - stored_auroc
    )

    print()
    print("-" * 78)

    print(
        "Reproduced Macro AUROC :",
        f"{macro_auroc:.8f}",
    )

    print(
        "Checkpoint Macro AUROC :",
        f"{stored_auroc:.8f}",
    )

    print(
        "Absolute difference    :",
        f"{difference:.10f}",
    )

    print("-" * 78)

    if difference <= AUROC_TOLERANCE:
        print(
            "RESULT: VALIDATION AUROC REPRODUCTION PASSED"
        )
    else:
        print(
            "RESULT: VALIDATION AUROC REPRODUCTION FAILED"
        )

        print()
        print(
            "STOP HERE. Do not select thresholds "
            "or evaluate the test set."
        )

    print("=" * 78)


    

        # -----------------------------------------------------
    # Validation-based threshold selection
    # -----------------------------------------------------

    if difference > AUROC_TOLERANCE:
        raise RuntimeError(
            "Validation AUROC reproduction failed. "
            "Threshold selection aborted."
        )

    threshold_results = select_youden_thresholds(
        targets=targets,
        probabilities=probabilities,
    )

    print()
    print("=" * 78)
    print("VALIDATION-BASED YOUDEN THRESHOLDS")
    print("=" * 78)

    print(
        f"{'Disease':20}"
        f"{'Threshold':>12}"
        f"{'Sensitivity':>14}"
        f"{'Specificity':>14}"
        f"{'Youden J':>12}"
    )

    print("-" * 78)

    for result in threshold_results:

        print(
            f"{result['disease']:20}"
            f"{result['threshold']:12.6f}"
            f"{result['sensitivity']:14.6f}"
            f"{result['specificity']:14.6f}"
            f"{result['youden_j']:12.6f}"
        )

    print("-" * 78)

    print()
    print(
        "NOTE: Thresholds above were derived from "
        "VALIDATION DATA ONLY."
    )

    print(
        "Thresholds will now be compared against "
        "the existing frozen threshold artifact."
    )

    print(
        "Test data has NOT been used."
    )

    print("=" * 78)


        # -----------------------------------------------------
    # Youden maximum tie check
    # -----------------------------------------------------

    tie_results = check_youden_ties(
        targets=targets,
        probabilities=probabilities,
    )

    print()
    print("=" * 78)
    print("YOUDEN THRESHOLD TIE CHECK")
    print("=" * 78)

    any_ties = False

    for result in tie_results:

        num_maxima = result["num_maxima"]

        if num_maxima > 1:
            any_ties = True

            print()
            print(
                f"{result['disease']}: "
                f"{num_maxima} equally optimal thresholds"
            )

            print(
                "Thresholds:",
                [
                    round(x, 6)
                    for x in result["thresholds"]
                ],
            )

            print(
                "Maximum J:",
                f"{result['max_j']:.6f}",
            )

        else:
            print(
                f"{result['disease']:20}: "
                "unique optimum"
            )

    print()
    print("-" * 78)

    if any_ties:
        print(
            "RESULT: One or more diseases have tied "
            "maximum-J thresholds."
        )

        raise RuntimeError(
            "Threshold freezing aborted because "
            "maximum-J ties were detected."
        )

    else:
        print(
            "RESULT: All 14 diseases have a unique "
            "maximum-J threshold."
        )

        print(
            "Threshold selection is ready for freezing."
        )

    print("-" * 78)

    print(
        "Test data has NOT been used."
    )

    print("=" * 78)

    # -----------------------------------------------------
    # Freeze thresholds
    # -----------------------------------------------------

        # -----------------------------------------------------
    # Frozen threshold artifact verification
    # -----------------------------------------------------

    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Expected frozen threshold artifact not found: "
            f"{THRESHOLDS_PATH}"
        )

    with open(
        THRESHOLDS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        frozen_artifact = json.load(f)

    frozen_thresholds = frozen_artifact[
        "thresholds"
    ]

    for result in threshold_results:

        disease = result["disease"]

        stored_threshold = float(
            frozen_thresholds[disease]
        )

        reproduced_threshold = float(
            result["threshold"]
        )

        if not np.isclose(
            stored_threshold,
            reproduced_threshold,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Frozen threshold mismatch for "
                f"{disease}."
            )

    print()
    print("=" * 78)
    print("FROZEN THRESHOLD VERIFICATION")
    print("=" * 78)

    print(
        "All 14 reproduced validation thresholds "
        "match thresholds.json."
    )

    print(
        "Frozen thresholds were NOT modified."
    )

    print("=" * 78)


        # -----------------------------------------------------
    # Validation metrics using frozen thresholds
    # -----------------------------------------------------

    validation_results = (
        calculate_validation_metrics(
            targets=targets,
            probabilities=probabilities,
            threshold_results=threshold_results,
            class_aurocs=class_aurocs,
        )
    )

    save_validation_metrics(
        validation_results
    )

    save_validation_predictions(
        filenames=filenames,
        targets=targets,
        logits=logits,
        probabilities=probabilities,
        threshold_results=threshold_results,
    )

    print()
    print("=" * 78)
    print("VALIDATION ARTIFACTS SAVED")
    print("=" * 78)

    print(
        "Metrics     :",
        VALIDATION_METRICS_PATH,
    )

    print(
        "Predictions :",
        VALIDATION_PREDICTIONS_PATH,
    )

    print(
        "Images      :",
        len(filenames),
    )

    print("=" * 78)

if __name__ == "__main__":
    main()