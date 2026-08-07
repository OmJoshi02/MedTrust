from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.dataset import CLASSES
from src.dataloader import create_dataloaders
from src.model import create_model


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
THRESHOLDS_PATH = Path("outputs/evaluation/thresholds.json")
OUTPUT_DIR = Path("outputs/evaluation")

TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
TEST_METRICS_PATH = OUTPUT_DIR / "test_metrics.csv"
TEST_SUMMARY_PATH = OUTPUT_DIR / "test_summary.json"

EXPECTED_TEST_IMAGES = 16805

BATCH_SIZE = 16
NUM_WORKERS = 4


# ============================================================
# Load frozen thresholds
# ============================================================

def load_thresholds():
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLDS_PATH}"
        )

    with open(
        THRESHOLDS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        threshold_data = json.load(file)

    # --------------------------------------------------------
    # Verify provenance
    # --------------------------------------------------------

    if threshold_data.get("selection_split") != "validation":
        raise RuntimeError(
            "Thresholds were not selected from validation data."
        )

    if threshold_data.get("selection_method") != "youden_j":
        raise RuntimeError(
            "Unexpected threshold selection method."
        )

    saved_classes = threshold_data.get("classes")

    if saved_classes != list(CLASSES):
        raise RuntimeError(
            "Threshold class order does not match CLASSES."
        )

    threshold_dict = threshold_data.get(
        "thresholds",
        {}
    )

    thresholds = []

    for disease in CLASSES:
        if disease not in threshold_dict:
            raise RuntimeError(
                f"Missing threshold for {disease}."
            )

        threshold = float(
            threshold_dict[disease]["threshold"]
        )

        if not np.isfinite(threshold):
            raise RuntimeError(
                f"Non-finite threshold for {disease}."
            )

        thresholds.append(threshold)

    thresholds = np.asarray(
        thresholds,
        dtype=np.float64,
    )

    return thresholds, threshold_data


# ============================================================
# Load frozen best model
# ============================================================

def load_model(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint missing model_state_dict."
        )

    if "classes" in checkpoint:
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


# ============================================================
# Test inference
# ============================================================

def run_test_inference(
    model,
    test_loader,
    device,
):
    all_targets = []
    all_logits = []
    all_probabilities = []
    all_filenames = []

    total_batches = len(test_loader)

    print("\nRunning FINAL TEST inference...")
    print("-" * 88)

    with torch.inference_mode():

        for batch_idx, batch in enumerate(
            test_loader,
            start=1,
        ):
            images, targets, filenames = batch

            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Non-finite logits in batch {batch_idx}."
                )

            if not torch.isfinite(probabilities).all():
                raise RuntimeError(
                    f"Non-finite probabilities in batch {batch_idx}."
                )

            all_targets.append(
                targets.cpu().numpy()
            )

            all_logits.append(
                logits.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_filenames.extend(
                filenames
            )

            if (
                batch_idx % 250 == 0
                or batch_idx == total_batches
            ):
                print(
                    f"Batch "
                    f"{batch_idx:4d}/{total_batches}"
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
        targets,
        logits,
        probabilities,
        all_filenames,
    )


# ============================================================
# Calculate final metrics
# ============================================================

def calculate_test_metrics(
    targets,
    probabilities,
    thresholds,
):
    rows = []

    aurocs = []

    print("\nFINAL TEST METRICS BY DISEASE")
    print("-" * 118)

    print(
        f"{'Disease':<23}"
        f"{'AUROC':>8}"
        f"{'Thresh':>9}"
        f"{'Sens':>9}"
        f"{'Spec':>9}"
        f"{'Prec':>9}"
        f"{'F1':>9}"
        f"{'TP':>7}"
        f"{'TN':>8}"
        f"{'FP':>8}"
        f"{'FN':>7}"
    )

    print("-" * 118)

    for idx, disease in enumerate(CLASSES):

        y_true = targets[
            :, idx
        ].astype(np.int32)

        y_score = probabilities[
            :, idx
        ].astype(np.float64)

        threshold = float(
            thresholds[idx]
        )

        positives = int(
            y_true.sum()
        )

        negatives = int(
            len(y_true) - positives
        )

        # ----------------------------------------------------
        # AUROC
        # ----------------------------------------------------

        if positives == 0 or negatives == 0:
            auroc = np.nan
        else:
            auroc = float(
                roc_auc_score(
                    y_true,
                    y_score,
                )
            )

            aurocs.append(auroc)

        # ----------------------------------------------------
        # Apply FROZEN validation threshold
        # ----------------------------------------------------

        y_pred = (
            y_score >= threshold
        ).astype(np.int32)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else np.nan
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        precision = float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        recall = float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        f1 = float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        prevalence = (
            positives / len(y_true)
        )

        predicted_positive = int(
            y_pred.sum()
        )

        rows.append(
            {
                "disease": disease,
                "auroc": auroc,
                "threshold": threshold,
                "positives": positives,
                "negatives": negatives,
                "prevalence": prevalence,
                "predicted_positive": predicted_positive,
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "f1": f1,
            }
        )

        print(
            f"{disease:<23}"
            f"{auroc:>8.4f}"
            f"{threshold:>9.4f}"
            f"{sensitivity:>9.4f}"
            f"{specificity:>9.4f}"
            f"{precision:>9.4f}"
            f"{f1:>9.4f}"
            f"{tp:>7}"
            f"{tn:>8}"
            f"{fp:>8}"
            f"{fn:>7}"
        )

    macro_auroc = float(
        np.mean(aurocs)
    )

    metrics_df = pd.DataFrame(
        rows
    )

    macro_metrics = {
        "macro_auroc": macro_auroc,
        "macro_sensitivity": float(
            metrics_df["sensitivity"].mean()
        ),
        "macro_specificity": float(
            metrics_df["specificity"].mean()
        ),
        "macro_precision": float(
            metrics_df["precision"].mean()
        ),
        "macro_recall": float(
            metrics_df["recall"].mean()
        ),
        "macro_f1": float(
            metrics_df["f1"].mean()
        ),
    }

    return metrics_df, macro_metrics


# ============================================================
# Save predictions
# ============================================================

def save_test_predictions(
    filenames,
    targets,
    logits,
    probabilities,
    thresholds,
):
    data = {
        "filename": filenames,
    }

    for idx, disease in enumerate(CLASSES):

        safe_name = disease.lower()

        y_pred = (
            probabilities[:, idx]
            >= thresholds[idx]
        ).astype(np.int32)

        data[
            f"target_{safe_name}"
        ] = targets[:, idx].astype(
            np.int32
        )

        data[
            f"logit_{safe_name}"
        ] = logits[:, idx]

        data[
            f"prob_{safe_name}"
        ] = probabilities[:, idx]

        data[
            f"pred_{safe_name}"
        ] = y_pred

    predictions_df = pd.DataFrame(
        data
    )

    predictions_df.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 88)
    print("MEDTRUST — FINAL HELD-OUT TEST EVALUATION")
    print("=" * 88)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Test labels are used ONLY for final metric calculation."
    )

    print(
        "No threshold optimization is performed on the test set."
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice       :", device)

    if torch.cuda.is_available():
        print(
            "GPU          :",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Frozen thresholds
    # --------------------------------------------------------

    thresholds, threshold_data = (
        load_thresholds()
    )

    print(
        "Thresholds   : VERIFIED — validation only"
    )

    print(
        "Method       :",
        threshold_data["selection_method"],
    )

    print("\nFROZEN THRESHOLDS")
    print("-" * 55)

    for disease, threshold in zip(
        CLASSES,
        thresholds,
    ):
        print(
            f"{disease:<24}"
            f"{threshold:.6f}"
        )

    # --------------------------------------------------------
    # Test loader
    # --------------------------------------------------------

    _, _, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    test_size = len(
        test_loader.dataset
    )

    print(
        "\nTest images  :",
        f"{test_size:,}",
    )

    if test_size != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            "Unexpected test size. "
            f"Expected {EXPECTED_TEST_IMAGES}, "
            f"got {test_size}."
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, checkpoint = load_model(
        device
    )

    print(
        "Checkpoint   :",
        CHECKPOINT_PATH,
    )

    print(
        "Best epoch   :",
        checkpoint.get("epoch", "unknown"),
    )

    if "best_macro_auroc" in checkpoint:
        print(
            "Val AUROC    :",
            f"{float(checkpoint['best_macro_auroc']):.5f}",
        )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    (
        targets,
        logits,
        probabilities,
        filenames,
    ) = run_test_inference(
        model,
        test_loader,
        device,
    )

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    print("\nTEST INFERENCE CHECK")
    print("-" * 88)

    print(
        "Images processed   :",
        f"{len(filenames):,}",
    )

    print(
        "Targets shape      :",
        targets.shape,
    )

    print(
        "Logits shape       :",
        logits.shape,
    )

    print(
        "Probabilities shape:",
        probabilities.shape,
    )

    print(
        "Probability range  :",
        f"{probabilities.min():.6f}",
        "to",
        f"{probabilities.max():.6f}",
    )

    expected_shape = (
        EXPECTED_TEST_IMAGES,
        len(CLASSES),
    )

    if targets.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected target shape: {targets.shape}"
        )

    if probabilities.shape != expected_shape:
        raise RuntimeError(
            "Unexpected probability shape: "
            f"{probabilities.shape}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics_df, macro_metrics = (
        calculate_test_metrics(
            targets,
            probabilities,
            thresholds,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        TEST_METRICS_PATH,
        index=False,
    )

    save_test_predictions(
        filenames,
        targets,
        logits,
        probabilities,
        thresholds,
    )

    summary = {
        "split": "test",
        "num_images": EXPECTED_TEST_IMAGES,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint.get(
            "epoch"
        ),
        "threshold_source": "validation",
        "threshold_method": "youden_j",
        "threshold_file": str(
            THRESHOLDS_PATH
        ),
        **macro_metrics,
    }

    with open(
        TEST_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 88)
    print("FINAL TEST MACRO METRICS")
    print("=" * 88)

    print(
        f"Macro AUROC       : "
        f"{macro_metrics['macro_auroc']:.5f}"
    )

    print(
        f"Macro sensitivity : "
        f"{macro_metrics['macro_sensitivity']:.5f}"
    )

    print(
        f"Macro specificity : "
        f"{macro_metrics['macro_specificity']:.5f}"
    )

    print(
        f"Macro precision   : "
        f"{macro_metrics['macro_precision']:.5f}"
    )

    print(
        f"Macro recall      : "
        f"{macro_metrics['macro_recall']:.5f}"
    )

    print(
        f"Macro F1          : "
        f"{macro_metrics['macro_f1']:.5f}"
    )

    print("\nOUTPUT FILES")
    print("-" * 88)

    print(
        "Test predictions :",
        TEST_PREDICTIONS_PATH,
    )

    print(
        "Test metrics     :",
        TEST_METRICS_PATH,
    )

    print(
        "Test summary     :",
        TEST_SUMMARY_PATH,
    )

    print("\n" + "=" * 88)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 88)

    print(
        "\nThe validation-derived thresholds were "
        "applied unchanged."
    )

    print(
        "No test-set threshold optimization was performed."
    )

    print("=" * 88)


if __name__ == "__main__":
    main()