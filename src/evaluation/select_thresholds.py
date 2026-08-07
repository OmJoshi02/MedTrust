from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.dataset import CLASSES


# ============================================================
# Configuration
# ============================================================

PREDICTIONS_PATH = Path(
    "outputs/evaluation/validation_predictions.csv"
)

OUTPUT_DIR = Path(
    "outputs/evaluation"
)

THRESHOLDS_PATH = OUTPUT_DIR / "thresholds.json"

METRICS_PATH = OUTPUT_DIR / "validation_metrics.csv"

EXPECTED_IMAGES = 16828


# ============================================================
# Youden J threshold selection
# ============================================================

def select_youden_threshold(y_true, y_score):
    """
    Select the threshold that maximizes Youden's J statistic:

        J = sensitivity + specificity - 1
          = TPR - FPR

    Threshold selection is performed on VALIDATION data only.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
    )

    # sklearn may include inf as the first threshold.
    finite_mask = np.isfinite(thresholds)

    fpr = fpr[finite_mask]
    tpr = tpr[finite_mask]
    thresholds = thresholds[finite_mask]

    if len(thresholds) == 0:
        raise RuntimeError(
            "No finite ROC thresholds were available."
        )

    youden_j = tpr - fpr

    best_index = int(
        np.argmax(youden_j)
    )

    return {
        "threshold": float(thresholds[best_index]),
        "youden_j": float(youden_j[best_index]),
        "sensitivity": float(tpr[best_index]),
        "specificity": float(1.0 - fpr[best_index]),
    }


# ============================================================
# Classification metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_score,
    threshold,
):
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

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    predicted_positive = int(
        y_pred.sum()
    )

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "predicted_positive": predicted_positive,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 92)
    print("MEDTRUST — VALIDATION-ONLY THRESHOLD SELECTION")
    print("=" * 92)

    print(
        "Method       : Youden's J statistic"
    )

    print(
        "Data source  : VALIDATION predictions only"
    )

    print(
        "Test data    : NOT USED"
    )

    print(
        "Input        :",
        PREDICTIONS_PATH,
    )

    # --------------------------------------------------------
    # Load validation predictions
    # --------------------------------------------------------

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing validation predictions: "
            f"{PREDICTIONS_PATH}"
        )

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    print(
        "Images       :",
        f"{len(df):,}",
    )

    if len(df) != EXPECTED_IMAGES:
        raise RuntimeError(
            "Unexpected validation prediction count. "
            f"Expected {EXPECTED_IMAGES}, got {len(df)}."
        )

    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    thresholds_output = {
        "selection_split": "validation",
        "selection_method": "youden_j",
        "num_validation_images": len(df),
        "decision_rule": (
            "positive if sigmoid_score >= threshold"
        ),
        "classes": list(CLASSES),
        "thresholds": {},
    }

    metric_rows = []

    # --------------------------------------------------------
    # Per-disease threshold selection
    # --------------------------------------------------------

    print("\n" + "-" * 92)

    print(
        f"{'Disease':<24}"
        f"{'Threshold':>11}"
        f"{'Youden J':>11}"
        f"{'Sens':>10}"
        f"{'Spec':>10}"
        f"{'F1':>10}"
        f"{'Precision':>12}"
    )

    print("-" * 92)

    for disease in CLASSES:

        safe_name = disease.lower()

        target_column = (
            f"target_{safe_name}"
        )

        probability_column = (
            f"prob_{safe_name}"
        )

        if target_column not in df.columns:
            raise RuntimeError(
                f"Missing column: {target_column}"
            )

        if probability_column not in df.columns:
            raise RuntimeError(
                f"Missing column: {probability_column}"
            )

        y_true = (
            df[target_column]
            .to_numpy(dtype=np.int32)
        )

        y_score = (
            df[probability_column]
            .to_numpy(dtype=np.float64)
        )

        # ----------------------------------------------------
        # Validation checks
        # ----------------------------------------------------

        unique_targets = np.unique(
            y_true
        )

        if not np.all(
            np.isin(unique_targets, [0, 1])
        ):
            raise RuntimeError(
                f"Invalid targets for {disease}: "
                f"{unique_targets}"
            )

        if len(unique_targets) != 2:
            raise RuntimeError(
                f"{disease} does not contain both "
                f"positive and negative validation samples."
            )

        if not np.isfinite(y_score).all():
            raise RuntimeError(
                f"Non-finite scores detected for {disease}."
            )

        # ----------------------------------------------------
        # Select threshold
        # ----------------------------------------------------

        selected = select_youden_threshold(
            y_true,
            y_score,
        )

        threshold = selected[
            "threshold"
        ]

        # ----------------------------------------------------
        # Metrics using frozen candidate threshold
        # ----------------------------------------------------

        metrics = calculate_metrics(
            y_true,
            y_score,
            threshold,
        )

        positives = int(
            y_true.sum()
        )

        negatives = int(
            len(y_true) - positives
        )

        prevalence = (
            positives / len(y_true)
        )

        thresholds_output[
            "thresholds"
        ][disease] = {
            "threshold": threshold,
            "youden_j": selected["youden_j"],
            "validation_sensitivity": metrics["sensitivity"],
            "validation_specificity": metrics["specificity"],
        }

        metric_rows.append(
            {
                "disease": disease,
                "threshold": threshold,
                "youden_j": selected["youden_j"],
                "positives": positives,
                "negatives": negatives,
                "prevalence": prevalence,
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "predicted_positive": metrics[
                    "predicted_positive"
                ],
            }
        )

        print(
            f"{disease:<24}"
            f"{threshold:>11.4f}"
            f"{selected['youden_j']:>11.4f}"
            f"{metrics['sensitivity']:>10.4f}"
            f"{metrics['specificity']:>10.4f}"
            f"{metrics['f1']:>10.4f}"
            f"{metrics['precision']:>12.4f}"
        )

    # --------------------------------------------------------
    # Save thresholds
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        THRESHOLDS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            thresholds_output,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Macro summaries
    # --------------------------------------------------------

    macro_sensitivity = float(
        metrics_df["sensitivity"].mean()
    )

    macro_specificity = float(
        metrics_df["specificity"].mean()
    )

    macro_precision = float(
        metrics_df["precision"].mean()
    )

    macro_f1 = float(
        metrics_df["f1"].mean()
    )

    print("-" * 92)

    print("\nVALIDATION MACRO CLASSIFICATION METRICS")
    print("-" * 92)

    print(
        f"Macro sensitivity : {macro_sensitivity:.5f}"
    )

    print(
        f"Macro specificity : {macro_specificity:.5f}"
    )

    print(
        f"Macro precision   : {macro_precision:.5f}"
    )

    print(
        f"Macro F1          : {macro_f1:.5f}"
    )

    print("\nOUTPUTS")
    print("-" * 92)

    print(
        "Thresholds         :",
        THRESHOLDS_PATH,
    )

    print(
        "Validation metrics :",
        METRICS_PATH,
    )

    print("\n" + "=" * 92)

    print(
        "VALIDATION THRESHOLD SELECTION COMPLETE"
    )

    print("=" * 92)

    print(
        "\nIMPORTANT: These thresholds were selected "
        "using validation data only."
    )

    print(
        "They must be frozen before final test evaluation."
    )

    print(
        "Do NOT modify them based on test-set results."
    )

    print("=" * 92)


if __name__ == "__main__":
    main()