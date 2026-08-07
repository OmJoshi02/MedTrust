from pathlib import Path
import csv
import json

import numpy as np

from src.dataset import CLASSES


# =========================================================
# PATHS
# =========================================================

OUTPUT_DIR = Path("outputs/evaluation")

THRESHOLDS_PATH = OUTPUT_DIR / "thresholds.json"

VALIDATION_METRICS_PATH = (
    OUTPUT_DIR / "validation_metrics.csv"
)

TEST_METRICS_PATH = (
    OUTPUT_DIR / "test_metrics.csv"
)

VALIDATION_PREDICTIONS_PATH = (
    OUTPUT_DIR / "validation_predictions.csv"
)

TEST_PREDICTIONS_PATH = (
    OUTPUT_DIR / "test_predictions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "evaluation_summary.json"
)


# =========================================================
# HELPERS
# =========================================================

def load_csv_metrics(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != len(CLASSES):
        raise ValueError(
            f"Expected {len(CLASSES)} metric rows "
            f"in {path}, found {len(rows)}."
        )

    diseases = [
        row["disease"]
        for row in rows
    ]

    if diseases != list(CLASSES):
        raise ValueError(
            f"Class order mismatch in {path}."
        )

    return rows


def macro_mean(
    rows,
    metric,
):
    values = np.array(
        [
            float(row[metric])
            for row in rows
        ],
        dtype=np.float64,
    )

    return float(
        np.nanmean(values)
    )


def count_prediction_rows(
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        # Subtract CSV header.
        count = sum(
            1 for _ in f
        ) - 1

    return count


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 78)
    print("MEDTRUST — CREATE EVALUATION SUMMARY")
    print("=" * 78)

    # -----------------------------------------------------
    # Load frozen threshold artifact
    # -----------------------------------------------------

    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Threshold artifact not found: "
            f"{THRESHOLDS_PATH}"
        )

    with open(
        THRESHOLDS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        threshold_artifact = json.load(f)

    if threshold_artifact["classes"] != list(CLASSES):
        raise ValueError(
            "Threshold artifact class order mismatch."
        )

    if (
        threshold_artifact[
            "test_data_used_for_selection"
        ]
        is not False
    ):
        raise ValueError(
            "Threshold artifact does not confirm "
            "test independence."
        )

    # -----------------------------------------------------
    # Load metrics
    # -----------------------------------------------------

    validation_rows = load_csv_metrics(
        VALIDATION_METRICS_PATH
    )

    test_rows = load_csv_metrics(
        TEST_METRICS_PATH
    )

    # -----------------------------------------------------
    # Verify prediction counts
    # -----------------------------------------------------

    validation_samples = (
        count_prediction_rows(
            VALIDATION_PREDICTIONS_PATH
        )
    )

    test_samples = (
        count_prediction_rows(
            TEST_PREDICTIONS_PATH
        )
    )

    if validation_samples != 16828:
        raise ValueError(
            "Validation prediction count mismatch."
        )

    if test_samples != 16805:
        raise ValueError(
            "Test prediction count mismatch."
        )

    # -----------------------------------------------------
    # Macro summaries
    # -----------------------------------------------------

    validation_summary = {
        "macro_auroc": macro_mean(
            validation_rows,
            "auroc",
        ),
        "macro_precision": macro_mean(
            validation_rows,
            "precision",
        ),
        "macro_recall": macro_mean(
            validation_rows,
            "recall",
        ),
        "macro_sensitivity": macro_mean(
            validation_rows,
            "sensitivity",
        ),
        "macro_specificity": macro_mean(
            validation_rows,
            "specificity",
        ),
        "macro_f1": macro_mean(
            validation_rows,
            "f1",
        ),
    }

    test_summary = {
        "macro_auroc": macro_mean(
            test_rows,
            "auroc",
        ),
        "macro_precision": macro_mean(
            test_rows,
            "precision",
        ),
        "macro_recall": macro_mean(
            test_rows,
            "recall",
        ),
        "macro_sensitivity": macro_mean(
            test_rows,
            "sensitivity",
        ),
        "macro_specificity": macro_mean(
            test_rows,
            "specificity",
        ),
        "macro_f1": macro_mean(
            test_rows,
            "f1",
        ),
    }

    # -----------------------------------------------------
    # Final summary artifact
    # -----------------------------------------------------

    summary = {
        "project": "MedTrust",

        "module": (
            "Final Model Evaluation and "
            "Decision Threshold Selection"
        ),

        "task": (
            "14-class multi-label chest X-ray "
            "disease classification"
        ),

        "model": "DenseNet-121",

        "checkpoint": (
            threshold_artifact["checkpoint"]
        ),

        "checkpoint_epoch": (
            threshold_artifact[
                "checkpoint_epoch"
            ]
        ),

        "classes": list(CLASSES),

        "num_classes": len(CLASSES),

        "splits": {
            "validation_images": validation_samples,
            "test_images": test_samples,
            "patient_leakage": 0,
        },

        "threshold_selection": {
            "split": "validation",
            "method": threshold_artifact[
                "selection_method"
            ],
            "formula": threshold_artifact[
                "selection_formula"
            ],
            "threshold_count": len(
                threshold_artifact[
                    "thresholds"
                ]
            ),
            "test_data_used": False,
            "status": "frozen",
        },

        "validation": validation_summary,

        "test": test_summary,

        "artifacts": {
            "thresholds": str(
                THRESHOLDS_PATH
            ),
            "validation_metrics": str(
                VALIDATION_METRICS_PATH
            ),
            "validation_predictions": str(
                VALIDATION_PREDICTIONS_PATH
            ),
            "test_metrics": str(
                TEST_METRICS_PATH
            ),
            "test_predictions": str(
                TEST_PREDICTIONS_PATH
            ),
        },

        "scientific_notes": {
            "auroc_uses_continuous_scores": True,

            "binary_metrics_use_frozen_validation_thresholds": True,

            "raw_sigmoid_scores_are_not_assumed_calibrated_probabilities": True,

            "test_set_used_for_final_evaluation_only": True,
        },

        "module_status": "complete",
    }

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print()
    print("Validation")
    print("-" * 78)

    print(
        "Images      :",
        validation_samples,
    )

    print(
        "Macro AUROC :",
        f"{validation_summary['macro_auroc']:.6f}",
    )

    print()
    print("Final test")
    print("-" * 78)

    print(
        "Images             :",
        test_samples,
    )

    print(
        "Macro AUROC        :",
        f"{test_summary['macro_auroc']:.6f}",
    )

    print(
        "Macro Precision    :",
        f"{test_summary['macro_precision']:.6f}",
    )

    print(
        "Macro Sensitivity  :",
        f"{test_summary['macro_sensitivity']:.6f}",
    )

    print(
        "Macro Specificity  :",
        f"{test_summary['macro_specificity']:.6f}",
    )

    print(
        "Macro F1           :",
        f"{test_summary['macro_f1']:.6f}",
    )

    print()
    print(
        "Summary saved to:",
        SUMMARY_PATH,
    )

    print()
    print("=" * 78)
    print("MODULE 1 STATUS: COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()