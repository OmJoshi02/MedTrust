"""
Evaluate frozen MedTrust temperature scaling.

The calibrator was fitted using validation data only.

This module:
    1. Loads frozen validation-derived temperatures.
    2. Evaluates calibration on validation data.
    3. Evaluates calibration on the untouched test data.
    4. Compares uncalibrated and calibrated:
        - NLL
        - Brier score
        - ECE
        - AUROC

No calibration parameters are fitted or modified here.
"""

import csv
import json
from pathlib import Path

import numpy as np

from src.calibration.metrics import (
    calculate_binary_calibration_metrics,
)


# =========================================================
# CONFIGURATION
# =========================================================

VALIDATION_PREDICTIONS_PATH = Path(
    "outputs/evaluation/validation_predictions.csv"
)

TEST_PREDICTIONS_PATH = Path(
    "outputs/evaluation/test_predictions.csv"
)

TEMPERATURE_PATH = Path(
    "outputs/calibration/temperature.json"
)

OUTPUT_DIR = Path(
    "outputs/calibration"
)

VALIDATION_METRICS_PATH = (
    OUTPUT_DIR
    / "validation_calibration_metrics.csv"
)

TEST_METRICS_PATH = (
    OUTPUT_DIR
    / "test_calibration_metrics.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "calibration_summary.json"
)

N_BINS = 15


CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


CLASS_TO_CSV_NAME = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "effusion",
    "Infiltration": "infiltration",
    "Mass": "mass",
    "Nodule": "nodule",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Pleural_Thickening": "pleural_thickening",
    "Hernia": "hernia",
}


# =========================================================
# NUMERIC HELPERS
# =========================================================

def stable_sigmoid(x):
    """
    Numerically stable sigmoid for NumPy arrays.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    result = np.empty_like(
        x,
        dtype=np.float64,
    )

    positive = x >= 0
    negative = ~positive

    result[positive] = (
        1.0
        / (
            1.0
            + np.exp(-x[positive])
        )
    )

    exp_x = np.exp(
        x[negative]
    )

    result[negative] = (
        exp_x
        / (1.0 + exp_x)
    )

    return result


# =========================================================
# LOAD FROZEN TEMPERATURES
# =========================================================

def load_temperatures():

    if not TEMPERATURE_PATH.exists():
        raise FileNotFoundError(
            "Frozen temperature artifact not found: "
            f"{TEMPERATURE_PATH}"
        )

    with open(
        TEMPERATURE_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        artifact = json.load(f)

    if artifact.get(
        "test_data_used_for_fitting"
    ) is not False:
        raise RuntimeError(
            "Calibration artifact does not explicitly "
            "confirm validation-only fitting."
        )

    stored_classes = artifact.get(
        "classes"
    )

    if stored_classes != CLASSES:
        raise RuntimeError(
            "Calibration class order mismatch."
        )

    temperature_map = artifact.get(
        "temperatures"
    )

    if temperature_map is None:
        raise RuntimeError(
            "Temperature map missing from artifact."
        )

    temperatures = np.asarray(
        [
            float(
                temperature_map[disease]
            )
            for disease in CLASSES
        ],
        dtype=np.float64,
    )

    if not np.isfinite(
        temperatures
    ).all():
        raise RuntimeError(
            "Non-finite temperature detected."
        )

    if np.any(
        temperatures <= 0.0
    ):
        raise RuntimeError(
            "All temperatures must be positive."
        )

    return temperatures, artifact


# =========================================================
# LOAD PREDICTION ARTIFACT
# =========================================================

def load_predictions(
    path,
):
    """
    Load targets, logits and saved probabilities from one
    evaluation prediction CSV.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction artifact not found: {path}"
        )

    targets = []
    logits = []
    probabilities = []

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"{path} has no CSV header."
            )

        required_columns = {
            "filename"
        }

        for disease in CLASSES:

            csv_name = CLASS_TO_CSV_NAME[
                disease
            ]

            required_columns.update({
                f"target_{csv_name}",
                f"logit_{csv_name}",
                f"prob_{csv_name}",
            })

        missing = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing:
            raise ValueError(
                f"{path} is missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            target_row = []
            logit_row = []
            probability_row = []

            for disease in CLASSES:

                csv_name = CLASS_TO_CSV_NAME[
                    disease
                ]

                target_row.append(
                    float(
                        row[
                            f"target_{csv_name}"
                        ]
                    )
                )

                logit_row.append(
                    float(
                        row[
                            f"logit_{csv_name}"
                        ]
                    )
                )

                probability_row.append(
                    float(
                        row[
                            f"prob_{csv_name}"
                        ]
                    )
                )

            targets.append(
                target_row
            )

            logits.append(
                logit_row
            )

            probabilities.append(
                probability_row
            )

    targets = np.asarray(
        targets,
        dtype=np.float64,
    )

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    verify_prediction_data(
        targets=targets,
        logits=logits,
        probabilities=probabilities,
        path=path,
    )

    return (
        targets,
        logits,
        probabilities,
    )


# =========================================================
# VERIFY PREDICTION DATA
# =========================================================

def verify_prediction_data(
    targets,
    logits,
    probabilities,
    path,
):

    expected_shape = (
        targets.shape[0],
        len(CLASSES),
    )

    if targets.shape != expected_shape:
        raise ValueError(
            f"Unexpected target shape in {path}: "
            f"{targets.shape}"
        )

    if logits.shape != expected_shape:
        raise ValueError(
            f"Unexpected logit shape in {path}: "
            f"{logits.shape}"
        )

    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Unexpected probability shape in {path}: "
            f"{probabilities.shape}"
        )

    if targets.shape[0] == 0:
        raise ValueError(
            f"Empty prediction artifact: {path}"
        )

    if not np.isfinite(
        targets
    ).all():
        raise ValueError(
            f"Non-finite targets in {path}."
        )

    if not np.isfinite(
        logits
    ).all():
        raise ValueError(
            f"Non-finite logits in {path}."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            f"Non-finite probabilities in {path}."
        )

    if not np.all(
        np.isin(
            targets,
            [0.0, 1.0],
        )
    ):
        raise ValueError(
            f"Non-binary targets in {path}."
        )

    if (
        np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError(
            f"Probabilities outside [0,1] in {path}."
        )

    # Verify saved probabilities correspond to saved logits.
    reconstructed = stable_sigmoid(
        logits
    )

    max_difference = float(
        np.max(
            np.abs(
                reconstructed
                - probabilities
            )
        )
    )

    if max_difference > 1e-6:
        raise RuntimeError(
            "Saved probabilities do not correspond "
            f"to saved logits in {path}. "
            f"Max difference = {max_difference}"
        )


# =========================================================
# EVALUATE ONE SPLIT
# =========================================================

def evaluate_split(
    targets,
    logits,
    probabilities,
    temperatures,
):

    calibrated_logits = (
        logits
        / temperatures.reshape(1, -1)
    )

    calibrated_probabilities = (
        stable_sigmoid(
            calibrated_logits
        )
    )

    results = []

    for class_idx, disease in enumerate(
        CLASSES
    ):

        before = (
            calculate_binary_calibration_metrics(
                targets=targets[:, class_idx],
                probabilities=probabilities[
                    :, class_idx
                ],
                n_bins=N_BINS,
            )
        )

        after = (
            calculate_binary_calibration_metrics(
                targets=targets[:, class_idx],
                probabilities=calibrated_probabilities[
                    :, class_idx
                ],
                n_bins=N_BINS,
            )
        )

        results.append({
            "disease": disease,

            "temperature": float(
                temperatures[class_idx]
            ),

            "nll_before": before["nll"],
            "nll_after": after["nll"],
            "nll_change": (
                after["nll"]
                - before["nll"]
            ),

            "brier_before": before["brier"],
            "brier_after": after["brier"],
            "brier_change": (
                after["brier"]
                - before["brier"]
            ),

            "ece_before": before["ece"],
            "ece_after": after["ece"],
            "ece_change": (
                after["ece"]
                - before["ece"]
            ),

            "auroc_before": before["auroc"],
            "auroc_after": after["auroc"],
            "auroc_change": (
                after["auroc"]
                - before["auroc"]
            ),
        })

    return results


# =========================================================
# MACRO SUMMARY
# =========================================================

def calculate_macro_summary(
    results,
):

    metric_names = [
        "nll",
        "brier",
        "ece",
        "auroc",
    ]

    summary = {}

    for metric in metric_names:

        before_values = np.asarray(
            [
                row[
                    f"{metric}_before"
                ]
                for row in results
            ],
            dtype=np.float64,
        )

        after_values = np.asarray(
            [
                row[
                    f"{metric}_after"
                ]
                for row in results
            ],
            dtype=np.float64,
        )

        summary[metric] = {
            "before": float(
                np.nanmean(
                    before_values
                )
            ),
            "after": float(
                np.nanmean(
                    after_values
                )
            ),
            "change": float(
                np.nanmean(
                    after_values
                )
                - np.nanmean(
                    before_values
                )
            ),
        }

    return summary


# =========================================================
# SAVE CSV
# =========================================================

def save_metrics_csv(
    path,
    results,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "disease",
        "temperature",

        "nll_before",
        "nll_after",
        "nll_change",

        "brier_before",
        "brier_after",
        "brier_change",

        "ece_before",
        "ece_after",
        "ece_change",

        "auroc_before",
        "auroc_after",
        "auroc_change",
    ]

    with open(
        path,
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
# PRINT SUMMARY
# =========================================================

def print_macro_summary(
    split_name,
    summary,
):

    print()
    print("=" * 78)
    print(
        f"{split_name.upper()} CALIBRATION SUMMARY"
    )
    print("=" * 78)

    print(
        f"{'Metric':12}"
        f"{'Before':>16}"
        f"{'After':>16}"
        f"{'Change':>16}"
    )

    print("-" * 78)

    for metric in [
        "nll",
        "brier",
        "ece",
        "auroc",
    ]:

        values = summary[
            metric
        ]

        print(
            f"{metric.upper():12}"
            f"{values['before']:16.8f}"
            f"{values['after']:16.8f}"
            f"{values['change']:16.8f}"
        )

    print("-" * 78)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 78)
    print("MEDTRUST — CALIBRATION EVALUATION")
    print("=" * 78)

    temperatures, temperature_artifact = (
        load_temperatures()
    )

    print(
        "Calibration method :",
        temperature_artifact["method"],
    )

    print(
        "Calibration split  :",
        temperature_artifact[
            "calibration_split"
        ],
    )

    print(
        "ECE bins           :",
        N_BINS,
    )

    print(
        "Test used for fit  :",
        temperature_artifact[
            "test_data_used_for_fitting"
        ],
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    (
        val_targets,
        val_logits,
        val_probabilities,
    ) = load_predictions(
        VALIDATION_PREDICTIONS_PATH
    )

    validation_results = evaluate_split(
        targets=val_targets,
        logits=val_logits,
        probabilities=val_probabilities,
        temperatures=temperatures,
    )

    validation_summary = (
        calculate_macro_summary(
            validation_results
        )
    )

    save_metrics_csv(
        VALIDATION_METRICS_PATH,
        validation_results,
    )

    print_macro_summary(
        "validation",
        validation_summary,
    )

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    (
        test_targets,
        test_logits,
        test_probabilities,
    ) = load_predictions(
        TEST_PREDICTIONS_PATH
    )

    test_results = evaluate_split(
        targets=test_targets,
        logits=test_logits,
        probabilities=test_probabilities,
        temperatures=temperatures,
    )

    test_summary = (
        calculate_macro_summary(
            test_results
        )
    )

    save_metrics_csv(
        TEST_METRICS_PATH,
        test_results,
    )

    print_macro_summary(
        "test",
        test_summary,
    )

    # -----------------------------------------------------
    # AUROC invariance check
    # -----------------------------------------------------

    max_val_auroc_change = max(
        abs(
            row["auroc_change"]
        )
        for row in validation_results
        if np.isfinite(
            row["auroc_change"]
        )
    )

    max_test_auroc_change = max(
        abs(
            row["auroc_change"]
        )
        for row in test_results
        if np.isfinite(
            row["auroc_change"]
        )
    )

    print()
    print("=" * 78)
    print("AUROC INVARIANCE CHECK")
    print("=" * 78)

    print(
        "Validation max |ΔAUROC| :",
        f"{max_val_auroc_change:.12e}",
    )

    print(
        "Test max |ΔAUROC|       :",
        f"{max_test_auroc_change:.12e}",
    )

    if (
        max_val_auroc_change <= 1e-12
        and max_test_auroc_change <= 1e-12
    ):
        print(
            "RESULT: PASSED"
        )
    else:
        print(
            "RESULT: REVIEW REQUIRED"
        )

    print("=" * 78)

    # -----------------------------------------------------
    # Save final summary
    # -----------------------------------------------------

    summary_artifact = {
        "project": "MedTrust",
        "module": "Probability Calibration Evaluation",

        "method": temperature_artifact[
            "method"
        ],

        "calibration_split": "validation",

        "ece_bins": N_BINS,

        "validation_samples": int(
            val_targets.shape[0]
        ),

        "test_samples": int(
            test_targets.shape[0]
        ),

        "validation": validation_summary,
        "test": test_summary,

        "max_validation_auroc_change": float(
            max_val_auroc_change
        ),

        "max_test_auroc_change": float(
            max_test_auroc_change
        ),

        "test_data_used_for_fitting": False,
    }

    if SUMMARY_PATH.exists():
        raise FileExistsError(
            "Calibration summary already exists: "
            f"{SUMMARY_PATH}"
        )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary_artifact,
            f,
            indent=4,
        )

    print()
    print("Artifacts saved")
    print("-" * 78)

    print(
        "Validation metrics :",
        VALIDATION_METRICS_PATH,
    )

    print(
        "Test metrics       :",
        TEST_METRICS_PATH,
    )

    print(
        "Summary            :",
        SUMMARY_PATH,
    )

    print("=" * 78)


if __name__ == "__main__":
    main()