"""
ECE sensitivity analysis for MedTrust.

Evaluates Expected Calibration Error using different
numbers of equal-width bins.

IMPORTANT:
    - No temperatures are fitted here.
    - Frozen validation-derived temperatures are used.
    - Test data is evaluation-only.
    - Results are descriptive and are not used to select
      a new calibrator.
"""

import csv
import json
from pathlib import Path

import numpy as np

from src.calibration.metrics import (
    expected_calibration_error,
)


VALIDATION_PATH = Path(
    "outputs/evaluation/validation_predictions.csv"
)

TEST_PATH = Path(
    "outputs/evaluation/test_predictions.csv"
)

TEMPERATURE_PATH = Path(
    "outputs/calibration/temperature.json"
)

OUTPUT_PATH = Path(
    "outputs/calibration/ece_sensitivity.csv"
)


BIN_COUNTS = [
    10,
    15,
    20,
]


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


def stable_sigmoid(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    result = np.empty_like(x)

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


def load_temperatures():

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
            "Calibration artifact is not "
            "validation-only."
        )

    if artifact.get("classes") != CLASSES:

        raise RuntimeError(
            "Class order mismatch."
        )

    temperatures = np.asarray(
        [
            float(
                artifact["temperatures"][
                    disease
                ]
            )
            for disease in CLASSES
        ],
        dtype=np.float64,
    )

    if (
        not np.isfinite(
            temperatures
        ).all()
        or np.any(
            temperatures <= 0
        )
    ):

        raise RuntimeError(
            "Invalid frozen temperatures."
        )

    return temperatures


def load_split(path):

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

        for row in reader:

            target_row = []
            logit_row = []
            probability_row = []

            for disease in CLASSES:

                name = CLASS_TO_CSV_NAME[
                    disease
                ]

                target_row.append(
                    float(
                        row[
                            f"target_{name}"
                        ]
                    )
                )

                logit_row.append(
                    float(
                        row[
                            f"logit_{name}"
                        ]
                    )
                )

                probability_row.append(
                    float(
                        row[
                            f"prob_{name}"
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

    if (
        targets.shape
        != logits.shape
        or targets.shape
        != probabilities.shape
    ):

        raise RuntimeError(
            f"Shape mismatch in {path}."
        )

    return (
        targets,
        logits,
        probabilities,
    )


def analyze_split(
    split_name,
    path,
    temperatures,
):

    (
        targets,
        logits,
        probabilities,
    ) = load_split(path)

    calibrated_probabilities = (
        stable_sigmoid(
            logits
            / temperatures.reshape(
                1,
                -1,
            )
        )
    )

    rows = []

    print()
    print("=" * 72)
    print(
        f"{split_name.upper()} ECE SENSITIVITY"
    )
    print("=" * 72)

    print(
        f"{'Bins':>8}"
        f"{'Before':>16}"
        f"{'After':>16}"
        f"{'Change':>16}"
    )

    print("-" * 72)

    for n_bins in BIN_COUNTS:

        before_values = []
        after_values = []

        for class_idx, disease in enumerate(
            CLASSES
        ):

            before = (
                expected_calibration_error(
                    targets[
                        :, class_idx
                    ],
                    probabilities[
                        :, class_idx
                    ],
                    n_bins=n_bins,
                )
            )

            after = (
                expected_calibration_error(
                    targets[
                        :, class_idx
                    ],
                    calibrated_probabilities[
                        :, class_idx
                    ],
                    n_bins=n_bins,
                )
            )

            before_values.append(
                before
            )

            after_values.append(
                after
            )

            rows.append({
                "split": split_name,
                "disease": disease,
                "n_bins": n_bins,
                "ece_before": before,
                "ece_after": after,
                "ece_change": (
                    after - before
                ),
            })

        macro_before = float(
            np.mean(
                before_values
            )
        )

        macro_after = float(
            np.mean(
                after_values
            )
        )

        macro_change = (
            macro_after
            - macro_before
        )

        print(
            f"{n_bins:8d}"
            f"{macro_before:16.8f}"
            f"{macro_after:16.8f}"
            f"{macro_change:16.8f}"
        )

    print("-" * 72)

    return rows


def save_results(rows):

    fieldnames = [
        "split",
        "disease",
        "n_bins",
        "ece_before",
        "ece_after",
        "ece_change",
    ]

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "ECE sensitivity artifact already exists: "
            f"{OUTPUT_PATH}"
        )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    print("=" * 72)
    print(
        "MEDTRUST — ECE SENSITIVITY ANALYSIS"
    )
    print("=" * 72)

    print(
        "Bin counts          :",
        BIN_COUNTS,
    )

    print(
        "Temperature fitting : DISABLED"
    )

    print(
        "Test used for fit   : False"
    )

    temperatures = (
        load_temperatures()
    )

    validation_rows = (
        analyze_split(
            split_name="validation",
            path=VALIDATION_PATH,
            temperatures=temperatures,
        )
    )

    test_rows = (
        analyze_split(
            split_name="test",
            path=TEST_PATH,
            temperatures=temperatures,
        )
    )

    all_rows = (
        validation_rows
        + test_rows
    )

    expected_rows = (
        2
        * len(CLASSES)
        * len(BIN_COUNTS)
    )

    if len(all_rows) != expected_rows:

        raise RuntimeError(
            "Unexpected ECE sensitivity "
            "row count."
        )

    save_results(
        all_rows
    )

    print()
    print("=" * 72)
    print(
        "ECE sensitivity artifact:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "Rows:",
        len(all_rows),
    )

    print(
        "Calibration parameters modified:",
        False,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()