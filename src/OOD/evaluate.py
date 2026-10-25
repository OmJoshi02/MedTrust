"""
Validation-time OOD evaluation and method benchmarking.

This script evaluates:

    1. MSP
    2. Energy
    3. Mahalanobis

using:

    ID validation samples
    OOD validation samples

It then:

    - calculates AUROC/AUPR
    - selects a threshold for each method
    - compares the methods
    - selects the best method
    - saves the results

IMPORTANT:
    Test data is NOT used here.
"""

from pathlib import Path

import json
import numpy as np

from .common import ensure_dir
from .config import (
    FEATURE_OUTPUT_DIR,
    SCORE_OUTPUT_DIR,
    EVALUATION_OUTPUT_DIR,
)
from .scores import (
    msp_score,
    energy_score,
)
from .metrics import (
    evaluate_method,
)
from .threshold import (
    select_threshold,
)


ID_VAL_FEATURES = (
    FEATURE_OUTPUT_DIR / "id_val.npz"
)

OOD_VAL_FEATURES = (
    FEATURE_OUTPUT_DIR / "ood_val.npz"
)

ID_VAL_MAHALANOBIS = (
    SCORE_OUTPUT_DIR / "id_val_mahalanobis.npz"
)

OOD_VAL_MAHALANOBIS = (
    SCORE_OUTPUT_DIR / "ood_val_mahalanobis.npz"
)

VALIDATION_RESULTS_PATH = (
    EVALUATION_OUTPUT_DIR
    / "validation_results.json"
)

SELECTED_METHOD_PATH = (
    EVALUATION_OUTPUT_DIR
    / "selected_method.json"
)


def load_npz(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )

    return np.load(
        path,
        allow_pickle=False,
    )


def load_validation_data():
    """
    Load ID and OOD validation logits.

    Returns:
        id_logits
        ood_logits
    """

    id_data = load_npz(
        ID_VAL_FEATURES
    )

    ood_data = load_npz(
        OOD_VAL_FEATURES
    )

    if "logits" not in id_data:
        raise RuntimeError(
            "ID validation artifact does not contain logits."
        )

    if "logits" not in ood_data:
        raise RuntimeError(
            "OOD validation artifact does not contain logits."
        )

    return (
        np.asarray(
            id_data["logits"],
            dtype=np.float64,
        ),
        np.asarray(
            ood_data["logits"],
            dtype=np.float64,
        ),
    )


def load_mahalanobis_validation_data():
    """
    Load precomputed Mahalanobis scores.
    """

    id_data = load_npz(
        ID_VAL_MAHALANOBIS
    )

    ood_data = load_npz(
        OOD_VAL_MAHALANOBIS
    )

    if "mahalanobis_score" not in id_data:
        raise RuntimeError(
            "ID Mahalanobis artifact does not contain scores."
        )

    if "mahalanobis_score" not in ood_data:
        raise RuntimeError(
            "OOD Mahalanobis artifact does not contain scores."
        )

    return (
        np.asarray(
            id_data["mahalanobis_score"],
            dtype=np.float64,
        ),
        np.asarray(
            ood_data["mahalanobis_score"],
            dtype=np.float64,
        ),
    )


def calculate_validation_scores():
    """
    Calculate all validation OOD scores.
    """

    id_logits, ood_logits = (
        load_validation_data()
    )

    scores = {}

    scores["msp"] = {
        "id": msp_score(id_logits),
        "ood": msp_score(ood_logits),
    }

    scores["energy"] = {
        "id": energy_score(id_logits),
        "ood": energy_score(ood_logits),
    }

    id_mahalanobis, ood_mahalanobis = (
        load_mahalanobis_validation_data()
    )

    scores["mahalanobis"] = {
        "id": id_mahalanobis,
        "ood": ood_mahalanobis,
    }

    return scores


def evaluate_validation_methods(
    validation_scores,
):
    """
    Evaluate every OOD method and select its threshold.
    """

    results = {}

    for method, values in validation_scores.items():

        id_scores = values["id"]
        ood_scores = values["ood"]

        threshold_result = select_threshold(
            id_scores=id_scores,
            ood_scores=ood_scores,
        )

        threshold = threshold_result[
            "threshold"
        ]

        evaluation = evaluate_method(
            id_scores=id_scores,
            ood_scores=ood_scores,
            threshold=threshold,
        )

        evaluation.update(
            {
                "threshold_method": threshold_result[
                    "method"
                ],
                "youden_j": threshold_result[
                    "youden_j"
                ],
                "threshold_tpr": threshold_result[
                    "tpr"
                ],
                "threshold_fpr": threshold_result[
                    "fpr"
                ],
            }
        )

        results[method] = evaluation

    return results


def select_best_method(results):
    """
    Select the best OOD method.

    Primary criterion:
        highest AUROC

    Tie-breaker:
        highest F1
    """

    if not results:
        raise RuntimeError(
            "No OOD evaluation results available."
        )

    best_method = sorted(
        results.keys(),
        key=lambda method: (
            results[method]["auroc"],
            results[method]["f1"],
        ),
        reverse=True,
    )[0]

    return best_method


def save_results(
    results,
    best_method,
):
    ensure_dir(
        EVALUATION_OUTPUT_DIR
    )

    payload = {
        "data_split": "validation",
        "test_data_used": False,
        "methods": results,
        "best_method": best_method,
        "selection_criterion": (
            "highest validation AUROC, "
            "then highest F1"
        ),
    }

    with open(
        VALIDATION_RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
        )

    selected = {
        "selected_method": best_method,
        "selection_split": "validation",
        "criterion": "highest AUROC",
        "threshold": results[
            best_method
        ]["threshold"],
    }

    with open(
        SELECTED_METHOD_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selected,
            file,
            indent=4,
        )


def print_results(
    results,
    best_method,
):
    print()
    print("=" * 70)
    print("OOD VALIDATION BENCHMARK")
    print("=" * 70)

    print()

    for method, result in results.items():

        print(
            f"Method: {method.upper()}"
        )

        print(
            f"  AUROC       : "
            f"{result['auroc']:.4f}"
        )

        print(
            f"  AUPR        : "
            f"{result['aupr']:.4f}"
        )

        print(
            f"  Threshold   : "
            f"{result['threshold']:.6f}"
        )

        print(
            f"  Sensitivity : "
            f"{result['sensitivity']:.4f}"
        )

        print(
            f"  Specificity : "
            f"{result['specificity']:.4f}"
        )

        print(
            f"  F1          : "
            f"{result['f1']:.4f}"
        )

        print()

    print(
        "BEST METHOD:",
        best_method.upper(),
    )

    print()
    print(
        "Validation results saved to:"
    )
    print(
        VALIDATION_RESULTS_PATH
    )

    print(
        SELECTED_METHOD_PATH
    )

    print("=" * 70)


def main():
    print("=" * 70)
    print("MEDTRUST — OOD VALIDATION EVALUATION")
    print("=" * 70)

    validation_scores = (
        calculate_validation_scores()
    )

    results = (
        evaluate_validation_methods(
            validation_scores
        )
    )

    best_method = select_best_method(
        results
    )

    save_results(
        results=results,
        best_method=best_method,
    )

    print_results(
        results=results,
        best_method=best_method,
    )


if __name__ == "__main__":
    main()