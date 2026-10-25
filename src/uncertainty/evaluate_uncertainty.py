"""
Evaluate standalone MC-Dropout uncertainty on validation or test data.

This script does NOT fit or modify the disease model, calibration artifact,
decision thresholds, or existing inference pipeline.

For scientific evaluation:
1. Use --split validation to inspect uncertainty behaviour and select
   uncertainty thresholds.
2. Freeze those thresholds separately.
3. Use --split test only for final reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import CLASSES, NIHChestXrayDataset
from src.model import create_model
from src.transformer import val_transform, test_transform
from src.uncertainty import MCDropoutPredictor


CHECKPOINT_PATH = Path("checkpoints/best_model.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MedTrust MC-Dropout uncertainty."
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        required=True,
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--val-split-file", default="splits/val_list.txt")
    parser.add_argument("--test-split-file", default="splits/test_list.txt")
    parser.add_argument("--passes", type=int, default=30)
    parser.add_argument("--dropout-rate", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for development/debugging.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/uncertainty",
    )
    return parser.parse_args()


def load_model(device: torch.device):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("classes") != list(CLASSES):
        raise RuntimeError(
            "Checkpoint class order does not match CLASSES."
        )

    model = create_model(
        num_classes=len(CLASSES),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute AUROC without adding a new dependency.

    Returns NaN when the class contains only one target value.
    """
    y_true = np.asarray(y_true).astype(np.int32)
    scores = np.asarray(scores).astype(np.float64)

    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]

    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")

    # Mann-Whitney U / pairwise probability formulation.
    greater = 0.0
    ties = 0.0

    for positive in positives:
        greater += float(np.sum(positive > negatives))
        ties += float(np.sum(positive == negatives))

    return (greater + 0.5 * ties) / (
        len(positives) * len(negatives)
    )


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    split_file = (
        args.val_split_file
        if args.split == "validation"
        else args.test_split_file
    )

    transform = (
        val_transform
        if args.split == "validation"
        else test_transform
    )

    dataset = NIHChestXrayDataset(
        image_dir=args.image_dir,
        metadata_csv=args.metadata_csv,
        split_file=split_file,
        transform=transform,
    )

    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max-images must be positive.")

        indices = list(range(min(args.max_images, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = load_model(device)

    predictor = MCDropoutPredictor(
        model=model,
        passes=args.passes,
        dropout_rate=args.dropout_rate,
        device=device,
    )

    rows = []

    for batch_index, (images, targets, filenames) in enumerate(loader):
        result = predictor.predict(images)

        samples = (
            result.probabilities
            .detach()
            .cpu()
            .numpy()
        )

        # samples: [passes, batch, classes]
        mean_probability = samples.mean(axis=0)
        variance = samples.var(axis=0)

        for sample_index, filename in enumerate(filenames):
            row = {
                "filename": filename,
            }

            for class_index, disease in enumerate(CLASSES):
                row[f"{disease}_probability"] = float(
                    mean_probability[sample_index, class_index]
                )
                row[f"{disease}_variance"] = float(
                    variance[sample_index, class_index]
                )

            rows.append(row)

        if (batch_index + 1) % 100 == 0:
            print(
                f"Processed {batch_index + 1} batches / "
                f"{len(loader)}"
            )

    csv_path = output_dir / f"{args.split}_uncertainty.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()) if rows else ["filename"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Build validation/test summary only from the current split.
    summary = {
        "split": args.split,
        "passes": args.passes,
        "dropout_rate": args.dropout_rate,
        "images": len(rows),
        "classes": list(CLASSES),
        "checkpoint": str(CHECKPOINT_PATH),
        "test_data_used_for_parameter_fitting": False,
        "note": (
            "This evaluator does not fit uncertainty parameters. "
            "Use validation results for any later threshold selection "
            "and freeze them before final test reporting."
        ),
        "mean_variance_by_disease": {},
        "auroc_by_disease": {},
    }

    if rows:
        targets_matrix = np.stack(
            [
                np.asarray(
                    dataset[i][1].numpy(),
                    dtype=np.float64,
                )
                for i in range(len(dataset))
            ],
            axis=0,
        )

        for disease in CLASSES:
            variance_values = np.asarray(
                [r[f"{disease}_variance"] for r in rows],
                dtype=np.float64,
            )
            probability_values = np.asarray(
                [r[f"{disease}_probability"] for r in rows],
                dtype=np.float64,
            )

            class_index = CLASSES.index(disease)

            summary["mean_variance_by_disease"][disease] = float(
                np.mean(variance_values)
            )

            summary["auroc_by_disease"][disease] = binary_auroc(
                targets_matrix[:, class_index],
                probability_values,
            )

    summary_path = output_dir / f"{args.split}_uncertainty_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("MEDTRUST — UNCERTAINTY EVALUATION")
    print("=" * 80)
    print(f"Split             : {args.split}")
    print(f"Images            : {len(rows)}")
    print(f"MC-Dropout passes : {args.passes}")
    print(f"Dropout rate      : {args.dropout_rate:.2f}")
    print(f"CSV               : {csv_path}")
    print(f"Summary           : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
