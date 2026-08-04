import pandas as pd
import torch
import torch.nn as nn

from src.dataset import CLASSES


DEFAULT_MAX_POS_WEIGHT = 50.0


def calculate_pos_weights(
    metadata_csv,
    train_split,
    max_pos_weight=DEFAULT_MAX_POS_WEIGHT,
):
    """
    Calculate positive-class weights using TRAINING DATA ONLY.

    For each disease:

        pos_weight = negative_samples / positive_samples

    Extremely large weights are capped to reduce the risk
    of unstable gradients from very rare classes.

    Validation and test data are never used.
    """

    metadata = pd.read_csv(metadata_csv)

    with open(train_split, "r") as f:
        train_files = {
            line.strip()
            for line in f
            if line.strip()
        }

    train_df = metadata[
        metadata["Image Index"].isin(train_files)
    ].copy()

    if len(train_df) != len(train_files):
        raise ValueError(
            "Some training images are missing from metadata."
        )

    weights = []

    for disease in CLASSES:

        positive = (
            train_df["Finding Labels"]
            .str.split("|")
            .apply(lambda labels: disease in labels)
            .sum()
        )

        negative = len(train_df) - positive

        if positive == 0:
            raise ValueError(
                f"No positive training samples found for {disease}"
            )

        weight = negative / positive

        if max_pos_weight is not None:
            weight = min(weight, max_pos_weight)

        weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def create_loss_function(
    device,
    metadata_csv="dataset/Data_Entry_2017_v2020.csv",
    train_split="splits/train_list.txt",
    max_pos_weight=DEFAULT_MAX_POS_WEIGHT,
):
    """
    Create weighted BCE loss for 14-label classification.
    """

    pos_weights = calculate_pos_weights(
        metadata_csv=metadata_csv,
        train_split=train_split,
        max_pos_weight=max_pos_weight,
    )

    pos_weights = pos_weights.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )

    return criterion, pos_weights