import torch
from torch.utils.data import DataLoader

from src.dataset import NIHChestXrayDataset
from src.transformer import (
    train_transform,
    val_transform,
    test_transform,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

IMAGE_DIR = "dataset/images"
METADATA_CSV = "dataset/Data_Entry_2017_v2020.csv"

TRAIN_SPLIT = "splits/train_list.txt"
VAL_SPLIT = "splits/val_list.txt"
TEST_SPLIT = "splits/test_list.txt"

BATCH_SIZE = 16
NUM_WORKERS = 4


def create_datasets():
    """
    Create Train / Validation / Test datasets.

    Training:
        preprocessing + augmentation

    Validation/Test:
        preprocessing only
    """

    train_dataset = NIHChestXrayDataset(
        image_dir=IMAGE_DIR,
        metadata_csv=METADATA_CSV,
        split_file=TRAIN_SPLIT,
        transform=train_transform,
    )

    val_dataset = NIHChestXrayDataset(
        image_dir=IMAGE_DIR,
        metadata_csv=METADATA_CSV,
        split_file=VAL_SPLIT,
        transform=val_transform,
    )

    test_dataset = NIHChestXrayDataset(
        image_dir=IMAGE_DIR,
        metadata_csv=METADATA_CSV,
        split_file=TEST_SPLIT,
        transform=test_transform,
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
):
    """
    Create PyTorch DataLoaders.

    Returns:
        train_loader
        val_loader
        test_loader
    """

    train_dataset, val_dataset, test_dataset = create_datasets()

    # pin_memory speeds up CPU -> GPU transfer when CUDA is used.
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_dataloaders()

    print("=" * 65)
    print("MEDTRUST DATALOADER")
    print("=" * 65)

    print("Batch size       :", BATCH_SIZE)
    print("Workers          :", NUM_WORKERS)
    print("CUDA available   :", torch.cuda.is_available())

    print()
    print("Number of batches")
    print("-" * 65)

    print("Train      :", len(train_loader))
    print("Validation :", len(val_loader))
    print("Test       :", len(test_loader))

    print("=" * 65)