"""
Data utilities for MedTrust OOD detection.

ID data:
    Reuses the existing NIHChestXrayDataset.

OOD data:
    Loaded from a simple folder structure.

Expected OOD structure:

    ood_data/
    ├── val/
    │   └── <OOD images>
    │
    └── test/
        └── <OOD images>

OOD images do not require disease labels.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.dataset import NIHChestXrayDataset
from src.transformer import eval_transform

from .config import (
    IMAGE_DIR,
    METADATA_CSV,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    OOD_VAL_DIR,
    OOD_TEST_DIR,
)


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# ID DATASET
# ============================================================

def create_id_dataset(
    split="train",
):
    """
    Create an existing MedTrust NIH dataset.

    Parameters
    ----------
    split:
        train / val / test
    """

    if split == "train":
        split_file = TRAIN_SPLIT

    elif split == "val":
        split_file = VAL_SPLIT

    elif split == "test":
        split_file = TEST_SPLIT

    else:
        raise ValueError(
            "split must be one of: train, val, test"
        )

    return NIHChestXrayDataset(
        image_dir=IMAGE_DIR,
        metadata_csv=METADATA_CSV,
        split_file=split_file,
        transform=eval_transform,
    )


# ============================================================
# OOD FOLDER DATASET
# ============================================================

class OODFolderDataset(Dataset):
    """
    Dataset for unlabeled OOD images.

    Each image is treated as:

        OOD = 1

    The actual image contents are not assumed to belong
    to any of the 14 NIH diseases.
    """

    def __init__(
        self,
        root_dir,
        transform=None,
    ):

        self.root_dir = Path(
            root_dir
        )

        self.transform = (
            transform
            if transform is not None
            else eval_transform
        )

        if not self.root_dir.exists():

            raise FileNotFoundError(
                f"OOD directory not found: "
                f"{self.root_dir}"
            )

        if not self.root_dir.is_dir():

            raise NotADirectoryError(
                f"OOD path is not a directory: "
                f"{self.root_dir}"
            )

        self.files = sorted(
            [
                path
                for path in self.root_dir.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]
        )

        if len(self.files) == 0:

            raise RuntimeError(
                f"No supported images found in "
                f"OOD directory: {self.root_dir}"
            )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        image_path = self.files[index]

        image = Image.open(
            image_path
        ).convert("L")

        if self.transform is not None:

            image = self.transform(
                image
            )

        return (
            image,
            image_path.name,
        )


# ============================================================
# OOD DATASET FACTORIES
# ============================================================

def create_ood_val_dataset():

    return OODFolderDataset(
        root_dir=OOD_VAL_DIR,
        transform=eval_transform,
    )


def create_ood_test_dataset():

    return OODFolderDataset(
        root_dir=OOD_TEST_DIR,
        transform=eval_transform,
    )


# ============================================================
# SUMMARY
# ============================================================

def describe_dataset(dataset):

    return {
        "num_samples": len(dataset),
        "dataset_type": type(dataset).__name__,
    }


if __name__ == "__main__":

    print("=" * 70)
    print("MEDTRUST OOD DATA")
    print("=" * 70)

    print(
        "ID train dataset:",
        create_id_dataset("train"),
    )

    print(
        "ID validation dataset:",
        create_id_dataset("val"),
    )

    print(
        "ID test dataset:",
        create_id_dataset("test"),
    )

    if OOD_VAL_DIR.exists():

        try:

            dataset = (
                create_ood_val_dataset()
            )

            print(
                "OOD validation samples:",
                len(dataset),
            )

        except Exception as exc:

            print(
                "OOD validation:",
                exc,
            )

    if OOD_TEST_DIR.exists():

        try:

            dataset = (
                create_ood_test_dataset()
            )

            print(
                "OOD test samples:",
                len(dataset),
            )

        except Exception as exc:

            print(
                "OOD test:",
                exc,
            )