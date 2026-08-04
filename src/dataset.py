from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# Fixed output order for the entire MedTrust project.
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

NUM_CLASSES = len(CLASSES)


def encode_labels(label_string):
    """
    Convert NIH Finding Labels into a 14-dimensional multi-hot tensor.

    Example:
        Cardiomegaly|Emphysema
        ->
        [0,1,0,0,0,0,0,0,0,0,1,0,0,0]

    No Finding
        ->
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    """

    labels = set(label_string.split("|"))

    target = [
        1.0 if disease in labels else 0.0
        for disease in CLASSES
    ]

    return torch.tensor(target, dtype=torch.float32)


class NIHChestXrayDataset(Dataset):
    """
    NIH ChestX-ray14 dataset loader for MedTrust.

    Reads:
        - split filename list
        - NIH metadata CSV
        - original PNG images

    Returns:
        image
        target (14-dimensional multi-hot tensor)
        filename

    Original images are never modified.
    """

    def __init__(
        self,
        image_dir,
        metadata_csv,
        split_file,
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.metadata_csv = Path(metadata_csv)
        self.split_file = Path(split_file)
        self.transform = transform

        # -----------------------------------------------
        # Read split filenames
        # -----------------------------------------------
        with open(self.split_file, "r") as f:
            self.filenames = [
                line.strip()
                for line in f
                if line.strip()
            ]

        # -----------------------------------------------
        # Read metadata
        # -----------------------------------------------
        metadata = pd.read_csv(self.metadata_csv)

        required_columns = {
            "Image Index",
            "Finding Labels",
            "Patient ID",
        }

        missing_columns = required_columns - set(metadata.columns)

        if missing_columns:
            raise ValueError(
                f"Metadata missing required columns: {missing_columns}"
            )

        if metadata["Image Index"].duplicated().any():
            raise ValueError(
                "Duplicate Image Index values found in metadata."
            )

        # Fast filename -> metadata lookup
        metadata = metadata.set_index("Image Index")

        # -----------------------------------------------
        # Verify every split image exists in metadata
        # -----------------------------------------------
        missing_metadata = [
            filename
            for filename in self.filenames
            if filename not in metadata.index
        ]

        if missing_metadata:
            raise ValueError(
                f"{len(missing_metadata)} split images are missing "
                f"from metadata. Example: {missing_metadata[:5]}"
            )

        # Keep only metadata belonging to this split,
        # in exactly the same order as the split file.
        self.metadata = metadata.loc[self.filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        row = self.metadata.loc[filename]

        image_path = self.image_dir / filename

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # NIH images are grayscale chest radiographs.
        # Keep them grayscale for now.
        image = Image.open(image_path).convert("L")

        target = encode_labels(row["Finding Labels"])

        if self.transform is not None:
            image = self.transform(image)

        return image, target, filename
