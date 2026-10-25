
"""Extract NIH ID training and validation artifacts only."""

from .common import get_device, set_seed
from .config import FEATURE_OUTPUT_DIR
from .extract import extract_id_split
from .model_adapter import MedTrustModelAdapter


def main():
    set_seed(42)
    device = get_device()

    print("=" * 70)
    print("MEDTRUST OOD — NIH ID-ONLY FEATURE EXTRACTION")
    print("=" * 70)
    print("Device:", device)

    model = MedTrustModelAdapter(device=device)

    extract_id_split(
        model=model,
        split="train",
        output_path=FEATURE_OUTPUT_DIR / "id_train.npz",
    )

    extract_id_split(
        model=model,
        split="val",
        output_path=FEATURE_OUTPUT_DIR / "id_val.npz",
    )

    print()
    print("ID-only feature extraction complete.")
    print("No OOD image folders were used.")
    print("No test split was used.")


if __name__ == "__main__":
    main()
