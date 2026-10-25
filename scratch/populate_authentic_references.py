import shutil
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 15 Authentic NIH train-split radiographs covering all 14 diseases + No Finding
AUTHENTIC_TRAIN_CASES = [
    "00000002_000.png",  # No Finding
    "00000008_000.png",  # Cardiomegaly
    "00000005_007.png",  # Effusion|Infiltration
    "00000013_010.png",  # Effusion|Pneumonia|Pneumothorax
    "00000013_023.png",  # Infiltration|Mass|Pneumothorax
    "00000013_026.png",  # Cardiomegaly|Emphysema
    "00000013_030.png",  # Atelectasis|Cardiomegaly|Emphysema|Mass|Pneumothorax
    "00000013_043.png",  # Mass|Pleural_Thickening
    "00000020_001.png",  # Fibrosis|Infiltration
    "00000032_001.png",  # Cardiomegaly|Edema|Effusion
    "00000032_012.png",  # Atelectasis|Consolidation|Edema|Pneumonia
    "00000003_001.png",  # Hernia
    "00000004_000.png",  # Mass|Nodule
    "00000017_001.png",  # Infiltration
    "00000022_002.png",  # Atelectasis
]

def main():
    src_dir = PROJECT_ROOT / "dataset" / "images"
    dest_dir = PROJECT_ROOT / "reference_images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Verify all are in train_list.txt
    with open(PROJECT_ROOT / "splits" / "train_list.txt") as f:
        train_set = set(line.strip() for line in f if line.strip())

    copied = 0
    for img_name in AUTHENTIC_TRAIN_CASES:
        if img_name not in train_set:
            print(f"[!] Warning: {img_name} not in train_list.txt! Skipping.")
            continue
        src_path = src_dir / img_name
        dest_path = dest_dir / img_name
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            copied += 1
            print(f"Copied {img_name} to reference_images/ ({src_path.stat().st_size:,} bytes)")
        else:
            print(f"[!] Warning: Source image {src_path} not found.")

    print(f"\nSuccessfully populated {copied} authentic NIH reference radiographs.")

if __name__ == "__main__":
    main()

