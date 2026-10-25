import csv
from pathlib import Path

def main():
    with open("splits/train_list.txt") as f:
        train_images = set(line.strip() for line in f if line.strip())

    sample_cases = {}
    with open("dataset/Data_Entry_2017_v2020.csv", mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row["Image Index"]
            if idx in train_images:
                labels = row["Finding Labels"]
                for lbl in labels.split("|"):
                    if lbl not in sample_cases:
                        sample_cases[lbl] = row

    print("Authentic NIH Train-Split Candidates:")
    for lbl, row in sorted(sample_cases.items()):
        p = Path("dataset/images") / row["Image Index"]
        print(f"  {lbl:<20}: {row['Image Index']} | exists: {p.exists()} | {row['Finding Labels']}")

if __name__ == "__main__":
    main()

