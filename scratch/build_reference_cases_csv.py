import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_reference_csv():
    nih_csv = PROJECT_ROOT / "dataset" / "Data_Entry_2017_v2020.csv"
    ref_dir = PROJECT_ROOT / "reference_images"
    target_csv = ref_dir / "reference_cases.csv"

    # Index NIH catalog by Image Index
    nih_records = {}
    with open(nih_csv, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row["Image Index"].strip()
            nih_records[idx] = row

    rows_to_write = []

    # Get all image files in reference_images
    image_files = sorted([
        f.name for f in ref_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])

    for fname in image_files:
        # Prompt Section 6: Never include sample_query.jpg in reference store
        if fname == "sample_query.jpg":
            continue

        if fname in nih_records:
            nih_row = nih_records[fname]
            raw_age = nih_row.get("Patient Age", "").strip()
            # Handle possible trailing Y or non-digit
            age_clean = ""
            for ch in raw_age:
                if ch.isdigit():
                    age_clean += ch
                else:
                    break
            rows_to_write.append({
                "filename": fname,
                "image_index": fname,
                "finding_labels": nih_row.get("Finding Labels", "No Finding").strip(),
                "patient_age": age_clean if age_clean else "Unknown",
                "patient_gender": nih_row.get("Patient Gender", "U").strip(),
                "view_position": nih_row.get("View Position", "PA").strip(),
                "dataset_source": "NIH ChestX-ray14",
                "annotation_type": "Dataset Annotation",
            })
        elif fname.startswith("xray00"):
            # Synthetic demonstration radiographs
            demo_labels = {
                "xray001.jpg": "Cardiomegaly|Effusion",
                "xray002.jpg": "Pneumonia|Infiltration",
                "xray003.jpg": "Pneumothorax",
                "xray004.jpg": "Nodule|Mass",
                "xray005.jpg": "Atelectasis",
            }
            demo_demo = {
                "xray001.jpg": ("58", "M", "PA"),
                "xray002.jpg": ("45", "F", "PA"),
                "xray003.jpg": ("31", "M", "PA"),
                "xray004.jpg": ("62", "F", "PA"),
                "xray005.jpg": ("67", "M", "PA"),
            }
            age, gender, view = demo_demo.get(fname, ("Unknown", "U", "PA"))
            rows_to_write.append({
                "filename": fname,
                "image_index": f"DEMO-{fname}",
                "finding_labels": demo_labels.get(fname, "No Finding"),
                "patient_age": age,
                "patient_gender": gender,
                "view_position": view,
                "dataset_source": "Synthetic Benchmark Demonstration",
                "annotation_type": "Demonstration Benchmark Label",
            })

    fieldnames = [
        "filename",
        "image_index",
        "finding_labels",
        "patient_age",
        "patient_gender",
        "view_position",
        "dataset_source",
        "annotation_type",
    ]

    with open(target_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)

    print(f"Generated {target_csv} with {len(rows_to_write)} validated reference cases.")
    for r in rows_to_write:
        print(f"  {r['filename']:<20} | {r['image_index']:<20} | {r['finding_labels']:<35} | {r['dataset_source']}")

if __name__ == "__main__":
    build_reference_csv()

