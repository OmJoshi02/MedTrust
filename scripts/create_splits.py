from pathlib import Path
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

SEED = 42

META = Path("dataset/Data_Entry_2017_v2020.csv")
OUT = Path("splits")

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


def label_vector(label_string):
    labels = set(label_string.split("|"))
    return np.array(
        [int(c in labels) for c in CLASSES],
        dtype=np.int8
    )


def main():
    OUT.mkdir(exist_ok=True)

    df = pd.read_csv(META)

    # --------------------------------------------------
    # Basic NIH-CXR14 verification
    # --------------------------------------------------
    assert len(df) == 112120, \
        f"Expected 112120 rows, found {len(df)}"

    assert df["Image Index"].is_unique, \
        "Duplicate image filenames found."

    print("=" * 75)
    print("MEDTRUST — NIH CHESTX-RAY14 PATIENT SPLIT")
    print("=" * 75)

    print(f"Images   : {len(df):,}")
    print(f"Patients : {df['Patient ID'].nunique():,}")
    print(f"Classes  : {len(CLASSES)}")
    print(f"Seed     : {SEED}")

    # --------------------------------------------------
    # Construct ONE multilabel vector for every patient.
    #
    # If any X-ray of the patient contains a pathology,
    # that pathology is positive for that patient's
    # stratification vector.
    # --------------------------------------------------
    patient_vectors = {}

    for patient_id, group in df.groupby("Patient ID"):

        vector = np.zeros(len(CLASSES), dtype=np.int8)

        for labels in group["Finding Labels"]:
            vector = np.maximum(
                vector,
                label_vector(labels)
            )

        patient_vectors[patient_id] = vector

    patient_ids = np.array(sorted(patient_vectors.keys()))

    Y = np.stack([
        patient_vectors[p]
        for p in patient_ids
    ])

    # --------------------------------------------------
    # SPLIT 1
    #
    # 70% patients -> TRAIN
    # 30% patients -> TEMP
    #
    # Multilabel stratification attempts to preserve
    # disease prevalence.
    # --------------------------------------------------
    split_train = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=SEED,
    )

    train_idx, temp_idx = next(
        split_train.split(patient_ids, Y)
    )

    train_patients = patient_ids[train_idx]

    temp_patients = patient_ids[temp_idx]
    temp_Y = Y[temp_idx]

    # --------------------------------------------------
    # SPLIT 2
    #
    # Divide remaining 30% equally:
    #
    # 15% validation
    # 15% test
    # --------------------------------------------------
    split_val_test = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=SEED,
    )

    val_idx, test_idx = next(
        split_val_test.split(temp_patients, temp_Y)
    )

    val_patients = temp_patients[val_idx]
    test_patients = temp_patients[test_idx]

    train_patients = set(train_patients)
    val_patients = set(val_patients)
    test_patients = set(test_patients)

    # --------------------------------------------------
    # CRITICAL: patient leakage checks
    # --------------------------------------------------
    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)

    all_split_patients = (
        train_patients |
        val_patients |
        test_patients
    )

    assert len(all_split_patients) == df["Patient ID"].nunique()

    # --------------------------------------------------
    # Expand patients back to X-rays.
    #
    # Original CSV order is preserved inside each split.
    # --------------------------------------------------
    train_df = df[
        df["Patient ID"].isin(train_patients)
    ].copy()

    val_df = df[
        df["Patient ID"].isin(val_patients)
    ].copy()

    test_df = df[
        df["Patient ID"].isin(test_patients)
    ].copy()

    # --------------------------------------------------
    # Image leakage checks
    # --------------------------------------------------
    train_images = set(train_df["Image Index"])
    val_images = set(val_df["Image Index"])
    test_images = set(test_df["Image Index"])

    assert train_images.isdisjoint(val_images)
    assert train_images.isdisjoint(test_images)
    assert val_images.isdisjoint(test_images)

    assert (
        len(train_df) +
        len(val_df) +
        len(test_df)
    ) == len(df)

    # --------------------------------------------------
    # Save filename lists.
    #
    # NO image is moved/renamed/modified.
    # --------------------------------------------------
    train_df["Image Index"].to_csv(
        OUT / "train_list.txt",
        index=False,
        header=False
    )

    val_df["Image Index"].to_csv(
        OUT / "val_list.txt",
        index=False,
        header=False
    )

    test_df["Image Index"].to_csv(
        OUT / "test_list.txt",
        index=False,
        header=False
    )

    # Also save patient IDs for auditing/reproducibility.
    np.savetxt(
        OUT / "train_patients.txt",
        sorted(train_patients),
        fmt="%d"
    )

    np.savetxt(
        OUT / "val_patients.txt",
        sorted(val_patients),
        fmt="%d"
    )

    np.savetxt(
        OUT / "test_patients.txt",
        sorted(test_patients),
        fmt="%d"
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------
    total_images = len(df)
    total_patients = df["Patient ID"].nunique()

    print()
    print("SPLIT SUMMARY")
    print("-" * 75)

    print(
        f"{'Split':12}"
        f"{'Images':>12}"
        f"{'Image %':>12}"
        f"{'Patients':>12}"
        f"{'Patient %':>12}"
    )

    for name, subset, patients in [
        ("Train", train_df, train_patients),
        ("Validation", val_df, val_patients),
        ("Test", test_df, test_patients),
    ]:

        print(
            f"{name:12}"
            f"{len(subset):12,d}"
            f"{100*len(subset)/total_images:11.2f}%"
            f"{len(patients):12,d}"
            f"{100*len(patients)/total_patients:11.2f}%"
        )

    print()
    print("PATIENT LEAKAGE")
    print("-" * 75)

    print(
        "Train <-> Validation :",
        len(train_patients & val_patients)
    )

    print(
        "Train <-> Test       :",
        len(train_patients & test_patients)
    )

    print(
        "Validation <-> Test  :",
        len(val_patients & test_patients)
    )

    # --------------------------------------------------
    # Image-level disease prevalence
    # --------------------------------------------------
    print()
    print("IMAGE-LEVEL DISEASE PREVALENCE")
    print("-" * 75)

    print(
        f"{'Disease':20}"
        f"{'Train %':>12}"
        f"{'Val %':>12}"
        f"{'Test %':>12}"
    )

    for disease in CLASSES:

        values = []

        for subset in [train_df, val_df, test_df]:

            positive = (
                subset["Finding Labels"]
                .str.split("|")
                .apply(lambda x: disease in x)
                .sum()
            )

            prevalence = (
                100 * positive / len(subset)
            )

            values.append(prevalence)

        print(
            f"{disease:20}"
            f"{values[0]:11.3f}%"
            f"{values[1]:11.3f}%"
            f"{values[2]:11.3f}%"
        )

    # --------------------------------------------------
    # Rare disease counts
    # --------------------------------------------------
    print()
    print("POSITIVE IMAGE COUNTS")
    print("-" * 75)

    print(
        f"{'Disease':20}"
        f"{'Train':>12}"
        f"{'Val':>12}"
        f"{'Test':>12}"
    )

    for disease in CLASSES:

        counts = []

        for subset in [train_df, val_df, test_df]:

            count = (
                subset["Finding Labels"]
                .str.split("|")
                .apply(lambda x: disease in x)
                .sum()
            )

            counts.append(int(count))

        print(
            f"{disease:20}"
            f"{counts[0]:12,d}"
            f"{counts[1]:12,d}"
            f"{counts[2]:12,d}"
        )

    print()
    print("FILES CREATED")
    print("-" * 75)

    print("splits/train_list.txt")
    print("splits/val_list.txt")
    print("splits/test_list.txt")
    print("splits/train_patients.txt")
    print("splits/val_patients.txt")
    print("splits/test_patients.txt")

    print()
    print("Original images modified : NO")
    print("Original NIH lists modified: NO")

    print("=" * 75)


if __name__ == "__main__":
    main()