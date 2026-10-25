# MedTrust OOD Detection

## Purpose

The MedTrust OOD module detects whether an input chest X-ray is
similar to the distribution of images used to train the MedTrust
DenseNet-121 classifier.

The module is post-hoc.

The trained classifier is frozen.

No classifier retraining is performed.

---

## Supported Methods

### 1. MSP

Uses the maximum sigmoid disease probability.

Formula:

OOD Score = 1 - max(sigmoid(logits))

Higher score means more OOD-like.

---

### 2. Energy

Uses the model logits to calculate an energy-based score.

Higher returned score means more OOD-like.

---

### 3. Mahalanobis

Uses the 1024-dimensional DenseNet feature representation.

Training ID features are used to estimate:

- feature mean
- feature variance

Distance from the ID feature distribution becomes the OOD score.

Higher distance means more OOD-like.

---

## Data Separation

The module follows:

Training:
    Used for Mahalanobis statistics.

Validation:
    Used for OOD threshold selection.

Test:
    Used only for final evaluation.

Therefore the test set does not influence
threshold selection.

---

## Directory Structure

Expected:

    MedTrust-main/
    │
    ├── checkpoints/
    │   └── best_model.pt
    │
    ├── dataset/
    │
    ├── splits/
    │
    ├── ood_data/
    │   ├── val/
    │   └── test/
    │
    ├── outputs/
    │   └── ood/
    │
    └── src/
        └── ood/

---

## OOD Data

OOD images can be placed in:

    ood_data/val/

and:

    ood_data/test/

The images do not require disease labels.

Supported formats:

- PNG
- JPG
- JPEG
- BMP
- TIFF

---

## Complete Pipeline

From the project root:

    python -m src.ood.run_pipeline

The pipeline performs:

    1. Model verification
    2. Feature extraction
    3. Mahalanobis fitting
    4. Mahalanobis scoring
    5. Validation benchmarking
    6. Final test evaluation

---

## Individual Commands

Verify:

    python -m src.ood verify

Feature extraction:

    python -m src.ood.extract

Fit Mahalanobis:

    python -m src.ood.fit_mahalanobis

Calculate Mahalanobis scores:

    python -m src.ood.add_mahalanobis

Validation evaluation:

    python -m src.ood evaluate

Final test evaluation:

    python -m src.ood evaluate-test

---

## Single Image Inference

After the OOD detector has been calibrated:

    python -m src.ood predict image.png

Mahalanobis is the default method.

Other methods:

    python -m src.ood predict image.png --method msp

    python -m src.ood predict image.png --method energy

    python -m src.ood predict image.png --method mahalanobis

---

## Example Output

    MEDTRUST OOD INFERENCE RESULT

    Image           : image.png
    Method          : mahalanobis
    OOD Score       : 14.3721
    OOD Threshold   : 8.9134
    Status          : OOD
    Trust Decision  : DO NOT TRUST PREDICTION

    Disease Predictions
    Pneumonia                0.82
    Atelectasis              0.31

---

## Interpretation

If:

    OOD Score < Threshold

the image is considered:

    ID

The image is sufficiently similar to the training distribution.

If:

    OOD Score >= Threshold

the image is considered:

    OOD

The model prediction should not automatically be treated as trustworthy.

---

## Important

OOD detection does not determine whether a disease is actually
present.

It determines whether the input is sufficiently similar to the
distribution on which the classifier was trained.

Therefore:

    Disease prediction
        and
    OOD detection

are separate decisions.

The final trust-aware system can use both signals.