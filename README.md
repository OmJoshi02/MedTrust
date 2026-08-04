# MedTrust

## Trust-Aware Explainable AI for Chest X-ray Disease Classification

MedTrust is a research-oriented framework for multi-label chest X-ray
disease classification that combines deep-learning predictions with
explainability and multiple forms of reliability evidence.

The objective is not only to predict chest X-ray pathologies, but also
to provide interpretable evidence about the reliability of individual
predictions and their explanations.

---

## Dataset

MedTrust uses the **NIH ChestX-ray14** dataset.

- Total X-rays: 112,120
- Total patients: 30,805
- Number of pathology outputs: 14
- Task: Multi-label classification

The project uses a patient-disjoint approximately 70/15/15 split:

| Split | Images | Patients |
|---|---:|---:|
| Train | 78,487 | 21,563 |
| Validation | 16,828 | 4,621 |
| Test | 16,805 | 4,621 |

Patient overlap between the three splits is zero.

The original dataset images are not stored in this Git repository.

---

## Pathology Classes

The classifier predicts:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. Hernia

`No Finding` is not modeled as a 15th pathology output. It is
represented by an all-zero 14-dimensional target vector.

---

## Disease Classification Model

The current baseline uses:

- DenseNet-121
- ImageNet pretrained weights
- 224 x 224 input
- 14 independent output logits
- Weighted BCEWithLogitsLoss
- AdamW optimizer
- Automatic Mixed Precision (AMP)
- Early stopping

Training was configured for a maximum of 20 epochs and stopped at epoch
12 due to early stopping.

Best validation Macro AUROC:

**0.82777**

The trained checkpoint is intentionally excluded from Git.

---

## MedTrust V1 Architecture

```text
Chest X-ray
    |
    v
Input Validation
    |
    v
Preprocessing
    |
    v
DenseNet-121
    |
    v
14-Disease Prediction
    |
    +--------------------+--------------------+
    |                    |                    |
    v                    v                    v
Confidence /         Grad-CAM             Image Quality
Calibration          Explainability        Analysis
                         |
                         v
                  Explanation
                   Reliability
    |                    |                    |
    +--------------------+--------------------+
                         |
                    Uncertainty
                         |
                  OOD Detection
                         |
                Similarity Analysis
                         |
                         v
                Trust Reasoning Engine
                         |
                         v
                Trust-Aware Result
                         |
                         v
                  Report / UI