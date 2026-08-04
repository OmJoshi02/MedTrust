# MedTrust V1 — Codex Agent Instructions

## 1. Project Identity

Project name: MedTrust

Full title:
Trust-Aware Explainable AI for Chest X-ray Disease Classification

MedTrust is a research/final-year project for multi-label chest X-ray
disease classification with explainability and trust/reliability analysis.

The project must prioritize:

- reproducibility
- prevention of data leakage
- scientifically defensible evaluation
- modular implementation
- explicit testing
- preservation of existing trained artifacts
- honest reporting of experimental results


## 2. Current Baseline — FROZEN

Dataset:
NIH ChestX-ray14

Total images:
112,120

Total patients:
30,805

The dataset has already been verified.

Original X-rays are stored under:

dataset/images/

NEVER rename, move, resize, overwrite, reorganize, or otherwise modify
the original X-ray files.

Preprocessing must happen dynamically.


## 3. Disease Classes

The model performs 14-label multi-label classification.

The class order MUST remain exactly:

0  Atelectasis
1  Cardiomegaly
2  Effusion
3  Infiltration
4  Mass
5  Nodule
6  Pneumonia
7  Pneumothorax
8  Consolidation
9  Edema
10 Emphysema
11 Fibrosis
12 Pleural_Thickening
13 Hernia

"No Finding" is NOT a 15th output class.

"No Finding" is represented by:

[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


## 4. Frozen Patient-Level Split

The project uses a patient-disjoint approximately 70/15/15 split.

Train:
78,487 images
21,563 patients

Validation:
16,828 images
4,621 patients

Test:
16,805 images
4,621 patients

Patient overlap between all splits:
ZERO

Frozen files:

splits/train_list.txt
splits/val_list.txt
splits/test_list.txt

splits/train_patients.txt
splits/val_patients.txt
splits/test_patients.txt

NEVER regenerate, reshuffle, replace, or modify these split files unless
the human user explicitly requests it.


## 5. Test-Set Leakage Policy

The test set is the FINAL HOLDOUT SET.

Test labels MUST NOT be used for:

- model selection
- threshold optimization
- calibration fitting
- hyperparameter selection
- OOD threshold tuning
- uncertainty parameter tuning
- trust-engine tuning
- similarity hyperparameter tuning
- choosing between competing implementations

Validation data may be used for those tasks when methodologically
appropriate.

Once parameters are selected using training/validation data, freeze them
before final test evaluation.


## 6. Existing Preprocessing

Existing preprocessing is defined in:

src/transformer.py

Current model input:

3 x 224 x 224

NIH grayscale X-rays are converted to 3 channels.

ImageNet normalization is used because DenseNet-121 uses ImageNet
pretrained weights.

Training augmentation currently includes mild:

- rotation
- translation
- scaling
- brightness adjustment
- contrast adjustment

Validation/test preprocessing is deterministic.

Do NOT silently alter this preprocessing contract.


## 7. Existing Dataset Pipeline

Relevant existing files include:

src/dataset.py
src/transformer.py
src/dataloader.py
src/loss.py
src/model.py
train.py

Inspect and reuse existing implementations before creating replacements.

Do NOT duplicate working functionality without justification.


## 8. Existing Model — TRAINING COMPLETE

Architecture:

DenseNet-121

Initialization:

ImageNet pretrained

Classifier:

1024 -> 14

Outputs:

14 independent logits

Training loss:

Weighted BCEWithLogitsLoss

Optimizer:

AdamW

Initial learning rate:

1e-4

Weight decay:

1e-4

Mixed precision:

Enabled

Maximum configured epochs:

20

Training stopped:

Epoch 12 due to early stopping

Early-stopping patience:

5

Best validation Macro AUROC:

0.82777

Primary model artifact:

checkpoints/best_model.pt

Resume checkpoint:

checkpoints/last_checkpoint.pt

Training history:

logs/training_history.csv

Training is considered COMPLETE for MedTrust V1.

NEVER overwrite, delete, modify, or automatically retrain the trained
checkpoint.

Do not start a new classifier training run unless explicitly requested.


## 9. Git Safety

The known pre-Codex baseline commit is:

fab4e1c

Message:

Establish MedTrust DenseNet-121 training baseline

Before large modifications, inspect git status.

Do not rewrite Git history.

Do not force-push.

Do not commit dataset images or trained checkpoints.

Do not delete user files merely because they appear unused.


## 10. MedTrust V1 Target Modules

Implement MedTrust V1 incrementally in this order:

1. Model Evaluation
2. Confidence and Calibration
3. Grad-CAM Explainability
4. Explanation Reliability
5. Image Quality Analysis
6. Prediction Uncertainty
7. Out-of-Distribution Detection
8. Similarity Analysis
9. Trust Reasoning Engine
10. Unified MedTrust Pipeline
11. Report Generation
12. User Interface
13. End-to-End Evaluation


## 11. Module 1 — Model Evaluation

Use:

checkpoints/best_model.pt

Do NOT retrain.

Implement at minimum:

- per-disease AUROC
- Macro AUROC
- precision
- recall
- F1
- sensitivity
- specificity
- confusion statistics

Decision thresholds must be selected using VALIDATION data only.

Do not optimize thresholds using the test set.

Save thresholds in a machine-readable artifact.

Keep validation results and final test results separate.


## 12. Module 2 — Confidence and Calibration

Preserve raw sigmoid probabilities.

Measure calibration.

Implement appropriate calibration analysis such as:

- reliability analysis
- Expected Calibration Error where applicable

If probability calibration is applied, calibration parameters must be
fit without using test labels.

Preserve both raw and calibrated probabilities.

Confidence and calibration are related but are NOT synonymous.


## 13. Module 3 — Explainability

Primary MedTrust V1 explainer:

Grad-CAM

Requirements:

- DenseNet-121 compatible
- multi-label compatible
- disease-specific
- heatmap generation
- X-ray overlay generation
- no modification of original images

Do NOT automatically add SHAP, LIME, or Integrated Gradients.

Additional explanation methods require explicit approval.


## 14. Module 4 — Explanation Reliability

A Grad-CAM heatmap alone is NOT sufficient for MedTrust.

The project must investigate whether explanations are reliable.

Implement scientifically defensible analysis where practical involving:

- explanation stability
- perturbation robustness
- consistency
- localization/fidelity-style evidence

Keep individual reliability indicators visible.

Do NOT hide all explanation evidence behind an arbitrary score.


## 15. Module 5 — Image Quality

Implement interpretable technical image-quality indicators.

Potential indicators include:

- blur/sharpness
- brightness/exposure
- contrast
- noise where justified
- image dimensions
- input validity

Image-quality results are engineering indicators.

Do NOT describe them as clinical image-quality certification.


## 16. Module 6 — Prediction Uncertainty

Implement a scientifically defensible uncertainty estimation method.

Uncertainty MUST remain conceptually separate from raw sigmoid
probability/confidence.

Document:

- method
- assumptions
- output
- interpretation
- limitations


## 17. Module 7 — OOD Detection

Implement an Out-of-Distribution detection component using a justified
method.

DenseNet feature representations may be reused.

Reference distributions must not introduce test leakage.

OOD results should be expressed conservatively, for example:

- in-distribution-like
- borderline/uncertain
- out-of-distribution-like

Do NOT claim perfect OOD detection.


## 18. Module 8 — Similarity Analysis

Reuse DenseNet feature embeddings where appropriate.

Implement:

- feature extraction
- reference embedding storage
- similarity calculation
- Top-K similar-case retrieval

Similarity is supporting evidence.

It is NOT proof that two patients have the same diagnosis.

Do not expose unnecessary patient-identifying information.


## 19. Module 9 — Trust Reasoning Engine

Potential evidence inputs include:

- calibrated confidence
- explanation reliability
- image quality
- uncertainty
- OOD evidence
- similarity evidence

CRITICAL:

Do NOT invent a medically validated "trust probability."

Any aggregate trust/reliability indicator must have:

- documented inputs
- documented normalization
- documented formula/rules
- justified parameters
- experimental validation
- clear limitations

Raw component indicators must remain accessible.

Any combined score must be described as an engineering/research
reliability indicator unless clinical validation actually exists.


## 20. Module 10 — Unified Pipeline

Target inference flow:

Chest X-ray
    ->
Input Validation
    ->
Preprocessing
    ->
DenseNet-121
    ->
14-Disease Prediction
    ->
Trust Evidence Modules
    ->
Trust Reasoning
    ->
Structured MedTrust Result


## 21. Module 11 — Report Generation

The report should be capable of presenting:

- predicted diseases
- probabilities
- decision thresholds
- calibration/confidence information
- Grad-CAM
- explanation reliability
- image-quality indicators
- uncertainty
- OOD evidence
- similar cases
- trust/reliability summary
- limitations/warnings

Never fabricate missing results.


## 22. Module 12 — User Interface

Target interaction:

Upload Chest X-ray
    ->
Analyze
    ->
Disease Predictions
    ->
Explainability
    ->
Trust Evidence
    ->
Trust-Aware Report

The application is a research/decision-support prototype.

Do NOT present it as an autonomous medical diagnostic system.


## 23. Preferred Code Organization

Prefer modular organization such as:

src/evaluation/
src/calibration/
src/explainability/
src/quality/
src/uncertainty/
src/ood/
src/similarity/
src/trust/
src/reporting/

tests/
configs/
docs/
outputs/

Do not put the entire MedTrust implementation into one large Python
script.


## 24. Dependency Policy

Do NOT silently install packages.

Before introducing a new dependency:

1. check whether the functionality already exists in installed packages
2. explain why the dependency is needed
3. keep dependencies minimal
4. ensure compatibility with the existing Python/PyTorch environment

Current development environment includes:

Python 3.12.7
PyTorch 2.13.0+cu130
TorchVision 0.28.0+cu130
CUDA available
NVIDIA GeForce RTX 4050 Laptop GPU
approximately 5.64 GB VRAM

The project intentionally uses the existing base environment.


## 25. Development Procedure

For EVERY major module:

1. Read this AGENTS.md.
2. Inspect relevant existing code.
3. Explain the proposed design.
4. Identify files to create/change.
5. Implement the smallest complete version.
6. Run syntax checks.
7. Run unit tests.
8. Run a small smoke test.
9. Report actual results.
10. Stop before implementing the next major module unless explicitly
    instructed to continue.

Do NOT implement all V1 modules as one uncontrolled change.


## 26. Testing Requirements

Every important module should have tests.

Where practical test:

- expected tensor shapes
- expected data types
- deterministic behavior
- invalid inputs
- missing files
- CPU fallback
- GPU execution where relevant
- NaN/Inf handling
- serialization/deserialization
- leakage-sensitive behavior


## 27. Scientific Integrity

NEVER fabricate:

- metrics
- experimental results
- citations
- clinical claims
- benchmark comparisons
- test outcomes

If something has not been evaluated, explicitly state:

NOT YET EVALUATED

Do not silently choose a method merely because it produces a more
favorable result.


## 28. Medical Safety / Claims

MedTrust V1 is a research prototype.

Do not describe outputs as definitive medical diagnoses.

Do not claim:

- clinical validation
- regulatory approval
- guaranteed reliability
- guaranteed explanation correctness
- guaranteed OOD detection

unless such evidence actually exists.


## 29. Agent Behavior

When first entering this repository:

DO NOT immediately write code.

First:

- inspect the repository
- understand existing implementation
- identify inconsistencies
- identify leakage risks
- produce a phased plan

Wait for human approval before beginning major implementation.

When uncertain about a scientifically important design decision, surface
the uncertainty instead of silently making the choice.
