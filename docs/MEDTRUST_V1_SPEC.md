# MedTrust V1 — Technical Specification

## Project Title

MedTrust: Trust-Aware Explainable AI for Chest X-ray Disease Classification


## 1. Objective

MedTrust V1 is a research prototype for multi-label chest X-ray disease
classification that supplements model predictions with interpretable
evidence about prediction reliability.

The system should not only answer:

"What diseases does the model predict?"

It should also provide evidence related to:

"How much confidence should we place in this prediction and its explanation?"


## 2. Core Architecture

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
14-Disease Multi-label Prediction
    |
    +------------------------------------------------------+
    |              |              |              |         |
    v              v              v              v         v
Confidence      Grad-CAM       Image         Uncertainty   Model
& Calibration   Explainability Quality       Analysis      Features
                   |
                   v
             Explanation
              Reliability
    |              |              |              |         |
    +--------------+--------------+--------------+---------+
                                  |
                         +--------+--------+
                         |                 |
                         v                 v
                   OOD Detection     Similarity Analysis
                         |                 |
                         +--------+--------+
                                  |
                                  v
                         Trust Reasoning Engine
                                  |
                                  v
                        Trust-Aware Result
                                  |
                                  v
                         Report Generation
                                  |
                                  v
                            MedTrust UI


## 3. Existing Foundation

The following components already exist and should be reused:

- NIH ChestX-ray14 dataset loader
- multi-label target encoding
- patient-disjoint 70/15/15 split
- training/validation/test DataLoaders
- preprocessing and augmentation
- ImageNet-pretrained DenseNet-121
- 14-output classifier
- weighted BCE loss
- mixed-precision training
- checkpointing
- validation AUROC calculation
- early stopping

Training is complete.

Best validation Macro AUROC:

0.82777


## 4. Disease Prediction Output

The model predicts these 14 pathologies:

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

The system is multi-label.

Multiple diseases may be predicted simultaneously.

"No Finding" is inferred when no disease satisfies the final decision
criteria. It is not a separately trained output.


## 5. Model Evaluation Module

Purpose:

Establish final classifier performance before adding trust-aware
analysis.

Required outputs:

- validation predictions
- validation probabilities
- per-disease validation AUROC
- Macro validation AUROC
- disease-specific decision thresholds
- test predictions
- test probabilities
- per-disease test AUROC
- Macro test AUROC
- precision
- recall
- F1
- sensitivity
- specificity
- TP
- TN
- FP
- FN

Threshold selection must use validation data only.

The test set must remain a final holdout.


## 6. Confidence Module

Purpose:

Expose model prediction strength without confusing raw probability with
reliability.

For each disease, preserve:

- raw logit
- sigmoid probability
- decision threshold
- predicted positive/negative state

Confidence must not automatically be interpreted as calibrated
probability of correctness.


## 7. Calibration Module

Purpose:

Determine whether predicted probabilities correspond appropriately to
observed outcomes.

Required analysis should include appropriate calibration metrics and
reliability analysis.

If calibration is applied:

- calibration fitting must not use test labels
- raw probabilities must be preserved
- calibrated probabilities must be stored separately
- the calibration method must be documented


## 8. Explainability Module

Primary V1 method:

Grad-CAM

Purpose:

Generate disease-specific visual evidence showing image regions that
most influenced the selected disease prediction.

Required outputs:

- raw Grad-CAM heatmap
- normalized heatmap
- overlay on the chest X-ray
- disease/class identifier
- associated prediction probability

Grad-CAM must support multi-label prediction.


## 9. Explanation Reliability Module

Purpose:

Evaluate whether the generated explanation itself behaves reliably.

V1 should investigate measurable properties such as:

### Stability

Does the explanation remain similar after small, non-destructive input
perturbations?

### Consistency

Do repeated or closely related inputs produce reasonably consistent
explanation regions?

### Fidelity / Perturbation Evidence

Does altering highly activated regions meaningfully affect the relevant
prediction?

Possible outputs include individual similarity or perturbation metrics.

Do not collapse these metrics into an unexplained score.


## 10. Image Quality Module

Purpose:

Identify technical input-quality conditions that may reduce confidence
in downstream analysis.

Potential indicators:

- blur/sharpness
- brightness
- exposure
- contrast
- noise
- dimensions
- input format validity

Required result:

structured quality indicators plus warnings where appropriate.

This module must not claim clinical radiographic quality certification.


## 11. Uncertainty Module

Purpose:

Estimate predictive uncertainty separately from raw confidence.

The exact V1 uncertainty method should be selected after reviewing the
existing model and computational constraints.

The implementation must document:

- uncertainty method
- number of inference passes if applicable
- per-disease uncertainty
- interpretation
- limitations


## 12. OOD Detection Module

Purpose:

Determine whether an input appears substantially different from the
reference distribution represented by the development data.

DenseNet feature embeddings may be used.

Required conceptual outputs:

- OOD score/evidence
- reference threshold/rule
- status such as:
  - in-distribution-like
  - borderline
  - out-of-distribution-like

OOD thresholds must not be optimized using final test labels.


## 13. Similarity Analysis Module

Purpose:

Retrieve reference X-rays with feature representations similar to the
query image.

Preferred basis:

DenseNet feature embeddings.

Required functionality:

- embedding extraction
- reference embedding database/index
- similarity calculation
- Top-K retrieval

Potential result for each retrieved reference:

- image reference
- similarity value
- known labels where permitted

Similarity is supporting evidence, not diagnostic proof.


## 14. Trust Evidence

MedTrust should preserve the individual evidence streams:

- prediction confidence
- calibration
- explanation reliability
- image quality
- uncertainty
- OOD evidence
- similarity evidence

These components must remain independently inspectable.


## 15. Trust Reasoning Engine

Purpose:

Integrate multiple evidence streams into an interpretable assessment of
prediction reliability.

The engine should NOT begin with an arbitrary weighted formula.

Development should proceed by:

1. defining each trust indicator
2. normalizing indicators where necessary
3. studying their behavior on validation data
4. measuring relationships with prediction correctness
5. defining interpretable rules or an aggregation method
6. validating the proposed formulation

Possible output categories may eventually include:

- higher reliability evidence
- moderate reliability evidence
- lower reliability evidence

If a numerical aggregate is introduced, it must be described as a
research/engineering reliability indicator unless clinically validated.


## 16. Unified MedTrust Result

For a single chest X-ray, the final pipeline should be capable of
returning a structured result containing:

{
    input_information,
    disease_predictions,
    confidence_information,
    calibration_information,
    explanations,
    explanation_reliability,
    image_quality,
    uncertainty,
    ood_analysis,
    similar_cases,
    trust_evidence,
    trust_summary,
    warnings
}

The exact schema should be formalized during implementation.


## 17. Report Generator

The report should summarize the unified MedTrust result in a
human-readable format.

The report should clearly separate:

- model prediction
- explanation
- supporting trust evidence
- warnings
- limitations

The report must avoid unsupported clinical claims.


## 18. User Interface

Target V1 workflow:

1. User uploads a chest X-ray.
2. System validates the input.
3. System preprocesses the image.
4. DenseNet-121 generates disease probabilities.
5. Decision thresholds generate prediction states.
6. Grad-CAM generates disease-specific explanations.
7. Trust evidence modules analyze the case.
8. Trust Reasoning Engine integrates evidence.
9. MedTrust presents the result.

The interface should expose component-level evidence rather than showing
only one final score.


## 19. Reuse of DenseNet-121

MedTrust should reuse the existing trained DenseNet-121 wherever
methodologically appropriate.

DenseNet outputs can provide:

Classifier logits
    ->
Disease Prediction / Confidence / Calibration

Gradients + convolutional features
    ->
Grad-CAM

Feature embeddings
    ->
OOD Detection / Similarity Analysis

This reduces unnecessary additional model training.


## 20. Expected V1 Pipeline

Input Chest X-ray
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
        +--------------------+
        |                    |
        v                    v
     Logits              Features
        |                    |
        v                    +------------------+
   Probabilities             |                  |
        |                    v                  v
        |                   OOD             Similarity
        |
        +--> Prediction
        |
        +--> Confidence
        |
        +--> Calibration
        |
        +--> Uncertainty

DenseNet gradients/features
        |
        v
     Grad-CAM
        |
        v
Explanation Reliability

Input image
        |
        v
Image Quality

All Evidence
        |
        v
Trust Reasoning Engine
        |
        v
Unified MedTrust Result
        |
        v
Report / UI


## 21. Development Order

Phase 1:
Final classifier evaluation

Phase 2:
Confidence and calibration

Phase 3:
Grad-CAM

Phase 4:
Explanation reliability

Phase 5:
Image quality analysis

Phase 6:
Prediction uncertainty

Phase 7:
OOD detection

Phase 8:
Similarity analysis

Phase 9:
Trust reasoning

Phase 10:
Unified pipeline

Phase 11:
Report generation

Phase 12:
User interface

Phase 13:
End-to-end experiments and documentation


## 22. Definition of Done

MedTrust V1 is complete only when:

- trained DenseNet checkpoint loads correctly
- frozen split policy is preserved
- final classifier evaluation is reproducible
- disease-specific thresholds are frozen correctly
- confidence/calibration outputs are available
- Grad-CAM works per disease
- explanation reliability is quantitatively analyzed
- image-quality indicators work
- uncertainty estimation works
- OOD analysis works
- similarity retrieval works
- trust evidence is integrated transparently
- unified inference pipeline works
- reports can be generated
- UI can process a chest X-ray end-to-end
- core modules have tests
- final experimental results are documented
- no test-set leakage has occurred
- limitations are explicitly documented
