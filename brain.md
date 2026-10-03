# MedTrust V1 — Project Brain & Architectural Knowledge Base

> **Project Name:** MedTrust  
> **Full Title:** Trust-Aware Explainable AI for Chest X-ray Disease Classification  
> **Document Role:** Persistent knowledge base of repository scan, architectural audit, module statuses, metrics ledger, and roadmap.  
> **Last Updated:** 2026-09-25T20:10:00+05:30  
> **Baseline Commit:** `fab4e1c` | **Latest Branch Head:** `main` (`3d9aa70`)  

---

## 1. Executive Summary & Project Identity

MedTrust is a clinical decision-support research framework for multi-label chest X-ray disease classification on the **NIH ChestX-ray14** dataset. Beyond predicting disease labels, MedTrust evaluates and communicates the **reliability, trustworthiness, and interpretability** of its predictions to healthcare practitioners.

### Core Guiding Principles (from `AGENTS.md` & `MEDTRUST_V1_SPEC.md`)
1. **Zero Data Leakage:** The test set (16,805 images, 4,621 patients) is a strict holdout. No thresholds, temperatures, OOD statistics, or hyperparameters are fit on test data.
2. **Frozen Baseline:** DenseNet-121 classifier (`checkpoints/best_model.pt`) achieved best validation Macro AUROC of **0.82777** at epoch 7 (early stopped at epoch 12). Classifier weights are frozen.
3. **Multi-Label Pathology Order:** Exactly 14 pathologies in fixed order. "No Finding" is strictly `[0, 0, ..., 0]`.
4. **Honest Scientific Reporting:** No fabricated metrics, no fake "trust probability" or claims of autonomous diagnostic certification.

---

## 2. Master Status & Progress Matrix (14 MedTrust Modules)

| # | MedTrust V1 Module | Target Status | Current Status | Code Coverage % | Artifacts / Key Files | Verified Capabilities |
|---|---|---|---|:---:|---|---|
| **1** | **Chest X-ray Input** | ✅ Implemented | ✅ Complete | **100%** | `src/dataset.py`, `src/dataloader.py`, `splits/`, `test_images/` | Dataset loader, zero-leakage splits, single/batch inference input from `test_images/`. |
| **2** | **Preprocessing** | ✅ Implemented | ✅ Complete | **100%** | `src/transformer.py` | Fully frozen. Grayscale $\to$ 3ch, $224 \times 224$, ImageNet normalization, deterministic eval. |
| **3** | **Disease Prediction (DenseNet-121)** | ✅ Implemented | ✅ Complete | **100%** | `src/model.py`, `src/evaluation/`, `checkpoints/best_model.pt` | Test evaluation complete (Macro test AUROC 0.82902). Decision thresholds frozen via Youden's J. |
| **4** | **Confidence & Calibration** | ✅ Implemented | ✅ Complete | **95%** | `outputs/calibration/temperature.json`, `predict.py`, `src/inference/trust_inference.py` | Both raw and calibrated probabilities available in runtime inference. |
| **5** | **Explainability (Grad-CAM)** | ✅ Implemented | ✅ Complete | **95%** | `src/explainability/gradcam.py`, `visualize_gradcam.py`, `test.py` | Disease-specific Grad-CAM targeting `features.norm5` with 3-panel overlays. |
| **6** | **Explanation Reliability Analyzer** | ❌ Incomplete | ✅ Complete | **100%** | `src/explainability/reliability.py`, `tests/test_explanation_reliability.py` | Reuses existing Grad-CAM. Multi-metric stability (Pearson, Spearman, MAD, IoU) + region occlusion fidelity. Fully tested. |
| **7** | **Image Quality Analyzer** | 🟡 Incomplete | ❌ Not Implemented | **0%** | Specified in `MEDTRUST_V1_SPEC.md` §10 | Sharpness (Laplacian variance), contrast, brightness/clipping, SNR, dimension checks. |
| **8** | **Uncertainty Analyzer** | ❌ Incomplete | ❌ Not Implemented | **0%** | Specified in `MEDTRUST_V1_SPEC.md` §11 | Test-Time Augmentation (TTA) or MC-Dropout predictive variance per pathology. |
| **9** | **Calibration Analyzer** | 🟡 Partial | ✅ Complete | **100%** | `src/calibration/calibration_analyzer.py`, `tests/test_calibration_analyzer.py` | Runtime diagnostic using frozen validation reliability bins. Discrepancy gap, bin range, empirical positive rate, and status. |
| **10** | **OOD Detection** | ❌ Incomplete | ❌ Not Implemented | **0%** | Specified in `MEDTRUST_V1_SPEC.md` §12 | DenseNet 1024-d feature extraction, reference distribution scoring (Mahalanobis / Energy). |
| **11** | **Similarity Analyzer** | ❌ Not Implemented | ❌ Not Implemented | **0%** | Specified in `MEDTRUST_V1_SPEC.md` §13 | Reference embedding database, Top-$K$ cosine similarity retrieval. |
| **12** | **Trust Reasoning Engine** | ❌ Not Implemented | ❌ Not Implemented | **0%** | Specified in `MEDTRUST_V1_SPEC.md` §15 | Multi-factor transparent rule-based reliability synthesis. |
| **13** | **Report Generator** | ❌ Not Implemented | ❌ Not Implemented | **0%** | Specified in `MEDTRUST_V1_SPEC.md` §17 | Structured Markdown/HTML/PDF clinical decision-support report compiler. |
| **14** | **User Interface / Dashboard** | ❌ Not Implemented | ❌ Not Implemented | **0%** | Specified in `MEDTRUST_V1_SPEC.md` §18 | Interactive web application (Streamlit/Gradio). |

---

## 3. Deep Dive into Newly Completed Modules

### Module 6: Explanation Reliability Analyzer (`src/explainability/reliability.py`)
* **Class:** `ExplanationReliabilityAnalyzer`
* **Architecture:**
  * Reuses the existing `src.explainability.gradcam.GradCAM` class (no duplication).
  * Evaluates perturbation stability across 4 reproducible transformations:
    * Brightness ($1.05\times$)
    * Contrast ($1.05\times$)
    * Translation ($4$ pixels)
    * Seeded Gaussian noise ($\sigma = 0.01$, `random_seed=42`)
  * Multi-metric spatial comparison:
    * **Pearson correlation** ($r$)
    * **Spearman rank correlation** ($\rho$)
    * **Mean Absolute Difference (MAD)** ($L_1$)
    * **Intersection-over-Union (IoU)** of top $20\%$ high-attribution regions
  * **Region Occlusion Fidelity Test:**
    * Identifies top attribution pixels (default: top $20\%$).
    * Masks them to neutral ImageNet gray in the input tensor.
    * Computes model probability drop $\Delta p = p_{orig} - p_{occluded}$.
  * **Non-Mutating:** Bitwise verified that model weights, gradients, and buffers are unchanged.
  * **Unit Tests:** 8 tests in `tests/test_explanation_reliability.py` (bounds, flat CAMs, determinism, parameter immutability, fidelity).

### Module 9: Calibration Analyzer (`src/calibration/calibration_analyzer.py`)
* **Class:** `CalibrationAnalyzer`
* **Architecture:**
  * Strictly diagnostic runtime lookup (does NOT recalibrate model weights or fit parameters).
  * Consumes frozen validation reliability bins (`outputs/calibration/reliability/validation_reliability_bins.csv`).
  * Mathematical discrepancy definition:
    $$\text{calibration\_gap} = \text{probability} - \text{empirical\_positive\_rate}$$
    $$\text{absolute\_gap} = |\text{calibration\_gap}|$$
  * Descriptive diagnostic status:
    * `WELL_CALIBRATED`: $|gap| \le 0.10$
    * `MODERATE_DISCREPANCY`: $0.10 < |gap| \le 0.20$
    * `HIGH_DISCREPANCY_WARNING`: $|gap| > 0.20$
    * `UNPOPULATED_BIN_WARNING`: Bin has 0 validation samples or NaN rate.
  * Robust boundary handling: Half-open intervals $[lower, upper)$, closed on final bin $[lower, upper]$, deterministic handling of $0.0$ and $1.0$.
  * Supports single-disease lookup (`analyze_prediction`) and multi-disease batch analysis (`analyze_all`).
  * **Unit Tests:** 8 tests in `tests/test_calibration_analyzer.py` (boundary conditions, invalid inputs, NaN/empty bins, multi-input types).

---

## 4. End-to-End Integration Test (`tests/test_integration_modules_6_and_9.py`)

Verifies the entire sequence without modifying production inference:
$$\text{X-ray} \to \text{Preprocessing} \to \text{Classifier} \to \text{Grad-CAM} \to \text{Mod 6 (Reliability)} \to \text{Calibrated Prediction} \to \text{Mod 9 (Calibration Diagnostic)}$$
* Passed on **NVIDIA GeForce RTX 4050 GPU**.
* Total test suite: **17 passing unit and integration tests**.
