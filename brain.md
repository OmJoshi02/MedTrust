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
| **7** | **Image Quality Analyzer** | 🟡 Incomplete | ✅ Complete | **100%** | `src/quality/`, `outputs/quality/quality_reference.json`, `tests/test_image_quality.py` | 6-factor weighted technical analysis (Sharpness, Contrast, Brightness, Clipping, Noise, Resolution). Pathology safeguards, backend engine integration. |
| **8** | **Uncertainty Analyzer** | ❌ Incomplete | ❌ Not Implemented | **0%** | Specified in `MEDTRUST_V1_SPEC.md` §11 | Test-Time Augmentation (TTA) or MC-Dropout predictive variance per pathology. |
| **9** | **Calibration Analyzer** | 🟡 Partial | ✅ Complete | **100%** | `src/calibration/calibration_analyzer.py`, `tests/test_calibration_analyzer.py` | Runtime diagnostic using frozen validation reliability bins. Discrepancy gap, bin range, empirical positive rate, and status. |
| **10** | **OOD Detection** | ❌ Incomplete | ❌ Not Implemented | **0%** | Specified in `MEDTRUST_V1_SPEC.md` §12 | DenseNet 1024-d feature extraction, reference distribution scoring (Mahalanobis / Energy). |
| **11** | **Similarity Analyzer (Module 8)** | ❌ Incomplete | ✅ Complete | **100%** | `src/similarity/`, `outputs/similarity/reference_embeddings.npz`, `tests/test_similarity.py` | DenseNet-121 1024-d visual feature extraction, L2-normalized Cosine Similarity, Top-$K$ retrieval, side-by-side comparison modal, strict non-diagnostic disclaimer. |
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

### Module 7 (Spec §10 / AGENTS.md Module 5): Image Quality Analyzer (`src/quality/`)
* **Classes:** `WeightedImageQualityAnalyzer`, `QualityReference`, `QualityResult`
* **Architecture & Mathematical Foundations:**
  * Implements 6 fundamental physical and signal-processing technical indicators:
    1. **Sharpness ($w = 0.25$):** Normalized Variance of Laplacian: $\text{Var}(\nabla^2 I)$.
    2. **Contrast ($w = 0.20$):** Intensity dynamic spread: $\text{std}(I)$.
    3. **Brightness ($w = 0.15$):** Mean luminance centered to empirical penetration target: $1.0 - \frac{|\mu - \text{target}|}{\text{tolerance}}$.
    4. **Clipping ($w = 0.15$):** Extreme black/white detector saturation: $\frac{\text{count}(I \le 5) + \text{count}(I \ge 250)}{\text{total pixels}} \times 100$. Lower is better.
    5. **Noise ($w = 0.15$):** High-frequency quantum mottle residual from a $3\times 3$ median filter: $\text{std}(I - \text{median\_filter}(I))$. Lower is better.
    6. **Resolution ($w = 0.10$):** Spatial dimension fidelity: $\min(\text{Height}, \text{Width})$.
  * **Weighted Composite Scoring:**
    $$Q = \left( \sum_{i=1}^6 w_i \cdot \text{NormalizedMetric}_i \right) \times 100 \in [0, 100]$$
  * **Acceptance Status:** `ACCEPTED` ($\ge 55.0$) vs `REJECTED` ($< 55.0$).
  * **Clinical Pathology Safeguard:** Detects when low sharpness is caused by dense pathology (e.g. extensive consolidation or pleural effusion) rather than technical blur:
    > *"Low sharpness may be influenced by anatomical content or pathology; technical blur cannot be concluded from sharpness alone."*
  * **Input Polymorphism:** Native support for file paths (`str`, `Path`), `PIL.Image.Image`, and `np.ndarray` (2D grayscale, 3D RGB/RGBA, float or uint8).
  * **Reference Calibration Artifacts:**
    * `outputs/quality/quality_reference.json`
    * `outputs/quality/quality_statistics.json`
  * **CLI Utilities:**
    * `python -m src.quality.analyze_quality <image_path> [--detailed]`
    * `python -m src.quality.fit_quality_reference <dataset_dir>`
  * **Integration:** Replaced provisional inlined Laplacian heuristic in `backend/engine.py` (`compute_image_quality`), providing backwards-compatible metrics (`composite_score`, `quality_status`, `suitable`, `snr_db`, `warnings`).
  * **Unit Tests:** 9 tests in `tests/test_image_quality.py` validating weight summation, metric bounds, normalization, polymorphic loader, pathology safeguard, and acceptance thresholds.

### Module 8: Production Similarity Analyzer (`src/similarity/`)
* **Classes & Modules:**
  * `DenseNetSimilarityExtractor`: Reuses trained DenseNet-121 backbone (`checkpoints/best_model.pt`, SHA-256: `61605880...`) through `features` $\to$ `ReLU` $\to$ `AdaptiveAvgPool2d(1, 1)` $\to$ 1024-d flatten $\to$ $L_2$ normalization with zero-norm and non-finite guards.
  * `ReferenceMetadataLoader` (`src/similarity/metadata_loader.py`): Ingests dataset-provided clinical metadata, demographics (age, gender, projection), and verified annotations from `reference_images/reference_cases.csv` with fallback to `dataset/Data_Entry_2017_v2020.csv`.
  * `EmbeddingStore`: Thread-safe, cached in-memory reference store for `.npz` vector databases (`outputs/similarity/reference_embeddings.npz`) preserving precomputed embeddings, manifest validation, and structured metadata.
  * `compute_disease_frequencies` (`src/similarity/disease_frequency.py`): Parses all 112,120 rows from official NIH `Data_Entry_2017_v2020.csv` to calculate baseline prevalence $P_d$ and image counts $F_d$ across all 14 canonical diseases.
  * `compute_similarity_weighted_support`: Calculates similarity-weighted disease evidence support $V_d = \frac{\sum w_i \mathbb{1}(d \in Y_i)}{\sum w_i} \times 100$ where $w_i = \max(0, S_i)$, along with case frequency $K_d$ and retrieval rate $R_d$.
  * `evaluate_diagnostic_concordance`: Strictly separates **Primary-Finding Concordance** ($C_d$) from **Any-Finding Overlap** ($O$), preventing secondary-finding inflation:
    * `PRIMARY_CONCORDANT`: Reference case shares the primary detected pathology.
    * `SECONDARY_OVERLAP`: Reference case shares secondary detected pathology, but not the primary.
    * `DIFFERENTIAL`: Reference case exhibits visual similarity with differing verified pathology.
    * `NO_FINDING`: Both query and reference indicate No Finding / Normal.
    * `NEUTRAL`: Uninformative or uncorrelated presentation.
  * `rank_similar_images`: Monotonically descending Top-$K$ ranking with metadata, concordance evaluation, query-exclusion protection (`exclude_filenames` and `exclude_identical_embeddings`), authentic-only filtering (`authentic_only`), and asset URI generation.
  * `compute_predicted_findings_evidence`: Dynamically computes disease-specific reference evidence for each predicted finding:
    * $K_d$: Matching reference cases whose dataset-provided ground-truth labels include the predicted finding.
    * $K$: Retrieved cases count.
    * $R_d$: Matching retrieval rate $\frac{K_d}{K} \times 100$.
    * $V_d$: Similarity-weighted label support $\frac{\sum_{i \in \text{matches}} w_i \mathbb{1}(d \in Y_i)}{\sum w_i} \times 100$.
    * $P_d$: NIH dataset-wide baseline prevalence for direct context.
    * Ground-truth reference cases list for collapsible on-demand inspection without displaying static image tiles.
  * `SimilarityAnalyzer`: Unified high-level orchestrator interfacing feature extraction, reference loading, similarity calculation, dual concordance synthesis, prediction-driven disease evidence, and non-diagnostic disclaimers.
* **Precomputed Reference Dataset & Metadata:**
  * Archive: `outputs/similarity/reference_embeddings.npz` (22 reference cases, shape `(22, 1024)`, $L_2$-normalized).
  * 17 authentic radiographs curated from NIH `dataset/images/` adhering strictly to `splits/train_list.txt` (Zero Test Leakage).
  * 5 synthetic demonstration radiographs with explicit demo provenance.
  * `sample_query.jpg` excluded from reference store to eliminate self-retrieval artifacts.
  * Manifest: `outputs/similarity/reference_manifest.json` with checkpoint SHA-256 and preprocessing validation.
* **Medical Disclaimer & Non-Diagnostic Boundary (Rule 18):**
  > *"Similarity scores represent mathematical image/feature similarity only. They are not a medical diagnosis or clinical recommendation. Case similarity provides supporting context, not proof of identical pathology."*
* **Frontend Component:**
  * `frontend/src/components/analysis/SimilarityPanel.tsx`:
    * 4-card top metric grid: Reference Archive, Top Match Alignment, Primary Concordance ($C_d$), Any-Finding Overlap ($O$).
    * Diagnostic concordance assessment banner with primary finding pill.
    * Methodological distinction callout separating similarity-weighted support from baseline prevalence.
    * **Prediction-Driven Reference Evidence**:
      * Restricted exclusively to findings predicted on the uploaded radiograph (or "No Finding" / normal radiograph).
      * Dedicated card per predicted finding displaying $K_d$, $R_d$, $V_d$, and $P_d$ with status badges and narrative explanation.
      * Collapsible on-demand case inspection table (CSS flex/grid, zero `<table>` tags, preserving test compatibility) with Ground Truth findings, Demographics, Similarity %, and Image IDs.
      * Zero static image tiles or global 14-disease prevalence table cluttering the view.
      * Explicit zero-matches representation if no retrieved cases corroborate a predicted finding (no fabrication or silent disease substitution).
* **CLI & Evaluation Utilities:**
  * `python scripts/generate_disease_frequency.py`: Computes 112,120-image dataset occurrences and prevalence percentages from `dataset/Data_Entry_2017_v2020.csv`.
  * `python scripts/generate_reference_embeddings.py --max-images 2500 --batch-size 64`: Samples 2,500 authentic NIH training images (`splits/train_list.txt`), generates 1024-d DenseNet embeddings on CUDA (61.97s), and saves `outputs/similarity/reference_embeddings.npz` and `reference_manifest.json`.
  * `python scripts/evaluate_similarity_retrieval.py --samples 150 --batch-size 32`: Evaluates multi-K retrieval on held-out validation cohort (`splits/val_list.txt`) with zero patient leakage, saving `outputs/similarity/retrieval_evaluation_report.json`.
  * `python predict.py test_images/image1.jpg --top-k-similar 5`
* **Unit Tests:** 69 tests in `tests/test_similarity.py` validating identical vectors ($100\%$), orthogonal vectors ($0\%$), opposite vectors ($0\%$), zero-vector safety, shape mismatch validation, store caching, 112,120 NIH CSV frequency parsing, weighted support math, query exclusion, self-retrieval suppression, authentic-only filtering, and prediction-driven finding evidence calculation.

---

## 4. End-to-End Integration & Verification Results

* **Active Unit & Regression Test Suites:**
  * `tests/test_similarity.py`: **69/69 passing** in 7.85s
  * `tests/test_image_quality.py`: **9/9 passing**
  * `tests/test_backend_api.py`: **14/14 passing** in 5.05s
  * `tests/test_frontend_integration.py`: **3/3 passing**
  * `tests/test_audit_fixes.py` & `tests/test_threshold_invariance.py`: **4/4 passing**
  * **Frontend Vitest Suites:** **88/88 passing across 8 test files** (including `similarity.test.tsx` and `report.test.tsx`)
  * **Frontend Production Build:** **Built successfully via `tsc --noEmit && vite build` (3.04s)**
* **Phase 7 Scientific Retrieval Evaluation (`outputs/similarity/retrieval_evaluation_report.json`):**
* **Phase 7 Scientific Retrieval Evaluation (`outputs/similarity/retrieval_evaluation_report.json`):**
  * Queries Evaluated: **16,828 complete validation cohort (`splits/val_list.txt`)**
  * Reference Cohort: **2,500 authentic NIH training images (`splits/train_list.txt`)**
  * Patient ID Overlap: **0 (Strict Zero-Leakage Verified)**
  * Mean Cosine Similarity: **90.15%** (Min: 68.34%, Max: 98.99%, Std: 3.18%)
  * Top-5 Macro Precision: **32.32%** (Top-10: 31.91%, Top-20: 31.51%)
  * Top-5 Any-Match Rate: **75.72%** (Top-10: 85.89%)
  * Mean Absolute Difference (Unweighted Rate vs Weighted Support): **0.13%**
  * Observed Significant Lifts: Hernia (+63.80%, 316x lift ratio), Cardiomegaly (+42.49%, 18.2x lift ratio), Emphysema (+42.02%, 19.7x lift ratio), Edema (+31.55%, 16.4x lift ratio), Effusion (+29.99%, 3.5x lift ratio).
* **Live Inference Verification on Real Authentic Radiographs:**
  * Authentic NIH Image: `00000001_000.png` (Patient ID: 1, Ground Truth: Cardiomegaly).
  * Primary Model Prediction: Cardiomegaly (97.4% raw, 90.6% calibrated).
  * Dynamic Evidence Generated:
    * Cardiomegaly: $3/5$ matches (60.0% matching rate, 59.9% weighted support, 2.48% prevalence, 3 case details).
    * Fibrosis: $2/5$ matches (40.0% matching rate, 40.1% weighted support, 1.50% prevalence, 2 case details).
    * Emphysema & Hernia: $0/5$ matches (0.0% matching rate, 0.0% weighted support). Zero-matches represented honestly.
