"""
MedTrust V2 — Evidence Comparison & Clinical Concordance Engine
(src/similarity_v2/evidence_engine.py)

Calculates finding-specific similarity evidence, diagnostic concordance, and
label support metrics across the actual retrieved Top-K reference cases.

Mathematical Invariants:
1. Denominator K is constant across all metrics for a given query (K in {5, 10, 20}).
2. Matching count K_d: count of retrieved cases whose authentic dataset annotations contain finding d.
3. Matching rate R_d: (K_d / K) * 100.0 (%).
4. Weighted support V_d: sum(max(0, s_i) for i in matches) / sum(max(0, s_i) for all i) * 100.0 (%).
5. Baseline prevalence P_d is reported separately from NIH master frequency statistics.
6. Zero fabricated composite scores; clear distinction between model predictions and dataset annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FREQ_PATH = PROJECT_ROOT / "outputs" / "similarity_v2" / "disease_frequency.json"
LEGACY_FREQ_PATH = PROJECT_ROOT / "outputs" / "similarity" / "disease_frequency.json"

STANDARD_DISCLAIMER: str = (
    "Case similarity provides supporting context based on visual feature proximity in DenseNet-121 latent space. "
    "It is NOT a medical diagnosis, calibrated disease probability, or proof that two patients "
    "share the same condition. Reference cases are retrieved from NIH ChestX-ray14. "
    "Reference findings are authentic NIH ChestX-ray14 dataset annotations. "
    "A 0-of-K result indicates no retrieved reference case carries that finding label, not zero disease probability."
)

METHODOLOGICAL_EXPLANATION: str = (
    "NIH Dataset Prevalence measures label frequency across the entire dataset. "
    "Retrieved-Case Label Support measures label representation among visually similar reference images. "
    "Similarity-weighted support (V_d) measures label representation among visually similar reference radiographs. "
    "Baseline prevalence (P_d) measures label frequency across the 112,120-image NIH ChestX-ray14 dataset. "
    "Neither is a calibrated disease probability."
)

CANONICAL_DISEASES: Tuple[str, ...] = (
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
)


def normalize_disease_name(name: str) -> str:
    """Normalize finding name for robust comparison."""
    return str(name).strip().lower().replace("_", " ")


class EvidenceComparisonEngine:
    """
    Computes structured evidence comparisons between a query radiograph and retrieved reference cases.
    """

    def __init__(self, frequency_path: Optional[Path] = None) -> None:
        self.frequency_path = frequency_path or (
            DEFAULT_FREQ_PATH if DEFAULT_FREQ_PATH.exists() else LEGACY_FREQ_PATH
        )
        self._prevalence_map: Optional[Dict[str, float]] = None
        self._frequency_raw: Optional[Dict[str, Any]] = None

    def _load_frequency_data(self) -> Dict[str, Any]:
        """Load and cache raw frequency json data."""
        if self._frequency_raw is not None:
            return self._frequency_raw

        if self.frequency_path.exists():
            try:
                with open(self.frequency_path, "r", encoding="utf-8") as f:
                    self._frequency_raw = json.load(f)
                    return self._frequency_raw
            except Exception as exc:
                print(f"[EvidenceComparisonEngine] Warning loading raw frequency: {exc}")

        self._frequency_raw = {}
        return self._frequency_raw

    def get_global_dataset_statistics(self) -> List[Dict[str, Any]]:
        """Return 14 canonical disease dataset statistics."""
        data = self._load_frequency_data()
        counts = data.get("disease_counts", {})
        prevalences = data.get("disease_prevalences", {})
        denom = int(data.get("denominator", data.get("total_valid_records", 112120)))

        stats: List[Dict[str, Any]] = []
        for disease in CANONICAL_DISEASES:
            c = int(counts.get(disease, 0))
            p = float(prevalences.get(disease, (c / denom) * 100.0 if denom else 0.0))
            stats.append({
                "disease_name": disease,
                "disease": disease,
                "occurrence_count": c,
                "prevalence_percentage": round(p, 4),
                "denominator": denom,
            })
        return stats

    def get_no_finding_statistics(self) -> Dict[str, Any]:
        """Return No Finding dataset statistics."""
        data = self._load_frequency_data()
        denom = int(data.get("denominator", data.get("total_valid_records", 112120)))
        nf = data.get("no_finding", {})
        if isinstance(nf, dict):
            c = int(nf.get("occurrence_count", data.get("no_finding_count", 60361)))
            p = float(nf.get("prevalence_percentage", data.get("no_finding_prevalence", 53.8361)))
        else:
            c = int(data.get("no_finding_count", 60361))
            p = float(data.get("no_finding_prevalence", 53.8361))
        return {
            "disease_name": "No Finding",
            "occurrence_count": c,
            "prevalence_percentage": round(p, 4),
            "denominator": denom,
        }

    def compute_weighted_label_support(
        self,
        retrieved_cases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute 14-disease similarity-weighted label support V_d across retrieved cases."""
        k = len(retrieved_cases)
        global_stats = {s["disease"]: s for s in self.get_global_dataset_statistics()}
        if k == 0:
            return [
                {
                    "disease": d,
                    "disease_name": d,
                    "weighted_support": 0.0,
                    "retrieved_count": 0,
                    "retrieval_rate": 0.0,
                    "retrieved_denominator": 0,
                    "dataset_prevalence": global_stats.get(d, {}).get("prevalence_percentage", 0.0),
                    "dataset_occurrence_count": global_stats.get(d, {}).get("occurrence_count", 0),
                    "dataset_denominator": 112120,
                    "is_available": False,
                    "reason": "No reference cases retrieved",
                    "top_k": 0,
                }
                for d in CANONICAL_DISEASES
            ]

        weights = [max(0.0, float(c.get("similarity_score", 0.0))) for c in retrieved_cases]
        total_w = sum(weights)

        support_records: List[Dict[str, Any]] = []
        for disease in CANONICAL_DISEASES:
            norm_d = normalize_disease_name(disease)
            matched_indices = []
            for idx, c in enumerate(retrieved_cases):
                c_findings = [normalize_disease_name(f) for f in (c.get("findings") or [])]
                c_labels = str(c.get("finding_labels") or "").lower()
                if norm_d in c_findings or norm_d in c_labels:
                    matched_indices.append(idx)

            count = len(matched_indices)
            rate = round((count / k) * 100.0, 2)
            g_stat = global_stats.get(disease, {})

            if total_w > 0:
                weighted_sum = sum(weights[i] for i in matched_indices)
                v_d = round((weighted_sum / total_w) * 100.0, 2)
                avail = True
                reason = None
            else:
                v_d = 0.0
                avail = False
                reason = "Zero total similarity weight across retrieved cases"

            support_records.append({
                "disease": disease,
                "disease_name": disease,
                "weighted_support": v_d,
                "retrieved_count": count,
                "retrieval_rate": rate,
                "retrieved_denominator": k,
                "dataset_prevalence": g_stat.get("prevalence_percentage", 0.0),
                "dataset_occurrence_count": g_stat.get("occurrence_count", 0),
                "dataset_denominator": 112120,
                "is_available": avail,
                "reason": reason,
                "top_k": k,
            })
        return support_records

    def _get_prevalences(self) -> Dict[str, float]:
        """Load and cache NIH global disease prevalence dictionary."""
        if self._prevalence_map is not None:
            return self._prevalence_map

        prev_map: Dict[str, float] = {}
        if self.frequency_path.exists():
            try:
                with open(self.frequency_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    prev_raw = data.get("disease_prevalences", {})
                    if isinstance(prev_raw, dict):
                        for d_name, p_val in prev_raw.items():
                            prev_map[d_name] = float(p_val)
                            prev_map[normalize_disease_name(d_name)] = float(p_val)
                    elif isinstance(prev_raw, list):
                        for item in prev_raw:
                            d_name = item.get("disease_name")
                            p_val = item.get("prevalence_percentage")
                            if d_name and p_val is not None:
                                prev_map[d_name] = float(p_val)
                                prev_map[normalize_disease_name(d_name)] = float(p_val)
                    if "no_finding" in data:
                        nf = data["no_finding"]
                        nf_prev = nf.get("prevalence_percentage") if isinstance(nf, dict) else data.get("no_finding_prevalence")
                        if nf_prev is not None:
                            prev_map["No Finding"] = float(nf_prev)
                            prev_map["no finding"] = float(nf_prev)
            except Exception as exc:
                print(f"[EvidenceComparisonEngine] Warning loading prevalence: {exc}")

        # Canonical fallback if file missing
        if not prev_map:
            prev_map = {
                "Infiltration": 17.74, "Effusion": 11.88, "Atelectasis": 10.31,
                "Nodule": 5.65, "Mass": 5.16, "Pneumothorax": 4.73,
                "Consolidation": 4.16, "Pleural_Thickening": 3.02, "Cardiomegaly": 2.48,
                "Emphysema": 2.24, "Edema": 2.05, "Fibrosis": 1.50,
                "Pneumonia": 1.28, "Hernia": 0.20, "No Finding": 53.84,
            }
            for k, v in list(prev_map.items()):
                prev_map[normalize_disease_name(k)] = v

        self._prevalence_map = prev_map
        return self._prevalence_map

    def evaluate_retrieval_evidence(
        self,
        retrieved_cases: List[Dict[str, Any]],
        predicted_findings: Sequence[str],
        primary_finding: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate comprehensive evidence and concordance across retrieved reference cases.

        Args:
            retrieved_cases: List of Top-K matched reference case dictionaries.
            predicted_findings: Sequence of diseases predicted positive for the query image.
            primary_finding: Disease identified as the query's primary finding.

        Returns:
            Structured dictionary with finding-specific evidence, concordance rates, and summary text.
        """
        k = len(retrieved_cases)
        if k == 0:
            return {
                "top_k": 0,
                "predicted_findings_evidence": [],
                "primary_concordance_rate": None,
                "primary_concordant_count": None,
                "any_finding_overlap_rate": None,
                "any_finding_overlap_count": None,
                "concordance_summary": "No reference cases retrieved.",
                "disclaimer": STANDARD_DISCLAIMER,
                "explanation_text": METHODOLOGICAL_EXPLANATION,
            }

        prev_map = self._get_prevalences()

        # Resolve query finding sets
        clean_pred_findings = [str(f).strip() for f in predicted_findings if str(f).strip()]
        if not clean_pred_findings:
            clean_pred_findings = [primary_finding] if primary_finding else ["No Finding"]

        norm_query_findings = {normalize_disease_name(f) for f in clean_pred_findings}
        norm_primary = normalize_disease_name(primary_finding) if primary_finding else None

        # Precompute weights for denominator
        weights = [max(0.0, float(c.get("similarity_score", 0.0))) for c in retrieved_cases]
        total_weight = sum(weights)

        # 1. Per-Predicted-Finding Evidence (K_d, R_d, V_d, P_d)
        findings_evidence: List[Dict[str, Any]] = []

        for target in clean_pred_findings:
            norm_target = normalize_disease_name(target)
            is_normal = norm_target in ("no finding", "normal")

            matching_indices: List[int] = []
            matching_cases_list: List[Dict[str, Any]] = []

            for idx, case in enumerate(retrieved_cases):
                # Check authentic dataset annotations
                raw_findings = case.get("findings") or []
                raw_labels = str(case.get("finding_labels") or "").lower()
                norm_case_findings = [normalize_disease_name(f) for f in raw_findings]

                has_match = False
                if is_normal:
                    has_match = (
                        len(norm_case_findings) == 0
                        or "no finding" in norm_case_findings
                        or "no finding" in raw_labels
                    )
                else:
                    has_match = (
                        norm_target in norm_case_findings
                        or norm_target in raw_labels
                        or any(norm_target in f for f in norm_case_findings)
                    )

                if has_match:
                    matching_indices.append(idx)
                    matching_cases_list.append(case)

            k_d = len(matching_indices)
            r_d = round((k_d / k) * 100.0, 1)

            if total_weight > 0 and k_d > 0:
                weighted_sum = sum(weights[i] for i in matching_indices)
                v_d = round((weighted_sum / total_weight) * 100.0, 1)
            else:
                v_d = 0.0

            p_d = prev_map.get(target) or prev_map.get(norm_target)
            if p_d is not None:
                p_d = round(p_d, 2)

            prev_ratio = round(v_d / p_d, 2) if (p_d and p_d > 0) else None

            if k_d > 0:
                explanation = (
                    f"{k_d} of {k} retrieved reference cases ({r_d}% matching rate, {v_d}% weighted support) "
                    f"corroborate predicted {target}."
                )
            else:
                explanation = (
                    f"0 of {k} retrieved reference cases (0.0% matching rate, 0.0% weighted support) "
                    f"match predicted {target}. No reference support found in top matches."
                )

            findings_evidence.append({
                "predicted_finding": target,
                "finding": target,
                "disease": target,
                "matching_cases": k_d,
                "retrieved_cases": k,
                "matching_rate": r_d,
                "weighted_support": v_d,
                "dataset_prevalence": p_d,
                "prevalence_ratio": prev_ratio,
                "is_matched": k_d > 0,
                "matching_references": matching_cases_list,
                "explanation": explanation,
            })

        # 2. Dual Concordance Statistics: Primary Finding vs Any-Finding Overlap
        primary_concordant_count: Optional[int] = None
        primary_concordance_rate: Optional[float] = None
        any_overlap_count: int = 0

        for case in retrieved_cases:
            case_findings = [normalize_disease_name(f) for f in (case.get("findings") or [])]
            case_labels = str(case.get("finding_labels") or "").lower()

            # Check primary concordance
            if norm_primary:
                is_prim_normal = norm_primary in ("no finding", "normal")
                prim_match = False
                if is_prim_normal:
                    prim_match = len(case_findings) == 0 or "no finding" in case_findings or "no finding" in case_labels
                else:
                    prim_match = norm_primary in case_findings or norm_primary in case_labels

                if prim_match:
                    primary_concordant_count = (primary_concordant_count or 0) + 1

            # Check any-finding overlap
            overlap = False
            for q_find in norm_query_findings:
                if q_find in ("no finding", "normal"):
                    if len(case_findings) == 0 or "no finding" in case_findings or "no finding" in case_labels:
                        overlap = True
                        break
                else:
                    if q_find in case_findings or q_find in case_labels:
                        overlap = True
                        break
            if overlap:
                any_overlap_count += 1

        if primary_concordant_count is None and norm_primary:
            primary_concordant_count = 0

        if primary_concordant_count is not None:
            primary_concordance_rate = round((primary_concordant_count / k) * 100.0, 1)

        any_overlap_rate = round((any_overlap_count / k) * 100.0, 1)

        # Formulate concise summary
        if primary_concordance_rate is not None and primary_finding:
            concordance_summary = (
                f"{primary_concordant_count} of {k} ({primary_concordance_rate:.1f}%) retrieved reference cases "
                f"share the primary finding '{primary_finding}'. Any-finding overlap is {any_overlap_count}/{k} ({any_overlap_rate:.1f}%)."
            )
        else:
            concordance_summary = (
                f"{any_overlap_count} of {k} ({any_overlap_rate:.1f}%) retrieved reference cases "
                f"share at least one predicted finding."
            )

        weighted_support = self.compute_weighted_label_support(retrieved_cases)
        global_stats = self.get_global_dataset_statistics()
        no_finding_stats = self.get_no_finding_statistics()

        return {
            "top_k": k,
            "predicted_findings_evidence": findings_evidence,
            "primary_concordance_rate": primary_concordance_rate,
            "primary_concordant_count": primary_concordant_count,
            "any_finding_overlap_rate": any_overlap_rate,
            "any_finding_overlap_count": any_overlap_count,
            "primary_query_finding": primary_finding,
            "concordance_summary": concordance_summary,
            "global_dataset_statistics": global_stats,
            "no_finding_statistics": no_finding_stats,
            "weighted_label_support": weighted_support,
            "retrieved_case_support": weighted_support,
            "disclaimer": STANDARD_DISCLAIMER,
            "explanation_text": METHODOLOGICAL_EXPLANATION,
        }

