"""
MedTrust V2 — Similarity Retrieval Engine
(src/similarity_v2/retrieval_engine.py)

Performs Top-K case-based similarity retrieval against the persistent reference
intelligence library, returning structured case matches, clinical concordance,
and prediction-driven evidence.

Invariants:
1. Purely embedding-based retrieval at runtime: requires zero raw reference images or raw CSVs.
2. Supports configurable retrieval depths K in {5, 10, 20} (and arbitrary valid K >= 1).
3. Searches the complete reference candidate index (N = 2,500) and returns Top-K.
4. Excludes query radiograph if its identity exists in the reference library.
5. Deterministic tie-breaking on similarity scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "outputs" / "similarity_v2" / "reference_reports"

from src.similarity_v2.embedding_index import PersistentEmbeddingIndex
from src.similarity_v2.evidence_engine import EvidenceComparisonEngine, STANDARD_DISCLAIMER


class SimilarityRetrievalEngineV2:
    """
    Production-grade case similarity retrieval engine for MedTrust V2.
    """

    def __init__(
        self,
        index: Optional[PersistentEmbeddingIndex] = None,
        evidence_engine: Optional[EvidenceComparisonEngine] = None,
        reports_dir: Optional[Union[str, Path]] = None,
        feature_extractor: Optional[Any] = None,
    ) -> None:
        self.index = index or PersistentEmbeddingIndex()
        self.evidence_engine = evidence_engine or EvidenceComparisonEngine()
        self.reports_dir = Path(reports_dir or DEFAULT_REPORTS_DIR)
        self._extractor = feature_extractor

    def _get_extractor(self) -> Any:
        """Lazy initialization of the DenseNet feature extractor."""
        if self._extractor is None:
            from src.similarity.feature_extractor import DenseNetSimilarityExtractor
            self._extractor = DenseNetSimilarityExtractor()
        return self._extractor

    def get_reference_report(self, reference_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve precomputed intelligence report for a specific reference case.

        Args:
            reference_id: Filename or ID of the reference image (e.g. '00000003_000.png').

        Returns:
            Dictionary containing the full structured report, or None if unavailable.
        """
        clean_name = Path(reference_id).name
        stem = Path(clean_name).stem
        candidates = [
            self.reports_dir / f"{clean_name}.json",
            self.reports_dir / f"{stem}.json",
            self.reports_dir / clean_name,
        ]

        for p in candidates:
            if p.exists() and p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as exc:
                    print(f"[SimilarityRetrievalEngineV2] Error loading report {p}: {exc}")

        # Fallback: synthesize lightweight report from index metadata if individual JSON missing
        meta = self.index.get_metadata(clean_name)
        if meta:
            return {
                "schema_version": "2.0.0-synthesized",
                "reference_id": clean_name,
                "dataset_provenance": meta.get("dataset_source", "NIH ChestX-ray14"),
                "dataset_split": meta.get("dataset_split", "train"),
                "patient_id": meta.get("patient_id", ""),
                "demographics": {
                    "patient_age": meta.get("patient_age"),
                    "patient_gender": meta.get("patient_gender", "U"),
                    "view_position": meta.get("view_position", "PA"),
                },
                "dataset_annotations": {
                    "findings": meta.get("findings", ["No Finding"]),
                    "finding_labels": meta.get("finding_labels", "No Finding"),
                    "annotation_type": meta.get("annotation_type", "Dataset Annotation"),
                },
                "status": "METADATA_ONLY",
            }

        return None

    def find_similar_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        predicted_findings: Optional[Sequence[str]] = None,
        primary_finding: Optional[str] = None,
        query_findings: Optional[Sequence[str]] = None,
        exclude_ids: Optional[Union[str, Sequence[str]]] = None,
        exclude_identical: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute Top-K retrieval using an already extracted 1,024-d query feature vector.
        """
        k = max(1, int(top_k))
        search_hits = self.index.search(
            query_vector=query_vector,
            top_k=k,
            exclude_ids=exclude_ids,
            exclude_identical=exclude_identical,
        )

        total_candidates = self.index.get_count()

        matches: List[Dict[str, Any]] = []
        for rank_idx, (ref_id, score, matrix_idx) in enumerate(search_hits, start=1):
            meta = self.index.get_metadata(ref_id) or {}
            raw_findings = meta.get("findings")
            if not raw_findings:
                lbls = meta.get("finding_labels") or ""
                raw_findings = [p.strip() for p in lbls.replace("|", ",").split(",") if p.strip()]
            if not raw_findings:
                raw_findings = ["No Finding"]

            finding_labels = meta.get("finding_labels") or "|".join(raw_findings)

            # Demographic string
            age = meta.get("patient_age")
            gender = meta.get("patient_gender")
            view = meta.get("view_position")
            demo_parts = [f"{age}y" if age else None, gender if gender and gender != "U" else None, view]
            demo_str = " · ".join([p for p in demo_parts if p]) or "N/A"

            # Evaluate detailed concordance
            prim_norm = str(primary_finding).strip().lower().replace("_", " ") if primary_finding else None
            norm_raw = [str(f).strip().lower().replace("_", " ") for f in raw_findings]
            matches_primary = bool(prim_norm and prim_norm in norm_raw and prim_norm not in ("no finding", "normal"))
            shared_findings = [
                f for f in raw_findings
                if any(str(f).strip().lower().replace("_", " ") == str(q).strip().lower().replace("_", " ")
                       for q in (predicted_findings or query_findings or []))
                and str(f).strip().lower().replace("_", " ") not in ("no finding", "normal")
            ]
            has_overlap = bool(matches_primary or len(shared_findings) > 0)
            if matches_primary:
                status = "PRIMARY_CONCORDANT"
                label = "Primary Concordant"
                desc = f"Reference case shares primary detected finding: {primary_finding}."
            elif has_overlap:
                status = "SECONDARY_OVERLAP"
                label = "Secondary Overlap"
                desc = f"Reference case shares secondary finding(s): {', '.join(shared_findings)}."
            elif any("no finding" in f for f in norm_raw) and not prim_norm:
                status = "NO_FINDING"
                label = "Normal / No Finding"
                desc = "Both query evaluation and reference case show no focal positive findings."
            else:
                status = "DIFFERENTIAL"
                label = "Differential Lookalike"
                desc = f"Visual morphological similarity with differing verified pathology ({', '.join(raw_findings)})."

            match_item = {
                "rank": rank_idx,
                "reference_id": ref_id,
                "image_id": ref_id,
                "filename": ref_id,
                "image_url": f"/reference_images/{ref_id}",
                "similarity_score": score,
                "demographics": demo_str,
                "findings": raw_findings,
                "finding_labels": finding_labels,
                "dataset_source": meta.get("dataset_source", "NIH ChestX-ray14"),
                "provenance": meta.get("dataset_source", "NIH ChestX-ray14"),
                "is_synthetic": False,
                "report_available": (self.reports_dir / f"{ref_id}.json").exists(),
                "report_url": f"/api/similarity/cases/{ref_id}",
                "metadata": meta,
                "concordance": {
                    "matches_primary": matches_primary,
                    "has_overlap": has_overlap,
                    "status": status,
                    "label": label,
                    "description": desc,
                    "shared_findings": shared_findings,
                    "primary_finding": primary_finding,
                    "reference_findings": raw_findings,
                },
            }
            matches.append(match_item)

        # Resolve prediction findings target
        resolved_predicted = list(predicted_findings or query_findings or [])
        if not resolved_predicted:
            resolved_predicted = [primary_finding] if primary_finding else ["No Finding"]

        evidence_res = self.evidence_engine.evaluate_retrieval_evidence(
            retrieved_cases=matches,
            predicted_findings=resolved_predicted,
            primary_finding=primary_finding,
        )

        top_match = matches[0] if matches else None

        return {
            "status": "SUCCESS",
            "version": "2.0.0",
            "top_k": len(matches),
            "requested_k": k,
            "total_references": total_candidates,
            "candidates_searched": total_candidates,
            "matches": matches,
            "top_match": top_match,
            "primary_concordance_rate": evidence_res.get("primary_concordance_rate"),
            "primary_concordant_count": evidence_res.get("primary_concordant_count"),
            "concordance_rate": evidence_res.get("primary_concordance_rate"),
            "concordant_cases_count": evidence_res.get("primary_concordant_count"),
            "any_finding_overlap_rate": evidence_res.get("any_finding_overlap_rate"),
            "any_finding_overlap_count": evidence_res.get("any_finding_overlap_count"),
            "primary_query_finding": primary_finding,
            "concordance_summary": evidence_res.get("concordance_summary"),
            "predicted_findings": resolved_predicted,
            "predicted_findings_evidence": evidence_res.get("predicted_findings_evidence", []),
            "predicted_evidence": evidence_res.get("predicted_findings_evidence", []),
            "global_dataset_statistics": evidence_res.get("global_dataset_statistics", []),
            "no_finding_statistics": evidence_res.get("no_finding_statistics"),
            "weighted_label_support": evidence_res.get("weighted_label_support", []),
            "retrieved_case_support": evidence_res.get("retrieved_case_support", []),
            "disclaimer": STANDARD_DISCLAIMER,
            "explanation_text": evidence_res.get("explanation_text", ""),
        }

    def find_similar_cases(
        self,
        image_input: Union[str, Path, bytes, Image.Image, torch.Tensor],
        top_k: int = 5,
        predicted_findings: Optional[Sequence[str]] = None,
        primary_finding: Optional[str] = None,
        query_findings: Optional[Sequence[str]] = None,
        exclude_filenames: Optional[Union[str, Sequence[str]]] = None,
        authentic_only: bool = True,
        exclude_identical_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract features from input image and retrieve Top-K most similar reference cases.
        """
        extractor = self._get_extractor()
        query_vector = extractor.extract_features(image_input, normalize=True)

        return self.find_similar_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            predicted_findings=predicted_findings,
            primary_finding=primary_finding,
            query_findings=query_findings,
            exclude_ids=exclude_filenames,
            exclude_identical=exclude_identical_embeddings,
        )

    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for find_similar_cases to satisfy MedTrust analyzer protocol."""
        return self.find_similar_cases(*args, **kwargs)

