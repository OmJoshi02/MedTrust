"""
MedTrust V1 — Similarity Analyzer (src/similarity/similarity_analyzer.py)

Module 8: Similarity Analysis.
High-level orchestrator that extracts features from query radiographs,
compares them against precomputed reference embeddings via Cosine Similarity,
and returns ranked Top-K case matches with dual metrics:
1. NIH Dataset Disease Prevalence (Data_Entry_2017_v2020.csv)
2. Retrieved-Case Similarity-Weighted Label Support
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.similarity.embedding_store import EmbeddingStore, embedding_store as default_store
from src.similarity.feature_extractor import DenseNetSimilarityExtractor
from src.similarity.disease_frequency import (
    CANONICAL_DISEASES,
    NO_FINDING_LABEL,
    load_cached_disease_frequency_json,
    load_cached_disease_frequencies,
)
from src.similarity.similarity_calculator import (
    DISCLAIMER_TEXT,
    EXPLANATION_TEXT,
    compute_cosine_similarity,
    compute_predicted_findings_evidence,
    compute_similarity_weighted_support,
    rank_similar_images,
)
from src.similarity_v2 import get_similarity_engine_v2


class SimilarityAnalyzer:
    """
    Unified analyzer for case-based radiograph similarity retrieval.
    Acts as a thin adapter delegating to Similarity Analyzer V2 while preserving
    backward compatibility for custom stores and legacy callers.
    """

    def __init__(
        self,
        extractor: Optional[DenseNetSimilarityExtractor] = None,
        store: Optional[EmbeddingStore] = None,
        model: Optional[nn.Module] = None,
        embeddings_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        use_v2: Optional[bool] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_v2 is None:
            # Delegate to V2 if no custom store or custom embeddings_path is supplied
            self.use_v2 = (store is None and embeddings_path is None)
        else:
            self.use_v2 = bool(use_v2)

        if self.use_v2:
            self._v2_engine = get_similarity_engine_v2()
        else:
            self._v2_engine = None

        if extractor is not None:
            self.extractor = extractor
        else:
            self.extractor = DenseNetSimilarityExtractor(model=model, device=self.device)

        if store is not None:
            self.store = store
        elif embeddings_path is not None:
            self.store = EmbeddingStore(embeddings_path=embeddings_path)
        else:
            self.store = default_store

        # Load global disease prevalence and frequency metrics from JSON artifact
        self.frequency_data: Dict[str, Any] = {}
        self.prevalence_map: Dict[str, float] = {}
        self.occurrence_map: Dict[str, int] = {}
        self.dataset_denominator: int = 112120
        self.no_finding_stats: Dict[str, Any] = {
            "disease_name": NO_FINDING_LABEL,
            "occurrence_count": 0,
            "prevalence_percentage": 0.0,
            "denominator": 112120,
        }

        try:
            freq_json = load_cached_disease_frequency_json()
            self.frequency_data = freq_json
            self.dataset_denominator = int(freq_json.get("denominator", freq_json.get("total_valid_records", 112120)))
            self.occurrence_map = {d: int(cnt) for d, cnt in freq_json.get("disease_counts", {}).items()}
            self.prevalence_map = {d: float(prev) for d, prev in freq_json.get("disease_prevalences", {}).items()}
            if "no_finding" in freq_json:
                nf = freq_json["no_finding"]
                self.no_finding_stats = {
                    "disease_name": nf.get("disease_name", NO_FINDING_LABEL),
                    "occurrence_count": int(nf.get("occurrence_count", 0)),
                    "prevalence_percentage": float(nf.get("prevalence_percentage", 0.0)),
                    "denominator": self.dataset_denominator,
                }
        except Exception as exc:
            # Fallback to CSV loader if JSON load encounters issue
            try:
                freq_table = load_cached_disease_frequencies()
                self.prevalence_map = {
                    d: float(rec["prevalence_percentage"])
                    for d, rec in freq_table.items()
                }
                self.occurrence_map = {
                    d: int(rec["image_count"])
                    for d, rec in freq_table.items()
                }
                if NO_FINDING_LABEL in freq_table:
                    nf_rec = freq_table[NO_FINDING_LABEL]
                    self.no_finding_stats = {
                        "disease_name": NO_FINDING_LABEL,
                        "occurrence_count": int(nf_rec["image_count"]),
                        "prevalence_percentage": float(nf_rec["prevalence_percentage"]),
                        "denominator": int(nf_rec.get("denominator", 112120)),
                    }
            except Exception:
                pass

    def find_similar_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        base_url: str = "/reference_images/",
        query_findings: Optional[List[str]] = None,
        primary_finding: Optional[str] = None,
        exclude_filenames: Optional[Union[str, Sequence[str]]] = None,
        predicted_findings: Optional[Sequence[str]] = None,
        authentic_only: bool = False,
        exclude_identical_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """
        Find top-K similar cases from a pre-extracted 1024-d feature vector.
        """
        if self._v2_engine is not None:
            return self._v2_engine.find_similar_by_vector(
                query_vector=query_vector,
                top_k=top_k,
                predicted_findings=predicted_findings,
                primary_finding=primary_finding,
                query_findings=query_findings,
                exclude_ids=exclude_filenames,
                exclude_identical=exclude_identical_embeddings,
            )

        filenames, embeddings = self.store.get_data()
        metadata = self.store.get_metadata()

        matches = rank_similar_images(
            query_vector=query_vector,
            reference_filenames=filenames,
            reference_embeddings=embeddings,
            top_k=top_k,
            base_url=base_url,
            metadata=metadata,
            query_findings=query_findings,
            primary_finding=primary_finding,
            exclude_filenames=exclude_filenames,
            authentic_only=authentic_only,
            exclude_identical_embeddings=exclude_identical_embeddings,
        )

        top_match = matches[0] if matches else None

        # Resolve dynamic predicted findings for prediction-driven evidence
        if predicted_findings is not None:
            resolved_predicted_findings = [str(f).strip() for f in predicted_findings if str(f).strip()]
        elif query_findings is not None and len(query_findings) > 0:
            resolved_predicted_findings = [str(f).strip() for f in query_findings if str(f).strip()]
        elif primary_finding is not None and str(primary_finding).strip():
            resolved_predicted_findings = [str(primary_finding).strip()]
        else:
            resolved_predicted_findings = [NO_FINDING_LABEL]

        if not resolved_predicted_findings:
            resolved_predicted_findings = [NO_FINDING_LABEL]

        # Calculate prediction-driven reference evidence exclusively for predicted finding(s)
        predicted_findings_evidence = compute_predicted_findings_evidence(
            matches=matches,
            predicted_findings=resolved_predicted_findings,
            global_prevalences=self.prevalence_map,
            global_frequency_data=self.frequency_data,
        )

        # Calculate dual concordance statistics: Primary-Finding vs Any-Finding Overlap
        primary_concordance_rate: Optional[float] = None
        primary_concordant_count: Optional[int] = None
        any_finding_overlap_rate: Optional[float] = None
        any_finding_overlap_count: Optional[int] = None
        concordance_summary: Optional[str] = None

        if primary_finding is not None:
            primary_matches = [
                m for m in matches if m.get("concordance", {}).get("matches_primary")
            ]
            primary_concordant_count = len(primary_matches)
            primary_concordance_rate = (
                round((primary_concordant_count / len(matches)) * 100.0, 1)
                if matches
                else 0.0
            )

        if query_findings is not None or primary_finding is not None:
            overlap_matches = [
                m for m in matches if m.get("concordance", {}).get("has_overlap")
            ]
            any_finding_overlap_count = len(overlap_matches)
            any_finding_overlap_rate = (
                round((any_finding_overlap_count / len(matches)) * 100.0, 1)
                if matches
                else 0.0
            )

            target_disease = primary_finding or (query_findings[0] if query_findings else "predicted pathology")
            if primary_finding:
                concordance_summary = (
                    f"{primary_concordant_count} of {len(matches)} ({primary_concordance_rate:.1f}%) retrieved reference cases "
                    f"corroborate primary {target_disease} finding; "
                    f"{any_finding_overlap_count} of {len(matches)} ({any_finding_overlap_rate:.1f}%) share any detected finding."
                )
            else:
                concordance_summary = (
                    f"{any_finding_overlap_count} of {len(matches)} ({any_finding_overlap_rate:.1f}%) retrieved reference cases "
                    f"share detected positive finding(s)."
                )

        # Compute Metric Two: Similarity-weighted label support table across 14 canonical diseases
        weighted_label_support = compute_similarity_weighted_support(
            matches=matches,
            canonical_diseases=CANONICAL_DISEASES,
            global_prevalences=self.prevalence_map,
            global_frequency_data=self.frequency_data,
        )

        # Compute Metric One: Structured Global NIH Dataset Statistics across 14 canonical diseases
        global_dataset_statistics = [
            {
                "disease_name": d,
                "occurrence_count": self.occurrence_map.get(d, 0),
                "prevalence_percentage": self.prevalence_map.get(d, 0.0),
                "denominator": self.dataset_denominator,
            }
            for d in CANONICAL_DISEASES
        ]

        result_payload: Dict[str, Any] = {
            "status": "SUCCESS",
            "top_k": len(matches),
            "total_references": len(filenames),
            "matches": matches,
            "top_match": top_match,
            "disclaimer": DISCLAIMER_TEXT,
            "explanation_text": EXPLANATION_TEXT,
            "summary": (
                f"Retrieved Top-{len(matches)} most visually similar reference cases from "
                f"a repository of {len(filenames)} reference radiographs (DenseNet-121 1024-d representation)."
            ),
            "predicted_findings_evidence": predicted_findings_evidence,
            "predicted_evidence": predicted_findings_evidence,
            "predicted_findings": resolved_predicted_findings,
            "global_dataset_statistics": global_dataset_statistics,
            "no_finding_statistics": self.no_finding_stats,
            "weighted_label_support": weighted_label_support,
            "retrieved_case_support": weighted_label_support,
        }

        if primary_concordance_rate is not None:
            result_payload["primary_concordance_rate"] = primary_concordance_rate
            result_payload["primary_concordant_count"] = primary_concordant_count
            # Backward compatibility aliases
            result_payload["concordance_rate"] = primary_concordance_rate
            result_payload["concordant_cases_count"] = primary_concordant_count

        if any_finding_overlap_rate is not None:
            result_payload["any_finding_overlap_rate"] = any_finding_overlap_rate
            result_payload["any_finding_overlap_count"] = any_finding_overlap_count

        if primary_finding is not None:
            result_payload["primary_query_finding"] = primary_finding

        if concordance_summary is not None:
            result_payload["concordance_summary"] = concordance_summary

        return result_payload

    def find_similar_cases(
        self,
        image_input: Union[str, Path, bytes, Image.Image, torch.Tensor],
        top_k: int = 5,
        base_url: str = "/reference_images/",
        query_findings: Optional[List[str]] = None,
        primary_finding: Optional[str] = None,
        exclude_filenames: Optional[Union[str, Sequence[str]]] = None,
        predicted_findings: Optional[Sequence[str]] = None,
        authentic_only: bool = False,
        exclude_identical_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract features from input image and retrieve Top-K most similar reference cases.

        Args:
            image_input: Chest X-ray file path, raw bytes, PIL Image, or preprocessed Tensor.
            top_k: Number of most similar reference cases to return (default: 5).
            base_url: URL prefix for serving reference images.
            query_findings: Optional list of detected positive findings for the query image.
            primary_finding: Optional primary disease detected for the query image.
            exclude_filenames: Optional filename or list of filenames to exclude from retrieval.
            predicted_findings: Optional list of predicted disease findings to evaluate.
            authentic_only: If True, exclude synthetic demonstration images.
            exclude_identical_embeddings: If True, exclude identical feature embeddings (self-match).

        Returns:
            Structured dictionary with similarity matches, prediction-driven evidence, concordance, and disclaimer.
        """
        query_vector = self.extractor.extract_features(image_input, normalize=True)
        return self.find_similar_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            base_url=base_url,
            query_findings=query_findings,
            primary_finding=primary_finding,
            exclude_filenames=exclude_filenames,
            predicted_findings=predicted_findings,
            authentic_only=authentic_only,
            exclude_identical_embeddings=exclude_identical_embeddings,
        )

    def analyze(
        self,
        image_input: Union[str, Path, bytes, Image.Image, torch.Tensor],
        top_k: int = 5,
        base_url: str = "/reference_images/",
        query_findings: Optional[List[str]] = None,
        primary_finding: Optional[str] = None,
        exclude_filenames: Optional[Union[str, Sequence[str]]] = None,
        predicted_findings: Optional[Sequence[str]] = None,
        authentic_only: bool = False,
        exclude_identical_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """Alias for find_similar_cases to maintain standard MedTrust analyzer interface."""
        return self.find_similar_cases(
            image_input=image_input,
            top_k=top_k,
            base_url=base_url,
            query_findings=query_findings,
            primary_finding=primary_finding,
            exclude_filenames=exclude_filenames,
            predicted_findings=predicted_findings,
            authentic_only=authentic_only,
            exclude_identical_embeddings=exclude_identical_embeddings,
        )
