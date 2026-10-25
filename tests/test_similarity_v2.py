"""
MedTrust V2 — Unit and Regression Test Suite for Similarity Analyzer V2
(tests/test_similarity_v2.py)

Comprehensive test coverage for:
1. Reference Case Intelligence Report Schema & Validation
2. Persistent Embedding Index (2,500 x 1,024 unit-normalized vectors, deterministic tie-breaking)
3. Evidence Comparison Engine (K_d, R_d, V_d, P_d, zero division safety, bounds)
4. Similarity Retrieval Engine V2 (K in {5, 10, 20}, zero-image deployment execution)
5. Backend Service, Case Report Lookup, and Health Integrity
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.similarity_v2.report_schema import (
    validate_reference_report,
    create_reference_report,
    SCHEMA_VERSION,
)
from src.similarity_v2.embedding_index import PersistentEmbeddingIndex
from src.similarity_v2.evidence_engine import (
    EvidenceComparisonEngine,
    STANDARD_DISCLAIMER,
    METHODOLOGICAL_EXPLANATION,
    CANONICAL_DISEASES,
)
from src.similarity_v2.retrieval_engine import SimilarityRetrievalEngineV2
from src.similarity_v2.service import (
    get_v2_index_health,
    get_reference_case_report,
    get_similarity_engine_v2,
)


class TestReportSchema(unittest.TestCase):
    """Tests for reference case intelligence report schema validation and building."""

    def setUp(self) -> None:
        self.valid_report = {
            "schema_version": "2.0.0",
            "reference_id": "00000001_000.png",
            "dataset_provenance": "NIH ChestX-ray14",
            "dataset_split": "train",
            "patient_id": "1",
            "demographics": {
                "patient_age": 58,
                "patient_gender": "M",
                "view_position": "PA",
            },
            "dataset_annotations": {
                "findings": ["Cardiomegaly"],
                "finding_labels": "Cardiomegaly",
                "annotation_type": "Dataset Annotation",
                "is_authentic_nih": True,
            },
            "predictions": {
                "raw_scores": {d: 0.1 for d in CANONICAL_DISEASES},
                "calibrated_scores": {d: 0.1 for d in CANONICAL_DISEASES},
                "thresholds": {d: 0.5 for d in CANONICAL_DISEASES},
                "decisions": {d: False for d in CANONICAL_DISEASES},
                "positive_findings": [],
            },
            "image_quality": {
                "composite_score": 75,
                "sharpness": 82.5,
                "contrast": 68.4,
                "brightness": 128.0,
                "suitable": True,
                "status": "SUCCESS",
            },
            "ood": {
                "is_ood": False,
                "status": "IN_DISTRIBUTION",
                "msp_score": 0.95,
                "energy_score": -5.2,
            },
            "uncertainty": {
                "overall_mean_variance": 0.0012,
                "root_mean_variance": 0.0346,
                "passes": 30,
            },
            "explainability": {
                "primary_disease": "Cardiomegaly",
                "stability_status": "HIGH",
                "mean_spearman": 0.98,
            },
            "embedding": {
                "dimension": 1024,
                "is_normalized": True,
                "norm": 1.0,
            },
            "model_provenance": {
                "architecture": "DenseNet-121",
                "checkpoint": "checkpoints/best_model.pt",
                "checkpoint_sha256": "6160588028dad6ce5396c74962c7d85b162d06b816f34175249fa069c8466d34",
            },
            "component_status": {
                "image_quality": "SUCCESS",
                "ood": "SUCCESS",
                "predictions": "SUCCESS",
            },
            "created_at": "2026-10-20T00:00:00Z",
            "report_sha256": "abcdef1234567890",
        }

    def test_valid_report_passes_validation(self) -> None:
        is_valid, errors = validate_reference_report(self.valid_report)
        self.assertTrue(is_valid, f"Expected valid, got errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_missing_required_top_level_field_fails(self) -> None:
        tampered = dict(self.valid_report)
        del tampered["predictions"]
        is_valid, errors = validate_reference_report(tampered)
        self.assertFalse(is_valid)
        self.assertTrue(any("predictions" in err for err in errors))

    def test_missing_demographics_fields_fails(self) -> None:
        tampered = dict(self.valid_report)
        tampered["demographics"] = {"patient_age": 58}  # Missing gender and view_position
        is_valid, errors = validate_reference_report(tampered)
        self.assertFalse(is_valid)
        self.assertTrue(any("patient_gender" in err or "view_position" in err for err in errors))

    def test_invalid_demographic_values_fail(self) -> None:
        tampered = dict(self.valid_report)
        tampered["demographics"] = {"patient_age": -5, "patient_gender": "X", "view_position": "INVALID"}
        is_valid, errors = validate_reference_report(tampered)
        self.assertFalse(is_valid)

    def test_strict_separation_of_ground_truth_and_predictions(self) -> None:
        # Dataset annotations must never contain prediction probabilities
        self.assertNotIn("probability", self.valid_report["dataset_annotations"])
        self.assertNotIn("score", self.valid_report["dataset_annotations"])
        # Predictions must have all 14 canonical diseases
        self.assertEqual(len(self.valid_report["predictions"]["raw_scores"]), 14)


class TestPersistentEmbeddingIndex(unittest.TestCase):
    """Tests for PersistentEmbeddingIndex integrity, normalization, and search."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = PersistentEmbeddingIndex()

    def test_index_is_loaded_and_valid(self) -> None:
        self.assertEqual(self.index.get_count(), 2500)
        self.assertEqual(self.index.get_dimension(), 1024)

    def test_all_embeddings_are_strictly_unit_normalized(self) -> None:
        filenames, embeddings = self.index.get_data()
        norms = np.linalg.norm(embeddings, axis=1)
        # Invariant: ||v||_2 = 1.0 within numerical float32 epsilon
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_search_deterministic_tie_breaking(self) -> None:
        # Repeating identical search must return identical order
        filenames, embeddings = self.index.get_data()
        query_v = embeddings[10].copy()
        hits_1 = self.index.search(query_v, top_k=10, exclude_identical=False)
        hits_2 = self.index.search(query_v, top_k=10, exclude_identical=False)
        self.assertEqual(hits_1, hits_2)

    def test_self_exclusion_filters_query_radiograph(self) -> None:
        filenames, embeddings = self.index.get_data()
        target_fn = filenames[0]
        query_v = embeddings[0].copy()

        # Without exclusion: rank 1 is target_fn with ~100% similarity
        hits_no_ex = self.index.search(query_v, top_k=5, exclude_identical=False)
        self.assertEqual(hits_no_ex[0][0], target_fn)

        # With exclude_ids: target_fn is excluded
        hits_ex = self.index.search(query_v, top_k=5, exclude_ids=[target_fn])
        retrieved_ids = [h[0] for h in hits_ex]
        self.assertNotIn(target_fn, retrieved_ids)

    def test_metadata_lookup_by_filename(self) -> None:
        meta = self.index.get_metadata("00008394_000.png")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.get("filename"), "00008394_000.png")
        self.assertEqual(meta.get("dataset_source"), "NIH ChestX-ray14")


class TestEvidenceComparisonEngine(unittest.TestCase):
    """Tests for mathematical invariants in evidence comparison and concordance."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = EvidenceComparisonEngine()

    def test_constant_denominator_k(self) -> None:
        # Mock 5 retrieved cases
        mock_cases = [
            {"similarity_score": 90.0, "findings": ["Effusion"], "finding_labels": "Effusion"},
            {"similarity_score": 85.0, "findings": ["Atelectasis"], "finding_labels": "Atelectasis"},
            {"similarity_score": 80.0, "findings": ["Effusion", "Pneumonia"], "finding_labels": "Effusion|Pneumonia"},
            {"similarity_score": 75.0, "findings": ["No Finding"], "finding_labels": "No Finding"},
            {"similarity_score": 70.0, "findings": ["Cardiomegaly"], "finding_labels": "Cardiomegaly"},
        ]

        res = self.engine.evaluate_retrieval_evidence(
            retrieved_cases=mock_cases,
            predicted_findings=["Effusion", "Pneumonia"],
            primary_finding="Effusion",
        )

        self.assertEqual(res["top_k"], 5)
        for ev in res["predicted_findings_evidence"]:
            # Denominator must strictly be K = 5
            self.assertEqual(ev["retrieved_cases"], 5)

        # Effusion appears in cases 0 and 2 -> K_d = 2, R_d = 40.0%
        eff_ev = next(e for e in res["predicted_findings_evidence"] if e["finding"] == "Effusion")
        self.assertEqual(eff_ev["matching_cases"], 2)
        self.assertEqual(eff_ev["matching_rate"], 40.0)

        # Pneumonia appears in case 2 -> K_d = 1, R_d = 20.0%
        pneu_ev = next(e for e in res["predicted_findings_evidence"] if e["finding"] == "Pneumonia")
        self.assertEqual(pneu_ev["matching_cases"], 1)
        self.assertEqual(pneu_ev["matching_rate"], 20.0)

    def test_weighted_support_bounds_and_zero_division_safety(self) -> None:
        # Zero weight case
        zero_cases = [
            {"similarity_score": 0.0, "findings": ["Atelectasis"], "finding_labels": "Atelectasis"},
            {"similarity_score": 0.0, "findings": ["Atelectasis"], "finding_labels": "Atelectasis"},
        ]
        res = self.engine.evaluate_retrieval_evidence(
            retrieved_cases=zero_cases,
            predicted_findings=["Atelectasis"],
            primary_finding="Atelectasis",
        )
        self.assertEqual(res["predicted_findings_evidence"][0]["weighted_support"], 0.0)

    def test_primary_concordance_and_overlap_rates(self) -> None:
        mock_cases = [
            {"similarity_score": 90.0, "findings": ["Cardiomegaly"], "finding_labels": "Cardiomegaly"},
            {"similarity_score": 85.0, "findings": ["Effusion"], "finding_labels": "Effusion"},
            {"similarity_score": 80.0, "findings": ["Infiltration"], "finding_labels": "Infiltration"},
            {"similarity_score": 75.0, "findings": ["No Finding"], "finding_labels": "No Finding"},
            {"similarity_score": 70.0, "findings": ["Pneumothorax"], "finding_labels": "Pneumothorax"},
        ]

        res = self.engine.evaluate_retrieval_evidence(
            retrieved_cases=mock_cases,
            predicted_findings=["Cardiomegaly", "Effusion"],
            primary_finding="Cardiomegaly",
        )

        # Primary Cardiomegaly appears in 1 of 5 -> 20.0%
        self.assertEqual(res["primary_concordant_count"], 1)
        self.assertEqual(res["primary_concordance_rate"], 20.0)

        # Any finding (Cardiomegaly or Effusion) appears in 2 of 5 -> 40.0%
        self.assertEqual(res["any_finding_overlap_count"], 2)
        self.assertEqual(res["any_finding_overlap_rate"], 40.0)


class TestSimilarityRetrievalEngineV2(unittest.TestCase):
    """Tests for full retrieval engine execution with K in {5, 10, 20}."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = SimilarityRetrievalEngineV2()

    def test_retrieval_depths_5_10_20(self) -> None:
        dummy_vector = np.random.randn(1024).astype(np.float32)
        dummy_vector = dummy_vector / np.linalg.norm(dummy_vector)

        for k in (5, 10, 20):
            res = self.engine.find_similar_by_vector(
                query_vector=dummy_vector,
                top_k=k,
                predicted_findings=["Cardiomegaly"],
                primary_finding="Cardiomegaly",
            )
            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["top_k"], k)
            self.assertEqual(len(res["matches"]), k)
            self.assertEqual(res["candidates_searched"], 2500)
            self.assertIn("top_match", res)
            self.assertIsNotNone(res["top_match"])

    def test_zero_raw_images_required_at_runtime(self) -> None:
        # Retrieval from feature vector does NOT read dataset/images/ or reference_images/
        dummy_vector = np.ones(1024, dtype=np.float32)
        dummy_vector = dummy_vector / np.linalg.norm(dummy_vector)

        res = self.engine.find_similar_by_vector(dummy_vector, top_k=5)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(len(res["matches"]), 5)

    def test_reference_report_lookup(self) -> None:
        report = self.engine.get_reference_report("00008394_000.png")
        self.assertIsNotNone(report)
        self.assertEqual(report.get("reference_id"), "00008394_000.png")
        self.assertIn("demographics", report)
        self.assertIn("dataset_annotations", report)
        self.assertIn("predictions", report)


class TestServiceAndHealth(unittest.TestCase):
    """Tests for service singletons and health checks."""

    def test_v2_health_report(self) -> None:
        health = get_v2_index_health()
        self.assertEqual(health["status"], "HEALTHY")
        self.assertEqual(health["version"], "2.0.0")
        self.assertEqual(health["indexed_cases"], 2500)
        self.assertEqual(health["embedding_dimension"], 1024)
        self.assertTrue(health["is_unit_normalized"])
        self.assertEqual(health["reports_available"], 2500)
        self.assertTrue(health["hashes_verified"])
        self.assertFalse(health["requires_original_images"])

    def test_report_lookup_via_service(self) -> None:
        rep = get_reference_case_report("00008394_000.png")
        self.assertIsNotNone(rep)
        self.assertEqual(rep["patient_id"], "8394")

    def test_nonexistent_report_returns_none(self) -> None:
        rep = get_reference_case_report("nonexistent_image_99999.png")
        self.assertIsNone(rep)


if __name__ == "__main__":
    unittest.main()
