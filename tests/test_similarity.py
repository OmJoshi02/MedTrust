"""
MedTrust V1 — Unit and Regression Tests for Module 8 (Similarity Analysis) (tests/test_similarity.py)

Covers:
- Cosine similarity mathematical invariants & numerical stability
- Top-K ranking ordering and boundary cases
- Reference embedding store loading, caching, and error handling
- DenseNet-121 feature extractor output shape, normalization, and determinism
- Unified SimilarityAnalyzer end-to-end integration and disclaimer compliance
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.similarity.similarity_calculator import (
    DISCLAIMER_TEXT,
    EXPLANATION_TEXT,
    compute_cosine_similarity,
    compute_similarity_weighted_support,
    compute_predicted_findings_evidence,
    rank_similar_images,
)
from src.similarity.disease_frequency import (
    CANONICAL_DISEASES,
    NO_FINDING_LABEL,
    compute_disease_frequencies,
    load_cached_disease_frequency_json,
    load_cached_disease_frequencies,
)
from src.similarity.embedding_store import EmbeddingStore
from src.similarity.feature_extractor import DenseNetSimilarityExtractor
from src.similarity.similarity_analyzer import SimilarityAnalyzer


class TestCosineSimilarityMath(unittest.TestCase):
    """Tests for mathematical invariants of cosine similarity computation."""

    def test_identical_vectors_have_100_percent_similarity(self):
        v = np.random.randn(1024).astype(np.float32)
        v = v / np.linalg.norm(v)
        ref = np.array([v, v * 0.5])

        scores = compute_cosine_similarity(v, ref)
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], 100.0, places=4)
        self.assertAlmostEqual(scores[1], 100.0, places=4)  # Collinear vector has cos = 1.0

    def test_orthogonal_vectors_have_zero_similarity(self):
        v1 = np.zeros(1024, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(1024, dtype=np.float32)
        v2[1] = 1.0
        ref = np.array([v2])

        scores = compute_cosine_similarity(v1, ref)
        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0], 0.0, places=4)

    def test_opposite_vectors_clip_to_zero(self):
        v1 = np.ones(1024, dtype=np.float32)
        v2 = -np.ones(1024, dtype=np.float32)
        ref = np.array([v2])

        scores = compute_cosine_similarity(v1, ref)
        # Cosine is -1.0, non-negative projection maps to 0.0%
        self.assertAlmostEqual(scores[0], 0.0, places=4)

    def test_zero_vector_does_not_crash(self):
        v_zero = np.zeros(1024, dtype=np.float32)
        ref = np.random.randn(3, 1024).astype(np.float32)

        scores = compute_cosine_similarity(v_zero, ref)
        self.assertEqual(len(scores), 3)
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_dimension_mismatch_raises_value_error(self):
        v = np.random.randn(512).astype(np.float32)
        ref = np.random.randn(3, 1024).astype(np.float32)
        with self.assertRaises(ValueError):
            compute_cosine_similarity(v, ref)

    def test_ranking_orders_descending(self):
        q = np.zeros(1024, dtype=np.float32)
        q[0] = 1.0

        r1 = np.zeros(1024, dtype=np.float32)  # Orthogonal -> 0%
        r1[1] = 1.0

        r2 = q.copy()  # Identical -> 100%

        r3 = np.zeros(1024, dtype=np.float32)  # Partial alignment
        r3[0] = 0.5
        r3[1] = 0.5
        r3 = r3 / np.linalg.norm(r3)

        ref_matrix = np.array([r1, r2, r3])
        filenames = ["r1.jpg", "r2.jpg", "r3.jpg"]

        ranked = rank_similar_images(q, filenames, ref_matrix, top_k=3)
        self.assertEqual(len(ranked), 3)

        # Ranks must be 1, 2, 3 in strictly non-increasing score order
        self.assertEqual(ranked[0]["filename"], "r2.jpg")
        self.assertAlmostEqual(ranked[0]["similarity_score"], 100.0, places=2)
        self.assertEqual(ranked[1]["filename"], "r3.jpg")
        self.assertEqual(ranked[2]["filename"], "r1.jpg")
        self.assertAlmostEqual(ranked[2]["similarity_score"], 0.0, places=2)

        for i in range(len(ranked) - 1):
            self.assertGreaterEqual(ranked[i]["similarity_score"], ranked[i + 1]["similarity_score"])

    def test_top_k_bounds(self):
        q = np.random.randn(1024).astype(np.float32)
        ref = np.random.randn(10, 1024).astype(np.float32)
        filenames = [f"img_{i}.jpg" for i in range(10)]

        ranked_3 = rank_similar_images(q, filenames, ref, top_k=3)
        self.assertEqual(len(ranked_3), 3)

        ranked_all = rank_similar_images(q, filenames, ref, top_k=100)
        self.assertEqual(len(ranked_all), 10)


class TestEmbeddingStore(unittest.TestCase):
    """Tests for reference embedding persistence and cache behavior."""

    def test_missing_file_raises_filenotfound(self):
        store = EmbeddingStore("/nonexistent/path/to/embeddings.npz")
        with self.assertRaises(FileNotFoundError):
            store.load()

    def test_valid_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_embeddings.npz"
            filenames = ["case_a.png", "case_b.png"]
            embeddings = np.random.randn(2, 1024).astype(np.float32)

            np.savez_compressed(file_path, filenames=filenames, embeddings=embeddings)

            store = EmbeddingStore(file_path)
            loaded_names, loaded_embeds = store.load()

            self.assertEqual(loaded_names, filenames)
            self.assertEqual(loaded_embeds.shape, (2, 1024))
            self.assertEqual(store.get_count(), 2)
            self.assertTrue(store.is_loaded)

    def test_shape_mismatch_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "corrupt.npz"
            np.savez_compressed(
                file_path,
                filenames=["a.jpg"],
                embeddings=np.random.randn(1, 512).astype(np.float32),  # Not 1024
            )
            store = EmbeddingStore(file_path)
            with self.assertRaises(ValueError):
                store.load()


class TestFeatureExtractor(unittest.TestCase):
    """Tests for DenseNet-121 feature vector extraction."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.extractor = DenseNetSimilarityExtractor(device=cls.device)

    def test_extract_features_shape_and_norm(self):
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        vector = self.extractor.extract_features(dummy_img, normalize=True)

        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (1024,))
        self.assertEqual(vector.dtype, np.float32)

        norm = np.linalg.norm(vector)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_extract_features_deterministic(self):
        dummy_img = Image.new("L", (224, 224), color=200)
        v1 = self.extractor.extract_features(dummy_img)
        v2 = self.extractor.extract_features(dummy_img)
        np.testing.assert_array_almost_equal(v1, v2, decimal=6)

    def test_extract_from_tensor_input(self):
        tensor = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
        vector = self.extractor.extract_features(tensor)
        self.assertEqual(vector.shape, (1024,))


class TestSimilarityAnalyzer(unittest.TestCase):
    """Integration tests for the unified SimilarityAnalyzer."""

    def test_find_similar_cases_disclaimer_and_result_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            emb_file = Path(tmpdir) / "ref.npz"
            # Create a store with 3 reference vectors
            v_ref = np.random.randn(3, 1024).astype(np.float32)
            v_ref = v_ref / np.linalg.norm(v_ref, axis=1, keepdims=True)
            np.savez_compressed(
                emb_file,
                filenames=["ref1.jpg", "ref2.jpg", "ref3.jpg"],
                embeddings=v_ref,
            )

            store = EmbeddingStore(emb_file)
            extractor = DenseNetSimilarityExtractor(device=torch.device("cpu"))
            analyzer = SimilarityAnalyzer(extractor=extractor, store=store)

            # Query with a known vector matching ref1 exactly
            res = analyzer.find_similar_by_vector(v_ref[0], top_k=2)

            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["top_k"], 2)
            self.assertEqual(res["total_references"], 3)
            self.assertEqual(len(res["matches"]), 2)

            # Top match must be ref1 with 100% similarity
            self.assertEqual(res["matches"][0]["filename"], "ref1.jpg")
            self.assertAlmostEqual(res["matches"][0]["similarity_score"], 100.0, places=2)

            # Disclaimer MUST be present
            self.assertIn("not a medical diagnosis", res["disclaimer"].lower())
            self.assertEqual(res["disclaimer"], DISCLAIMER_TEXT)

    def test_concordance_calculation_in_analyzer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            emb_file = Path(tmpdir) / "ref_meta.npz"
            v_ref = np.random.randn(3, 1024).astype(np.float32)
            v_ref = v_ref / np.linalg.norm(v_ref, axis=1, keepdims=True)
            meta = [
                {"filename": "c1.jpg", "findings": ["Cardiomegaly", "Effusion"], "finding_labels": "Cardiomegaly|Effusion"},
                {"filename": "c2.jpg", "findings": ["Pneumonia"], "finding_labels": "Pneumonia"},
                {"filename": "c3.jpg", "findings": ["Cardiomegaly"], "finding_labels": "Cardiomegaly"},
            ]
            np.savez_compressed(
                emb_file,
                filenames=["c1.jpg", "c2.jpg", "c3.jpg"],
                embeddings=v_ref,
                metadata=meta,
            )

            store = EmbeddingStore(emb_file)
            extractor = DenseNetSimilarityExtractor(device=torch.device("cpu"))
            analyzer = SimilarityAnalyzer(extractor=extractor, store=store)

            # Query with Cardiomegaly as primary
            res = analyzer.find_similar_by_vector(
                v_ref[0],
                top_k=3,
                query_findings=["Cardiomegaly"],
                primary_finding="Cardiomegaly",
            )

            self.assertIn("concordance_rate", res)
            self.assertIn("concordant_cases_count", res)
            self.assertIn("primary_concordance_rate", res)
            self.assertIn("any_finding_overlap_rate", res)
            self.assertIn("weighted_label_support", res)
            self.assertEqual(res["primary_query_finding"], "Cardiomegaly")
            # 2 out of 3 reference cases have Cardiomegaly (c1 and c3)
            self.assertEqual(res["primary_concordant_count"], 2)
            self.assertAlmostEqual(res["primary_concordance_rate"], 66.7, places=1)
            self.assertEqual(res["any_finding_overlap_count"], 2)
            self.assertAlmostEqual(res["any_finding_overlap_rate"], 66.7, places=1)

            # Check individual matches
            for m in res["matches"]:
                self.assertIn("concordance", m)
                if m["filename"] in ("c1.jpg", "c3.jpg"):
                    self.assertEqual(m["concordance"]["status"], "PRIMARY_CONCORDANT")
                    self.assertTrue(m["concordance"]["matches_primary"])
                else:
                    self.assertEqual(m["concordance"]["status"], "DIFFERENTIAL")
                    self.assertFalse(m["concordance"]["matches_primary"])

    def test_query_exclusion_from_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            emb_file = Path(tmpdir) / "ref_excl.npz"
            v_ref = np.random.randn(3, 1024).astype(np.float32)
            v_ref = v_ref / np.linalg.norm(v_ref, axis=1, keepdims=True)
            np.savez_compressed(
                emb_file,
                filenames=["sample_query.jpg", "case_b.png", "case_c.png"],
                embeddings=v_ref,
            )

            store = EmbeddingStore(emb_file)
            extractor = DenseNetSimilarityExtractor(device=torch.device("cpu"))
            analyzer = SimilarityAnalyzer(extractor=extractor, store=store)

            # Query excluding sample_query.jpg
            res = analyzer.find_similar_by_vector(
                v_ref[0],
                top_k=3,
                exclude_filenames=["sample_query.jpg"],
            )

            retrieved_names = [m["filename"] for m in res["matches"]]
            self.assertNotIn("sample_query.jpg", retrieved_names)
            self.assertEqual(len(res["matches"]), 2)


class TestMetadataLoader(unittest.TestCase):
    """Tests for ReferenceMetadataLoader and CSV parsing."""

    def test_parse_finding_labels(self):
        from src.similarity.metadata_loader import parse_finding_labels

        self.assertEqual(parse_finding_labels("Cardiomegaly|Effusion"), ["Cardiomegaly", "Effusion"])
        self.assertEqual(parse_finding_labels("Pneumonia, Infiltration"), ["Pneumonia", "Infiltration"])
        self.assertEqual(parse_finding_labels("No Finding"), ["No Finding"])
        self.assertEqual(parse_finding_labels(""), ["No Finding"])
        self.assertEqual(parse_finding_labels(None), ["No Finding"])

    def test_reference_cases_csv_loading(self):
        from src.similarity.metadata_loader import ReferenceMetadataLoader

        loader = ReferenceMetadataLoader()
        meta = loader.get_metadata_for_filename("xray001.jpg")

        self.assertEqual(meta["filename"], "xray001.jpg")
        self.assertIn("Cardiomegaly", meta["findings"])
        self.assertIn("Effusion", meta["findings"])
        self.assertEqual(meta["patient_age"], 58)
        self.assertEqual(meta["patient_gender"], "M")
        self.assertEqual(meta["view_position"], "PA")

    def test_fallback_for_unlisted_filename(self):
        from src.similarity.metadata_loader import ReferenceMetadataLoader

        loader = ReferenceMetadataLoader()
        meta = loader.get_metadata_for_filename("unknown_image_999.jpg")
        self.assertEqual(meta["filename"], "unknown_image_999.jpg")
        self.assertIn("findings", meta)
        self.assertEqual(meta["findings"], ["Unspecified"])


class TestDiagnosticConcordance(unittest.TestCase):
    """Tests for evaluate_diagnostic_concordance."""

    def test_concordant_shared_findings(self):
        from src.similarity.similarity_calculator import evaluate_diagnostic_concordance

        ref_meta = {"findings": ["Cardiomegaly", "Effusion"], "finding_labels": "Cardiomegaly|Effusion"}
        res = evaluate_diagnostic_concordance(
            reference_metadata=ref_meta,
            query_findings=["Cardiomegaly"],
            primary_finding="Cardiomegaly",
        )
        self.assertEqual(res["status"], "PRIMARY_CONCORDANT")
        self.assertEqual(res["label"], "Primary Concordant")
        self.assertTrue(res["matches_primary"])
        self.assertTrue(res["has_overlap"])
        self.assertIn("Cardiomegaly", res["shared_findings"])

    def test_secondary_overlap(self):
        from src.similarity.similarity_calculator import evaluate_diagnostic_concordance

        ref_meta = {"findings": ["Infiltration"], "finding_labels": "Infiltration"}
        res = evaluate_diagnostic_concordance(
            reference_metadata=ref_meta,
            query_findings=["Pneumothorax", "Infiltration"],
            primary_finding="Pneumothorax",
        )
        self.assertEqual(res["status"], "SECONDARY_OVERLAP")
        self.assertEqual(res["label"], "Secondary Overlap")
        self.assertFalse(res["matches_primary"])
        self.assertTrue(res["has_overlap"])
        self.assertIn("Infiltration", res["shared_findings"])

    def test_differential_lookalike(self):
        from src.similarity.similarity_calculator import evaluate_diagnostic_concordance

        ref_meta = {"findings": ["Atelectasis"], "finding_labels": "Atelectasis"}
        res = evaluate_diagnostic_concordance(
            reference_metadata=ref_meta,
            query_findings=["Pneumonia"],
            primary_finding="Pneumonia",
        )
        self.assertEqual(res["status"], "DIFFERENTIAL")
        self.assertEqual(res["label"], "Differential Lookalike")
        self.assertFalse(res["matches_primary"])
        self.assertFalse(res["has_overlap"])
        self.assertEqual(len(res["shared_findings"]), 0)
        self.assertIn("differential lookalike", res["description"].lower())

    def test_neutral_no_finding(self):
        from src.similarity.similarity_calculator import evaluate_diagnostic_concordance

        ref_meta = {"findings": ["No Finding"], "finding_labels": "No Finding"}
        res = evaluate_diagnostic_concordance(
            reference_metadata=ref_meta,
            query_findings=["No Finding"],
            primary_finding="No Finding",
        )
        self.assertEqual(res["status"], "NO_FINDING")

    def test_disease_name_normalization(self):
        from src.similarity.similarity_calculator import evaluate_diagnostic_concordance

        ref_meta = {"findings": ["Pleural_Thickening"], "finding_labels": "Pleural_Thickening"}
        res = evaluate_diagnostic_concordance(
            reference_metadata=ref_meta,
            query_findings=["pleural thickening"],
            primary_finding="Pleural Thickening",
        )
        self.assertEqual(res["status"], "PRIMARY_CONCORDANT")
        self.assertTrue(res["matches_primary"])


class TestDiseaseFrequencyAndSupport(unittest.TestCase):
    """Unit tests for NIH CSV parsing, disease frequency calculation, and weighted label support."""

    def test_compute_disease_frequencies_synthetic_csv(self):
        from src.similarity.disease_frequency import compute_disease_frequencies

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_nih.csv"
            out_csv = Path(tmpdir) / "out_freq.csv"

            # Create synthetic CSV with duplicates, multilabel, No Finding, malformed row
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                f.write("Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position\n")
                f.write("00001.png,Atelectasis|Cardiomegaly,0,1,50,M,PA\n")
                f.write("00002.png,Cardiomegaly,0,2,60,F,PA\n")
                f.write("00003.png,No Finding,0,3,40,M,AP\n")
                f.write("00001.png,Atelectasis,0,1,50,M,PA\n")  # Duplicate ID
                f.write(",Cardiomegaly,0,4,30,F,PA\n")  # Empty ID (excluded)
                f.write("00004.png,Pneumonia|UnknownLabel,0,5,70,M,PA\n")

            res = compute_disease_frequencies(csv_path=csv_path, output_csv_path=out_csv, output_json_path=None)

            # Valid unique IDs: 00001, 00002, 00003, 00004 -> Total = 4
            self.assertEqual(res["total_valid_records"], 4)
            self.assertEqual(res["duplicate_records_count"], 1)
            self.assertEqual(res["excluded_records_count"], 1)
            self.assertIn("UnknownLabel", res["unknown_labels_found"])

            # Counts: Atelectasis (1), Cardiomegaly (2), Pneumonia (1), No Finding (1)
            self.assertEqual(res["disease_counts"]["Atelectasis"], 1)
            self.assertEqual(res["disease_counts"]["Cardiomegaly"], 2)
            self.assertEqual(res["disease_counts"]["Pneumonia"], 1)
            self.assertEqual(res["no_finding_count"], 1)

            # Prevalences: Cardiomegaly = 2/4 * 100 = 50.0%
            self.assertEqual(res["disease_prevalences"]["Cardiomegaly"], 50.0)
            self.assertEqual(res["disease_prevalences"]["Atelectasis"], 25.0)

            # Output CSV written
            self.assertTrue(out_csv.exists())

    def test_compute_similarity_weighted_support_math(self):
        from src.similarity.similarity_calculator import compute_similarity_weighted_support

        matches = [
            {
                "rank": 1,
                "similarity_score": 80.0,
                "metadata": {"findings": ["Cardiomegaly", "Effusion"]},
            },
            {
                "rank": 2,
                "similarity_score": 60.0,
                "metadata": {"findings": ["Cardiomegaly"]},
            },
            {
                "rank": 3,
                "similarity_score": 40.0,
                "metadata": {"findings": ["Pneumonia"]},
            },
        ]
        # Total weight W = 80 + 60 + 40 = 180.0
        # Cardiomegaly: in match 1 (80) and match 2 (60) -> W_d = 140 -> V_d = 140 / 180 * 100 = 77.78%
        # Effusion: in match 1 (80) -> W_d = 80 -> V_d = 80 / 180 * 100 = 44.44%
        # Pneumonia: in match 3 (40) -> W_d = 40 -> V_d = 40 / 180 * 100 = 22.22%
        # Hernia: 0.0%

        support = compute_similarity_weighted_support(matches)
        support_dict = {item["disease"]: item for item in support}

        self.assertEqual(len(support), 14)
        self.assertAlmostEqual(support_dict["Cardiomegaly"]["weighted_support"], 77.78, places=1)
        self.assertEqual(support_dict["Cardiomegaly"]["retrieved_count"], 2)
        self.assertAlmostEqual(support_dict["Cardiomegaly"]["retrieval_rate"], 66.67, places=1)

        self.assertAlmostEqual(support_dict["Effusion"]["weighted_support"], 44.44, places=1)
        self.assertEqual(support_dict["Effusion"]["retrieved_count"], 1)

        self.assertAlmostEqual(support_dict["Hernia"]["weighted_support"], 0.0, places=2)
        self.assertEqual(support_dict["Hernia"]["retrieved_count"], 0)


class TestConcordanceDistinctionPneumothoraxRegression(unittest.TestCase):
    """
    CRITICAL REGRESSION TEST:
    Ensures that secondary findings in retrieved cases do NOT inflate primary-finding concordance.

    Scenario:
    - Query primary finding: Pneumothorax
    - Query detected findings: [Pneumothorax, Pneumonia, Infiltration]
    - 5 Retrieved cases:
        1. Pneumothorax (Primary concordant)
        2. Pneumonia (Secondary overlap)
        3. Infiltration (Secondary overlap)
        4. Pneumonia|Infiltration (Secondary overlap)
        5. Atelectasis (Differential lookalike)

    Requirement:
    - Primary-Finding Concordance MUST be 20.0% (1/5), NOT 80.0%!
    - Any-Finding Overlap MUST be 80.0% (4/5)!
    """

    def test_pneumothorax_concordance_regression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            emb_file = Path(tmpdir) / "pneu_reg.npz"
            v_ref = np.random.randn(5, 1024).astype(np.float32)
            v_ref = v_ref / np.linalg.norm(v_ref, axis=1, keepdims=True)
            meta = [
                {"filename": "r1.png", "findings": ["Pneumothorax"], "finding_labels": "Pneumothorax"},
                {"filename": "r2.png", "findings": ["Pneumonia"], "finding_labels": "Pneumonia"},
                {"filename": "r3.png", "findings": ["Infiltration"], "finding_labels": "Infiltration"},
                {"filename": "r4.png", "findings": ["Pneumonia", "Infiltration"], "finding_labels": "Pneumonia|Infiltration"},
                {"filename": "r5.png", "findings": ["Atelectasis"], "finding_labels": "Atelectasis"},
            ]
            np.savez_compressed(
                emb_file,
                filenames=[f"r{i}.png" for i in range(1, 6)],
                embeddings=v_ref,
                metadata=meta,
            )

            store = EmbeddingStore(emb_file)
            extractor = DenseNetSimilarityExtractor(device=torch.device("cpu"))
            analyzer = SimilarityAnalyzer(extractor=extractor, store=store)

            res = analyzer.find_similar_by_vector(
                v_ref[0],
                top_k=5,
                query_findings=["Pneumothorax", "Pneumonia", "Infiltration"],
                primary_finding="Pneumothorax",
            )

            # Strict verification of the dual concordance metrics
            self.assertEqual(res["primary_concordant_count"], 1)
            self.assertAlmostEqual(res["primary_concordance_rate"], 20.0, places=1)

            self.assertEqual(res["any_finding_overlap_count"], 4)
            self.assertAlmostEqual(res["any_finding_overlap_rate"], 80.0, places=1)

            # Backward compatibility check
            self.assertAlmostEqual(res["concordance_rate"], 20.0, places=1)
            self.assertEqual(res["concordant_cases_count"], 1)

            # Match statuses
            statuses = [m["concordance"]["status"] for m in res["matches"]]
            self.assertEqual(statuses.count("PRIMARY_CONCORDANT"), 1)
            self.assertEqual(statuses.count("SECONDARY_OVERLAP"), 3)
            self.assertEqual(statuses.count("DIFFERENTIAL"), 1)


class TestConsolidationCase42Regression(unittest.TestCase):
    """
    CRITICAL REGRESSION TEST FOR CASE 42:
    Validates the exact scenario observed in Report 42:
    - Primary finding: Consolidation
    - Query detected findings: [Atelectasis, Effusion, Infiltration, Mass, Pneumonia, Pneumothorax, Consolidation, Emphysema, Pleural_Thickening]
    - 5 Retrieved cases from NIH catalog:
        1. 00000013_010.png: Effusion|Pneumonia|Pneumothorax (Secondary overlap)
        2. 00000020_001.png: Fibrosis|Infiltration (Secondary overlap)
        3. 00000013_023.png: Infiltration|Mass|Pneumothorax (Secondary overlap)
        4. 00000032_012.png: Atelectasis|Consolidation|Edema|Pneumonia (Primary concordant)
        5. 00000013_030.png: Atelectasis|Cardiomegaly|Emphysema|Mass|Pneumothorax (Secondary overlap)

    Requirement:
    - Primary Concordance MUST be 20.0% (1/5).
    - Any-Finding Overlap MUST be 100.0% (5/5).
    - Only Match 4 is PRIMARY_CONCORDANT. Matches 1, 2, 3, 5 are SECONDARY_OVERLAP.
    """

    def test_case_42_concordance_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            emb_file = Path(tmpdir) / "case42_reg.npz"
            v_ref = np.random.randn(5, 1024).astype(np.float32)
            v_ref = v_ref / np.linalg.norm(v_ref, axis=1, keepdims=True)
            meta = [
                {"filename": "00000013_010.png", "finding_labels": "Effusion|Pneumonia|Pneumothorax"},
                {"filename": "00000020_001.png", "finding_labels": "Fibrosis|Infiltration"},
                {"filename": "00000013_023.png", "finding_labels": "Infiltration|Mass|Pneumothorax"},
                {"filename": "00000032_012.png", "finding_labels": "Atelectasis|Consolidation|Edema|Pneumonia"},
                {"filename": "00000013_030.png", "finding_labels": "Atelectasis|Cardiomegaly|Emphysema|Mass|Pneumothorax"},
            ]
            np.savez_compressed(
                emb_file,
                filenames=[m["filename"] for m in meta],
                embeddings=v_ref,
                metadata=meta,
            )

            store = EmbeddingStore(emb_file)
            extractor = DenseNetSimilarityExtractor(device=torch.device("cpu"))
            analyzer = SimilarityAnalyzer(extractor=extractor, store=store)

            res = analyzer.find_similar_by_vector(
                v_ref[0],
                top_k=5,
                query_findings=[
                    "Atelectasis", "Effusion", "Infiltration", "Mass", "Pneumonia",
                    "Pneumothorax", "Consolidation", "Emphysema", "Pleural_Thickening",
                ],
                primary_finding="Consolidation",
            )

            self.assertEqual(res["primary_concordant_count"], 1)
            self.assertAlmostEqual(res["primary_concordance_rate"], 20.0, places=1)
            self.assertEqual(res["any_finding_overlap_count"], 5)
            self.assertAlmostEqual(res["any_finding_overlap_rate"], 100.0, places=1)

            # Find match 4 by filename (since rank ordering depends on random vectors)
            m4 = next(m for m in res["matches"] if m["filename"] == "00000032_012.png")
            self.assertTrue(m4["concordance"]["matches_primary"])
            self.assertEqual(m4["concordance"]["status"], "PRIMARY_CONCORDANT")
            for m in res["matches"]:
                if m["filename"] != "00000032_012.png":
                    self.assertFalse(m["concordance"]["matches_primary"])
                    self.assertTrue(m["concordance"]["has_overlap"])
                    self.assertEqual(m["concordance"]["status"], "SECONDARY_OVERLAP")


class TestEnsureSimilarityConcordanceMigration(unittest.TestCase):
    """
    Validates backend.server.ensure_similarity_concordance:
    - Corrects legacy payload with 100% concordance to actual 20% primary concordance.
    - Preserves exact 0.0% when primary finding has zero matches.
    - Standardizes individual match concordance statuses.
    """

    def test_legacy_payload_migration(self):
        from backend.server import ensure_similarity_concordance

        legacy_matches = [
            {
                "rank": 1,
                "filename": "00000013_010.png",
                "similarity_score": 70.17,
                "metadata": {"finding_labels": "Effusion|Pneumonia|Pneumothorax"},
                "concordance": {"status": "CONCORDANT", "matches_primary": False},
            },
            {
                "rank": 2,
                "filename": "00000020_001.png",
                "similarity_score": 67.64,
                "metadata": {"finding_labels": "Fibrosis|Infiltration"},
                "concordance": {"status": "CONCORDANT", "matches_primary": False},
            },
            {
                "rank": 3,
                "filename": "00000013_023.png",
                "similarity_score": 61.8,
                "metadata": {"finding_labels": "Infiltration|Mass|Pneumothorax"},
                "concordance": {"status": "CONCORDANT", "matches_primary": False},
            },
            {
                "rank": 4,
                "filename": "00000032_012.png",
                "similarity_score": 61.71,
                "metadata": {"finding_labels": "Atelectasis|Consolidation|Edema|Pneumonia"},
                "concordance": {"status": "CONCORDANT", "matches_primary": True},
            },
            {
                "rank": 5,
                "filename": "00000013_030.png",
                "similarity_score": 61.4,
                "metadata": {"finding_labels": "Atelectasis|Cardiomegaly|Emphysema|Mass|Pneumothorax"},
                "concordance": {"status": "CONCORDANT", "matches_primary": False},
            },
        ]

        legacy_sim_data = {
            "status": "SUCCESS",
            "top_k": 5,
            "total_references": 22,
            "matches": legacy_matches,
            "concordance_rate": 100.0,
            "concordant_cases_count": 5,
            "primary_concordance_rate": None,
            "any_finding_overlap_rate": None,
        }

        mock_result = {
            "prediction": {
                "primary_finding": {"disease": "Consolidation"},
                "findings": [
                    {"disease": d, "decision": "POSITIVE"}
                    for d in [
                        "Atelectasis", "Effusion", "Infiltration", "Mass", "Pneumonia",
                        "Pneumothorax", "Consolidation", "Emphysema", "Pleural_Thickening",
                    ]
                ],
            }
        }

        migrated = ensure_similarity_concordance(legacy_sim_data, mock_result)
        self.assertIsNotNone(migrated)
        self.assertEqual(migrated["primary_concordant_count"], 1)
        self.assertAlmostEqual(migrated["primary_concordance_rate"], 20.0, places=1)
        self.assertEqual(migrated["any_finding_overlap_count"], 5)
        self.assertAlmostEqual(migrated["any_finding_overlap_rate"], 100.0, places=1)
        self.assertEqual(migrated["matches"][3]["concordance"]["status"], "PRIMARY_CONCORDANT")
        self.assertEqual(migrated["matches"][0]["concordance"]["status"], "SECONDARY_OVERLAP")

    def test_zero_concordance_preserves_zero(self):
        from backend.server import ensure_similarity_concordance

        matches = [
            {
                "rank": 1,
                "filename": "00000001_000.png",
                "similarity_score": 75.0,
                "metadata": {"finding_labels": "Cardiomegaly"},
                "concordance": {"status": "CONCORDANT"},
            }
        ]
        sim_data = {
            "matches": matches,
            "primary_concordance_rate": None,
            "any_finding_overlap_rate": None,
        }
        mock_result = {
            "prediction": {
                "primary_finding": {"disease": "Hernia"},
                "findings": [{"disease": "Hernia", "decision": "POSITIVE"}],
            }
        }

        migrated = ensure_similarity_concordance(sim_data, mock_result)
        self.assertEqual(migrated["primary_concordant_count"], 0)
        self.assertEqual(migrated["primary_concordance_rate"], 0.0)
        self.assertEqual(migrated["any_finding_overlap_count"], 0)
        self.assertEqual(migrated["any_finding_overlap_rate"], 0.0)
        self.assertEqual(migrated["matches"][0]["concordance"]["status"], "DIFFERENTIAL")


class TestModule8DiseaseFrequency(unittest.TestCase):
    """
    Dedicated tests for Metric One: NIH ChestX-ray14 Disease Frequency & Global Prevalence.
    Covers all 12 specified requirements from Section 8 of Module 8 specification.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_sample_csv(self, rows: list[list[str]]) -> Path:
        csv_p = self.test_dir / "sample_data_entry.csv"
        with open(csv_p, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Index", "Finding Labels", "Follow-up #", "Patient ID", "Patient Age", "Patient Gender", "View Position"])
            for r in rows:
                writer.writerow(r)
        return csv_p

    def test_01_correct_csv_parsing(self):
        """Test 1: CSV parser properly parses headers and valid rows."""
        csv_p = self._create_sample_csv([
            ["img1.png", "Atelectasis", "0", "1", "45", "M", "PA"],
            ["img2.png", "Cardiomegaly", "1", "1", "46", "M", "AP"],
        ])
        res = compute_disease_frequencies(csv_path=csv_p, output_json_path=None, output_csv_path=None)
        self.assertEqual(res["total_valid_records"], 2)
        self.assertEqual(res["disease_counts"]["Atelectasis"], 1)
        self.assertEqual(res["disease_counts"]["Cardiomegaly"], 1)

    def test_02_correct_counting_single_label_records(self):
        """Test 2: Correct frequency counts for single-label images."""
        csv_p = self._create_sample_csv([
            ["img1.png", "Pneumothorax", "0", "1", "30", "F", "PA"],
            ["img2.png", "Pneumothorax", "0", "2", "35", "M", "PA"],
            ["img3.png", "Consolidation", "0", "3", "50", "F", "AP"],
        ])
        res = compute_disease_frequencies(csv_path=csv_p, output_json_path=None, output_csv_path=None)
        self.assertEqual(res["disease_counts"]["Pneumothorax"], 2)
        self.assertEqual(res["disease_counts"]["Consolidation"], 1)
        self.assertAlmostEqual(res["disease_prevalences"]["Pneumothorax"], 66.6667, places=3)
        self.assertAlmostEqual(res["disease_prevalences"]["Consolidation"], 33.3333, places=3)

    def test_03_correct_counting_multilabel_records(self):
        """Test 3: Multilabel records contribute to each labeled disease independently."""
        csv_p = self._create_sample_csv([
            ["img1.png", "Cardiomegaly|Effusion|Pneumonia", "0", "1", "60", "M", "PA"],
            ["img2.png", "Effusion|Infiltration", "0", "2", "70", "F", "AP"],
        ])
        res = compute_disease_frequencies(csv_path=csv_p, output_json_path=None, output_csv_path=None)
        self.assertEqual(res["disease_counts"]["Cardiomegaly"], 1)
        self.assertEqual(res["disease_counts"]["Effusion"], 2)
        self.assertEqual(res["disease_counts"]["Pneumonia"], 1)
        self.assertEqual(res["disease_counts"]["Infiltration"], 1)
        self.assertEqual(res["disease_counts"]["Atelectasis"], 0)

    def test_04_each_disease_counted_at_most_once_per_image(self):
        """Test 4: Redundant labels within a single image record are counted at most once."""
        csv_p = self._create_sample_csv([
            ["img1.png", "Effusion|Effusion|Effusion", "0", "1", "55", "M", "PA"],
        ])
        res = compute_disease_frequencies(csv_path=csv_p, output_json_path=None, output_csv_path=None)
        self.assertEqual(res["disease_counts"]["Effusion"], 1)
        self.assertEqual(res["total_valid_records"], 1)

    def test_05_correct_prevalence_denominator(self):
        """Test 5: Prevalence percentage uses exact denominator N = total_valid_records."""
        csv_p = self._create_sample_csv([
            ["img1.png", "Mass", "0", "1", "40", "F", "PA"],
            ["img2.png", "No Finding", "0", "2", "41", "M", "PA"],
            ["img3.png", "No Finding", "0", "3", "42", "F", "AP"],
            ["img4.png", "Mass", "0", "4", "43", "M", "PA"],
        ])
        res = compute_disease_frequencies(csv_path=csv_p, output_json_path=None, output_csv_path=None)
        self.assertEqual(res["denominator"], 4)
        self.assertEqual(res["total_valid_records"], 4)
        self.assertEqual(res["disease_counts"]["Mass"], 2)
        self.assertAlmostEqual(res["disease_prevalences"]["Mass"], 50.0, places=4)

    def test_06_separate_no_finding_handling(self):
        """Test 6: 'No Finding' is reported separately and NOT counted as a 15th disease class."""
        csv_p = self._create_sample_csv([
            ["img1.png", "No Finding", "0", "1", "20", "M", "PA"],
            ["img2.png", "Normal", "0", "2", "21", "F", "PA"],
            ["img3.png", "Atelectasis", "0", "3", "22", "M", "PA"],
        ])
        res = compute_disease_frequencies(csv_path=csv_p, output_json_path=None, output_csv_path=None)
        self.assertEqual(len(res["diseases"]), 14)
        self.assertNotIn("No Finding", res["disease_counts"])
        self.assertEqual(res["no_finding_count"], 2)
        self.assertAlmostEqual(res["no_finding_prevalence"], 66.6667, places=3)
        self.assertEqual(res["disease_counts"]["Atelectasis"], 1)

    def test_07_duplicate_identifiers(self):
        """Test 7: Duplicate image indices are detected and excluded from denominator."""
        csv_p = self._create_sample_csv([
            ["img1.png", "Cardiomegaly", "0", "1", "50", "M", "PA"],
            ["img1.png", "Cardiomegaly", "1", "1", "50", "M", "PA"],  # Duplicate ID
            ["img2.png", "Effusion", "0", "2", "60", "F", "PA"],
        ])
        res = compute_disease_frequencies(csv_path=csv_p, output_json_path=None, output_csv_path=None)
        self.assertEqual(res["total_valid_records"], 2)
        self.assertEqual(res["duplicate_records_count"], 1)
        self.assertEqual(res["disease_counts"]["Cardiomegaly"], 1)

    def test_08_missing_and_malformed_records(self):
        """Test 8: Missing identifiers or malformed rows are counted as excluded, not silently dropped."""
        csv_p = self._create_sample_csv([
            ["", "Effusion", "0", "1", "50", "M", "PA"],  # Empty ID
            ["img1.png", "Effusion", "0", "2", "50", "M", "PA"],
        ])
        res = compute_disease_frequencies(csv_path=csv_p, output_json_path=None, output_csv_path=None)
        self.assertEqual(res["excluded_records_count"], 1)
        self.assertEqual(res["total_valid_records"], 1)

    def test_09_unknown_labels(self):
        """Test 9: Unknown finding labels are audited and reported without crashing."""
        csv_p = self._create_sample_csv([
            ["img1.png", "Pneumonia|UnrecognizedPathologyX", "0", "1", "40", "M", "PA"],
        ])
        res = compute_disease_frequencies(csv_path=csv_p, output_json_path=None, output_csv_path=None)
        self.assertIn("UnrecognizedPathologyX", res["unknown_labels_found"])
        self.assertEqual(res["unknown_labels_count"], 1)
        self.assertEqual(res["disease_counts"]["Pneumonia"], 1)

    def test_10_json_serialization_and_reload(self):
        """Test 10: JSON artifact saves with full schema and reloads identically."""
        json_out = self.test_dir / "disease_freq.json"
        csv_p = self._create_sample_csv([
            ["img1.png", "Hernia", "0", "1", "70", "M", "PA"],
            ["img2.png", "No Finding", "0", "2", "25", "F", "PA"],
        ])
        computed = compute_disease_frequencies(csv_path=csv_p, output_json_path=json_out, output_csv_path=None)
        self.assertTrue(json_out.exists())

        loaded = load_cached_disease_frequency_json(json_path=json_out, force_reload=True)
        self.assertEqual(loaded["total_valid_records"], computed["total_valid_records"])
        self.assertEqual(loaded["disease_counts"]["Hernia"], 1)
        self.assertEqual(loaded["no_finding"]["occurrence_count"], 1)
        self.assertEqual(len(loaded["diseases"]), 14)

    def test_11_no_hardcoded_statistics(self):
        """Test 11: Changing CSV data changes the computed counts and prevalences dynamically."""
        csv_a = self._create_sample_csv([["a.png", "Fibrosis", "0", "1", "30", "M", "PA"]])
        res_a = compute_disease_frequencies(csv_path=csv_a, output_json_path=None, output_csv_path=None)

        csv_b = self.test_dir / "csv_b.csv"
        with open(csv_b, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Index", "Finding Labels", "Follow-up #", "Patient ID", "Patient Age", "Patient Gender", "View Position"])
            writer.writerow(["b1.png", "Fibrosis", "0", "1", "30", "M", "PA"])
            writer.writerow(["b2.png", "Fibrosis", "0", "2", "31", "M", "PA"])
        res_b = compute_disease_frequencies(csv_path=csv_b, output_json_path=None, output_csv_path=None)

        self.assertEqual(res_a["disease_counts"]["Fibrosis"], 1)
        self.assertEqual(res_b["disease_counts"]["Fibrosis"], 2)
        self.assertNotEqual(res_a["total_valid_records"], res_b["total_valid_records"])

    def test_12_missing_csv_fails_clearly(self):
        """Test 12: Missing CSV file raises FileNotFoundError, never generating fake statistics."""
        missing_p = self.test_dir / "non_existent_data_entry.csv"
        with self.assertRaises(FileNotFoundError):
            compute_disease_frequencies(csv_path=missing_p, output_json_path=None, output_csv_path=None)


class TestModule8WeightedSupport(unittest.TestCase):
    """
    Dedicated tests for Metric Two: Similarity-Weighted Retrieved-Case Label Support.
    Covers all 12 specified requirements from Section 8 of Module 8 specification.
    """

    def test_01_correct_cosine_similarity(self):
        """Test 1: Cosine similarity formula produces exact mathematical dot-product alignment."""
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ref = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        sims = compute_cosine_similarity(q, ref)
        self.assertAlmostEqual(sims[0], 100.0, places=3)
        self.assertAlmostEqual(sims[1], 70.71, places=1)  # 1/sqrt(2) * 100
        self.assertAlmostEqual(sims[2], 0.0, places=3)

    def test_02_correct_top_k_ranking(self):
        """Test 2: References are strictly ranked in non-increasing similarity score order."""
        q = np.array([1.0, 0.0], dtype=np.float32)
        ref = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]], dtype=np.float32)
        names = ["c1.png", "c2.png", "c3.png"]
        ranked = rank_similar_images(q, names, ref, top_k=3)
        self.assertEqual(ranked[0]["filename"], "c2.png")
        self.assertGreaterEqual(ranked[0]["similarity_score"], ranked[1]["similarity_score"])
        self.assertGreaterEqual(ranked[1]["similarity_score"], ranked[2]["similarity_score"])

    def test_03_correct_weighted_support_for_known_labels(self):
        """
        Test 3: Analytically verified manual calculation for weighted support:
        3 references with scores [80.0, 60.0, 20.0], total weight W = 160.0.
        Ref 1: {"Pneumonia", "Effusion"}
        Ref 2: {"Effusion"}
        Ref 3: {"No Finding"}
        Effusion weight sum = 80 + 60 = 140 -> V_d = 140/160 * 100 = 87.5%
        Pneumonia weight sum = 80 -> V_d = 80/160 * 100 = 50.0%
        """
        matches = [
            {"rank": 1, "filename": "r1.png", "similarity_score": 80.0, "metadata": {"findings": ["Pneumonia", "Effusion"]}},
            {"rank": 2, "filename": "r2.png", "similarity_score": 60.0, "metadata": {"findings": ["Effusion"]}},
            {"rank": 3, "filename": "r3.png", "similarity_score": 20.0, "metadata": {"findings": ["No Finding"]}},
        ]
        support = compute_similarity_weighted_support(matches=matches)
        sup_dict = {s["disease"]: s for s in support}

        self.assertTrue(sup_dict["Effusion"]["is_available"])
        self.assertAlmostEqual(sup_dict["Effusion"]["weighted_support"], 87.5, places=2)
        self.assertEqual(sup_dict["Effusion"]["retrieved_count"], 2)
        self.assertAlmostEqual(sup_dict["Effusion"]["retrieval_rate"], 66.67, places=2)

        self.assertTrue(sup_dict["Pneumonia"]["is_available"])
        self.assertAlmostEqual(sup_dict["Pneumonia"]["weighted_support"], 50.0, places=2)
        self.assertEqual(sup_dict["Pneumonia"]["retrieved_count"], 1)
        self.assertAlmostEqual(sup_dict["Pneumonia"]["retrieval_rate"], 33.33, places=2)

        self.assertEqual(sup_dict["Cardiomegaly"]["retrieved_count"], 0)
        self.assertAlmostEqual(sup_dict["Cardiomegaly"]["weighted_support"], 0.0, places=2)

    def test_04_correct_raw_label_counts(self):
        """Test 4: Raw matching case counts K_d match ground-truth label presence exactly."""
        matches = [
            {"rank": 1, "filename": "r1.png", "similarity_score": 90.0, "metadata": {"findings": ["Edema"]}},
            {"rank": 2, "filename": "r2.png", "similarity_score": 85.0, "metadata": {"findings": ["Edema", "Cardiomegaly"]}},
            {"rank": 3, "filename": "r3.png", "similarity_score": 75.0, "metadata": {"findings": ["Cardiomegaly"]}},
            {"rank": 4, "filename": "r4.png", "similarity_score": 70.0, "metadata": {"findings": ["Edema"]}},
            {"rank": 5, "filename": "r5.png", "similarity_score": 60.0, "metadata": {"findings": ["Mass"]}},
        ]
        support = compute_similarity_weighted_support(matches=matches)
        sup_dict = {s["disease"]: s for s in support}
        self.assertEqual(sup_dict["Edema"]["retrieved_count"], 3)
        self.assertEqual(sup_dict["Cardiomegaly"]["retrieved_count"], 2)
        self.assertEqual(sup_dict["Mass"]["retrieved_count"], 1)
        self.assertEqual(sup_dict["Fibrosis"]["retrieved_count"], 0)

    def test_05_correct_raw_percentages(self):
        """Test 5: Raw percentages R_d = (K_d / K) * 100 are strictly calculated against K."""
        matches = [
            {"rank": 1, "filename": "r1.png", "similarity_score": 80.0, "metadata": {"findings": ["Atelectasis"]}},
            {"rank": 2, "filename": "r2.png", "similarity_score": 70.0, "metadata": {"findings": ["Atelectasis"]}},
            {"rank": 3, "filename": "r3.png", "similarity_score": 60.0, "metadata": {"findings": ["Infiltration"]}},
            {"rank": 4, "filename": "r4.png", "similarity_score": 50.0, "metadata": {"findings": ["No Finding"]}},
        ]
        support = compute_similarity_weighted_support(matches=matches)
        sup_dict = {s["disease"]: s for s in support}
        self.assertAlmostEqual(sup_dict["Atelectasis"]["retrieval_rate"], 50.0, places=2)  # 2/4 * 100
        self.assertAlmostEqual(sup_dict["Infiltration"]["retrieval_rate"], 25.0, places=2)  # 1/4 * 100
        self.assertAlmostEqual(sup_dict["Hernia"]["retrieval_rate"], 0.0, places=2)

    def test_06_multiple_findings_per_reference(self):
        """Test 6: Reference case with multiple findings contributes independently to each finding."""
        matches = [
            {"rank": 1, "filename": "r1.png", "similarity_score": 80.0, "metadata": {"findings": ["Pneumothorax", "Emphysema"]}},
        ]
        support = compute_similarity_weighted_support(matches=matches)
        sup_dict = {s["disease"]: s for s in support}
        self.assertAlmostEqual(sup_dict["Pneumothorax"]["weighted_support"], 100.0, places=2)
        self.assertAlmostEqual(sup_dict["Emphysema"]["weighted_support"], 100.0, places=2)
        self.assertAlmostEqual(sup_dict["Pneumonia"]["weighted_support"], 0.0, places=2)

    def test_07_empty_reference_collection(self):
        """Test 7: Empty retrieval results return unavailable status with explicit reason."""
        support = compute_similarity_weighted_support(matches=[])
        self.assertEqual(len(support), 14)
        for s in support:
            self.assertFalse(s["is_available"])
            self.assertEqual(s["retrieved_count"], 0)
            self.assertEqual(s["retrieved_denominator"], 0)
            self.assertEqual(s["reason"], "No reference cases retrieved")

    def test_08_zero_total_similarity_weights(self):
        """Test 8: Zero total similarity weight handles division-by-zero cleanly without crashing."""
        matches = [
            {"rank": 1, "filename": "r1.png", "similarity_score": 0.0, "metadata": {"findings": ["Pneumonia"]}},
            {"rank": 2, "filename": "r2.png", "similarity_score": 0.0, "metadata": {"findings": ["Effusion"]}},
        ]
        support = compute_similarity_weighted_support(matches=matches)
        self.assertEqual(len(support), 14)
        for s in support:
            self.assertFalse(s["is_available"])
            self.assertEqual(s["reason"], "Zero total similarity weight across retrieved cases")
            self.assertEqual(s["retrieved_denominator"], 2)

    def test_09_missing_metadata(self):
        """Test 9: Reference with missing metadata does not crash support calculation."""
        matches = [
            {"rank": 1, "filename": "r1.png", "similarity_score": 75.0, "metadata": None},
            {"rank": 2, "filename": "r2.png", "similarity_score": 65.0, "metadata": {}},
        ]
        support = compute_similarity_weighted_support(matches=matches)
        sup_dict = {s["disease"]: s for s in support}
        self.assertTrue(sup_dict["Pneumonia"]["is_available"])
        self.assertEqual(sup_dict["Pneumonia"]["retrieved_count"], 0)
        self.assertAlmostEqual(sup_dict["Pneumonia"]["weighted_support"], 0.0, places=2)

    def test_10_synthetic_records_do_not_masquerade_as_nih_references(self):
        """Test 10: Synthetic demonstration cases are flagged with is_synthetic=True and explicit provenance."""
        q = np.zeros(1024, dtype=np.float32)
        q[0] = 1.0
        ref_matrix = np.array([q, q])
        names = ["xray001.jpg", "00000002_000.png"]
        meta = [
            {"image_index": "DEMO-xray001.jpg", "dataset_source": "Synthetic Benchmark Demonstration", "findings": ["Cardiomegaly"]},
            {"image_index": "00000002_000.png", "dataset_source": "NIH ChestX-ray14", "findings": ["No Finding"]},
        ]
        ranked = rank_similar_images(q, names, ref_matrix, top_k=2, metadata=meta)
        m_demo = next(m for m in ranked if m["filename"] == "xray001.jpg")
        m_nih = next(m for m in ranked if m["filename"] == "00000002_000.png")

        self.assertTrue(m_demo["is_synthetic"])
        self.assertEqual(m_demo["provenance"], "Synthetic Benchmark Demonstration")
        self.assertFalse(m_nih["is_synthetic"])
        self.assertEqual(m_nih["provenance"], "NIH ChestX-ray14")

    def test_11_query_evaluation_image_exclusion(self):
        """Test 11: Query or evaluation images can be excluded from candidate reference collection."""
        q = np.zeros(1024, dtype=np.float32)
        q[0] = 1.0
        ref_matrix = np.array([q, q * 0.9, q * 0.8])
        names = ["query_case.png", "other_case1.png", "other_case2.png"]
        ranked = rank_similar_images(q, names, ref_matrix, top_k=3, exclude_filenames="query_case.png")
        result_names = [m["filename"] for m in ranked]
        self.assertNotIn("query_case.png", result_names)
        self.assertEqual(len(ranked), 2)

    def test_12_no_fabricated_support_values(self):
        """Test 12: Labels not present in retrieved reference metadata never receive positive support."""
        matches = [
            {"rank": 1, "filename": "r1.png", "similarity_score": 95.0, "metadata": {"findings": ["Cardiomegaly"]}},
        ]
        support = compute_similarity_weighted_support(matches=matches)
        sup_dict = {s["disease"]: s for s in support}
        for d in CANONICAL_DISEASES:
            if d != "Cardiomegaly":
                self.assertEqual(sup_dict[d]["retrieved_count"], 0)
                self.assertAlmostEqual(sup_dict[d]["weighted_support"], 0.0, places=2)
                self.assertAlmostEqual(sup_dict[d]["retrieval_rate"], 0.0, places=2)


class TestModule8ApiAndIntegration(unittest.TestCase):
    """
    Dedicated integration tests for API response models, denominator consistency,
    and distinct presentations for the two Module 8 metrics.
    """

    def test_01_api_returns_both_metrics(self):
        """Test 1: API response includes both global_dataset_statistics and weighted_label_support."""
        v = np.random.randn(1024).astype(np.float32)
        v = v / np.linalg.norm(v)
        ref = np.array([v])
        names = ["00000002_000.png"]
        meta = [{"image_index": "00000002_000.png", "findings": ["Atelectasis"], "dataset_source": "NIH ChestX-ray14"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            f_path = Path(tmpdir) / "test_store.npz"
            np.savez_compressed(f_path, filenames=names, embeddings=ref, metadata=meta)
            store = EmbeddingStore(f_path)
            analyzer = SimilarityAnalyzer(extractor=None, store=store)

            res = analyzer.find_similar_by_vector(v, top_k=1)
            self.assertIn("global_dataset_statistics", res)
            self.assertIn("weighted_label_support", res)
            self.assertIn("no_finding_statistics", res)
            self.assertIn("explanation_text", res)
            self.assertEqual(len(res["global_dataset_statistics"]), 14)
            self.assertEqual(len(res["weighted_label_support"]), 14)

    def test_02_api_and_frontend_use_same_denominator(self):
        """Test 2: Global statistics report denominator = 112,120 across all diseases and No Finding."""
        analyzer = SimilarityAnalyzer()
        res = analyzer.find_similar_by_vector(np.random.randn(1024).astype(np.float32), top_k=1)
        for stat in res["global_dataset_statistics"]:
            self.assertEqual(stat["denominator"], 112120)
        self.assertEqual(res["no_finding_statistics"]["denominator"], 112120)

    def test_03_normalizer_preserves_valid_zero_values(self):
        """Test 3: Zero weighted support and zero counts are preserved as valid 0.0 numbers."""
        matches = [
            {"rank": 1, "filename": "r1.png", "similarity_score": 80.0, "metadata": {"findings": ["Hernia"]}},
        ]
        support = compute_similarity_weighted_support(matches=matches)
        atelectasis = next(s for s in support if s["disease"] == "Atelectasis")
        self.assertTrue(atelectasis["is_available"])
        self.assertEqual(atelectasis["weighted_support"], 0.0)
        self.assertEqual(atelectasis["retrieved_count"], 0)
        self.assertEqual(atelectasis["retrieval_rate"], 0.0)

    def test_04_missing_metrics_display_explicit_unavailable_state(self):
        """Test 4: When retrieval is empty, is_available=False with clear reason is set."""
        support = compute_similarity_weighted_support(matches=[])
        self.assertFalse(support[0]["is_available"])
        self.assertIsNotNone(support[0]["reason"])

    def test_05_two_metrics_have_distinct_labels_and_explanations(self):
        """Test 5: Prevalence and Weighted Support are distinct concepts and have explicit explanation."""
        analyzer = SimilarityAnalyzer()
        res = analyzer.find_similar_by_vector(np.random.randn(1024).astype(np.float32), top_k=1)
        self.assertIn("NIH Dataset Prevalence measures label frequency", res["explanation_text"])
        self.assertIn("Retrieved-Case Label Support measures label representation", res["explanation_text"])
        self.assertIn("Neither is a calibrated disease probability", res["explanation_text"])

    def test_06_existing_report_behavior_remains_intact(self):
        """Test 6: SimilarityAnalyzer retains top_match, top_k, total_references, and medical disclaimer."""
        analyzer = SimilarityAnalyzer()
        res = analyzer.find_similar_by_vector(np.random.randn(1024).astype(np.float32), top_k=3)
        self.assertIn("top_match", res)
        self.assertEqual(res["top_k"], 3)
        self.assertIn("disclaimer", res)
        self.assertIn("Case similarity provides supporting context", res["disclaimer"])

    def test_07_existing_similarity_tests_continue_to_pass(self):
        """Test 7: Top match has rank, similarity_score, and valid image_url."""
        analyzer = SimilarityAnalyzer()
        res = analyzer.find_similar_by_vector(np.random.randn(1024).astype(np.float32), top_k=1)
        if res["top_match"]:
            self.assertIn("similarity_score", res["top_match"])
            self.assertIn("image_url", res["top_match"])
            self.assertIn("rank", res["top_match"])


class TestPredictionDrivenReferenceEvidence(unittest.TestCase):
    """
    Comprehensive tests for Prediction-Driven, Disease-Specific Reference Evidence.
    Verifies:
    - Mathematical calculation of K_d, R_d, V_d, P_d from retrieval results
    - Single vs multiple predicted findings
    - Explicit zero-matches representation without silent disease substitution
    - Authentic-only filtering (filtering out synthetic demonstration images)
    - Self-retrieval exclusion by filename and identical embedding
    - End-to-end integration via SimilarityAnalyzer
    """

    def setUp(self):
        self.matches = [
            {
                "rank": 1,
                "filename": "00000001_000.png",
                "similarity_score": 80.0,
                "findings": ["Cardiomegaly", "Effusion"],
                "metadata": {
                    "patient_age": 55,
                    "patient_gender": "M",
                    "view_position": "PA",
                    "finding_labels": "Cardiomegaly|Effusion",
                    "dataset_source": "NIH ChestX-ray14",
                },
            },
            {
                "rank": 2,
                "filename": "00000002_000.png",
                "similarity_score": 60.0,
                "findings": ["Cardiomegaly"],
                "metadata": {
                    "patient_age": 62,
                    "patient_gender": "F",
                    "view_position": "AP",
                    "finding_labels": "Cardiomegaly",
                    "dataset_source": "NIH ChestX-ray14",
                },
            },
            {
                "rank": 3,
                "filename": "00000003_000.png",
                "similarity_score": 20.0,
                "findings": ["Effusion"],
                "metadata": {
                    "patient_age": 45,
                    "patient_gender": "M",
                    "view_position": "PA",
                    "finding_labels": "Effusion",
                    "dataset_source": "NIH ChestX-ray14",
                },
            },
        ]
        self.global_prevalences = {
            "Cardiomegaly": 2.48,
            "Effusion": 11.88,
            "Pneumonia": 1.28,
            "No Finding": 53.84,
        }

    def test_single_finding_hand_calculated_metrics(self):
        """Cardiomegaly matches rank 1 (80.0) and rank 2 (60.0). Total weight = 160.0."""
        evidence = compute_predicted_findings_evidence(
            matches=self.matches,
            predicted_findings=["Cardiomegaly"],
            global_prevalences=self.global_prevalences,
        )
        self.assertEqual(len(evidence), 1)
        rec = evidence[0]
        self.assertEqual(rec["predicted_finding"], "Cardiomegaly")
        self.assertEqual(rec["matching_cases"], 2)
        self.assertEqual(rec["retrieved_cases"], 3)
        self.assertEqual(rec["matching_rate"], 66.7)  # 2 / 3 * 100
        self.assertEqual(rec["weighted_support"], 87.5)  # (80 + 60) / 160 * 100 = 140 / 160 * 100 = 87.5
        self.assertEqual(rec["dataset_prevalence"], 2.48)
        self.assertTrue(rec["is_matched"])
        self.assertEqual(len(rec["matching_references"]), 2)
        self.assertEqual(rec["matching_references"][0]["filename"], "00000001_000.png")
        self.assertEqual(rec["matching_references"][0]["demographics"], "55y · M · PA")
        self.assertEqual(rec["matching_references"][1]["filename"], "00000002_000.png")
        self.assertIn("corroborate predicted Cardiomegaly", rec["explanation"])

    def test_multiple_predicted_findings_independent_evidence(self):
        """Cardiomegaly and Effusion each get an independent evidence record."""
        evidence = compute_predicted_findings_evidence(
            matches=self.matches,
            predicted_findings=["Cardiomegaly", "Effusion"],
            global_prevalences=self.global_prevalences,
        )
        self.assertEqual(len(evidence), 2)
        cardio = next(e for e in evidence if e["predicted_finding"] == "Cardiomegaly")
        effusion = next(e for e in evidence if e["predicted_finding"] == "Effusion")

        self.assertEqual(cardio["matching_cases"], 2)
        self.assertEqual(cardio["weighted_support"], 87.5)

        # Effusion matches rank 1 (80.0) and rank 3 (20.0): (80 + 20) / 160 * 100 = 62.5%
        self.assertEqual(effusion["matching_cases"], 2)
        self.assertEqual(effusion["retrieved_cases"], 3)
        self.assertEqual(effusion["matching_rate"], 66.7)
        self.assertEqual(effusion["weighted_support"], 62.5)
        self.assertEqual(effusion["dataset_prevalence"], 11.88)

    def test_zero_matching_cases_does_not_fabricate_or_substitute(self):
        """Zero matches for Pneumonia must return explicit 0 matching cases without fabrication."""
        evidence = compute_predicted_findings_evidence(
            matches=self.matches,
            predicted_findings=["Pneumonia"],
            global_prevalences=self.global_prevalences,
        )
        self.assertEqual(len(evidence), 1)
        rec = evidence[0]
        self.assertEqual(rec["predicted_finding"], "Pneumonia")
        self.assertEqual(rec["matching_cases"], 0)
        self.assertEqual(rec["retrieved_cases"], 3)
        self.assertEqual(rec["matching_rate"], 0.0)
        self.assertEqual(rec["weighted_support"], 0.0)
        self.assertEqual(rec["dataset_prevalence"], 1.28)
        self.assertFalse(rec["is_matched"])
        self.assertEqual(len(rec["matching_references"]), 0)
        self.assertIn("0 of 3 retrieved reference cases", rec["explanation"])
        self.assertIn("No reference support found in top matches", rec["explanation"])

    def test_normal_no_finding_evaluation(self):
        """When predicting No Finding, matches with No Finding or empty findings match."""
        normal_matches = [
            {
                "rank": 1,
                "filename": "normal1.png",
                "similarity_score": 90.0,
                "findings": ["No Finding"],
                "metadata": {"finding_labels": "No Finding"},
            },
            {
                "rank": 2,
                "filename": "diseased1.png",
                "similarity_score": 10.0,
                "findings": ["Pneumonia"],
                "metadata": {"finding_labels": "Pneumonia"},
            },
        ]
        evidence = compute_predicted_findings_evidence(
            matches=normal_matches,
            predicted_findings=["No Finding"],
            global_prevalences=self.global_prevalences,
        )
        self.assertEqual(len(evidence), 1)
        rec = evidence[0]
        self.assertEqual(rec["predicted_finding"], "No Finding")
        self.assertEqual(rec["matching_cases"], 1)
        self.assertEqual(rec["matching_rate"], 50.0)
        self.assertEqual(rec["weighted_support"], 90.0)
        self.assertTrue(rec["is_matched"])

    def test_empty_matches_handles_gracefully(self):
        """Empty matches list returns valid 0.0 metrics without ZeroDivisionError."""
        evidence = compute_predicted_findings_evidence(
            matches=[],
            predicted_findings=["Cardiomegaly"],
            global_prevalences=self.global_prevalences,
        )
        self.assertEqual(len(evidence), 1)
        rec = evidence[0]
        self.assertEqual(rec["matching_cases"], 0)
        self.assertEqual(rec["retrieved_cases"], 0)
        self.assertEqual(rec["matching_rate"], 0.0)
        self.assertEqual(rec["weighted_support"], 0.0)
        self.assertFalse(rec["is_matched"])

    def test_authentic_only_filtering_in_ranking(self):
        """rank_similar_images filters out synthetic demonstration cases when authentic_only=True."""
        q = np.array([1.0, 0.0], dtype=np.float32)
        ref_embeddings = np.array([
            [0.9, 0.1],
            [0.85, 0.15],
            [0.8, 0.2],
        ], dtype=np.float32)
        filenames = ["demo_xray001.png", "00000001_000.png", "00000002_000.png"]
        metadata_list = [
            {"is_synthetic": True, "dataset_source": "synthetic-demo"},
            {"is_synthetic": False, "dataset_source": "NIH ChestX-ray14"},
            {"is_synthetic": False, "dataset_source": "NIH ChestX-ray14"},
        ]
        # authentic_only=False includes synthetic demo case
        all_matches = rank_similar_images(
            q, filenames, ref_embeddings, top_k=3, metadata=metadata_list, authentic_only=False
        )
        self.assertEqual(len(all_matches), 3)
        self.assertEqual(all_matches[0]["filename"], "demo_xray001.png")

        # authentic_only=True filters out synthetic demo case
        auth_matches = rank_similar_images(
            q, filenames, ref_embeddings, top_k=3, metadata=metadata_list, authentic_only=True
        )
        self.assertEqual(len(auth_matches), 2)
        self.assertEqual(auth_matches[0]["filename"], "00000001_000.png")
        self.assertEqual(auth_matches[1]["filename"], "00000002_000.png")

    def test_self_retrieval_exclusion(self):
        """Excludes query image by filename and by identical embedding (>=99.99%)."""
        q = np.array([1.0, 0.0], dtype=np.float32)
        ref_embeddings = np.array([
            [1.0, 0.0],
            [0.75, 0.25],
            [0.70, 0.30],
        ], dtype=np.float32)
        filenames = ["query_case.png", "00000001_000.png", "00000002_000.png"]

        # Filename exclusion
        matches_ex_name = rank_similar_images(
            q, filenames, ref_embeddings, top_k=2, exclude_filenames=["query_case.png"]
        )
        self.assertEqual(len(matches_ex_name), 2)
        self.assertNotIn("query_case.png", [m["filename"] for m in matches_ex_name])

        # Identical embedding exclusion (>=99.99%)
        matches_ex_ident = rank_similar_images(
            q, filenames, ref_embeddings, top_k=2, exclude_identical_embeddings=True
        )
        self.assertEqual(len(matches_ex_ident), 2)
        self.assertNotIn("query_case.png", [m["filename"] for m in matches_ex_ident])

    def test_similarity_analyzer_integration_with_predicted_findings(self):
        """SimilarityAnalyzer.find_similar_by_vector generates predicted_findings_evidence."""
        analyzer = SimilarityAnalyzer()
        v = np.random.randn(1024).astype(np.float32)
        v = v / np.linalg.norm(v)

        res = analyzer.find_similar_by_vector(
            v,
            top_k=3,
            predicted_findings=["Cardiomegaly", "Infiltration"],
            authentic_only=True,
            exclude_identical_embeddings=True,
        )
        self.assertIn("predicted_findings_evidence", res)
        self.assertIn("predicted_evidence", res)
        evidence = res["predicted_findings_evidence"]
        self.assertEqual(len(evidence), 2)
        findings_found = {e["predicted_finding"] for e in evidence}
        self.assertEqual(findings_found, {"Cardiomegaly", "Infiltration"})
        for e in evidence:
            self.assertIn("matching_cases", e)
            self.assertIn("retrieved_cases", e)
            self.assertIn("matching_rate", e)
            self.assertIn("weighted_support", e)
            self.assertIn("is_matched", e)
            self.assertIn("explanation", e)


class TestEmbeddingOnlyDeployment(unittest.TestCase):
    """
    Verification tests for MedTrust Module 8 Embedding-Only Deployment Architecture.
    Validates that similarity retrieval and finding-specific evidence computation
    function with 100% fidelity without requiring original reference X-ray image files.
    """

    def test_embedding_store_integrity_validation(self):
        """Verify that precomputed reference embeddings strictly meet all integrity invariants."""
        from src.similarity.embedding_store import embedding_store

        report = embedding_store.verify_integrity(verify_hashes=True)
        self.assertTrue(report["valid"], f"Store integrity invalid: {report.get('errors')}")
        self.assertEqual(report["count"], 2500)
        self.assertEqual(report["dimension"], 1024)
        self.assertTrue(report["is_normalized"])
        self.assertTrue(report["finite"])
        self.assertEqual(report["duplicate_count"], 0)
        self.assertEqual(report["metadata_count"], 2500)
        self.assertFalse(report["requires_original_images"])
        self.assertTrue(report["hashes_verified"])

    def test_metadata_json_direct_loading_without_csv(self):
        """Verify that ReferenceMetadataLoader operates solely from reference_metadata.json when CSVs are absent."""
        from src.similarity.metadata_loader import ReferenceMetadataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_csv = Path(tmpdir) / "nonexistent_ref.csv"
            dummy_nih = Path(tmpdir) / "nonexistent_nih.csv"
            loader = ReferenceMetadataLoader(
                csv_path=dummy_csv,
                nih_catalog_path=dummy_nih,
            )
            self.assertFalse(loader.csv_path.exists())
            self.assertFalse(loader.nih_catalog_path.exists())

            index = loader.load_index()
            self.assertEqual(len(index), 2500)
            sample_fn = "00000003_000.png"
            self.assertIn(sample_fn, index)
            self.assertIn("Hernia", index[sample_fn]["findings"])
            self.assertEqual(index[sample_fn]["patient_age"], 81)

    def test_embedding_store_json_fallback_when_npz_lacks_metadata(self):
        """Verify EmbeddingStore loads metadata from JSON when an archive lacks embedded metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.similarity.embedding_store import EmbeddingStore

            npz_path = Path(tmpdir) / "strip_metadata.npz"
            fnames = [f"case_{i:04d}.png" for i in range(5)]
            embeds = np.random.randn(5, 1024).astype(np.float32)
            embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
            np.savez_compressed(npz_path, filenames=fnames, embeddings=embeds)

            json_path = Path(tmpdir) / "meta.json"
            meta_records = [
                {
                    "filename": fn,
                    "image_index": fn,
                    "findings": ["Cardiomegaly" if i % 2 == 0 else "No Finding"],
                    "finding_labels": "Cardiomegaly" if i % 2 == 0 else "No Finding",
                    "patient_age": 50 + i,
                    "patient_gender": "M",
                    "view_position": "PA",
                    "dataset_source": "NIH ChestX-ray14",
                    "annotation_type": "Dataset Annotation",
                }
                for i, fn in enumerate(fnames)
            ]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta_records, f)

            store = EmbeddingStore(
                embeddings_path=npz_path,
                metadata_path=json_path,
            )
            loaded_names, loaded_embeds = store.load()
            self.assertEqual(len(loaded_names), 5)
            self.assertEqual(loaded_embeds.shape, (5, 1024))
            self.assertIsNotNone(store.get_metadata())
            self.assertEqual(len(store.get_metadata()), 5)
            self.assertEqual(store.get_metadata()[0]["findings"], ["Cardiomegaly"])

    def test_retrieval_without_reference_image_files(self):
        """
        Verify similarity retrieval and finding-specific evidence computation execute
        without any access to reference image files on disk.
        """
        from src.similarity.similarity_analyzer import SimilarityAnalyzer
        from unittest.mock import patch

        analyzer = SimilarityAnalyzer()
        query_vector = np.random.randn(1024).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)

        # Mock image reading functions to assert zero calls to open images
        with patch("PIL.Image.open") as mock_pil_open:
            result = analyzer.find_similar_by_vector(
                query_vector=query_vector,
                top_k=5,
                predicted_findings=["Effusion", "Pneumothorax"],
                authentic_only=True,
            )
            mock_pil_open.assert_not_called()

        self.assertIn("matches", result)
        self.assertEqual(len(result["matches"]), 5)
        for m in result["matches"]:
            self.assertIn("rank", m)
            self.assertIn("filename", m)
            self.assertIn("similarity_score", m)
            self.assertIn("findings", m)
            self.assertIn("metadata", m)
            self.assertGreaterEqual(m["similarity_score"], 0.0)
            self.assertLessEqual(m["similarity_score"], 100.0)

        self.assertIn("predicted_findings_evidence", result)
        evidence = result["predicted_findings_evidence"]
        self.assertEqual(len(evidence), 2)
        eff_ev = next(e for e in evidence if e["predicted_finding"] == "Effusion")
        self.assertEqual(eff_ev["retrieved_cases"], 5)
        self.assertIn("matching_cases", eff_ev)
        self.assertIn("matching_rate", eff_ev)
        self.assertIn("weighted_support", eff_ev)
        self.assertIn("dataset_prevalence", eff_ev)

    def test_end_to_end_engine_similarity_without_reference_images(self):
        """
        Verify MedTrustBackendEngine run_similarity executes successfully
        and produces complete structured similarity evidence without reading reference image files.
        """
        from backend.server import get_engine
        from unittest.mock import patch

        engine = get_engine()
        self.assertIsNotNone(engine)

        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Ensure that during similarity retrieval, no reference image file is accessed
        original_open = open

        def guarded_open(file, *args, **kwargs):
            file_str = str(file).lower()
            if "reference_images" in file_str and not (file_str.endswith(".csv") or file_str.endswith(".json")):
                raise AssertionError(f"Attempted to open reference image file on disk: {file}")
            return original_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=guarded_open):
            sim_result = engine.run_similarity(
                image=dummy_img,
                top_k=5,
                query_findings=["Atelectasis"],
                primary_finding="Atelectasis",
            )

        self.assertIsNotNone(sim_result)
        self.assertIn("top_match", sim_result)
        self.assertIn("matches", sim_result)
        self.assertEqual(len(sim_result["matches"]), 5)
        self.assertIn("predicted_findings_evidence", sim_result)
        self.assertEqual(len(sim_result["predicted_findings_evidence"]), 1)
        self.assertEqual(sim_result["predicted_findings_evidence"][0]["predicted_finding"], "Atelectasis")


if __name__ == "__main__":
    unittest.main()





