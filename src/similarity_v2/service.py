"""
MedTrust V2 — Similarity Subsystem Service
(src/similarity_v2/service.py)

Provides singleton service instances and health monitoring for Similarity Analyzer V2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.similarity_v2.embedding_index import PersistentEmbeddingIndex
from src.similarity_v2.evidence_engine import EvidenceComparisonEngine
from src.similarity_v2.retrieval_engine import SimilarityRetrievalEngineV2

_INDEX_INSTANCE: Optional[PersistentEmbeddingIndex] = None
_ENGINE_INSTANCE: Optional[SimilarityRetrievalEngineV2] = None


def get_embedding_index_v2() -> PersistentEmbeddingIndex:
    """Get or initialize singleton PersistentEmbeddingIndex."""
    global _INDEX_INSTANCE
    if _INDEX_INSTANCE is None:
        _INDEX_INSTANCE = PersistentEmbeddingIndex()
    return _INDEX_INSTANCE


def get_similarity_engine_v2() -> SimilarityRetrievalEngineV2:
    """Get or initialize singleton SimilarityRetrievalEngineV2."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        idx = get_embedding_index_v2()
        ev_eng = EvidenceComparisonEngine()
        _ENGINE_INSTANCE = SimilarityRetrievalEngineV2(index=idx, evidence_engine=ev_eng)
    return _ENGINE_INSTANCE


def get_reference_case_report(reference_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve structured reference report for a given case ID."""
    engine = get_similarity_engine_v2()
    return engine.get_reference_report(reference_id)


def get_v2_index_health() -> Dict[str, Any]:
    """Return health and readiness report for the V2 reference intelligence library."""
    idx = get_embedding_index_v2()
    integrity = idx.verify_integrity(verify_hashes=True)

    reports_dir = Path(idx.index_path).parent / "reference_reports"
    report_count = len(list(reports_dir.glob("*.json"))) if reports_dir.exists() else 0

    return {
        "status": "HEALTHY" if integrity.get("valid") else "UNHEALTHY",
        "version": "2.0.0",
        "indexed_cases": integrity.get("count", 0),
        "embedding_dimension": integrity.get("dimension", 1024),
        "is_unit_normalized": integrity.get("is_normalized", False),
        "reports_available": report_count,
        "hashes_verified": integrity.get("hashes_verified", False),
        "requires_original_images": False,
        "errors": integrity.get("errors", []),
    }

