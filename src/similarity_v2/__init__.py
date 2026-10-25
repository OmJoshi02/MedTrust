"""
MedTrust V2 — Production Case Similarity & Intelligence System
(src/similarity_v2/__init__.py)

Module 8 Replacement.
"""

from src.similarity_v2.report_schema import (
    CANONICAL_DISEASES,
    SCHEMA_VERSION,
    create_reference_report,
    validate_reference_report,
)
from src.similarity_v2.embedding_index import PersistentEmbeddingIndex
from src.similarity_v2.evidence_engine import (
    EvidenceComparisonEngine,
    STANDARD_DISCLAIMER,
    METHODOLOGICAL_EXPLANATION,
)
from src.similarity_v2.retrieval_engine import SimilarityRetrievalEngineV2
from src.similarity_v2.builder import ReferenceLibraryBuilder
from src.similarity_v2.service import (
    get_embedding_index_v2,
    get_similarity_engine_v2,
    get_reference_case_report,
    get_v2_index_health,
)

__all__ = [
    "CANONICAL_DISEASES",
    "SCHEMA_VERSION",
    "create_reference_report",
    "validate_reference_report",
    "PersistentEmbeddingIndex",
    "EvidenceComparisonEngine",
    "STANDARD_DISCLAIMER",
    "METHODOLOGICAL_EXPLANATION",
    "SimilarityRetrievalEngineV2",
    "ReferenceLibraryBuilder",
    "get_embedding_index_v2",
    "get_similarity_engine_v2",
    "get_reference_case_report",
    "get_v2_index_health",
]

