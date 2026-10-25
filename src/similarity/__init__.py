"""
MedTrust V1 — Module 8: Similarity Analysis Package (src/similarity/__init__.py)

Exports:
- DISCLAIMER_TEXT
- compute_cosine_similarity
- rank_similar_images
- EmbeddingStore
- DenseNetSimilarityExtractor
- SimilarityAnalyzer
"""

from src.similarity.disease_frequency import (
    CANONICAL_DISEASES,
    NO_FINDING_LABEL,
    compute_disease_frequencies,
    load_cached_disease_frequencies,
)
from src.similarity.similarity_calculator import (
    DISCLAIMER_TEXT,
    compute_cosine_similarity,
    compute_predicted_findings_evidence,
    compute_similarity_weighted_support,
    evaluate_diagnostic_concordance,
    rank_similar_images,
)
from src.similarity.embedding_store import (
    DEFAULT_EMBEDDINGS_PATH,
    EmbeddingStore,
    embedding_store,
)
from src.similarity.metadata_loader import (
    ReferenceMetadataLoader,
    metadata_loader,
    parse_finding_labels,
)
from src.similarity.feature_extractor import DenseNetSimilarityExtractor
from src.similarity.similarity_analyzer import SimilarityAnalyzer

__all__ = [
    "CANONICAL_DISEASES",
    "NO_FINDING_LABEL",
    "compute_disease_frequencies",
    "load_cached_disease_frequencies",
    "DISCLAIMER_TEXT",
    "compute_cosine_similarity",
    "compute_predicted_findings_evidence",
    "compute_similarity_weighted_support",
    "evaluate_diagnostic_concordance",
    "rank_similar_images",
    "DEFAULT_EMBEDDINGS_PATH",
    "EmbeddingStore",
    "embedding_store",
    "ReferenceMetadataLoader",
    "metadata_loader",
    "parse_finding_labels",
    "DenseNetSimilarityExtractor",
    "SimilarityAnalyzer",
]


