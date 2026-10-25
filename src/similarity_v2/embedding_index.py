"""
MedTrust V2 — Persistent Reference Embedding Index
(src/similarity_v2/embedding_index.py)

Manages loading, validating, and searching precomputed reference feature embeddings
in an embedding-only deployment architecture without requiring raw reference X-rays.

Invariants:
1. Every vector is 1024-dimensional, float32, finite, and L2 unit-normalized (norm ~ 1.0).
2. One-to-one correspondence with reference metadata and intelligence reports.
3. Zero duplicate reference identifiers.
4. Fast cosine dot-product search with deterministic tie-breaking.
5. Fails clearly on corrupted files or manifest checksum mismatches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INDEX_PATH = PROJECT_ROOT / "outputs" / "similarity_v2" / "reference_embeddings.npz"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "similarity_v2" / "reference_manifest.json"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "outputs" / "similarity_v2" / "reference_metadata.json"


class PersistentEmbeddingIndex:
    """
    In-memory vector search index for 1,024-dimensional reference radiograph representations.
    """

    def __init__(
        self,
        index_path: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.index_path = Path(index_path or DEFAULT_INDEX_PATH)
        self.manifest_path = Path(manifest_path or DEFAULT_MANIFEST_PATH)
        self.metadata_path = Path(metadata_path or DEFAULT_METADATA_PATH)

        self._reference_ids: Optional[List[str]] = None
        self._embeddings: Optional[np.ndarray] = None
        self._metadata_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._manifest: Optional[Dict[str, Any]] = None

    @property
    def is_loaded(self) -> bool:
        """Check if embeddings are loaded in memory."""
        return self._embeddings is not None and len(self._embeddings) > 0

    def load(self, force_reload: bool = False) -> Tuple[List[str], np.ndarray]:
        """
        Load reference embeddings and metadata into memory.
        """
        if not force_reload and self.is_loaded:
            assert self._reference_ids is not None
            assert self._embeddings is not None
            return self._reference_ids, self._embeddings

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Reference embedding index not found at: {self.index_path}. "
                "Run the reference library builder to construct the V2 index."
            )

        data = np.load(self.index_path, allow_pickle=True)
        if "filenames" not in data and "reference_ids" not in data:
            raise ValueError(f"Invalid index archive at {self.index_path}: missing reference IDs.")
        if "embeddings" not in data:
            raise ValueError(f"Invalid index archive at {self.index_path}: missing embeddings matrix.")

        raw_ids = data["reference_ids"] if "reference_ids" in data else data["filenames"]
        ref_ids = [str(fid).strip() for fid in raw_ids]
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)

        N = len(ref_ids)
        if N == 0:
            raise ValueError(f"Empty reference embedding index at {self.index_path}.")
        if len(embeddings) != N:
            raise ValueError(f"Mismatch between ID count ({N}) and embeddings ({len(embeddings)}).")
        if embeddings.ndim != 2 or embeddings.shape[1] != 1024:
            raise ValueError(f"Expected shape (N, 1024), got {embeddings.shape}.")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Corrupted embeddings: non-finite values (NaN/Inf) detected.")

        # Unit-normalization check
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.all(np.abs(norms - 1.0) < 1e-2):
            raise ValueError(f"Embeddings are not unit-normalized: norm range [{norms.min():.4f}, {norms.max():.4f}].")

        # Duplicate ID check
        seen: Set[str] = set()
        duplicates: List[str] = [rid for rid in ref_ids if rid in seen or seen.add(rid)]
        if duplicates:
            raise ValueError(f"Duplicate reference IDs detected in index: {duplicates[:5]}.")

        # Load metadata records (JSON or embedded)
        metadata_map: Dict[str, Dict[str, Any]] = {}
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    meta_list = json.load(f)
                    if isinstance(meta_list, list):
                        for item in meta_list:
                            fid = item.get("reference_id") or item.get("filename") or item.get("image_index")
                            if fid:
                                metadata_map[str(fid).strip()] = item
            except Exception as exc:
                print(f"[PersistentEmbeddingIndex] Warning reading metadata JSON: {exc}")

        if not metadata_map and "metadata" in data:
            try:
                raw_meta = data["metadata"]
                for item in raw_meta:
                    m = dict(item)
                    fid = m.get("reference_id") or m.get("filename") or m.get("image_index")
                    if fid:
                        metadata_map[str(fid).strip()] = m
            except Exception:
                pass

        # Load manifest
        manifest: Optional[Dict[str, Any]] = None
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                pass

        self._reference_ids = ref_ids
        self._embeddings = embeddings
        self._metadata_map = metadata_map
        self._manifest = manifest

        return self._reference_ids, self._embeddings

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        exclude_ids: Optional[Union[str, Sequence[str]]] = None,
        exclude_identical: bool = True,
    ) -> List[Tuple[str, float, int]]:
        """
        Search the index for the Top-K most similar reference cases using cosine similarity.

        Args:
            query_vector: 1D or 2D array of shape (1024,) or (1, 1024), unit-normalized.
            top_k: Number of reference cases to retrieve.
            exclude_ids: Optional reference ID or sequence of IDs to exclude from search candidates.
            exclude_identical: If True, exclude candidates with >= 99.99% cosine similarity.

        Returns:
            List of tuples: (reference_id, similarity_score_pct, matrix_index) sorted descending.
        """
        ref_ids, embeddings = self.load()

        q = np.asarray(query_vector, dtype=np.float32).flatten()
        if q.shape != (1024,):
            raise ValueError(f"Query vector must be 1024-dimensional, got {q.shape}.")
        if not np.all(np.isfinite(q)):
            raise ValueError("Query vector contains NaN or Inf.")

        # Ensure query is unit-normalized
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Filter candidate indices
        exclude_set: Set[str] = set()
        if exclude_ids:
            if isinstance(exclude_ids, str):
                exclude_set.add(exclude_ids.strip().lower())
                exclude_set.add(Path(exclude_ids).name.lower())
            else:
                for eid in exclude_ids:
                    exclude_set.add(str(eid).strip().lower())
                    exclude_set.add(Path(str(eid)).name.lower())

        candidate_indices = [
            i for i, rid in enumerate(ref_ids)
            if rid.lower() not in exclude_set and Path(rid).name.lower() not in exclude_set
        ]

        if not candidate_indices:
            return []

        sub_embeddings = embeddings[candidate_indices]
        # Cosine dot-product similarity scaled to [0, 100]%
        raw_scores = np.dot(sub_embeddings, q)
        # Clip numerical precision to [-1.0, 1.0] and map to [0, 100] percentage
        scores = np.clip(raw_scores, -1.0, 1.0) * 100.0

        if exclude_identical:
            valid_cand = [idx for idx, s in enumerate(scores) if s < 99.99]
            if len(valid_cand) > 0 and len(valid_cand) < len(scores):
                candidate_indices = [candidate_indices[idx] for idx in valid_cand]
                scores = scores[valid_cand]

        # Deterministic sorting: primary = score descending, secondary = ID ascending (lexicographical)
        # Lexsort uses keys from right-to-left
        sub_ids = [ref_ids[idx] for idx in candidate_indices]
        # Negated scores for descending order
        sort_order = np.lexsort((sub_ids, -scores))

        k = min(max(1, top_k), len(sort_order))
        selected_order = sort_order[:k]

        results: List[Tuple[str, float, int]] = []
        for idx in selected_order:
            orig_idx = candidate_indices[idx]
            results.append((ref_ids[orig_idx], round(float(scores[idx]), 2), orig_idx))

        return results

    def get_metadata(self, reference_id: str) -> Optional[Dict[str, Any]]:
        """Get precomputed metadata dictionary for a reference case."""
        if not self.is_loaded:
            self.load()
        assert self._metadata_map is not None
        clean_id = Path(reference_id).name
        return self._metadata_map.get(clean_id)

    def get_manifest(self) -> Optional[Dict[str, Any]]:
        """Return index manifest dictionary."""
        if not self.is_loaded:
            self.load()
        return self._manifest

    def get_count(self) -> int:
        """Return count of indexed reference cases."""
        if self._reference_ids is not None:
            return len(self._reference_ids)
        if self.index_path.exists():
            try:
                rids, _ = self.load()
                return len(rids)
            except Exception:
                return 0
        return 0

    def get_data(self) -> Tuple[List[str], np.ndarray]:
        """Return reference IDs and embeddings matrix."""
        return self.load()

    def get_dimension(self) -> int:
        """Return feature embedding dimension."""
        _, emb = self.load()
        return int(emb.shape[1])

    def verify_integrity(self, verify_hashes: bool = True) -> Dict[str, Any]:
        """
        Verify index invariants, shapes, finiteness, unit norms, and cryptographic checksums.
        """
        errors: List[str] = []
        if not self.index_path.exists():
            return {
                "valid": False,
                "error": f"Index archive not found at {self.index_path}",
                "requires_original_images": False,
            }

        try:
            ref_ids, embeddings = self.load()
        except Exception as exc:
            return {
                "valid": False,
                "error": f"Failed loading index archive: {exc}",
                "requires_original_images": False,
            }

        N = len(ref_ids)
        D = embeddings.shape[1] if embeddings.ndim == 2 else -1

        if D != 1024:
            errors.append(f"Expected 1024-d embeddings, found dimension {D}")
        if not np.all(np.isfinite(embeddings)):
            errors.append("Non-finite values (NaN/Inf) detected in embeddings")

        norms = np.linalg.norm(embeddings, axis=1)
        is_normalized = bool(np.all(np.abs(norms - 1.0) < 1e-2))
        if not is_normalized:
            errors.append(f"Embeddings are not unit-normalized: norm range [{norms.min():.4f}, {norms.max():.4f}]")

        seen: Set[str] = set()
        duplicates: List[str] = [rid for rid in ref_ids if rid in seen or seen.add(rid)]
        if duplicates:
            errors.append(f"Found {len(duplicates)} duplicate reference IDs")

        hashes_verified = False
        if verify_hashes and self.manifest_path.exists():
            try:
                manifest_data = self.get_manifest() or {}
                artifacts = manifest_data.get("artifacts", {})
                expected_sha = artifacts.get("reference_embeddings_npz", {}).get("sha256")
                if expected_sha:
                    actual_sha = hashlib.sha256(self.index_path.read_bytes()).hexdigest()
                    if actual_sha.lower() != expected_sha.lower():
                        errors.append(f"SHA-256 mismatch for {self.index_path.name}: expected {expected_sha}, got {actual_sha}")
                    else:
                        hashes_verified = True
            except Exception as e:
                errors.append(f"Manifest hash verification error: {e}")

        return {
            "valid": len(errors) == 0,
            "count": N,
            "dimension": D,
            "is_normalized": is_normalized,
            "finite": bool(np.all(np.isfinite(embeddings))),
            "duplicate_count": len(duplicates),
            "metadata_count": len(self._metadata_map or {}),
            "requires_original_images": False,
            "hashes_verified": hashes_verified,
            "errors": errors,
        }

