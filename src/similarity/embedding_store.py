"""
MedTrust V1 — Reference Embedding Storage & Integrity Store (src/similarity/embedding_store.py)

Module 8: Similarity Analysis.
Manages loading, validating, and caching precomputed reference image embeddings
and associated authentic dataset metadata.

Storage Format (.npz):
    - 'filenames': 1D array of strings naming each reference image on disk
    - 'embeddings': 2D float32 array of shape (N, 1024) containing L2-normalized feature vectors
    - 'metadata': Structured array / list of metadata dicts with dataset provenance
    - Optional 'manifest': JSON string documenting model architecture, checkpoint SHA-256, and preprocessing

Integrity Invariants:
1. Every embedding maps to exactly one metadata record (len(filenames) == len(embeddings) == len(metadata)).
2. Every vector is 1024-dimensional, finite (no NaN/Inf), and L2-normalized (norm ~ 1.0).
3. Detects and reports duplicate image identifiers.
4. Fails clearly on corrupted or incompatible files.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "outputs" / "similarity" / "reference_embeddings.npz"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "similarity" / "reference_manifest.json"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "outputs" / "similarity" / "reference_metadata.json"


class EmbeddingStore:
    """
    In-memory cache and integrity validator for reference feature embeddings.
    Operates in an embedding-only deployment architecture without requiring
    original reference X-ray image files on disk.
    """

    def __init__(
        self,
        embeddings_path: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.embeddings_path = Path(embeddings_path or DEFAULT_EMBEDDINGS_PATH)
        self.manifest_path = Path(manifest_path or DEFAULT_MANIFEST_PATH)
        self.metadata_path = Path(metadata_path or DEFAULT_METADATA_PATH)
        self._filenames: Optional[List[str]] = None
        self._embeddings: Optional[np.ndarray] = None
        self._metadata: Optional[List[Dict[str, Any]]] = None
        self._manifest: Optional[Dict[str, Any]] = None

    @property
    def is_loaded(self) -> bool:
        """Check if embeddings are currently loaded in memory."""
        return self._embeddings is not None and len(self._embeddings) > 0

    def load(self, force_reload: bool = False) -> Tuple[List[str], np.ndarray]:
        """
        Load and strictly validate reference embeddings from the .npz archive into memory.

        Returns:
            Tuple of (filenames list, embeddings 2D numpy array of shape (N, 1024))

        Raises:
            FileNotFoundError: If the embeddings file does not exist.
            ValueError: If the file is corrupted, empty, non-finite, or has shape/count mismatches.
        """
        if not force_reload and self.is_loaded:
            assert self._filenames is not None
            assert self._embeddings is not None
            return self._filenames, self._embeddings

        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Reference embeddings file not found at: {self.embeddings_path}. "
                "Run 'python scripts/generate_reference_embeddings.py' to generate embeddings."
            )

        try:
            data = np.load(self.embeddings_path, allow_pickle=True)
            if "filenames" not in data or "embeddings" not in data:
                raise ValueError(
                    f"Invalid embeddings archive at {self.embeddings_path}. "
                    "Must contain both 'filenames' and 'embeddings' arrays."
                )

            filenames = [str(f) for f in data["filenames"]]
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)

            N = len(filenames)
            if N == 0 or len(embeddings) == 0:
                raise ValueError(
                    f"The reference embeddings dataset at {self.embeddings_path} is empty (0 reference cases)."
                )

            if len(filenames) != len(embeddings):
                raise ValueError(
                    f"Mismatch between number of filenames ({len(filenames)}) and embeddings ({len(embeddings)})."
                )

            if embeddings.ndim != 2:
                raise ValueError(
                    f"Expected 2D embeddings matrix (N, D), got shape {embeddings.shape}."
                )

            if embeddings.shape[1] != 1024:
                raise ValueError(
                    f"Expected 1024-dimensional DenseNet-121 feature vectors, got dimension {embeddings.shape[1]}."
                )

            # Check for NaN / Inf
            if not np.all(np.isfinite(embeddings)):
                raise ValueError("Corrupted reference embeddings: found non-finite values (NaN or Inf).")

            # Check for duplicate image identifiers
            seen: Set[str] = set()
            duplicates: List[str] = []
            for fn in filenames:
                if fn in seen:
                    duplicates.append(fn)
                seen.add(fn)
            if duplicates:
                print(f"[EmbeddingStore] Warning: Found {len(duplicates)} duplicate image identifiers in reference store: {duplicates[:5]}")

            # Metadata parsing: NPZ embedded -> JSON catalog -> CSV loader
            metadata: Optional[List[Dict[str, Any]]] = None
            if "metadata" in data:
                try:
                    raw_meta = data["metadata"]
                    metadata = [dict(m) for m in raw_meta]
                except Exception:
                    metadata = None

            if (metadata is None or len(metadata) == 0) and self.metadata_path.exists():
                try:
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        loaded_json = json.load(f)
                        if isinstance(loaded_json, list) and len(loaded_json) == N:
                            metadata = loaded_json
                except Exception:
                    metadata = None

            if metadata is None or len(metadata) == 0:
                try:
                    from src.similarity.metadata_loader import metadata_loader
                    metadata = metadata_loader.get_metadata_list_for_filenames(filenames)
                except Exception:
                    metadata = None

            if metadata is not None and len(metadata) != N:
                raise ValueError(
                    f"Count mismatch between metadata records ({len(metadata)}) and embeddings ({N})."
                )

            # Manifest parsing: Disk manifest file takes precedence for deployment verification
            manifest: Optional[Dict[str, Any]] = None
            if self.manifest_path.exists():
                try:
                    with open(self.manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                except Exception:
                    manifest = None

            if manifest is None and "manifest" in data:
                try:
                    manifest = json.loads(str(data["manifest"]))
                except Exception:
                    manifest = None

            self._filenames = filenames
            self._embeddings = embeddings
            self._metadata = metadata
            self._manifest = manifest
            return self._filenames, self._embeddings

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise e
            raise ValueError(f"Failed to load reference embeddings file: {str(e)}") from e

    def get_data(self) -> Tuple[List[str], np.ndarray]:
        """Get cached data or trigger load if not yet cached."""
        return self.load()

    def get_metadata(self) -> Optional[List[Dict[str, Any]]]:
        """Get optional metadata list or None."""
        if not self.is_loaded:
            self.load()
        return self._metadata

    def get_manifest(self) -> Optional[Dict[str, Any]]:
        """Get provenance manifest or None."""
        if not self.is_loaded:
            self.load()
        return self._manifest

    def verify_integrity(self, verify_hashes: bool = False) -> Dict[str, Any]:
        """
        Validate store integrity in an embedding-only deployment environment:
        1. Checks presence of embeddings archive.
        2. Validates vector dimension (1024), unit normalization (L2 norm ~ 1.0), and finiteness.
        3. Validates one-to-one alignment between filenames, embeddings, and metadata.
        4. Validates zero duplicate filenames.
        5. Optionally checks SHA-256 hashes against reference_manifest.json.
        6. Confirms that deployment operates without requiring original image files on disk.
        """
        errors: List[str] = []
        if not self.embeddings_path.exists():
            return {
                "valid": False,
                "error": f"Embeddings archive not found at {self.embeddings_path}",
                "requires_original_images": False,
            }

        try:
            filenames, embeddings = self.load()
        except Exception as exc:
            return {
                "valid": False,
                "error": f"Failed loading embeddings archive: {exc}",
                "requires_original_images": False,
            }

        N = len(filenames)
        D = embeddings.shape[1] if embeddings.ndim == 2 else -1

        if D != 1024:
            errors.append(f"Expected 1024-d embeddings, found dimension {D}")

        if not np.all(np.isfinite(embeddings)):
            errors.append("Non-finite values (NaN/Inf) detected in embeddings")

        norms = np.linalg.norm(embeddings, axis=1)
        is_normalized = bool(np.all(np.abs(norms - 1.0) < 1e-3))
        if not is_normalized:
            errors.append(f"Embeddings are not unit-normalized: norm range [{norms.min():.4f}, {norms.max():.4f}]")

        seen: Set[str] = set()
        duplicates: List[str] = []
        for fn in filenames:
            if fn in seen:
                duplicates.append(fn)
            seen.add(fn)
        if duplicates:
            errors.append(f"Found {len(duplicates)} duplicate image identifiers")

        metadata = self._metadata or []
        if len(metadata) != N:
            errors.append(f"Metadata length mismatch: {len(metadata)} metadata items vs {N} embeddings")

        hashes_verified = False
        if verify_hashes and self.manifest_path.exists():
            try:
                manifest_data = self.get_manifest() or {}
                artifacts = manifest_data.get("artifacts", {})
                expected_sha = artifacts.get("reference_embeddings_npz", {}).get("sha256")
                if expected_sha:
                    actual_sha = hashlib.sha256(self.embeddings_path.read_bytes()).hexdigest()
                    if actual_sha.lower() != expected_sha.lower():
                        errors.append(f"SHA-256 mismatch for {self.embeddings_path.name}: expected {expected_sha}, got {actual_sha}")
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
            "metadata_count": len(metadata),
            "requires_original_images": False,
            "hashes_verified": hashes_verified,
            "errors": errors,
        }

    def get_count(self) -> int:
        """Return the count of loaded reference images."""
        if self._filenames is not None:
            return len(self._filenames)
        if self.embeddings_path.exists():
            try:
                filenames, _ = self.load()
                return len(filenames)
            except Exception:
                return 0
        return 0


# Default shared singleton instance
embedding_store = EmbeddingStore()
