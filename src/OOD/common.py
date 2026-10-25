"""
MedTrust OOD Detection - Common Utilities (src/OOD/common.py)

Reusable validation, filesystem, and tensor utilities for the OOD module.
"""

from pathlib import Path
import json
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


def get_device(device_str: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Return the specified or best available PyTorch device."""
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_file(path: Union[str, Path], description: str = "Required file") -> Path:
    """Verify that a target file exists and is accessible."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{description} not found at: {p}")
    return p


def validate_class_order(
    classes: Optional[List[str]],
    expected_classes: List[str],
) -> None:
    """Verify that class order in checkpoint/artifacts matches canonical classes."""
    if classes is None:
        return
    if list(classes) != list(expected_classes):
        raise ValueError(
            f"Class list mismatch:\nExpected: {list(expected_classes)}\nFound: {list(classes)}"
        )


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists; create parents if needed."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_finite_array(arr: Any, name: str = "Array") -> np.ndarray:
    """Ensure NumPy array contains only finite floating-point values."""
    np_arr = np.asarray(arr)
    if not np.all(np.isfinite(np_arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return np_arr


def validate_ood_scores(scores: Any) -> np.ndarray:
    """Ensure OOD scores form a 1D finite array of float64."""
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    validate_finite_array(flat, name="OOD scores")
    return flat


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save dictionary as formatted JSON."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    p = require_file(path, "JSON artifact")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
