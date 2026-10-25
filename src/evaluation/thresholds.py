"""
MedTrust V1 — Evaluation Decision Threshold Utilities (src/evaluation/thresholds.py)
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from src.dataset import CLASSES

DEFAULT_THRESHOLDS_PATH = Path("outputs/evaluation/thresholds.json")


def load_frozen_thresholds(
    path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    p = Path(path) if path is not None else DEFAULT_THRESHOLDS_PATH
    if not p.exists():
        repo_root = Path(__file__).resolve().parents[2]
        alt = repo_root / p
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"Thresholds artifact not found at: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("classes") != list(CLASSES):
        raise RuntimeError(
            f"Threshold artifact class order does not match canonical CLASSES.\n"
            f"Expected: {list(CLASSES)}\n"
            f"Found: {data.get('classes')}"
        )

    thresholds_dict = data.get("thresholds", {})
    thresholds = []
    for disease in CLASSES:
        if disease not in thresholds_dict or "threshold" not in thresholds_dict[disease]:
            raise KeyError(f"Missing threshold entry for class '{disease}' in {p}")
        thresh_val = float(thresholds_dict[disease]["threshold"])
        thresholds.append(thresh_val)

    return np.array(thresholds, dtype=np.float32)
