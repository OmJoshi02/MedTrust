"""
MedTrust V1 — Calibration Temperature Loader & Utilities (src/calibration/temperature_loader.py)
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from src.dataset import CLASSES

DEFAULT_TEMPERATURE_PATH = Path("outputs/calibration/temperature.json")


def load_frozen_temperatures(
    path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    p = Path(path) if path is not None else DEFAULT_TEMPERATURE_PATH
    if not p.exists():
        repo_root = Path(__file__).resolve().parents[2]
        alt = repo_root / p
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"Calibration artifact not found at: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("classes") != list(CLASSES):
        raise RuntimeError("Calibration artifact class order does not match canonical CLASSES.")

    temperatures_dict = data.get("temperatures", {})
    temperatures = [float(temperatures_dict[disease]) for disease in CLASSES]
    return np.array(temperatures, dtype=np.float32)


def apply_temperature_scaling(
    logits: torch.Tensor,
    temperatures: Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    if isinstance(temperatures, np.ndarray):
        temp_tensor = torch.from_numpy(temperatures).to(device=logits.device, dtype=logits.dtype)
    else:
        temp_tensor = temperatures.to(device=logits.device, dtype=logits.dtype)

    if logits.ndim == 2 and temp_tensor.ndim == 1:
        temp_tensor = temp_tensor.unsqueeze(0)

    return logits / temp_tensor
