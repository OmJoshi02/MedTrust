import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "outputs" / "similarity_v2" / "evaluation_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

PROTECTED_FILES = [
    "checkpoints/best_model.pt",
    "outputs/evaluation/thresholds.json",
    "outputs/calibration/temperature.json",
    "outputs/calibration/reliability/validation_reliability_bins.csv",
    "outputs/quality/quality_reference.json",
    "splits/train_list.txt",
    "splits/val_list.txt",
    "splits/test_list.txt",
    "splits/train_patients.txt",
    "splits/val_patients.txt",
    "splits/test_patients.txt",
]

hashes_before = {}
for rel_path in PROTECTED_FILES:
    p = PROJECT_ROOT / rel_path
    if p.exists():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        hashes_before[rel_path] = {
            "sha256": h,
            "size_bytes": p.stat().st_size,
            "exists": True,
        }
    else:
        hashes_before[rel_path] = {"exists": False}

out_path = AUDIT_DIR / "protected_artifact_hashes_before.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(hashes_before, f, indent=2)

print(f"Recorded {len(hashes_before)} baseline protected artifacts in {out_path}:")
for k, v in hashes_before.items():
    print(f"  {k}: {v.get('sha256', 'MISSING')} ({v.get('size_bytes', 0)} bytes)")

