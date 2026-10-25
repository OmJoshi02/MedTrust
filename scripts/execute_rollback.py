"""
Execute complete rollback of MedTrust calibration files to baseline state.
"""
import shutil
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = ROOT / "backup" / "calibration_baseline"
HASHES_FILE = BACKUP_DIR / "original_hashes.json"

if not HASHES_FILE.exists():
    raise RuntimeError(f"Baseline hashes file missing: {HASHES_FILE}")

with open(HASHES_FILE, "r", encoding="utf-8") as f:
    expected_hashes = json.load(f)

def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

print("=" * 80)
print("EXECUTING MEDTRUST CALIBRATION ROLLBACK")
print("=" * 80)

# Restore src/calibration files
backup_src = BACKUP_DIR / "src_calibration"
backup_out = BACKUP_DIR / "outputs_calibration"

verified_count = 0
for rel_path_str, expected_hash in expected_hashes.items():
    rel_path = Path(rel_path_str)
    if rel_path_str.startswith("src/calibration/"):
        rel_sub = rel_path_str[len("src/calibration/"):]
        src_file = backup_src / rel_sub
        dst_file = ROOT / "src" / "calibration" / rel_sub
    elif rel_path_str.startswith("outputs/calibration/"):
        rel_sub = rel_path_str[len("outputs/calibration/"):]
        src_file = backup_out / rel_sub
        dst_file = ROOT / "outputs" / "calibration" / rel_sub
    else:
        continue

    if not src_file.exists():
        raise FileNotFoundError(f"Backup file missing: {src_file}")

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)
    actual_hash = compute_sha256(dst_file)
    if actual_hash != expected_hash:
        raise ValueError(f"Hash mismatch on restored {dst_file}!\nExpected: {expected_hash}\nActual:   {actual_hash}")
    verified_count += 1
    print(f"  [RESTORED & VERIFIED] {rel_path_str}")

# Clean candidate directory if it exists
candidate_dir = ROOT / "outputs" / "candidate_calibration"
if candidate_dir.exists():
    shutil.rmtree(candidate_dir)
    print(f"  [CLEANED] {candidate_dir}")

print("=" * 80)
print(f"ROLLBACK SUCCESSFUL: All {verified_count} files restored and SHA-256 verified.")
print("=" * 80)

