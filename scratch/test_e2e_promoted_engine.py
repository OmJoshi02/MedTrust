import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.engine import MedTrustBackendEngine

engine = MedTrustBackendEngine()

print("Backend Engine Initialized on device:", engine.device)

# 1. Test In-Distribution image (disease.png)
id_img_path = PROJECT_ROOT / "test_images" / "disease.png"
with Image.open(id_img_path) as src:
    id_img = src.copy()

print("\n--- Testing In-Distribution Radiograph: test_images/disease.png ---")
id_res = engine.analyze_full(id_img)
print("  Status :", id_res.get("status"))
print("  is_ood :", id_res.get("is_ood"))
print("  Primary Finding:", id_res.get("prediction", {}).get("primary_finding"))
ood_meta = id_res.get("ood", {})
print("  OOD method results:")
for m, d in ood_meta.get("method_results", {}).items():
    print(f"    {m:<12} Score: {d['score']:>10.4f}  Threshold: {d['threshold']:>10.4f}  Status: {d['status']}")
assert id_res["is_ood"] == False, "In-distribution image was incorrectly rejected!"
assert id_res["status"] in ("ANALYZED", "SUCCESS"), "Analysis failed on in-distribution image!"
print(">>> In-Distribution Radiograph PASSES OOD Gating!")

# 2. Test Out-of-Distribution image (mri.jpg)
ood_img_path = PROJECT_ROOT / "test_images" / "mri.jpg"
with Image.open(ood_img_path) as src:
    ood_img = src.copy()

print("\n--- Testing Out-of-Distribution Scan: test_images/mri.jpg ---")
ood_res = engine.analyze_full(ood_img)
print("  Status :", ood_res.get("status"))
print("  is_ood :", ood_res.get("is_ood"))
print("  Message:", ood_res.get("message"))
ood_meta = ood_res.get("ood", {})
print("  OOD method results:")
for m, d in ood_meta.get("method_results", {}).items():
    print(f"    {m:<12} Score: {d['score']:>10.4f}  Threshold: {d['threshold']:>10.4f}  Status: {d['status']}")
assert ood_res["is_ood"] == True, "OOD image was not caught!"
assert ood_res["status"] == "REJECTED_OOD", "OOD pre-gating failed to halt diagnosis!"
print(">>> Out-of-Distribution Input PROPERLY GATED & HALTED!")

# Verify exact thresholds in results match Candidate 1
msp_th = ood_meta["method_results"]["msp"]["threshold"]
eng_th = ood_meta["method_results"]["energy"]["threshold"]
mah_th = ood_meta["method_results"]["mahalanobis"]["threshold"]

assert abs(msp_th - 0.737333) < 1e-5, f"MSP threshold mismatch: {msp_th}"
assert abs(eng_th - (-0.071788)) < 1e-5, f"Energy threshold mismatch: {eng_th}"
assert abs(mah_th - 975.531329) < 1e-5, f"Maha threshold mismatch: {mah_th}"

print("\n>>> ACTIVE BACKEND PROVABLY LOADS CANDIDATE 1 THRESHOLDS AND ENFORCES GATING!")
