import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.OOD.predict import predict_all_methods
from src.OOD.model_adapter import MedTrustModelAdapter
from src.transformer import eval_transform
from PIL import Image

# Load candidate thresholds
with open(PROJECT_ROOT / 'outputs/ood/artifacts/ood_thresholds_candidate1_exp.json') as f:
    cand_data = json.load(f)['thresholds']

cand_th = {
    'msp': cand_data['msp']['threshold'],
    'energy': cand_data['energy']['threshold'],
    'mahalanobis': cand_data['mahalanobis']['threshold'],
}

adapter = MedTrustModelAdapter()
mri_path = PROJECT_ROOT / 'test_images/mri.jpg'

with Image.open(mri_path) as src:
    img = src.convert('L')
tensor = eval_transform(img).unsqueeze(0)
out = adapter.predict(tensor)
logits = out['logits'].detach().cpu().numpy()
features = out['features'].detach().cpu().numpy()

# Test standard predict_all_methods with candidate thresholds
# Let's see if predict_all_methods can take custom thresholds or if it loads from threshold.py
import src.OOD.predict as ood_predict

# Monkeypatch load_threshold temporarily in memory
orig_load_threshold = ood_predict.load_threshold
try:
    ood_predict.load_threshold = lambda method: cand_th[method]
    res_cand = ood_predict.predict_all_methods(logits, features)
    print("Candidate 1 Evaluation on mri.jpg:")
    print("  Status:", res_cand['status'])
    print("  is_ood:", res_cand['is_ood'])
    print("  Trust :", res_cand['trust_decision'])
    for m, d in res_cand['method_results'].items():
        print(f"    {m:<12} Score: {d['score']:>10.4f}  Threshold: {d['threshold']:>10.4f}  Status: {d['status']}")
    assert res_cand['is_ood'] == True, "Failed to reject mri.jpg under Candidate 1!"
    print("Verification Passed: mri.jpg is successfully rejected under Candidate 1!")
finally:
    ood_predict.load_threshold = orig_load_threshold

