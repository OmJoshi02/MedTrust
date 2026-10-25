import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from backend.engine import MedTrustBackendEngine

engine = MedTrustBackendEngine()
img = Image.open('test_images/d2.jpg')
res = engine.analyze_full(img, query_filename='d2.jpg')

print('Status:', res.get('status'))
print('is_ood:', res.get('is_ood'))
ood = res.get('ood', {})
print('OOD status:', ood.get('status'))
print('OOD trust decision:', ood.get('trust_decision'))
print('OOD detectors:')
for k, v in ood.get('method_results', {}).items():
    print(f'  {k}: score={v["score"]:.4f}, thr={v["threshold"]:.4f}, is_ood={v["is_ood"]}')
print('res keys:', list(res.keys()))
if 'predictions' in res:
    print('predictions in res:', res['predictions'][:3])
if 'findings' in res:
    print('findings in res:', res['findings'][:3])
