import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.OOD.predict import predict_single_image
from src.OOD.model_adapter import MedTrustModelAdapter
from src.transformer import eval_transform
from PIL import Image
import numpy as np

ood_samples = ['test_images/foot.png', 'test_images/hand.png', 'test_images/mri.jpg']
data_stats = np.load('outputs/ood/artifacts/mahalanobis_statistics.npz')
mean = data_stats['mean']
safe_var = data_stats['safe_variance']
raw_var = data_stats['variance']
zero_var_idx = np.where(raw_var == 0)[0]
nonzero_var_idx = np.where(raw_var > 0)[0]

adapter = MedTrustModelAdapter()

for sample in ood_samples:
    p = Path(sample)
    if not p.exists():
        continue
    res = predict_single_image(str(p), method='combined')
    print('=== ' + sample + ' ===')
    print('Combined:', res['status'], 'Trust:', res['trust_decision'])
    for m, d in res['method_results'].items():
        s = d['score']
        th = d['threshold']
        st = d['status']
        print(f"  {m:<12} Score: {s:>10.4f}  Threshold: {th:>10.4f}  Status: {st}")
    pred = res['predictions'][0]
    prob = pred['probability'] * 100
    print(f"  Top Pred: {pred['disease']} ({prob:.1f}%)")
    
    # Check features in dead dims
    with Image.open(p) as src:
        img = src.convert('L')
    tensor = eval_transform(img).unsqueeze(0)
    out = adapter.predict(tensor)
    feat = out['features'].detach().cpu().numpy()[0]
    dead_feats = feat[zero_var_idx]
    active_dead_count = np.sum(dead_feats > 0)
    max_dead_feat = np.max(dead_feats)
    
    # Mahalanobis contribution
    centered = feat - mean
    sq_dist = (centered ** 2) / safe_var
    dead_contrib = np.sum(sq_dist[zero_var_idx])
    alive_contrib = np.sum(sq_dist[nonzero_var_idx])
    print(f"  Maha breakdown: Alive={alive_contrib:.1f}, Dead_contrib={dead_contrib:.1f} (Active dead dims: {active_dead_count}, Max dead val: {max_dead_feat:.6f})")
    print()
