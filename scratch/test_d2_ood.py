import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
import numpy as np
from PIL import Image
from src.transformer import eval_transform
from src.OOD.model_adapter import MedTrustModelAdapter
from src.OOD.scores import msp_score, energy_score, mahalanobis_score
from src.OOD.predict import load_mahalanobis_statistics
from src.dataset import CLASSES

# 1. Load image
img_path = 'test_images/d2.jpg'
with Image.open(img_path) as source:
    img = source.convert('L')
tensor = eval_transform(img).unsqueeze(0)

# 2. Model inference via MedTrustModelAdapter
adapter = MedTrustModelAdapter(device='cpu')
features, logits = adapter.extract_features_and_logits(tensor)

logits_np = logits.numpy()
features_np = features.numpy()
probs_np = torch.sigmoid(logits).numpy()[0]

# 3. Calculate scores
mean, safe_var = load_mahalanobis_statistics()
s_msp = float(msp_score(logits_np)[0])
s_energy = float(energy_score(logits_np)[0])
s_maha = float(mahalanobis_score(features_np, mean, safe_var)[0])

# 4. Compare with Baseline vs Promoted
with open('outputs/ood/artifacts/ood_thresholds.json.bak_baseline') as f:
    base = json.load(f)['thresholds']
with open('outputs/ood/artifacts/ood_thresholds.json') as f:
    promoted = json.load(f)['thresholds']

print('=== SCORES FOR test_images/d2.jpg ===')
print(f'MSP score:         {s_msp:.6f}')
print(f'Energy score:      {s_energy:.6f}')
print(f'Mahalanobis score: {s_maha:.6f}')
print()
print('=== UNDER BASELINE THRESHOLDS (Original Pre-Promotion) ===')
print(f'MSP:         {s_msp:.4f} vs threshold {base["msp"]["threshold"]:.4f} -> {"OOD" if s_msp >= base["msp"]["threshold"] else "ID"}')
print(f'Energy:      {s_energy:.4f} vs threshold {base["energy"]["threshold"]:.4f} -> {"OOD" if s_energy >= base["energy"]["threshold"] else "ID"}')
print(f'Mahalanobis: {s_maha:.4f} vs threshold {base["mahalanobis"]["threshold"]:.4f} -> {"OOD (FLAGGED)" if s_maha >= base["mahalanobis"]["threshold"] else "ID"}')
base_flag = (s_msp >= base["msp"]["threshold"]) or (s_energy >= base["energy"]["threshold"]) or (s_maha >= base["mahalanobis"]["threshold"])
print(f'Combined Baseline Status: {"OOD (REJECTED)" if base_flag else "ID"}')
print()
print('=== UNDER PROMOTED CANDIDATE 1 THRESHOLDS (Active Canonical) ===')
print(f'MSP:         {s_msp:.4f} vs threshold {promoted["msp"]["threshold"]:.4f} -> {"OOD" if s_msp >= promoted["msp"]["threshold"] else "ID"}')
print(f'Energy:      {s_energy:.4f} vs threshold {promoted["energy"]["threshold"]:.4f} -> {"OOD" if s_energy >= promoted["energy"]["threshold"] else "ID"}')
print(f'Mahalanobis: {s_maha:.4f} vs threshold {promoted["mahalanobis"]["threshold"]:.4f} -> {"OOD" if s_maha >= promoted["mahalanobis"]["threshold"] else "ID (PASSES)"}')
prom_flag = (s_msp >= promoted["msp"]["threshold"]) or (s_energy >= promoted["energy"]["threshold"]) or (s_maha >= promoted["mahalanobis"]["threshold"])
print(f'Combined Promoted Status: {"OOD" if prom_flag else "IN-DISTRIBUTION (PASSES / ACCEPTED)"}')
print()
print('=== DISEASE PREDICTIONS WHEN EVALUATED (ID) ===')
sorted_indices = np.argsort(probs_np)[::-1]
for idx in sorted_indices[:7]:
    print(f'  {CLASSES[idx]:<20}: {probs_np[idx]*100:.2f}%')

