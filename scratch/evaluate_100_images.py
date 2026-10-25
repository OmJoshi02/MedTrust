import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.dataset import CLASSES
from src.OOD.model_adapter import MedTrustModelAdapter
from src.OOD.predict import (
    load_threshold,
    load_mahalanobis_statistics,
    extract_prediction,
)
from src.OOD.scores import msp_score, energy_score, mahalanobis_score
from src.transformer import eval_transform

# 1. Load 100 images from test split
test_list_path = PROJECT_ROOT / "splits" / "test_list.txt"
with open(test_list_path, "r") as f:
    all_test_images = [line.strip() for line in f if line.strip()]

np.random.seed(42)
sampled_100 = np.random.choice(all_test_images, size=100, replace=False).tolist()

# Load metadata
df_meta = pd.read_csv(PROJECT_ROOT / "dataset" / "Data_Entry_2017_v2020.csv").set_index("Image Index")

# Load thresholds & Mahalanobis reference statistics
msp_thresh = load_threshold("msp")
energy_thresh = load_threshold("energy")
maha_thresh = load_threshold("mahalanobis")
maha_mean, maha_var = load_mahalanobis_statistics()

print(f"Loaded thresholds: MSP={msp_thresh:.6f}, Energy={energy_thresh:.6f}, Maha={maha_thresh:.6f}")

# Initialize model
adapter = MedTrustModelAdapter()
img_dir = PROJECT_ROOT / "dataset" / "images"

results = []
batch_size = 16

for i in range(0, len(sampled_100), batch_size):
    batch_names = sampled_100[i : i + batch_size]
    tensors = []
    valid_names = []
    
    for name in batch_names:
        img_path = img_dir / name
        if not img_path.exists():
            print(f"Warning: {name} not found")
            continue
        with Image.open(img_path) as src:
            img = src.convert("L")
        tensor = eval_transform(img)
        tensors.append(tensor)
        valid_names.append(name)
        
    if not tensors:
        continue
        
    batch_tensor = torch.stack(tensors)
    with torch.no_grad():
        out = adapter.predict(batch_tensor)
        
    feats = out["features"].detach().cpu().numpy()
    logits = out["logits"].detach().cpu().numpy()
    probs = out["probabilities"].detach().cpu().numpy()
    
    # Calculate scores
    s_msp = msp_score(logits)
    s_energy = energy_score(logits)
    s_maha = mahalanobis_score(feats, maha_mean, maha_var)
    
    for idx, name in enumerate(valid_names):
        row = df_meta.loc[name] if name in df_meta.index else None
        gt_findings = str(row["Finding Labels"]) if row is not None else "Unknown"
        age = int(row["Patient Age"]) if (row is not None and pd.notna(row["Patient Age"])) else "N/A"
        gender = str(row["Patient Gender"]) if row is not None else "N/A"
        view = str(row["View Position"]) if row is not None else "N/A"
        
        m_val = float(s_msp[idx])
        e_val = float(s_energy[idx])
        mah_val = float(s_maha[idx])
        
        m_ood = bool(m_val >= msp_thresh)
        e_ood = bool(e_val >= energy_thresh)
        mah_ood = bool(mah_val >= maha_thresh)
        comb_ood = bool(m_ood or e_ood or mah_ood)
        
        sample_probs = probs[idx : idx + 1]
        preds = extract_prediction(sample_probs)
        top_disease = preds[0]["disease"]
        top_prob = preds[0]["probability"]
        
        results.append({
            "image": name,
            "patient_age": age,
            "patient_gender": gender,
            "view_position": view,
            "ground_truth": gt_findings,
            "msp_score": m_val,
            "msp_ood": m_ood,
            "energy_score": e_val,
            "energy_ood": e_ood,
            "mahalanobis_score": mah_val,
            "mahalanobis_ood": mah_ood,
            "combined_ood": comb_ood,
            "combined_status": "OOD" if comb_ood else "ID",
            "trust_decision": "DO NOT TRUST PREDICTION" if comb_ood else "PREDICTION CAN BE CONSIDERED",
            "top_prediction": top_disease,
            "top_probability": top_prob,
            "all_predictions": [f"{p['disease']} ({p['probability']*100:.1f}%)" for p in preds if p['probability'] >= 0.5] or [f"{top_disease} ({top_prob*100:.1f}%)"]
        })

print(f"Successfully processed {len(results)} images.")

# Compute statistics
df_res = pd.DataFrame(results)

comb_ood_count = df_res["combined_ood"].sum()
comb_id_count = len(df_res) - comb_ood_count

msp_ood_count = df_res["msp_ood"].sum()
energy_ood_count = df_res["energy_ood"].sum()
maha_ood_count = df_res["mahalanobis_ood"].sum()

stats = {
    "total_images": len(df_res),
    "combined_id_count": int(comb_id_count),
    "combined_ood_count": int(comb_ood_count),
    "combined_id_rate": float(comb_id_count / len(df_res) * 100),
    "combined_ood_rate": float(comb_ood_count / len(df_res) * 100),
    "msp_flagged_count": int(msp_ood_count),
    "energy_flagged_count": int(energy_ood_count),
    "mahalanobis_flagged_count": int(maha_ood_count),
    "msp_score": {
        "mean": float(df_res["msp_score"].mean()),
        "std": float(df_res["msp_score"].std()),
        "min": float(df_res["msp_score"].min()),
        "max": float(df_res["msp_score"].max()),
        "median": float(df_res["msp_score"].median())
    },
    "energy_score": {
        "mean": float(df_res["energy_score"].mean()),
        "std": float(df_res["energy_score"].std()),
        "min": float(df_res["energy_score"].min()),
        "max": float(df_res["energy_score"].max()),
        "median": float(df_res["energy_score"].median())
    },
    "mahalanobis_score": {
        "mean": float(df_res["mahalanobis_score"].mean()),
        "std": float(df_res["mahalanobis_score"].std()),
        "min": float(df_res["mahalanobis_score"].min()),
        "max": float(df_res["mahalanobis_score"].max()),
        "median": float(df_res["mahalanobis_score"].median())
    }
}

print("Summary Stats:")
print(json.dumps(stats, indent=2))

# Save outputs to scratch
output_json = PROJECT_ROOT / "scratch" / "ood_100_images_results.json"
with open(output_json, "w") as f:
    json.dump({"stats": stats, "results": results}, f, indent=2)

df_res.to_csv(PROJECT_ROOT / "scratch" / "ood_100_images_results.csv", index=False)
print("Saved outputs to scratch/ood_100_images_results.json and .csv")
