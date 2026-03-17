"""
evaluate_model.py — Evaluate the trained image baseline on the validation set.
"""

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from src.models.image_baseline import get_model
from src.data.preprocess_images import OpenIDataset
import yaml

LABEL_COLS = ["cardiomegaly", "effusion", "edema", "pneumothorax", "consolidation"]

@torch.no_grad()
def evaluate():
    # 1. Load Config
    with open("configs/train_image.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Data
    studies = pd.read_csv("data/processed/studies.csv")
    val_splits = pd.read_csv("data/splits/val.csv")
    val_df = studies[studies["study_id"].isin(val_splits["study_id"])].reset_index(drop=True)
    
    val_ds = OpenIDataset(val_df, label_cols=LABEL_COLS)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # 3. Load Model
    model = get_model(cfg).to(device)
    model.load_state_dict(torch.load("artifacts/models/best_model.pt", map_location=device))
    model.eval()
    
    all_targets = []
    all_probs = []
    
    print("Evaluating...")
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        all_targets.append(labels.cpu())
        all_probs.append(probs.cpu())
        
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    # 4. Metrics
    print("\nResults:")
    aurocs = []
    for i, label in enumerate(LABEL_COLS):
        try:
            score = roc_auc_score(all_targets[:, i], all_probs[:, i])
            print(f"  {label:<15}: {score:.4f}")
            aurocs.append(score)
        except ValueError:
            print(f"  {label:<15}: N/A (single class)")
            
    macro_auroc = sum(aurocs) / len(aurocs) if aurocs else 0.0
    print(f"\nMacro AUROC: {macro_auroc:.4f}")
    
    # Save results
    import json
    import os
    results = {
        "per_class": {label: score for label, score in zip(LABEL_COLS, aurocs)},
        "macro_auroc": macro_auroc,
        "pass_gate": bool(macro_auroc >= 0.70)
    }
    os.makedirs("artifacts/results", exist_ok=True)
    with open("artifacts/results/image_baseline_val.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to artifacts/results/image_baseline_val.json")

    if macro_auroc >= 0.70:
        print("\n[PASS] Phase 1 Gate Met.")
    else:
        print("\n[FAIL] Phase 1 Gate Not Met (Target: 0.70).")

if __name__ == "__main__":
    evaluate()
