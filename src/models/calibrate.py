"""
calibrate.py — Apply temperature scaling to the trained image classifier.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from src.models.image_baseline import get_model
from src.data.preprocess_images import OpenIDataset
from src.utils.metrics import calculate_brier_score, plot_reliability_diagram, TemperatureScaler

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LABEL_COLS = ["cardiomegaly", "effusion", "edema", "pneumothorax", "consolidation"]

def get_logits_and_labels(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)

def main():
    # 1. Load Config & Model
    with open("configs/train_image.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    
    model_path = "artifacts/models/best_model.pt"
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Complete Stage 2 first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Loaded model from {model_path}")

    # 2. Load Calibration Data
    studies = pd.read_csv("data/processed/studies.csv")
    cal_splits = pd.read_csv("data/splits/calibration.csv")
    cal_df = studies[studies["study_id"].isin(cal_splits["study_id"])].reset_index(drop=True)
    
    cal_ds = OpenIDataset(cal_df, label_cols=LABEL_COLS)
    cal_loader = DataLoader(cal_ds, batch_size=32, shuffle=False)
    
    logger.info(f"Extracting logits for {len(cal_df)} calibration samples...")
    logits, labels = get_logits_and_labels(model, cal_loader, device)

    # 3. Optimize Temperature
    # We want to find T that minimizes NLL (BCE) on the calibration set
    nll_criterion = nn.BCEWithLogitsLoss()
    
    scaler = TemperatureScaler().to(device)
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    logits = logits.to(device)
    labels = labels.to(device)

    def eval_nll():
        optimizer.zero_grad()
        loss = nll_criterion(scaler(logits), labels)
        loss.backward()
        return loss

    logger.info("Optimizing temperature...")
    optimizer.step(eval_nll)
    
    optimal_t = scaler.temperature.item()
    logger.info(f"Optimal temperature found: {optimal_t:.4f}")

    # 4. Evaluation & Visualization
    # Calculate probs before and after
    probs_before = torch.sigmoid(logits).detach().cpu().numpy()
    probs_after = torch.sigmoid(logits / optimal_t).detach().cpu().numpy()
    labels_np = labels.cpu().numpy()

    brier_before, _ = calculate_brier_score(labels_np, probs_before)
    brier_after, _ = calculate_brier_score(labels_np, probs_after)
    
    logger.info(f"Brier score (Before): {brier_before:.4f}")
    logger.info(f"Brier score (After):  {brier_after:.4f}")

    # Save Reliability Diagrams
    os.makedirs("artifacts/figures", exist_ok=True)
    plot_reliability_diagram(
        labels_np, probs_before, 
        title=f"Reliability (Uncalibrated, Brier={brier_before:.3f})",
        save_path="artifacts/figures/reliability_uncalibrated.png"
    )
    plot_reliability_diagram(
        labels_np, probs_after, 
        title=f"Reliability (Calibrated T={optimal_t:.2f}, Brier={brier_after:.3f})",
        save_path="artifacts/figures/reliability_calibrated.png"
    )
    logger.info("Reliability diagrams saved to artifacts/figures/")

    # 5. Save Calibration Result
    calib_res = {
        "temperature": optimal_t,
        "brier_before": float(brier_before),
        "brier_after": float(brier_after)
    }
    with open("artifacts/models/calibration.yaml", "w") as f:
        yaml.dump(calib_res, f)
    logger.info("Calibration metadata saved to artifacts/models/calibration.yaml")

if __name__ == "__main__":
    main()
