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
from src.utils.metrics import (
    calculate_brier_score, 
    plot_combined_reliability_diagram, 
    TemperatureScaler,
    calculate_ece
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import pickle

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

    # 2. Load Calibration & Test Data
    studies = pd.read_csv("data/processed/studies.csv")
    
    cal_splits = pd.read_csv("data/splits/calibration.csv")
    cal_df = studies[studies["study_id"].isin(cal_splits["study_id"])].reset_index(drop=True)
    cal_ds = OpenIDataset(cal_df, label_cols=LABEL_COLS)
    cal_loader = DataLoader(cal_ds, batch_size=32, shuffle=False)
    
    test_splits = pd.read_csv("data/splits/test.csv")
    test_df = studies[studies["study_id"].isin(test_splits["study_id"])].reset_index(drop=True)
    test_ds = OpenIDataset(test_df, label_cols=LABEL_COLS)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    logger.info(f"Extracting logits for {len(cal_df)} calibration and {len(test_df)} test samples...")
    logits_cal, labels_cal = get_logits_and_labels(model, cal_loader, device)
    logits_test, labels_test = get_logits_and_labels(model, test_loader, device)

    # 3. Optimize Temperature
    # We want to find T that minimizes NLL (BCE) on the calibration set
    nll_criterion = nn.BCEWithLogitsLoss()
    
    scaler = TemperatureScaler().to(device)
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    logits_cal_dev = logits_cal.to(device)
    labels_cal_dev = labels_cal.to(device)

    losses = []
    def eval_nll():
        optimizer.zero_grad()
        loss = nll_criterion(scaler(logits_cal_dev), labels_cal_dev)
        loss.backward()
        losses.append(loss.item())
        return loss

    logger.info("Optimizing temperature scaling...")
    optimizer.step(eval_nll)
    
    optimal_t = scaler.temperature.item()
    logger.info(f"Optimal temperature found: {optimal_t:.4f}")
    logger.info(f"NLL Trace: {[round(l, 4) for l in losses]}")

    # 4. Fallback Methods: Platt Scaling & Isotonic Regression
    # Training on calibration set
    logits_cal_np = logits_cal.numpy().flatten()
    labels_cal_np = labels_cal.numpy().flatten()
    
    logger.info("Training Platt Scaling (Global)...")
    platt = LogisticRegression(penalty=None)
    platt.fit(logits_cal_np.reshape(-1, 1), labels_cal_np)
    
    logger.info("Training Isotonic Regression (Global)...")
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(logits_cal_np, labels_cal_np)

    # 5. Model Selection based on Calibration Set ECE (Adaptive)
    def get_probs(method, lgs, plat_model=None, iso_model=None, temp=1.0):
        if method == "uncalibrated":
            return torch.sigmoid(lgs).numpy()
        elif method == "temp_scaling":
            return torch.sigmoid(lgs / temp).numpy()
        elif method == "platt":
            return plat_model.predict_proba(lgs.numpy().flatten().reshape(-1, 1))[:, 1].reshape(lgs.shape)
        elif method == "isotonic":
            return iso_model.predict(lgs.numpy().flatten()).reshape(lgs.shape)
        return None

    methods = ["uncalibrated", "temp_scaling", "platt", "isotonic"]
    ece_cal_results = {}

    labels_cal_np_multi = labels_cal.numpy()
    for m in methods:
        p = get_probs(m, logits_cal, plat_model=platt, iso_model=iso, temp=optimal_t)
        # Using min_bin_size=30 for selection as well to be consistent
        ece = calculate_ece(labels_cal_np_multi, p, adaptive=True, min_bin_size=30)
        ece_cal_results[m] = ece
        logger.info(f"ECE (Cal-set, {m}, min_size=30): {ece:.4f}")

    # Best method selection
    best_method = min(ece_cal_results, key=ece_cal_results.get)
    logger.info(f"Best calibration method selected: {best_method}")
    
    # 6. Evaluation on Held-out Test Set
    labels_test_np_multi = labels_test.numpy()
    probs_test_before = get_probs("uncalibrated", logits_test)
    probs_test_after = get_probs(best_method, logits_test, plat_model=platt, iso_model=iso, temp=optimal_t)
    
    ece_test_before = calculate_ece(labels_test_np_multi, probs_test_before, adaptive=True, min_bin_size=30)
    ece_test_after = calculate_ece(labels_test_np_multi, probs_test_after, adaptive=True, min_bin_size=30)
    
    logger.info(f"True ECE (Test-set, Before, min_size=30): {ece_test_before:.4f}")
    logger.info(f"True ECE (Test-set, After, min_size=30):  {ece_test_after:.4f}")

    # 7. Per-Class ECE Stress Test
    logger.info("\nPer-Class ECE (Test Set):")
    miscalibrated_classes = []
    for i, label in enumerate(LABEL_COLS):
        label_ece_pre = calculate_ece(labels_test_np_multi[:, i:i+1], probs_test_before[:, i:i+1], adaptive=True, min_bin_size=30)
        label_ece_post = calculate_ece(labels_test_np_multi[:, i:i+1], probs_test_after[:, i:i+1], adaptive=True, min_bin_size=30)
        logger.info(f"  {label:<15}: {label_ece_pre:.4f} -> {label_ece_post:.4f}")
        if label_ece_post > 0.05:
            miscalibrated_classes.append(label)
    
    if miscalibrated_classes:
        logger.warning(f"Miscalibrated classes (ECE > 0.05): {miscalibrated_classes}")

    # 8. Report Samples per Bin & Dominance Sanity Check (Test Set)
    from src.utils.metrics import get_reliability_data
    acc, conf, counts = get_reliability_data(labels_test_np_multi, probs_test_after, adaptive=True, min_bin_size=30)
    logger.info(f"Bin counts (Test-set, {best_method}): {counts.tolist()}")
    
    # Assertions
    assert np.all(counts >= 30), f"Sanity Check Failed: Some bins have < 30 samples: {counts}"
    assert ece_test_after < ece_test_before, f"Sanity Check Failed: {best_method} Test-ECE ({ece_test_after:.4f}) >= Uncalibrated Test-ECE ({ece_test_before:.4f})"
    
    # Check dominance in last bin
    # We need the last bin mask
    y_prob_test_flat = probs_test_after.flatten()
    quantiles = np.linspace(0, 1, 10 + 1)
    bin_boundaries = np.unique(np.quantile(y_prob_test_flat, quantiles))
    # Correct boundaries after merging
    new_boundaries = [bin_boundaries[0]]
    for i in range(1, len(bin_boundaries)):
        if np.sum((y_prob_test_flat > new_boundaries[-1]) & (y_prob_test_flat <= bin_boundaries[i])) >= 30:
            new_boundaries.append(bin_boundaries[i])
    if new_boundaries[-1] != bin_boundaries[-1]:
        new_boundaries[-1] = bin_boundaries[-1]
    
    last_lower = new_boundaries[-2]
    last_upper = new_boundaries[-1]
    in_last_bin_flat = (y_prob_test_flat > last_lower) & (y_prob_test_flat <= last_upper)
    in_last_bin_multi = in_last_bin_flat.reshape(labels_test.shape)
    
    num_samples_last_bin = np.sum(in_last_bin_flat)
    logger.info(f"Last bin samples: {num_samples_last_bin}")
    
    max_dominance = 0
    dominant_class = ""
    for i, label in enumerate(LABEL_COLS):
        class_count = np.sum(in_last_bin_multi[:, i])
        dominance = class_count / num_samples_last_bin
        if dominance > max_dominance:
            max_dominance = dominance
            dominant_class = label
    
    logger.info(f"Max class dominance in last bin: {max_dominance:.4f} ({dominant_class})")
    assert max_dominance <= 0.8, f"Sanity Check Failed: {dominant_class} dominates last bin ({max_dominance:.2%})"

    # 9. Evaluation & Visualization (using Test Set)
    brier_test_before, _ = calculate_brier_score(labels_test_np_multi, probs_test_before)
    brier_test_after, _ = calculate_test_brier = calculate_brier_score(labels_test_np_multi, probs_test_after)
    # Correcting common typo in previous edits
    brier_test_after = brier_test_after[0] if isinstance(brier_test_after, tuple) else brier_test_after
    
    logger.info(f"Brier score (Test-set, Before): {brier_test_before:.4f}")
    logger.info(f"Brier score (Test-set, After):  {brier_test_after:.4f}")

    # Save Combined Reliability Diagram (Test Set)
    os.makedirs("artifacts/figures", exist_ok=True)
    plot_combined_reliability_diagram(
        labels_test_np_multi,
        {
            "Uncalibrated": probs_test_before,
            f"Calibrated ({best_method})": probs_test_after
        },
        min_bin_size=30,
        title="Calibration Comparison (Test Set, min_size=30)",
        save_path="artifacts/figures/reliability_comparison.png"
    )
    logger.info("Reliability comparison plot saved to artifacts/figures/reliability_comparison.png")

    # 9. Save Calibration Result
    calib_res = {
        "best_method": best_method,
        "temperature": optimal_t if best_method == "temp_scaling" else 1.0,
        "ece_before": float(ece_test_before),
        "ece_after": float(ece_test_after),
        "brier_before": float(brier_test_before),
        "brier_after": float(brier_test_after),
        "test_bin_counts": counts.tolist()
    }
    
    with open("artifacts/models/calibration.yaml", "w") as f:
        yaml.dump(calib_res, f)
        
    # Save Platt/Isotonic models if needed
    if best_method == "platt":
        with open("artifacts/models/platt_scaler.pkl", "wb") as f:
            pickle.dump(platt, f)
    elif best_method == "isotonic":
        with open("artifacts/models/isotonic_scaler.pkl", "wb") as f:
            pickle.dump(iso, f)

    logger.info("Calibration metadata saved to artifacts/models/calibration.yaml")

if __name__ == "__main__":
    main()
