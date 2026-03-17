"""
train_image.py — Multi-label training loop for OpenI image classifier.

Features:
  - Loads hyperparameters from configs/train_image.yaml
  - Supports --subset N for quick validation
  - Logs per-class and macro AUROC
  - Saves best checkpoint to artifacts/models/best_model.pt
"""

import os
import yaml
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.models.image_baseline import get_model
from src.data.preprocess_images import OpenIDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LABEL_COLS = ["cardiomegaly", "effusion", "edema", "pneumothorax", "consolidation"]

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        
        # BCEWithLogitsLoss expects float labels
        loss = criterion(logits, labels)
        
        # Assertions during subset run
        if torch.isnan(loss):
            raise ValueError("NaN loss detected")
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    return running_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    
    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        
        probs = torch.sigmoid(logits)
        all_targets.append(labels.cpu())
        all_probs.append(probs.cpu())
        
    avg_loss = running_loss / len(loader.dataset)
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    # Calculate per-class AUROC
    aurocs = {}
    valid_aurocs = []
    for i, label in enumerate(LABEL_COLS):
        try:
            score = roc_auc_score(all_targets[:, i], all_probs[:, i])
            aurocs[label] = score
            valid_aurocs.append(score)
        except ValueError:
            # Handle cases with only one class present in small subsets
            aurocs[label] = 0.0
            
    macro_auroc = sum(valid_aurocs) / len(valid_aurocs) if valid_aurocs else 0.0
    
    return avg_loss, macro_auroc, aurocs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None, help="Train on first N samples")
    args = parser.parse_args()
    
    # 1. Load Config
    with open("configs/train_image.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    # Assertion at startup
    assert "backbone" in cfg, "backbone missing from config"
    assert "lr" in cfg, "lr missing from config"
    assert "batch_size" in cfg, "batch_size missing from config"
    
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 2. Load Data
    studies = pd.read_csv("data/processed/studies.csv")
    train_splits = pd.read_csv("data/splits/train.csv")
    val_splits = pd.read_csv("data/splits/val.csv")
    
    train_df = studies[studies["study_id"].isin(train_splits["study_id"])].reset_index(drop=True)
    val_df = studies[studies["study_id"].isin(val_splits["study_id"])].reset_index(drop=True)
    
    # Assert OpenI has no -1 labels
    assert (train_df[LABEL_COLS] == -1).to_numpy().sum() == 0, \
        "Unexpected -1 labels found — OpenI should not have uncertainty labels"

    # Subset handling
    if args.subset:
        logger.info(f"Using subset of {args.subset} samples")
        train_df = train_df.head(args.subset)
        val_df = val_df.head(args.subset)
        
    # Check for sufficient positives
    if not args.subset:
        for label in LABEL_COLS:
            pos_count = train_df[label].sum()
            assert pos_count >= 10, \
                f"{label} has only {pos_count} positives — too few to train reliably"

    train_ds = OpenIDataset(train_df, label_cols=LABEL_COLS)
    val_ds = OpenIDataset(val_df, label_cols=LABEL_COLS)
    
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 0))
    
    # 3. Model & Optimizer
    model = get_model(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    
    # 4. Training Loop
    best_auroc = 0.0
    os.makedirs("artifacts/models", exist_ok=True)
    
    for epoch in range(cfg["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, macro_auroc, aurocs = validate(model, val_loader, criterion, device)
        
        logger.info(f"Epoch {epoch+1}/{cfg['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro AUROC: {macro_auroc:.4f}")
        for label, score in aurocs.items():
            logger.info(f"  {label:<15}: {score:.4f}")
            
        # Save best
        if macro_auroc > best_auroc:
            best_auroc = macro_auroc
            assert macro_auroc > 0.0, "Refusing to save checkpoint with 0 AUROC"
            torch.save(model.state_dict(), "artifacts/models/best_model.pt")
            logger.info(f"Saved new best model with AUROC {best_auroc:.4f}")
            
    logger.info(f"Training complete. Best Macro AUROC: {best_auroc:.4f}")
    
    # Phase 1 gate check (if not subset)
    if not args.subset:
        assert best_auroc >= 0.70, \
            f"Gate not met: macro AUROC {best_auroc:.3f} < 0.70 — try lr: 1e-4 or efficientnet_b0"

if __name__ == "__main__":
    main()
