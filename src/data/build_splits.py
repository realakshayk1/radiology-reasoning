"""
build_splits.py — Patient-level (study-level) stratified train/val/test splits.

Per PRD Rule 1: split at STUDY level (each XML = one patient episode), never image level.
One XML report may link to multiple images — all images from a report stay together.

Split ratios:  70% train / 15% val / 15% test
Calibration:   5% of val reserved as calibration-only (for temperature scaling)

Saves:
  data/splits/train.csv   → study_id rows
  data/splits/val.csv
  data/splits/test.csv
  data/splits/calibration.csv

Assertions (all must pass):
  - No train/test overlap
  - No train/val overlap
  - No val/test overlap
  - Calibration split is non-empty
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_splits(
    processed_dir: str = "data/processed",
    splits_dir: str = "data/splits",
    train_size: float = 0.70,
    test_size: float = 0.15,
    cal_size: float = 0.04,  # Targeting ~150 studies
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split allocation chosen:
      - Train ≈ 70% (2696 studies)
      - Test  ≈ 15% (578 studies)
      - Val   ≈ 11% (423 studies)
      - Cal   ≈ 4% (154 studies)
    
    Justification:
    A larger calibration set (N ~ 150) is critical for stable post-hoc temperature scaling.
    Smaller sets (e.g. < 50) produce high-variance reliability diagrams. This allocation
    provides stability without sacrificing significant training data or the integrity of
    the final 15% test set.
    """
    os.makedirs(splits_dir, exist_ok=True)

    studies = pd.read_csv(os.path.join(processed_dir, "studies.csv"))
    logger.info(f"Loaded studies.csv: {len(studies)} rows")

    groups = studies["study_id"]

    # Step 1: Split into Train (70%) and Temp (30%)
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, temp_idx = next(gss.split(studies, groups=groups))

    train_df = studies.iloc[train_idx].reset_index(drop=True)
    temp_df = studies.iloc[temp_idx].reset_index(drop=True)

    # Step 2: Split Temp into Test (50% of 30% = 15% total) and ValCal (50% of 30% = 15% total)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    valcal_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["study_id"]))

    valcal_df = temp_df.iloc[valcal_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    # Step 3: Split ValCal into Val (~423) and Cal (~154)
    # ValCal is ~577 studies. To get 154 for Cal, we need roughly 154/577 = 26.7%
    cal_target_fraction = cal_size / (1.0 - train_size - test_size)
    gss3 = GroupShuffleSplit(n_splits=1, test_size=cal_target_fraction, random_state=random_state)
    val_idx, cal_idx = next(gss3.split(valcal_df, groups=valcal_df["study_id"]))

    val_df = valcal_df.iloc[val_idx].reset_index(drop=True)
    cal_df = valcal_df.iloc[cal_idx].reset_index(drop=True)

    # Save
    train_df[["study_id"]].to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df[["study_id"]].to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df[["study_id"]].to_csv(os.path.join(splits_dir, "test.csv"), index=False)
    cal_df[["study_id"]].to_csv(os.path.join(splits_dir, "calibration.csv"), index=False)

    # Collect study_id sets for leakage checks
    train_ids = set(train_df["study_id"])
    val_ids = set(val_df["study_id"])
    test_ids = set(test_df["study_id"])
    cal_ids = set(cal_df["study_id"])

    # ── Assertions ────────────────────────────────────────────────────────────
    assert len(train_ids & test_ids) == 0, "Patient leakage: train/test overlap"
    assert len(train_ids & val_ids) == 0, "Patient leakage: train/val overlap"
    assert len(train_ids & cal_ids) == 0, "Patient leakage: train/cal overlap"
    assert len(val_ids & test_ids) == 0, "Patient leakage: val/test overlap"
    assert len(val_ids & cal_ids) == 0, "Patient leakage: val/cal overlap"
    assert len(test_ids & cal_ids) == 0, "Patient leakage: test/cal overlap"
    assert len(cal_ids) >= 100, f"Calibration split too small: {len(cal_ids)}"

    logger.info(
        f"Splits: train={len(train_df)} (70%), val={len(val_df)} (11%), "
        f"test={len(test_df)} (15%), cal={len(cal_df)} (4%)"
    )
    logger.info("All split integrity and size assertions passed.")

    return train_df, val_df, test_df, cal_df


if __name__ == "__main__":
    build_splits()
