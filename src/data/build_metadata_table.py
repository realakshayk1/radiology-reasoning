"""
build_metadata_table.py — Assemble studies.csv for the OpenI IU Chest X-ray dataset.

Schema (per PRD Section 9):
  study_id, image_path, report_text, findings, impression

Rules:
  - study_id = XML filename stem (e.g. "1" for "1.xml")
  - image_path = path to primary (first) PNG in data/raw/images/
  - report_text = findings + " " + impression (combined for retrieval)
  - findings / impression = raw text from XML (empty string if missing)
  - Skip records where the primary image file doesn't exist on disk

Assertions (must all pass before committing):
  - "study_id" in studies.columns
  - all image_paths exist on disk
  - view_position (represented as image_path) never null
"""

import os
import logging
import pandas as pd
from src.data.parse_reports import parse_all_reports

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_metadata_table(
    reports_dir: str = "data/raw/reports",
    images_dir: str = "data/raw/images",
    processed_dir: str = "data/processed",
) -> pd.DataFrame:
    os.makedirs(processed_dir, exist_ok=True)

    records = parse_all_reports(reports_dir)
    logger.info(f"Building metadata table from {len(records)} parsed records...")

    rows = []
    missing_images = []

    # Target findings for multi-label classification
    FINDINGS_KEYWORDS = {
        "cardiomegaly": ["cardiomegaly"],
        "effusion": ["effusion"],
        "edema": ["edema"],
        "pneumothorax": ["pneumothorax"],
        "consolidation": ["consolidation"],
        "no_finding": ["no acute", "normal"]
    }

    def has_negation(text, keyword, window=3):
        """Simple check for negation words preceding a keyword."""
        negations = {"no", "not", "without", "negative", "none", "denies", "denied"}
        words = text.split()
        try:
            kw_idx = words.index(keyword)
            # Check window before the keyword
            start = max(0, kw_idx - window)
            for i in range(start, kw_idx):
                if words[i] in negations:
                    return True
        except ValueError:
            pass
        return False

    for rec in records:
        # Primary image = first parentImage id
        primary_id = rec["image_ids"][0]
        image_path = os.path.abspath(os.path.join(images_dir, f"{primary_id}.png"))

        if not os.path.exists(image_path):
            missing_images.append(primary_id)
            logger.warning(f"Image not found, skipping: {image_path}")
            continue

        findings = rec["findings"]
        impression = rec["impression"]
        # Combined text for retrieval and label extraction
        report_text = " ".join(filter(None, [findings, impression])).strip()
        combined_lower = report_text.lower()
        # Basic cleanup for negation check (remove punctuation)
        clean_text = combined_lower.replace(".", " ").replace(",", " ").replace(";", " ")

        # Extract binary labels
        label_dict = {}
        for label, keywords in FINDINGS_KEYWORDS.items():
            if label == "no_finding":
                # For no_finding, positive if any keyword is present
                label_dict[label] = 1 if any(kw in clean_text for kw in keywords) else 0
            else:
                # For findings, positive if keyword is present AND NOT negated
                is_positive = 0
                for kw in keywords:
                    if kw in clean_text:
                        if not has_negation(clean_text, kw):
                            is_positive = 1
                            break
                label_dict[label] = is_positive

        rows.append({
            "study_id": rec["study_id"],
            "image_path": image_path,
            "report_text": report_text,
            "findings": findings,
            "impression": impression,
            **label_dict
        })

    studies = pd.DataFrame(rows)

    if missing_images:
        logger.warning(f"Skipped {len(missing_images)} records with missing images")
        logger.warning(f"  First few missing IDs: {missing_images[:5]}")

    # Save
    out_path = os.path.join(processed_dir, "studies.csv")
    studies.to_csv(out_path, index=False)
    logger.info(f"Saved {len(studies)} rows to {out_path}")

    # ── Assertions ───────────────────────────────────────────────────────────
    assert "study_id" in studies.columns, "study_id column missing"
    assert studies["image_path"].apply(os.path.exists).all(), "Missing image files"
    assert studies["image_path"].notna().all(), "image_path must never be null"
    
    label_cols = list(FINDINGS_KEYWORDS.keys())
    for col in label_cols:
        assert col in studies.columns, f"Label column {col} missing"
        assert studies[col].isin([0, 1]).all(), f"Invalid values in {col}"
    
    # OpenI should not have uncertainty labels (-1)
    # OpenI should not have uncertainty labels (-1)
    assert (studies[label_cols] == -1).to_numpy().sum() == 0, "Unexpected -1 labels found"
    
    assert len(studies) > 0, "studies.csv is empty"
    logger.info("All assertions passed.")

    return studies


if __name__ == "__main__":
    studies = build_metadata_table()
    print(f"\nstudies.csv shape: {studies.shape}")
    print(studies.head(3).to_string())
