"""
check_generation_validity.py — Validate generation schema over 20 real samples.
"""

import os
import pandas as pd
from src.retrieval.retrieve import RadiologyRetriever
from src.generation.generate_report import ReportGenerator
from src.api.schemas import RadiologyReport
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found. Cannot run generation check.")
        return

    # 1. Load Data
    studies = pd.read_csv("data/processed/studies.csv")
    val_splits = pd.read_csv("data/splits/val.csv")
    val_df = studies[studies["study_id"].isin(val_splits["study_id"])].sample(20, random_state=42).reset_index(drop=True)

    # 2. Init Retriever & Generator
    retriever = RadiologyRetriever()
    generator = ReportGenerator()

    # 3. Test Generations
    success_count = 0
    total = 20
    
    # Mock some predictions for the findings list (using labels from studies.csv)
    LABEL_COLS = ["cardiomegaly", "effusion", "edema", "pneumothorax", "consolidation"]

    for i, row in val_df.iterrows():
        study_id = row["study_id"]
        # Use real labels from the dataset for realistic findings
        preds_dict = {col: float(row[col]) for col in LABEL_COLS}
        
        # Retrieve chunks
        query = row["report_text"]
        hits = retriever.search(query, top_k=5)
        
        logger.info(f"[{i+1}/{total}] testing study_id: {study_id}")
        
        try:
            report = generator.generate(preds_dict, hits)
            
            # Assertions
            assert isinstance(report.escalate, bool), "escalate must be bool"
            assert report.confidence in ("high", "moderate", "low"), f"invalid confidence: {report.confidence}"
            
            # If it reached here without Exception and passed assertions, it's valid
            # Fix: check report.caution instead of report.findings for TECHNICAL ERROR
            if "TECHNICAL ERROR" not in report.caution:
                success_count += 1
                logger.info(f"  Result: PASS (Confidence: {report.confidence})")
            else:
                logger.warning(f"  Result: FAIL (Fallback report returned)")
                
        except Exception as e:
            logger.error(f"  Result: ERROR ({e})")
            # If it's a validation error, print some details
            if "validation error" in str(e).lower():
                print(f"DEBUG: {e}")

    validity_rate = success_count / total
    print(f"\nFinal Schema Validity Rate: {validity_rate:.2%}")

    assert validity_rate >= 0.95, f"Schema validity {validity_rate:.2%} below 95% target"
    print("[PASS] Check 2: Generation schema validity verified.")

if __name__ == "__main__":
    main()
