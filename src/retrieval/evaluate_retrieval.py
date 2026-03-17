"""
evaluate_retrieval.py — Evaluate retrieval hit rate on the validation set.

Metric: Top-k Hit Rate
For each study in the validation set, we use its raw report as a query.
We check if any of the top-k retrieved chunks belong to the same study.
"""

import pandas as pd
import numpy as np
from src.retrieval.retrieve import RadiologyRetriever
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def evaluate_hit_rate(k_values=[1, 3, 5]):
    # 1. Load Data
    studies = pd.read_csv("data/processed/studies.csv")
    val_splits = pd.read_csv("data/splits/val.csv")
    val_df = studies[studies["study_id"].isin(val_splits["study_id"])].reset_index(drop=True)
    
    # 2. Init Retriever
    try:
        retriever = RadiologyRetriever()
    except Exception as e:
        logger.error(f"Could not init retriever: {e}")
        return

    # 3. Evaluate
    logger.info(f"Evaluating hit rate on {len(val_df)} validation studies...")
    
    hits = {k: 0 for k in k_values}
    
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        study_id = row["study_id"]
        query = row["report_text"]
        
        if not isinstance(query, str) or not query.strip():
            continue
            
        # Search for top max(k)
        max_k = max(k_values)
        results = retriever.search(query, top_k=max_k + 1) # +1 because it might retrieve itself
        
        # Check if any of the top k results are from the SAME study_id
        for k in k_values:
            top_k_ids = [res["study_id"] for res in results[:k]]
            if study_id in top_k_ids:
                hits[k] += 1
                
    # 4. Results
    print("\nRetrieval Evaluation (Top-k Hit Rate):")
    random_baseline = 1 / retriever.index.ntotal
    hit_rate_5 = hits[5] / len(val_df)
    
    for k in k_values:
        rate = hits[k] / len(val_df)
        print(f"  Hit@{k}: {rate:.4f} ({hits[k]}/{len(val_df)})")
        
    print(f"  Random Baseline: {random_baseline:.6f}")
    
    # Gate Assertion
    assert hit_rate_5 > random_baseline, \
        f"Retrieval hit rate {hit_rate_5:.3f} does not beat random {random_baseline:.3f}"

if __name__ == "__main__":
    evaluate_hit_rate()
