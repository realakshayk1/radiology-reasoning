"""
build_faiss.py — Embed chunks and build a FAISS index for retrieval.

Steps:
1. Load retrieval_chunks.csv
2. Embed with all-MiniLM-L6-v2
3. Build FAISS IndexFlatIP (Inner Product)
4. Save index and metadata
"""

import os
import yaml
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def build_faiss(chunks_path="data/processed/retrieval_chunks.csv", 
                config_path="configs/retrieval.yaml",
                out_dir="artifacts/faiss"):
    
    if not os.path.exists(chunks_path):
        logger.error(f"Chunks file not found at {chunks_path}. Run Stage 4 first.")
        return

    # 1. Load config
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"model_name": "all-MiniLM-L6-v2"}
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)

    # 2. Load Chunks
    logger.info(f"Loading chunks from {chunks_path}...")
    df = pd.read_csv(chunks_path)
    # Ensure no NaN texts
    df = df.dropna(subset=["chunk_text"]).reset_index(drop=True)
    texts = df["chunk_text"].tolist()

    # 3. Embed
    model_name = cfg.get("model_name", "all-MiniLM-L6-v2")
    logger.info(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    logger.info(f"Embedding {len(texts)} chunks (this may take a minute on CPU)...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # L2 normalize for Inner Product similarity (equivalent to Cosine Similarity)
    faiss.normalize_L2(embeddings)
    
    # 4. Build Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # 5. Save
    index_path = os.path.join(out_dir, "index.bin")
    faiss.write_index(index, index_path)
    
    meta_path = os.path.join(out_dir, "chunks_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(df, f)
        
    logger.info(f"FAISS index built with {index.ntotal} vectors.")
    logger.info(f"Saved index to {index_path}")
    logger.info(f"Saved metadata to {meta_path}")

if __name__ == "__main__":
    build_faiss()
