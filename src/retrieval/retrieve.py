"""
retrieve.py — Search the FAISS index for relevant evidence chunks.
"""

import os
import pickle
import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class RadiologyRetriever:
    """Handles embedding queries and searching the FAISS index."""
    
    def __init__(self, index_path="artifacts/faiss/index.bin", 
                 meta_path="artifacts/faiss/chunks_meta.pkl",
                 config_path="configs/retrieval.yaml"):
        
        # 1. Load Config
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = {"model_name": "all-MiniLM-L6-v2", "top_k": 5}
            
        # 2. Load Model
        self.model_name = self.cfg.get("model_name", "all-MiniLM-L6-v2")
        logger.info(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        # 3. Load Index & Metadata
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"FAISS index or metadata not found at {index_path}. Run Stage 5 first.")
            
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
            
        logger.info(f"Retriever initialized with {self.index.ntotal} chunks.")

    def search(self, query: str, top_k: int = None):
        """Returns the top-k chunks most similar to the query."""
        if top_k is None:
            top_k = self.cfg.get("top_k", 5)
            
        # 1. Embed and normalize query
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        # 2. Search
        scores, indices = self.index.search(query_emb, top_k)
        
        # 3. Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            
            chunk = self.metadata.iloc[idx].to_dict()
            chunk["score"] = float(score)
            results.append(chunk)
            
        return results

if __name__ == "__main__":
    # Test retrieval
    try:
        retriever = RadiologyRetriever()
        query = "moderate cardiomegaly and pleural effusion"
        logger.info(f"Query: {query}")
        
        hits = retriever.search(query, top_k=3)
        for i, hit in enumerate(hits):
            print(f"\nHit {i+1} (Score: {hit['score']:.4f}):")
            print(f"  [{hit['section']}] {hit['chunk_text']}")
    except FileNotFoundError:
        logger.warning("FAISS index not found. Skipping test.")
