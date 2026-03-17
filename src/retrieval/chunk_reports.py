"""
chunk_reports.py — Chunk findings + impression text into sentence-level segments.
"""

import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def chunk_reports(studies_path="data/processed/studies.csv", out_path="data/processed/retrieval_chunks.csv"):
    if not os.path.exists(studies_path):
        logger.error(f"Studies file not found at {studies_path}. Complete Stage 1 first.")
        return

    logger.info(f"Loading studies from {studies_path}...")
    studies = pd.read_csv(studies_path)
    
    all_chunks = []
    chunk_counter = 0
    
    for _, row in studies.iterrows():
        study_id = row["study_id"]
        
        # Sections to chunk
        sections = {
            "findings": str(row.get("findings", "")),
            "impression": str(row.get("impression", ""))
        }
        
        for section_name, text in sections.items():
            if not text or text.lower() == "nan":
                continue
                
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 5:  # Skip very short fragments
                    continue
                    
                all_chunks.append({
                    "chunk_id": f"c{chunk_counter}",
                    "study_id": study_id,
                    "chunk_text": sentence,
                    "section": section_name
                })
                chunk_counter += 1
                
    chunks_df = pd.DataFrame(all_chunks)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    chunks_df.to_csv(out_path, index=False)
    
    logger.info(f"Created {len(chunks_df)} chunks from {len(studies)} studies.")
    logger.info(f"Saved to {out_path}")
    return chunks_df

if __name__ == "__main__":
    chunk_reports()
