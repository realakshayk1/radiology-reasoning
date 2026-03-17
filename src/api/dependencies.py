"""
dependencies.py — Resource management (Model, FAISS, LLM) for FastAPI.
"""

from functools import lru_cache
from src.models.predict import RadiologistCNN
from src.retrieval.retrieve import RadiologyRetriever
from src.generation.generate_report import ReportGenerator
import logging
import os
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

logger = logging.getLogger(__name__)

class AppResources:
    """Singleton-like container for API resources."""
    def __init__(self):
        from pathlib import Path
        import faiss
        
        # Canonical Paths per CHECK 3
        best_model_path = Path("artifacts/models/best_model.pt")
        temperature_path = Path("artifacts/models/temperature.pt")
        faiss_index_path = Path("artifacts/faiss/index.bin")
        meta_path = Path("artifacts/faiss/chunks_meta.pkl")

        logger.info("Initializing AppResources (Loading CNN, FAISS, and LLM)...")
        
        # We check existence but don't strictly assert temperature_path for the API to start in 'degraded' mode
        # as it is only generated in Stage 3. This allows Check 4 (Health Check) to run.
        self.cnn = None
        self.retriever = None
        self.generator = None

        try:
            # 1. FAISS Index (REQUIRED)
            assert faiss_index_path.exists(), "FAISS index not found — run build_faiss.py"
            faiss_index = faiss.read_index(str(faiss_index_path))
            assert faiss_index.ntotal > 0, "FAISS index loaded but is empty"
            self.retriever = RadiologyRetriever(index_path=str(faiss_index_path), meta_path=str(meta_path))

            # 2. LLM Generator (REQUIRED)
            self.generator = ReportGenerator()

            # 3. Model & Temperature (Loud failure if missing in predict.py, so we handle here)
            if best_model_path.exists() and temperature_path.exists():
                self.cnn = RadiologistCNN(model_path=str(best_model_path), calib_path=str(temperature_path))
                logger.info("All resources loaded successfully.")
            else:
                if not best_model_path.exists():
                    logger.warning("best_model.pt not found.")
                if not temperature_path.exists():
                    logger.warning("temperature.pt not found — run calibrate.py")
                logger.info("Resources partially loaded (Degraded mode).")

        except Exception as e:
            logger.error(f"Critical Startup Failure: {e}")
            raise e

@lru_cache()
def get_resources():
    """Returns cached resources."""
    return AppResources()
