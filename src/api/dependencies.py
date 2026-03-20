"""
dependencies.py — Model and FAISS index loading at API startup.

Fixes applied (March 2026):
  - Replaced @app.on_event("startup") with the modern FastAPI lifespan context
    manager. The old on_event pattern blocks the event loop on slow I/O, which
    caused the 16-hour hanging health check.
  - Loading now runs in a thread pool via asyncio.to_thread() so the event loop
    stays responsive while large model weights load from disk.
  - Added a READY flag: /health returns 503 while loading, 200 when done.
    This prevents the load balancer from routing traffic to a half-initialized pod.
  - Added a 120-second timeout so a corrupted model file doesn't hang forever.
  - Errors during load are caught and stored; /health surfaces them in the response.

Usage in main.py:
    from src.api.dependencies import lifespan, get_predictor, get_retriever, is_ready
    app = FastAPI(lifespan=lifespan)
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
_predictor: Optional[object] = None   # RadiologistCNN
_retriever: Optional[object] = None   # RadiologyRetriever
_generator: Optional[object] = None   # ReportGenerator
_ready: bool = False
_startup_error: Optional[str] = None

LOAD_TIMEOUT_SECONDS = 120


# ── Blocking loader (runs in thread pool) ─────────────────────────────────────
def _load_all() -> tuple:
    """
    Load all heavy objects synchronously.
    Called via asyncio.to_thread() so it doesn't block the event loop.
    """
    import yaml

    with open("configs/app.yaml") as f:
        app_cfg = yaml.safe_load(f)

    logger.info("Loading image classifier...")
    from src.models.predict import RadiologistCNN
    predictor = RadiologistCNN(
        model_path=app_cfg["model_path"],
        config_path="configs/train_image.yaml",
        calib_path=app_cfg.get("calib_path", "artifacts/models/calibration.yaml"),
    )

    logger.info("Loading FAISS retriever...")
    from src.retrieval.retrieve import RadiologyRetriever
    retriever = RadiologyRetriever(
        index_path=app_cfg["faiss_index_path"],
        meta_path=app_cfg["faiss_meta_path"],
    )

    logger.info("Initialising report generator...")
    from src.generation.generate_report import ReportGenerator
    generator = ReportGenerator(
        model_name=app_cfg.get("openai_model", "gpt-5.4-mini-2026-03-17"),
        top_k=app_cfg.get("top_k", 5),
    )

    return predictor, retriever, generator


# ── Lifespan context manager ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Replaces the deprecated @app.on_event("startup") / ("shutdown") pattern.

    Startup: load model + FAISS in a thread pool with a timeout.
    Shutdown: log session stats.
    """
    global _predictor, _retriever, _generator, _ready, _startup_error

    logger.info("RadReason API starting — loading models (timeout=%ds)...", LOAD_TIMEOUT_SECONDS)

    try:
        _predictor, _retriever, _generator = await asyncio.wait_for(
            asyncio.to_thread(_load_all),
            timeout=LOAD_TIMEOUT_SECONDS,
        )
        _ready = True
        logger.info("All models loaded. API is ready.")
    except asyncio.TimeoutError:
        _startup_error = f"Model loading timed out after {LOAD_TIMEOUT_SECONDS}s"
        logger.error(_startup_error)
        # Do not re-raise — let the app start so /health can report the error
    except Exception as e:
        _startup_error = str(e)
        logger.error("Startup failed: %s", e)

    yield  # ── application runs here ──

    # Shutdown
    from src.generation.generate_report import ReportGenerator
    stats = ReportGenerator.session_stats()
    logger.info(
        "Session stats — total: %d | retries: %d (%.1f%%) | escalations: %d (%.1f%%)",
        stats["total"],
        stats["retries"],
        stats["retry_rate"] * 100,
        stats["escalations"],
        stats["escalation_rate"] * 100,
    )


# ── FastAPI dependency injectors ──────────────────────────────────────────────
def is_ready() -> bool:
    return _ready


def get_predictor():
    if not _ready:
        raise HTTPException(status_code=503, detail="Service not ready — models still loading")
    return _predictor


def get_retriever():
    if not _ready:
        raise HTTPException(status_code=503, detail="Service not ready — models still loading")
    return _retriever


def get_generator():
    if not _ready:
        raise HTTPException(status_code=503, detail="Service not ready — models still loading")
    return _generator


def get_startup_error() -> Optional[str]:
    return _startup_error