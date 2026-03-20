"""
routers/health.py — Health and readiness endpoints.

GET /health
  - 200 {"status": "ok", "ready": true}   when fully loaded
  - 503 {"status": "starting", ...}        while models are loading
  - 503 {"status": "error", ...}           if startup failed

This prevents load balancers from routing traffic to a half-initialised pod,
and gives operators a clear signal when startup fails vs is still in progress.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.dependencies import is_ready, get_startup_error

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    if is_ready():
        return {"status": "ok", "ready": True}

    err = get_startup_error()
    if err:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "ready": False, "detail": err},
        )

    return JSONResponse(
        status_code=503,
        content={"status": "starting", "ready": False, "detail": "Models still loading"},
    )