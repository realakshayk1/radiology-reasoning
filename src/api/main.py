"""
main.py — FastAPI application logic for RadReason.
"""

import os
from dotenv import load_dotenv
load_dotenv() # Load OPENAI_API_KEY from .env

from fastapi import FastAPI, Depends, HTTPException
from src.api.schemas import ReportRequest, ReportResponse, Prediction, RetrievalResult
from src.api.dependencies import (
    lifespan,
    get_predictor,
    get_retriever,
    get_generator,
    is_ready,
    get_startup_error,
)
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="RadReason API", version="0.2.0", lifespan=lifespan)

@app.get("/health")
def health_check():
    """Verifies that all models and indices are loaded."""
    if is_ready():
        return {"status": "healthy", "message": "All models loaded."}
    
    error = get_startup_error()
    if error:
        raise HTTPException(status_code=500, detail=f"Startup failed: {error}")
        
    return {"status": "loading", "message": "Models are still being loaded into memory."}

@app.post("/predict", response_model=ReportResponse)
def predict_report(
    request: ReportRequest, 
    cnn=Depends(get_predictor),
    retriever=Depends(get_retriever),
    generator=Depends(get_generator)
):
    """Full pipeline: Predict -> Retrieve -> Generate."""
    # 1. Image Classification
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found at {request.image_path}")
        
    preds_dict = cnn.predict(request.image_path)
    predictions = [Prediction(label=k, prob=v) for k, v in preds_dict.items()]
    
    # 2. Evidence Retrieval
    # Query by the top predicted findings or clinical indication if provided
    query = request.clinical_indication or ""
    top_findings = [f"{k}" for k, v in preds_dict.items() if v >= 0.4]
    if top_findings:
        query += " " + " ".join(top_findings)
        
    hits = retriever.search(query.strip(), top_k=5)
    retrieved_evidence = [
        RetrievalResult(chunk_id=str(h["chunk_id"]), text=h["chunk_text"], score=h["score"], study_id=str(h["study_id"]))
        for h in hits
    ]
    
    # 3. Report Generation
    summary = generator.generate(preds_dict, hits)
    
    return ReportResponse(
        predictions=predictions,
        retrieved_evidence=retrieved_evidence,
        summary=summary
    )

@app.get("/metrics")
def get_metrics():
    """Returns model performance metrics (AUROC)."""
    import json
    results_path = "artifacts/results/image_baseline_val.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return {"message": "No metrics available yet."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
