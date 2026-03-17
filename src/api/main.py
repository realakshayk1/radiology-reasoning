"""
main.py — FastAPI application logic for RadReason.
"""

import os
from fastapi import FastAPI, Depends, HTTPException
from src.api.schemas import ReportRequest, ReportResponse, Prediction, RetrievalResult
from src.api.dependencies import get_resources, AppResources
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="RadReason API", version="0.1.0")

# Trigger early loading for Check 3 assertions
@app.on_event("startup")
def startup_event():
    get_resources()

@app.get("/health")
def health_check(resources: AppResources = Depends(get_resources)):
    """Verifies that all models and indices are loaded."""
    status = {
        "cnn_loaded": resources.cnn is not None,
        "retriever_loaded": resources.retriever is not None,
        "generator_loaded": resources.generator is not None
    }
    if all(status.values()):
        return {"status": "healthy", **status}
    else:
        return {"status": "degraded", **status}

@app.post("/predict", response_model=ReportResponse)
def predict_report(request: ReportRequest, resources: AppResources = Depends(get_resources)):
    """Full pipeline: Predict -> Retrieve -> Generate."""
    if not resources.cnn or not resources.retriever or not resources.generator:
        raise HTTPException(status_code=503, detail="API resources not fully loaded.")
        
    # 1. Image Classification
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found at {request.image_path}")
        
    preds_dict = resources.cnn.predict(request.image_path)
    predictions = [Prediction(label=k, prob=v) for k, v in preds_dict.items()]
    
    # 2. Evidence Retrieval
    # Query by the top predicted findings or clinical indication if provided
    query = request.clinical_indication or ""
    top_findings = [f"{k}" for k, v in preds_dict.items() if v >= 0.4]
    if top_findings:
        query += " " + " ".join(top_findings)
        
    hits = resources.retriever.search(query.strip(), top_k=5)
    retrieved_evidence = [
        RetrievalResult(chunk_id=h["chunk_id"], text=h["chunk_text"], score=h["score"], study_id=h["study_id"])
        for h in hits
    ]
    
    # 3. Report Generation
    summary = resources.generator.generate(preds_dict, hits)
    
    return ReportResponse(
        predictions=predictions,
        retrieved_evidence=retrieved_evidence,
        summary=summary
    )

@app.get("/metrics")
def get_metrics():
    """Returns model performance metrics (AUROC)."""
    # In a real app, this would load from artifacts/results/
    import json
    results_path = "artifacts/results/image_baseline_val.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return {"message": "No metrics available yet."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
