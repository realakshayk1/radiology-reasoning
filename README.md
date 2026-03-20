# RadReason: Multimodal Chest X-ray Copilot

RadReason is a multimodal medical imaging assistant that analyzes chest X-ray images, retrieves relevant clinical evidence from historical radiology reports, and generates structured, grounded radiology summaries.

## System Architecture

Three-stage pipeline:

1. **Vision** — EfficientNet-B0 (ImageNet pretrained, multi-label sigmoid head) predicts 5 thoracic findings from a CXR image. Probabilities are calibrated via temperature scaling.
2. **Retrieval** — FAISS over sentence-level chunks of Indiana University (OpenI) radiology reports. Queries embed with `all-MiniLM-L6-v2`; Inner Product similarity.
3. **Generation** — `gpt-5.4-mini-2026-03-17` synthesizes calibrated predictions + top-5 retrieved chunks into a structured JSON report, validated against a Pydantic schema.

**Stack**: PyTorch · torchvision · FAISS · OpenAI API · FastAPI · Streamlit · Docker

---

## Performance Metrics

| Metric | Result | Target | Status |
|---|---|---|---|
| Macro AUROC (val) | 0.66 | ≥ 0.70 | ⚠️ Retraining... |
| JSON Schema Validity | 100% | ≥ 95% | ✅ |
| Retrieval Top-3 Hit Rate | 0.2222 | > random | ✅ |
| End-to-End Latency | ~4.4s | < 10s | ✅ |

> **Note on AUROC**: The existing checkpoint was trained with `epochs: 5` (insufficient).
> Retraining with `epochs: 25` + cosine LR scheduler is expected to reach ≥ 0.70.
> See `configs/train_image.yaml` — backbone and LR are already set correctly.

---

## Quick Start

### Prerequisites
- Python 3.10
- `OPENAI_API_KEY` in `.env`
- OpenI IU Chest X-ray dataset downloaded to `data/raw/` (see Data Setup below)

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...
```

### 3. Build the data pipeline (requires raw data)
```bash
# Parse XML reports → studies.csv
python -m src.data.build_metadata_table

# Build patient-level splits
python -m src.data.build_splits

# Chunk reports for retrieval
python -m src.retrieval.chunk_reports

# Build FAISS index
python -m src.retrieval.build_faiss
```

### 4. Train the image classifier
```bash
# Validate pipeline on 50 samples first
python -m src.models.train_image --subset 50

# Full training (25 epochs, ~2-4h on CPU)
python -m src.models.train_image
```

### 5. Calibrate
```bash
python -m src.models.calibrate
# Saves: artifacts/models/calibration.yaml
# Saves: artifacts/figures/reliability_*.png
```

### 6. Run the API
```bash
uvicorn src.api.main:app --reload
# Health check: curl http://localhost:8000/health
# Metrics:      curl http://localhost:8000/metrics
```

### 7. Run the UI
```bash
streamlit run src/ui/streamlit_app.py
```

### Docker (full stack)
```bash
docker-compose up --build
# API:  http://localhost:8000
# UI:   http://localhost:8501
```
The UI container waits for `/health` to return 200 before starting.

---

## Data Setup

Dataset: **OpenI IU Chest X-ray** (Indiana University, freely available, no credentialing)

```
# Download images
wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
tar -xzf NLMCXR_png.tgz -C data/raw/images/

# Download XML reports
wget https://openi.nlm.nih.gov/imgs/collections/ecgen-radiology.tgz
tar -xzf ecgen-radiology.tgz -C data/raw/reports/
```

Never commit `data/raw/` — it is gitignored.

---

## Project Structure

```
src/
  data/         parse_reports, build_splits, preprocess_images, build_metadata_table
  models/       image_baseline, train_image, calibrate, predict
  retrieval/    chunk_reports, build_faiss, retrieve, evaluate_retrieval
  generation/   prompts, generate_report
  api/          main, schemas, dependencies
  ui/           streamlit_app
  utils/        metrics, paths, seed
configs/        train_image.yaml, retrieval.yaml
artifacts/      models/, faiss/, figures/, results/
data/           raw/ (gitignored), processed/, splits/
scripts/        check_generation_validity.py, evaluate_model.py, debug_*.py
```

---

## Validation Scripts

```bash
# Check generation schema validity (requires OPENAI_API_KEY + studies.csv + FAISS index)
python scripts/check_generation_validity.py

# Evaluate image classifier AUROC on val set
python scripts/evaluate_model.py

# Evaluate FAISS retrieval hit rate
python -m src.retrieval.evaluate_retrieval
```

---

*For research use only. Not a medical device. Not validated for clinical use.*