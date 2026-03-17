# RadReason: Multimodal Chest X-ray Copilot

RadReason is a multimodal medical imaging assistant that analyzes chest X-ray images, retrieves relevant clinical evidence from historical reports, and generates structured, grounded radiology summaries.

## System Architecture

The system follows a three-stage pipeline:
1.  **Vision**: A calibrated DenseNet121 classifier predicts thoracic findings from CXR images.
2.  **Retrieval**: A FAISS-based RAG system retrieves relevant evidence segments from a database of Indiana University radiology reports.
3.  **Generation**: A GPT-4o model synthesizes the predictions and evidence into a structured JSON report, enforced by Pydantic.

**Stack**: PyTorch | FAISS | GPT-4o | FastAPI | Streamlit | Docker

## Quick Start

### Local Development
1.  **Clone and Setup**:
    ```bash
    git clone [repo-url]
    cd radiology-reasoning
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```
2.  **Environment Variables**:
    Create a `.env` file and add your `OPENAI_API_KEY`.
3.  **Run API**:
    ```bash
    uvicorn src.api.main:app --reload
    ```
4.  **Run UI**:
    ```bash
    streamlit run src.ui/streamlit_app.py
    ```

### Docker
Start the entire stack with a single command:
```bash
docker-compose up --build
```

## Performance Metrics

| Metric | Result | Target |
|---|---|---|
| Macro AUROC | 0.66 | 0.70 |
| JSON Schema Validity | > 95% | 95% |
| Retrieval Top-3 Hit Rate | PASS | > Random |

## Project Structure

- `src/`: Core source code (models, retrieval, generation, API, UI).
- `data/`: Dataset splits and processed metadata.
- `artifacts/`: Model checkpoints, FAISS index, and results.
- `notebooks/`: EDA and evaluation experiments.
- `scripts/`: Utility scripts for training and validation.
- `configs/`: Hyperparameter settings.

---
*For research use only. Not a medical device.*
