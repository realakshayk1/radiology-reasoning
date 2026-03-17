# Dockerfile for RadReason API

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data needed for inference
# In production, data/raw should be mounted or retrieved from S3
COPY src/ src/
COPY artifacts/ artifacts/
COPY configs/ configs/
COPY data/processed/ data/processed/
COPY data/splits/ data/splits/

# Set environment variables
ENV PYTHONPATH=/app
ENV API_URL=http://localhost:8000

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
