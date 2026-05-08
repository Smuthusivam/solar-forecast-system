# Backend image: FastAPI + ML core (XGBoost, LightGBM, Prophet)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: libgomp for LightGBM/XGBoost, build tools for prophet/cmdstanpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first to leverage Docker layer caching
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

# Copy backend + ml_core (ml_core is imported by backend via sys.path)
COPY backend /app/backend
COPY ml_core /app/ml_core

# Storage volume mount points (uploads, exports, sqlite fallback)
RUN mkdir -p /app/backend/storage/db \
             /app/backend/storage/uploads \
             /app/backend/storage/exports \
             /app/backend/storage/sessions \
             /app/ml_core/saved_models

WORKDIR /app/backend

EXPOSE 8000

# Healthcheck hits the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Run uvicorn with 2 workers — bump for production load
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
