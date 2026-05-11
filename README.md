# Solar Irradiance Forecasting System

A full-stack machine learning application for solar irradiance prediction. Upload any solar or weather CSV dataset, automatically detect columns using Claude AI, preprocess and clean the data, train XGBoost and LightGBM models, detect and correct anomalies, and visualise forecast results — all through a modern web interface.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [ML Pipeline](#ml-pipeline)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Deployment](#deployment)

---

## Overview

This system accepts solar irradiance datasets in CSV format and runs a complete forecasting pipeline:

1. **Upload** — parse the CSV and auto-detect columns using Claude AI
2. **Preprocess** — clean, fill gaps, remove duplicates, and clip impossible values
3. **Analyse** — inspect data quality, missing values, distributions, and patterns
4. **Forecast** — train XGBoost and LightGBM models and evaluate on a held-out test set
5. **Detect & Correct** — find anomalies in the irradiance series and correct them using AI
6. **Export** — download the preprocessed dataset as CSV

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Docker Network                      │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │   Frontend   │    │   Backend    │    │ PostgreSQL│  │
│  │  React + Vite│───▶│  FastAPI     │───▶│  Port     │  │
│  │  nginx:443   │    │  Port 8000   │    │  5432     │  │
│  └──────────────┘    └──────────────┘    └───────────┘  │
│         │                   │                            │
│         │ /api/*            │ ml_core/                   │
│         └───────────────────┘                            │
└─────────────────────────────────────────────────────────┘
```

- **Frontend** — React SPA served by nginx, which reverse-proxies `/api/*` to the backend
- **Backend** — FastAPI application running with Uvicorn (2 workers)
- **Database** — PostgreSQL for storing forecast runs, datasets, and history
- **ML Core** — XGBoost + LightGBM pipeline, shared between backend and worker processes

---

## Features

### Data Upload & Preprocessing
- Drag-and-drop CSV upload with support for any solar or weather dataset format
- Auto-detects column mappings (GHI, temperature, humidity, wind speed, cloud cover) using Claude AI
- Handles NSRDB multi-row header format automatically
- Fills gaps up to 6 hours by linear interpolation; larger gaps by forward-fill
- Removes duplicate timestamps and clips physically impossible values (0–1300 W/m²)
- Detects and raises errors for empty or mostly-missing GHI columns

### Data Quality Dashboard
- Per-column statistics: missing count, missing %, unique values, min/max/mean/std
- Dataset health overview: duplicate rows, missing hours, missing cells
- Irradiance distribution statistics
- Pattern analysis: hourly, daily, weekly, and monthly irradiance averages

### Machine Learning Forecast
- Trains XGBoost and LightGBM simultaneously on a time-series train/test split
- Features: hour, day of year, month, cyclical encodings, lag features (1h, 2h, 24h, 48h, 168h), rolling statistics, weather interaction features
- Selects the best model by test RMSE
- Evaluates on held-out test set with RMSE, MAE, and R² metrics
- Optionally generates future forecasts beyond the last known data point
- Stores all runs and forecast points to the database for history replay

### Anomaly Detection & AI Correction
- Detects anomalies using Z-score, IQR, and rolling window methods
- AI-powered correction using Claude Sonnet — batches 20 anomalies per API call, 8 concurrent batches
- Falls back to linear interpolation for low-severity anomalies or when AI is unavailable
- Hard cap of 120 AI-corrected items to complete within 2 minutes
- Before-and-after model comparison showing improvement in RMSE, MAE, and R²

### History & Export
- Full run history with searchable, sortable table
- Replay any past forecast with predicted vs actual charts
- Export the preprocessed dataset as CSV

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, React Router 7, Recharts, Tailwind CSS 4, Vite |
| Backend | FastAPI, Uvicorn, psycopg 3, Pydantic v2 |
| Database | PostgreSQL 16 |
| ML Models | XGBoost 2.0, LightGBM 4.3, scikit-learn |
| AI | Anthropic Claude (column detection + anomaly correction) |
| Infrastructure | Docker, Docker Compose, nginx |
| Data | pandas, numpy |

---

## Getting Started

### Prerequisites

- Docker 24+ and Docker Compose v2
- An Anthropic API key — get one at [console.anthropic.com](https://console.anthropic.com)
- 4 GB RAM minimum (8 GB recommended for large datasets)

### Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd solar-forecast-system

# 2. Create your environment file
cp .env.example .env

# 3. Fill in required values
#    - ANTHROPIC_API_KEY
#    - POSTGRES_PASSWORD

# 4. Build and start all services
docker compose up -d --build

# 5. Open the app
open http://localhost
```

The first build takes 3–5 minutes. Subsequent starts are much faster.

### Verify Everything is Running

```bash
docker compose ps          # all three services should show "healthy" or "running"
curl http://localhost/health  # should return {"status":"ok"}
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values below.

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key for column detection and anomaly correction |
| `POSTGRES_PASSWORD` | Yes | — | PostgreSQL password |
| `POSTGRES_USER` | No | `solar` | PostgreSQL username |
| `POSTGRES_DB` | No | `solar_forecast` | PostgreSQL database name |
| `APP_ENV` | No | `production` | `production` or `development` |
| `ALLOWED_ORIGINS` | No | `http://localhost` | Comma-separated CORS origins |
| `DATABASE_URL` | Yes | — | Full PostgreSQL connection string, e.g. `postgresql://user:pass@host:5432/db` |
| `TEST_DATABASE_URL` | No | — | Separate PostgreSQL DB for running tests — never the same as `DATABASE_URL` |
| `DB_POOL_MIN` | No | `1` | Minimum connections in the psycopg pool |
| `DB_POOL_MAX` | No | `10` | Maximum connections in the psycopg pool |
| `VITE_API_BASE_URL` | No | `` (empty) | Leave empty for nginx proxy; set to backend URL for direct access |

---

## ML Pipeline

### Feature Engineering

Each row in the dataset is enriched with the following features before training:

| Feature | Description |
|---|---|
| `hour`, `month`, `day_of_week`, `day_of_year` | Calendar features |
| `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos` | Cyclical encodings (prevents 23:00 → 00:00 discontinuity) |
| `is_daytime` | Binary flag for hours 6–20 |
| `lag_1h`, `lag_2h`, `lag_24h`, `lag_48h`, `lag_168h` | Lagged irradiance values |
| `rolling_mean_3h`, `rolling_mean_24h` | Short and long rolling averages |
| `rolling_std_24h`, `rolling_max_24h` | Rolling variability and peak |
| `cloud_clearness` | `1 - cloud_cover / 100` (if cloud column present) |
| `temp_humidity_ratio` | Temperature / humidity interaction (if both present) |

### Train / Test Split

The dataset is split chronologically — no shuffling. Earlier data trains the model; the most recent slice evaluates it. This simulates real-world forecasting where you never train on future data.

### Model Selection

Both XGBoost and LightGBM are trained on the same features. The model with the lower test RMSE is selected as the best model and used for future forecast generation.

---

## API Reference

All endpoints are prefixed with `/api`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload CSV, detect columns, preprocess |
| `POST` | `/api/forecast/run` | Train models and return forecast results |
| `GET` | `/api/anomaly` | Detect anomalies in the uploaded dataset |
| `POST` | `/api/correction/run/{session_id}` | Detect and AI-correct anomalies |
| `GET` | `/api/correction/export/{session_id}` | Download corrected dataset as CSV |
| `GET` | `/api/export/csv` | Download preprocessed dataset as CSV |
| `GET` | `/api/history` | List all past forecast runs |
| `GET` | `/api/history/{run_id}/points` | Get forecast points for a past run |
| `GET` | `/health` | Service health check |

Full interactive API docs are available at `http://localhost:8000/docs` (Swagger UI).

---

## Project Structure

```
solar-forecast-system/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, CORS, middleware
│   │   ├── database.py          # psycopg3 connection pool, DDL init, get_db
│   │   ├── models/
│   │   │   ├── dataset.py       # Dataset dataclass
│   │   │   ├── forecast_run.py  # ForecastRun dataclass
│   │   │   ├── forecast_point.py# ForecastPoint dataclass
│   │   │   └── schemas.py       # Pydantic request/response schemas
│   │   ├── crud/
│   │   │   ├── dataset.py       # save_dataset
│   │   │   └── forecast.py      # save/get forecast runs and points
│   │   ├── routers/
│   │   │   ├── upload.py        # CSV upload and session management
│   │   │   ├── forecast.py      # Model training and forecasting
│   │   │   ├── anomaly.py       # Anomaly detection
│   │   │   ├── correction.py    # AI anomaly correction
│   │   │   ├── history.py       # Run history
│   │   │   └── export.py        # CSV export
│   │   └── services/
│   │       ├── ai_detector.py   # Claude column detection
│   │       └── anomaly_corrector.py  # Claude anomaly correction
│   ├── storage/
│   │   ├── sessions/            # Disk-backed session pickles (TTL 2h)
│   │   └── exports/             # Exported CSV files
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Upload.jsx       # Upload, preprocessing, EDA
│   │   │   ├── Dashboard.jsx    # Forecast results and charts
│   │   │   ├── AnomalyReport.jsx # Anomaly detection and correction
│   │   │   ├── Forecast.jsx     # Future forecast view
│   │   │   ├── Compare.jsx      # Model comparison
│   │   │   └── History.jsx      # Past run history
│   │   ├── components/
│   │   │   └── ui.jsx           # Shared UI components
│   │   └── services/
│   │       ├── api.js           # Axios API client
│   │       └── forecastState.js # Cross-page forecast state
│   ├── nginx.conf               # nginx reverse proxy config
│   └── Dockerfile
├── ml_core/
│   ├── pipeline.py              # XGBoost + LightGBM training pipeline
│   ├── preprocessing.py         # CSV parsing, cleaning, feature prep
│   ├── feature_engineering.py   # Feature generation
│   ├── anomaly.py               # Anomaly detection algorithms
│   └── evaluate.py              # RMSE, MAE, R² computation
├── docker-compose.yml
├── .env.example
└── DEPLOY.md                    # Production deployment guide
```

---

## Deployment

For full production deployment instructions including SSL setup, firewall configuration, and Docker volume management, see [DEPLOY.md](DEPLOY.md).

### HTTPS

The system supports HTTPS via Let's Encrypt. Certificates are mounted into the nginx container as a read-only volume. HTTP requests are automatically redirected to HTTPS.

### Session Management

User sessions (uploaded datasets) are stored as pickle files on a shared Docker volume with a 2-hour TTL. Sessions and their associated exports are automatically deleted when they expire. This ensures no user data persists on the server longer than necessary.

### Database

PostgreSQL is required. The backend uses psycopg 3 with a connection pool and creates all tables automatically on startup via `init_db()` — no migration tool needed for additive schema changes. Set `DATABASE_URL` in `.env` before starting the stack.

---

## Acknowledgements

Built as a Final Year Project. Uses the [Anthropic Claude API](https://www.anthropic.com) for intelligent column detection and AI-powered anomaly correction.
