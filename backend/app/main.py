"""
main.py — FastAPI application entry point.

Responsibilities:
  - Create the FastAPI app instance
  - Register all routers under /api
  - Configure CORS for React frontend (dev + production)
  - Run init_db() on startup so tables always exist
  - Expose GET /health for Railway/Render health checks
  - Add request logging middleware for debugging

Environment variables:
  ALLOWED_ORIGINS   Comma-separated list of allowed CORS origins.
                    Defaults to localhost:5173 (Vite dev server).
                    In production set to your Railway frontend URL.

  APP_ENV           "development" | "production"
                    Controls debug mode and log verbosity.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database import db_is_online, init_db

# ── Routers (imported here, built in later steps) ─────────────────────────────
from app.routers import anomaly, export, forecast, history, upload

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

APP_ENV = os.getenv("APP_ENV", "development")
IS_DEV  = APP_ENV == "development"

# CORS origins — extend this when you know your Railway frontend URL
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",   # Vite + fallback CRA port
)
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup + shutdown events
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan replaces the deprecated @app.on_event pattern.

    Startup:
      - Initialise database tables (idempotent — safe on every restart)
      - Log environment so you can confirm which DB is active in Railway logs

    Shutdown:
      - Placeholder for any cleanup (model unloading, temp file removal, etc.)
    """
    # ── STARTUP ──────────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("Solar Irradiance Forecasting System — starting up")
    logger.info("Environment : %s", APP_ENV)
    logger.info("CORS origins: %s", ALLOWED_ORIGINS)

    try:
        init_db()
        logger.info("Database    : OK (tables verified)")
    except Exception as exc:
        # Don't crash the app — db_is_online() will report False in /health
        logger.error("Database init failed: %s", exc)

    logger.info("═" * 60)

    yield   # ← application runs here

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    logger.info("Solar Forecast System shutting down — goodbye")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app instance
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Solar Irradiance Forecasting API",
    description = (
        "ML-powered solar irradiance forecasting system. "
        "Accepts any weather/solar CSV, auto-detects columns via Claude AI, "
        "runs XGBoost + LightGBM + Prophet ensemble, returns interactive results."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",      # Swagger UI  → http://localhost:8000/docs
    redoc_url   = "/redoc",     # ReDoc UI    → http://localhost:8000/redoc
    lifespan    = lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────────────

# CORS — must be added BEFORE routers
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],       # GET, POST, PUT, DELETE, OPTIONS
    allow_headers     = ["*"],       # Content-Type, Authorization, etc.
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log every request with method, path, status code, and duration.
    Helps you trace slow ML pipeline calls during development.

    Example output:
      POST /api/forecast/run → 200  (took 4.231s)
    """
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    logger.info(
        "%s %s → %d  (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Global exception handler
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch any unhandled exception and return a consistent JSON error envelope
    instead of a raw 500 HTML page (which React can't parse).

    The ErrorDetail schema (schemas.py) matches this structure.
    """
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "code":    "internal_server_error",
            "message": str(exc) if IS_DEV else "An unexpected error occurred.",
            "field":   None,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routers — all mounted under /api
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(upload.router,   prefix="/api", tags=["Upload"])
app.include_router(forecast.router, prefix="/api", tags=["Forecast"])
app.include_router(anomaly.router,  prefix="/api", tags=["Anomaly"])
app.include_router(history.router,  prefix="/api", tags=["History"])
app.include_router(export.router,   prefix="/api", tags=["Export"])


# ─────────────────────────────────────────────────────────────────────────────
# Health check — required by Railway / Render for deployment
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """
    Railway and Render ping this endpoint to confirm the service is alive.
    Returns 200 if healthy, 503 if the database is unreachable.

    Response matches HealthResponse schema (schemas.py).
    """
    db_ok = db_is_online()
    status_code = 200 if db_ok else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status":    "ok" if db_ok else "degraded",
            "version":   "1.0.0",
            "env":       APP_ENV,
            "db_online": db_ok,
        },
    )


@app.get("/", tags=["System"])
def root():
    """
    Root endpoint — confirms the API is running.
    Useful sanity check after deployment.
    """
    return {
        "message": "Solar Irradiance Forecasting API is running.",
        "docs":    "/docs",
        "health":  "/health",
    }