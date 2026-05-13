# FastAPI entry point — registers routers, CORS, middleware, and health check.

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Add project root to sys.path so ml_core is importable from anywhere in the app
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database import db_is_online, init_db
from app.routers import anomaly, export, forecast, history, upload, correction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

APP_ENV = os.getenv("APP_ENV", "development")
IS_DEV  = APP_ENV == "development"

# Read allowed origins from env; default to Vite dev server
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
)
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise DB tables on startup, log environment, then yield to the app.
    logger.info("═" * 60)
    logger.info("Solar Irradiance Forecasting System — starting up")
    logger.info("Environment : %s", APP_ENV)
    logger.info("CORS origins: %s", ALLOWED_ORIGINS)

    try:
        init_db()
        logger.info("Database    : OK (tables verified)")
    except Exception as exc:
        # Don't crash — db_is_online() will report False in /health
        logger.error("Database init failed: %s", exc)

    logger.info("═" * 60)

    yield

    logger.info("Solar Forecast System shutting down — goodbye")


app = FastAPI(
    title       = "Solar Irradiance Forecasting API",
    description = (
        "ML-powered solar irradiance forecasting system. "
        "Accepts any weather/solar CSV, auto-detects columns via Claude AI, "
        "trains XGBoost + LightGBM and selects the best model, returns interactive results."
    ),
    version  = "1.0.0",
    docs_url = "/docs",
    redoc_url= "/redoc",
    lifespan = lifespan,
)

# CORS must be added before routers
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log every request with method, path, status code, and duration.
    start    = time.perf_counter()
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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Return a consistent JSON error envelope instead of a raw 500 HTML page.
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "code":    "internal_server_error",
            "message": str(exc) if IS_DEV else "An unexpected error occurred.",
            "field":   None,
        },
    )


app.include_router(upload.router,   prefix="/api", tags=["Upload"])
app.include_router(forecast.router, prefix="/api", tags=["Forecast"])
app.include_router(anomaly.router,  prefix="/api", tags=["Anomaly"])
app.include_router(history.router,  prefix="/api", tags=["History"])
app.include_router(correction.router, prefix="/api")   
app.include_router(export.router,   prefix="/api", tags=["Export"])


@app.get("/health", tags=["System"])
def health_check():
    # Railway/Render ping this to confirm the service is alive.
    db_ok       = db_is_online()
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
    # Sanity check endpoint — confirms the API is running after deployment.
    return {
        "message": "Solar Irradiance Forecasting API is running.",
        "docs":    "/docs",
        "health":  "/health",
    }
