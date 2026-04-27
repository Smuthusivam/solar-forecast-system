"""
ai_detector.py — AI-powered column detection using the Anthropic Claude API.

Problem this solves:
  Users upload CSVs from different sources — weather stations, Kaggle datasets,
  government portals — each with different column names, languages, and formats.
  Examples of the same physical variable across real datasets:
    Irradiance : "GHI", "ghi_wm2", "Solar Radiation", "Solare Strahlung",
                 "irradiance_wpm2", "SOLARAD", "солнечная радиация"
    Temperature: "temp", "Temp_C", "air_temperature", "TAMB", "tmp", "Température"

  Claude reads the column names + sample rows and returns a clean JSON mapping
  regardless of naming convention or language.

Fallback strategy:
  If the Claude API is unavailable (network error, quota exceeded, invalid key),
  the service falls back to alias-based detection using a hardcoded dictionary
  of known column name variants. This keeps the system functional offline.

Environment variables:
  ANTHROPIC_API_KEY   Required for Claude API calls. Set in backend/.env
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-6"
MAX_TOKENS        = 500
REQUEST_TIMEOUT   = 30   # seconds — Claude API is fast, 30s is generous

# Confidence thresholds
HIGH_CONFIDENCE   = 0.90
MEDIUM_CONFIDENCE = 0.70
LOW_CONFIDENCE    = 0.50

# ─────────────────────────────────────────────────────────────────────────────
# Alias dictionary — fallback when Claude API is unavailable
# ─────────────────────────────────────────────────────────────────────────────

COLUMN_ALIASES: dict[str, list[str]] = {
    "irradiance": [
        "ghi", "ghi_wm2", "global_horizontal_irradiance", "solar_radiation",
        "solar_irradiance", "irradiance", "irrad", "solarad", "solrad",
        "radiation", "rad", "shortwave_radiation", "sw_radiation",
        "global_radiation", "globalrad", "sr", "rs",
    ],
    "temperature": [
        "temp", "temperature", "air_temperature", "tamb", "tmp",
        "temp_c", "temp_f", "ambient_temp", "t_air", "ta", "t2m",
        "drybulb", "dry_bulb_temperature",
    ],
    "humidity": [
        "humidity", "rh", "relative_humidity", "rhum", "hum",
        "specific_humidity", "dewpoint", "dew_point", "dp",
    ],
    "wind_speed": [
        "wind_speed", "windspeed", "ws", "wind", "wind_spd",
        "wind_velocity", "wv", "wspd", "w_speed", "spd",
    ],
    "cloud_cover": [
        "cloud_cover", "cloudcover", "clouds", "cloud", "cc",
        "cloud_fraction", "oktas", "total_cloud_cover", "tcc",
        "cloudiness", "nebulosity",
    ],
    "sunshine_hours": [
        "sunshine_hours", "sunshine", "sun_hours", "ssh",
        "sunshine_duration", "sun_duration", "bright_sunshine",
    ],
    "timestamp": [
        "timestamp", "datetime", "date_time", "time", "date",
        "dt", "ts", "utc", "local_time", "observation_time",
        "valid_time", "period_end", "period_start",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Claude API caller
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(columns: list[str], sample_rows: list[dict]) -> str:
    """
    Build the prompt sent to Claude.

    We give Claude:
      1. The column names
      2. Three sample rows of actual data values
      3. Clear instructions to return only JSON

    Showing sample values is critical — it lets Claude distinguish between
    a column named "temp" (could be temporal or temperature) by seeing
    whether the values are timestamps or numbers like 23.5.
    """
    col_list   = ", ".join(columns)
    sample_str = json.dumps(sample_rows[:3], indent=2, default=str)

    return f"""You are a data scientist analyzing a solar/weather CSV dataset.

Column names found in the CSV:
{col_list}

Sample data (first 3 rows):
{sample_str}

Your task: identify which column corresponds to each physical variable.

Return ONLY a JSON object with these exact keys:
{{
  "irradiance":      "<column name or null>",
  "temperature":     "<column name or null>",
  "humidity":        "<column name or null>",
  "wind_speed":      "<column name or null>",
  "cloud_cover":     "<column name or null>",
  "sunshine_hours":  "<column name or null>",
  "timestamp":       "<column name or null>",
  "confidence":      <float between 0.0 and 1.0>,
  "notes":           "<brief explanation of any uncertain mappings>"
}}

Rules:
- Use null (not "null") if a variable is not present in the dataset
- confidence reflects your overall certainty across all mappings
- Column names in your response must match exactly as given above
- Return ONLY the JSON object — no explanation, no markdown, no backticks"""


def _call_claude_api(prompt: str) -> dict[str, Any]:
    """
    Call the Anthropic Messages API and return the parsed JSON response.
    Raises ValueError if the response cannot be parsed as valid JSON.
    Raises httpx.HTTPError on network / API errors.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set in environment variables")

    headers = {
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }

    payload = {
        "model":      CLAUDE_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages":   [{"role": "user", "content": prompt}],
    }

    logger.info("Calling Claude API for column detection (model: %s)", CLAUDE_MODEL)

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        response = client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
        response.raise_for_status()

    data        = response.json()
    raw_content = data["content"][0]["text"].strip()

    logger.info("Claude API response received (%d chars)", len(raw_content))

    # Strip markdown code fences if Claude wraps the JSON anyway
    raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
    raw_content = re.sub(r"\s*```$",          "", raw_content)

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as exc:
        logger.error("Claude returned non-JSON response: %s", raw_content[:200])
        raise ValueError(f"Claude API returned invalid JSON: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Alias-based fallback detector
# ─────────────────────────────────────────────────────────────────────────────

def _alias_fallback(columns: list[str]) -> dict[str, Any]:
    """
    Detect columns using the hardcoded alias dictionary.
    Used when the Claude API is unavailable.

    Matching is case-insensitive and ignores common separators
    ( - _ space ) so "GHI_Wm2" matches "ghi_wm2".
    """
    def normalise(name: str) -> str:
        return re.sub(r"[\s\-_]+", "_", name.strip().lower())

    normalised_cols = {normalise(c): c for c in columns}
    result: dict[str, Any] = {k: None for k in COLUMN_ALIASES}
    matched = 0

    for variable, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            norm_alias = normalise(alias)
            if norm_alias in normalised_cols:
                result[variable] = normalised_cols[norm_alias]
                matched += 1
                break

    total     = len(COLUMN_ALIASES)
    # Confidence scales with how many variables were found
    confidence = round(matched / total, 2)

    result["confidence"] = confidence
    result["notes"]      = f"Alias fallback used — matched {matched}/{total} variables"

    logger.info("Alias fallback: matched %d/%d variables (confidence=%.2f)", matched, total, confidence)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Result builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_result(
    mapping: dict[str, Any],
    columns: list[str],
    used_fallback: bool,
) -> dict[str, Any]:
    """
    Validate the raw mapping (from Claude or fallback) and build the
    final result dict consumed by the upload router.

    Returns:
    {
        "detected":       DetectedColumns-compatible dict,
        "detection_mode": "direct" | "estimated",
        "confidence":     float,
        "warnings":       list[str],
        "used_fallback":  bool,
    }
    """
    warnings: list[str] = []
    if used_fallback:
        warnings.append("Claude API unavailable — used alias-based detection as fallback")

    # Validate that mapped column names actually exist in the CSV
    valid_columns = set(columns)
    detected: dict[str, str | None] = {}

    for field in ["irradiance", "temperature", "humidity",
                  "wind_speed", "cloud_cover", "sunshine_hours", "timestamp"]:
        raw_value = mapping.get(field)

        if raw_value and raw_value in valid_columns:
            detected[field] = raw_value
        else:
            if raw_value and raw_value not in valid_columns:
                warnings.append(
                    f"Claude mapped '{field}' → '{raw_value}' "
                    f"but that column doesn't exist in the CSV — ignored"
                )
            detected[field] = None

    # Determine detection mode
    # "direct"    → irradiance column found, use it as-is
    # "estimated" → no irradiance column, GHI will be computed from weather vars
    if detected["irradiance"]:
        detection_mode = "direct"
    else:
        detection_mode = "estimated"
        warnings.append(
            "No irradiance column detected — GHI will be estimated "
            "from weather variables using the Angstrom-Prescott formula"
        )

        # Warn if we also lack the minimum weather variables for estimation
        has_weather = any(
            detected.get(v) for v in ["temperature", "humidity", "cloud_cover"]
        )
        if not has_weather:
            warnings.append(
                "WARNING: No weather columns detected either — "
                "GHI estimation may be inaccurate"
            )

    confidence = float(mapping.get("confidence", LOW_CONFIDENCE))

    if confidence < MEDIUM_CONFIDENCE:
        warnings.append(
            f"Low detection confidence ({confidence:.0%}) — "
            "please verify the column mappings in the preview"
        )

    logger.info(
        "Detection complete | mode=%s confidence=%.2f irradiance=%s",
        detection_mode, confidence, detected.get("irradiance"),
    )

    return {
        "detected":       detected,
        "detection_mode": detection_mode,
        "confidence":     confidence,
        "warnings":       warnings,
        "used_fallback":  used_fallback,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public interface — called by upload router
# ─────────────────────────────────────────────────────────────────────────────

def detect_columns(
    columns:     list[str],
    sample_rows: list[dict],
) -> dict[str, Any]:
    """
    Main entry point for column detection.

    Strategy:
      1. Try Claude API first (most accurate, handles any language/format)
      2. On any failure → fall back to alias dictionary (always works offline)

    Args:
        columns:     List of column names from the uploaded CSV header
        sample_rows: First 3–5 rows as a list of dicts (column → value)

    Returns:
        {
            "detected":       dict mapping variable → column name (or None),
            "detection_mode": "direct" | "estimated",
            "confidence":     float 0.0–1.0,
            "warnings":       list of warning strings,
            "used_fallback":  bool,
        }

    Never raises — always returns a result (fallback if necessary).
    """
    used_fallback = False

    # ── Try Claude API ────────────────────────────────────────────────────────
    try:
        prompt  = _build_prompt(columns, sample_rows)
        mapping = _call_claude_api(prompt)
        logger.info("Claude API detection succeeded")

    except Exception as exc:
        logger.warning("Claude API failed (%s) — switching to alias fallback", exc)
        mapping       = _alias_fallback(columns)
        used_fallback = True

    # ── Build and return validated result ─────────────────────────────────────
    return _build_result(mapping, columns, used_fallback)