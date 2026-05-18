# AI-powered column detection using Claude API, with alias-based fallback when offline.

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

# Load the single root .env — works both locally and in Docker (where env vars are injected directly).
_ROOT_ENV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"
)
load_dotenv(_ROOT_ENV)

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-6"
MAX_TOKENS        = 500
REQUEST_TIMEOUT   = 30

HIGH_CONFIDENCE   = 0.90
MEDIUM_CONFIDENCE = 0.70
LOW_CONFIDENCE    = 0.50

# Known column name variants for each physical variable — used when Claude API is unavailable.
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
        "drybulb", "dry_bulb_temperature", "dry_bulb",
    ],
    "humidity": [
        "humidity", "rh", "relative_humidity", "rhum", "hum",
        "specific_humidity", "dewpoint", "dew_point", "dp",
        "relative_humidity_%", "rh_%",
    ],
    "wind_speed": [
        "wind_speed", "windspeed", "ws", "wind", "wind_spd",
        "wind_velocity", "wv", "wspd", "w_speed", "spd",
        "wind_speed_m/s",
    ],
    "cloud_cover": [
        "cloud_cover", "cloudcover", "clouds", "cloud", "cc",
        "cloud_fraction", "oktas", "total_cloud_cover", "tcc",
        "cloudiness", "nebulosity", "cloud_type", "cloudtype",
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


def _build_prompt(columns: list[str], sample_rows: list[dict]) -> str:
    col_list   = ", ".join(columns)
    sample_str = json.dumps(sample_rows[:3], indent=2, default=str)

    return f"""You are a data scientist analyzing a solar energy CSV dataset.

Column names found in the CSV:
{col_list}

Sample data (first 3 rows):
{sample_str}

Your task: identify which column corresponds to each physical variable.

IRRADIANCE RULES (most important):
- GHI (Global Horizontal Irradiance) is the correct irradiance column — map it to irradiance
- DNI and DHI are NOT irradiance — do NOT map them to irradiance
- Only use DNI/DHI if GHI is completely absent
- DC_POWER or AC_POWER map to irradiance only if no GHI/DNI/DHI column exists

CLOUD COVER RULES:
- Only map to cloud_cover if the column contains percentage values (0-100%) or fractional values (0-1)
- "Cloud Type" is a categorical code (0-9 integers), NOT cloud cover percentage — map it to null
- Valid cloud cover columns: "Cloud Cover", "Clouds", "cloud_fraction", "tcc", "oktas"

TIMESTAMP RULES:
- If the date is split into separate Year/Month/Day/Hour columns, set timestamp to null
- The preprocessing pipeline handles NSRDB-style split date columns automatically

GENERAL RULES:
- Columns containing "Units" (e.g. "GHI Units") are metadata — ignore them
- Never return unit strings like "w/m2", "%", "c", "m/s" as column names
- Return actual column header names exactly as shown above
- Use null if a variable is not present

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
  "notes":           "<brief explanation>"
}}

Return ONLY the JSON — no markdown, no backticks"""


def _call_claude_api(prompt: str) -> dict[str, Any]:
    # Call the Anthropic Messages API and return the parsed JSON response.
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

    # Strip markdown fences if Claude wraps the JSON anyway
    raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
    raw_content = re.sub(r"\s*```$",          "", raw_content)

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as exc:
        logger.error("Claude returned non-JSON response: %s", raw_content[:200])
        raise ValueError(f"Claude API returned invalid JSON: {exc}") from exc

# Match columns against the alias dictionary; confidence scales with how many matched.
def _alias_fallback(columns: list[str]) -> dict[str, Any]: 
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

    total      = len(COLUMN_ALIASES)
    confidence = round(matched / total, 2)

    result["confidence"] = confidence
    result["notes"]      = f"Alias fallback used — matched {matched}/{total} variables"

    logger.info("Alias fallback: matched %d/%d variables (confidence=%.2f)", matched, total, confidence)
    return result

# Validate the raw mapping and build the final result dict for the upload router.
def _build_result(
    mapping:       dict[str, Any],
    columns:       list[str],
    used_fallback: bool,
) -> dict[str, Any]:
    
    warnings = []
    if used_fallback:
        warnings.append("Claude API unavailable — used alias-based detection as fallback")

    valid_columns = set(columns)

    # Unit strings that Claude sometimes returns by mistake instead of real column names
    UNIT_STRINGS = {
        "w/m2", "w/m²", "kwh/m2", "%", "c", "°c", "k",
        "m/s", "km/h", "mph", "degree", "degrees", "mbar",
        "hpa", "cm", "mm", "hours", "year", "month", "day",
    }

    detected = {}
    for field in ["irradiance", "temperature", "humidity",
                  "wind_speed", "cloud_cover", "sunshine_hours", "timestamp"]:
        raw_value = mapping.get(field)

        if raw_value and str(raw_value).lower().strip() in UNIT_STRINGS:
            warnings.append(
                f"Claude returned a unit string '{raw_value}' for '{field}' — ignored"
            )
            detected[field] = None
        elif raw_value and raw_value in valid_columns:
            detected[field] = raw_value
        else:
            if raw_value and raw_value not in valid_columns:
                warnings.append(
                    f"Claude mapped '{field}' → '{raw_value}' "
                    f"but that column doesn't exist in the CSV — ignored"
                )
            detected[field] = None

    if detected["irradiance"]:
        detection_mode = "direct"
    else:
        raise ValueError(
            "No GHI/irradiance column found. "
            "Please upload a CSV that contains a direct solar irradiance or GHI column."
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

# Try Claude API first; fall back to alias matching on any failure.
def detect_columns(
    columns:     list[str],
    sample_rows: list[dict],
) -> dict[str, Any]:
    used_fallback = False

    try:
        prompt  = _build_prompt(columns, sample_rows)
        mapping = _call_claude_api(prompt)
        logger.info("Claude API detection succeeded")

    except Exception as exc:
        logger.warning("Claude API failed (%s) — switching to alias fallback", exc)
        mapping       = _alias_fallback(columns)
        used_fallback = True

    return _build_result(mapping, columns, used_fallback)
