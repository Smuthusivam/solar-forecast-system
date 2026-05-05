"""
anomaly_corrector.py
AI-powered anomaly correction using Claude API.
- Low severity anomalies: interpolated immediately, no API call.
- High/medium severity anomalies: corrected concurrently via Claude API.
"""

import json
import asyncio
import logging
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import anthropic

logger = logging.getLogger(__name__)
client = anthropic.AsyncAnthropic()  # async client for concurrent calls

# Physical bounds for solar irradiance (W/m²)
GHI_MIN = 0.0
GHI_MAX = 1200.0

# Hours considered nighttime — irradiance must be 0
NIGHT_HOURS = set(range(0, 5)) | set(range(21, 24))

# Only send the worst anomalies to AI — the rest get interpolated
AI_CAP = 20


def _build_correction_prompt(
    ts: str,
    bad_value: float,
    method: str,
    severity: str,
    context_rows: List[Dict],
    column_map: Dict[str, str],
) -> str:
    return f"""You are a solar irradiance data quality expert working with sensor time-series data.
A sensor recorded a physically abnormal value that was flagged by automated anomaly detection.

ANOMALY DETAILS:
- Timestamp: {ts}
- Flagged value: {bad_value:.2f} W/m²
- Detection method: {method}
- Severity: {severity}

SURROUNDING DATA (±3 hours of context around the anomaly):
{json.dumps(context_rows, indent=2, default=str)}

AVAILABLE WEATHER COLUMNS IN THIS DATASET:
{json.dumps(column_map, indent=2)}

YOUR TASK:
1. Analyze the surrounding values and available weather variables (cloud cover, temperature, humidity if present)
2. Consider the time of day — irradiance MUST be 0 at night (before 5am, after 9pm)
3. Consider the trend from neighboring valid readings
4. Produce the most physically plausible corrected irradiance value

Respond ONLY with this exact JSON (no explanation outside JSON, no markdown):
{{
  "corrected_value": <float between 0 and 1200>,
  "reasoning": "<1-2 sentence physical explanation>",
  "confidence": "<high|medium|low>"
}}"""


def _is_nighttime(ts) -> bool:
    try:
        return pd.Timestamp(ts).hour in NIGHT_HOURS
    except Exception:
        return False


def _interpolate_fallback(df: pd.DataFrame, idx: int, col: str) -> float:
    """Linear interpolation using nearest valid neighbors."""
    prev_val = next_val = None
    col_pos = df.columns.get_loc(col)

    for j in range(idx - 1, max(-1, idx - 6), -1):
        v = df.iloc[j, col_pos]
        if pd.notna(v):
            prev_val = float(v)
            break

    for j in range(idx + 1, min(len(df), idx + 6)):
        v = df.iloc[j, col_pos]
        if pd.notna(v):
            next_val = float(v)
            break

    if prev_val is not None and next_val is not None:
        return (prev_val + next_val) / 2
    return prev_val or next_val or 0.0


async def _call_claude(prompt: str) -> dict:
    """Single async Claude API call. Returns parsed result dict or raises."""
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.content[0].text.strip())


async def _correct_one(
    anomaly: Dict,
    df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    column_map: Dict[str, str],
    context_cols: List[str],
    irradiance_col: str,
    semaphore: asyncio.Semaphore,
) -> Dict:
    """Process a single anomaly — returns one correction_log entry."""
    ts = anomaly.get("timestamp")
    bad_value = float(anomaly.get("value", 0))
    method = anomaly.get("method", "unknown")
    severity = anomaly.get("severity", "medium")

    # Locate row by DatetimeIndex
    ts_parsed = pd.Timestamp(ts)
    ts_matches = df.index[df.index == ts_parsed].tolist()
    if not ts_matches:
        logger.warning("Timestamp %s not found in dataframe — skipping", ts)
        return None

    i = df.index.get_loc(ts_matches[0])
    col_loc = corrected_df.columns.get_loc(irradiance_col)

    # Nighttime: physics rule, no API call
    if _is_nighttime(ts):
        corrected_df.iloc[i, col_loc] = 0.0
        return {
            "timestamp": str(ts),
            "original_value": bad_value,
            "corrected_value": 0.0,
            "reasoning": "Nighttime hours (before 5am / after 9pm) — irradiance is physically zero.",
            "confidence": "high",
            "method_flagged_by": method,
            "severity": severity,
            "correction_source": "physics_rule",
        }

    # Low severity: interpolate, skip API call
    if severity == "low":
        fallback = _interpolate_fallback(corrected_df, i, irradiance_col)
        corrected_df.iloc[i, col_loc] = fallback
        return {
            "timestamp": str(ts),
            "original_value": bad_value,
            "corrected_value": round(fallback, 4),
            "reasoning": "Low severity anomaly — linear interpolation used (no AI call needed).",
            "confidence": "medium",
            "method_flagged_by": method,
            "severity": severity,
            "correction_source": "interpolation_fallback",
        }

    # High/medium: call Claude concurrently (bounded by semaphore)
    window_start = max(0, i - 3)
    window_end = min(len(df), i + 4)
    existing_cols = [c for c in context_cols if c in df.columns]
    context_rows = df.iloc[window_start:window_end][existing_cols].to_dict(orient="records")
    prompt = _build_correction_prompt(str(ts), bad_value, method, severity, context_rows, column_map)

    async with semaphore:
        try:
            result = await _call_claude(prompt)
            corrected_value = max(GHI_MIN, min(GHI_MAX, float(result["corrected_value"])))
            corrected_df.iloc[i, col_loc] = corrected_value
            return {
                "timestamp": str(ts),
                "original_value": bad_value,
                "corrected_value": round(corrected_value, 4),
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", "medium"),
                "method_flagged_by": method,
                "severity": severity,
                "correction_source": "ai",
            }
        except json.JSONDecodeError as e:
            logger.error("Claude returned non-JSON for %s: %s", ts, e)
        except Exception as e:
            logger.error("AI correction failed for %s: %s", ts, e)

    # Fallback if Claude failed
    fallback = _interpolate_fallback(corrected_df, i, irradiance_col)
    corrected_df.iloc[i, col_loc] = fallback
    return {
        "timestamp": str(ts),
        "original_value": bad_value,
        "corrected_value": round(fallback, 4),
        "reasoning": "AI unavailable — linear interpolation used as fallback.",
        "confidence": "low",
        "method_flagged_by": method,
        "severity": severity,
        "correction_source": "interpolation_fallback",
    }


async def _correct_all_async(
    df: pd.DataFrame,
    anomalies: List[Dict],
    column_map: Dict[str, str],
) -> Tuple[pd.DataFrame, List[Dict]]:
    corrected_df = df.copy()
    irradiance_col = "irradiance"
    corrected_df[irradiance_col] = corrected_df[irradiance_col].astype(float)

    weather_context_cols = [
        col for key, col in column_map.items()
        if key != "irradiance" and col in df.columns
    ]
    context_cols = [irradiance_col] + weather_context_cols

    # Prioritise by severity, send only the top AI_CAP to Claude, interpolate the rest
    severity_rank = {"high": 3, "medium": 2, "low": 1}
    sorted_anomalies = sorted(anomalies, key=lambda a: severity_rank.get(a.get("severity", "low"), 0), reverse=True)
    ai_anomalies   = sorted_anomalies[:AI_CAP]
    skip_anomalies = sorted_anomalies[AI_CAP:]

    logger.info("%d anomalies total — %d sent to AI, %d interpolated directly", len(anomalies), len(ai_anomalies), len(skip_anomalies))

    semaphore = asyncio.Semaphore(AI_CAP)

    ai_tasks = [
        _correct_one(anomaly, df, corrected_df, column_map, context_cols, irradiance_col, semaphore)
        for anomaly in ai_anomalies
    ]
    skip_tasks = [
        _correct_one({**anomaly, "severity": "low"}, df, corrected_df, column_map, context_cols, irradiance_col, semaphore)
        for anomaly in skip_anomalies
    ]

    results = await asyncio.gather(*ai_tasks, *skip_tasks)
    correction_log = [r for r in results if r is not None]

    ai_count = sum(1 for c in correction_log if c["correction_source"] == "ai")
    physics_count = sum(1 for c in correction_log if c["correction_source"] == "physics_rule")
    interp_count = sum(1 for c in correction_log if "fallback" in c["correction_source"])
    logger.info(
        "Correction complete: %d processed — %d AI, %d physics rule, %d interpolation",
        len(correction_log), ai_count, physics_count, interp_count,
    )

    return corrected_df, correction_log


def correct_anomalies_with_ai(
    df: pd.DataFrame,
    anomalies: List[Dict],
    column_map: Dict[str, str],
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Entry point called from the correction router (sync context).
    Runs the async pipeline via asyncio.run().
    """
    return asyncio.run(_correct_all_async(df, anomalies, column_map))


def compute_correction_stats(correction_log: List[Dict]) -> Dict:
    if not correction_log:
        return {}

    deltas = [abs(c["corrected_value"] - c["original_value"]) for c in correction_log]
    sources = [c["correction_source"] for c in correction_log]
    confidences = [c["confidence"] for c in correction_log]

    return {
        "total_corrected": len(correction_log),
        "ai_corrections": sources.count("ai"),
        "physics_rule_corrections": sources.count("physics_rule"),
        "interpolation_fallbacks": sum(1 for s in sources if "fallback" in s),
        "avg_correction_delta": round(float(np.mean(deltas)), 4),
        "max_correction_delta": round(float(np.max(deltas)), 4),
        "high_confidence": confidences.count("high"),
        "medium_confidence": confidences.count("medium"),
        "low_confidence": confidences.count("low"),
    }
