"""
anomaly_corrector.py — AI-powered anomaly correction using Claude Haiku.

For each flagged anomaly the pipeline does, in order:
  1. Nighttime rule    — set to 0.0, no API call needed
  2. Plausibility check — dismiss values that are fine given neighbours
  3. Low severity      — linear interpolation, no API call needed
  4. High / medium     — single Claude call asking only for a corrected number
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import anthropic

logger = logging.getLogger(__name__)
client = anthropic.AsyncAnthropic()

GHI_MIN        = 0.0
GHI_MAX        = 1200.0
NIGHT_HOURS    = set(range(0, 5)) | set(range(21, 24))
MAX_CONCURRENT = 10           # reduced — fewer concurrent calls = fewer rate-limit failures
_MODEL         = "claude-haiku-4-5"
_MAX_TOKENS    = 80           # we only need a single number back


# ── Physical helpers ──────────────────────────────────────────────────────────

def _is_nighttime(ts) -> bool:
    try:
        return pd.Timestamp(ts).hour in NIGHT_HOURS
    except Exception:
        return False


def _is_physically_plausible(value: float, ts, df: pd.DataFrame, idx: int) -> bool:
    """Return True if the value is reasonable and should NOT be corrected."""
    hour = pd.Timestamp(ts).hour

    if value < GHI_MIN or value > GHI_MAX:
        return False
    if hour in NIGHT_HOURS and value > 10:
        return False

    if "irradiance" not in df.columns:
        return True

    start = max(0, idx - 6)
    end   = min(len(df), idx + 7)
    neighbours = df["irradiance"].iloc[
        [j for j in range(start, end) if j != idx]
    ].dropna().values

    if len(neighbours) < 3:
        return True

    median = float(np.median(neighbours))
    if median == 0:
        return value == 0
    return abs(value - median) / median < 0.20


def _interpolate(df: pd.DataFrame, idx: int) -> float:
    """Linear interpolation from the nearest valid neighbours."""
    col_pos  = df.columns.get_loc("irradiance")
    prev_val = next_val = None

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
        return (prev_val + next_val) / 2.0
    return prev_val or next_val or 0.0


# ── Claude call ───────────────────────────────────────────────────────────────

def _build_prompt(ts: str, bad_value: float, prev_val: float, next_val: float, hour: int) -> str:
    """
    Minimal prompt — give Haiku only the numbers it needs, ask only for a number back.
    Keeping it short and unambiguous prevents Haiku from wrapping the answer in prose.
    """
    return (
        f"Solar irradiance sensor reading at hour {hour}: {bad_value:.1f} W/m²\n"
        f"Previous hour: {prev_val:.1f} W/m²  |  Next hour: {next_val:.1f} W/m²\n"
        f"Valid range: 0–1200 W/m². The reading looks anomalous.\n"
        f"Reply with only the corrected value as a single number (e.g. 342.5). No other text."
    )


def _parse_number(text: str) -> float | None:
    """Extract the first float from Haiku's reply."""
    text = text.strip()
    # Remove any markdown, units, or extra words — just grab the first numeric token
    match = re.search(r"[-+]?\d+\.?\d*", text)
    if match:
        return float(match.group())
    return None


async def _call_claude(
    ts: str,
    bad_value: float,
    prev_val: float,
    next_val: float,
    hour: int,
    semaphore: asyncio.Semaphore,
) -> float | None:
    """Call Claude and return a corrected float, or None on failure."""
    async with semaphore:
        try:
            response = await client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                messages=[{"role": "user", "content": _build_prompt(ts, bad_value, prev_val, next_val, hour)}],
            )
            raw = response.content[0].text
            logger.debug("Haiku raw reply for %s: %r", ts, raw)
            return _parse_number(raw)
        except Exception as exc:
            logger.warning("Claude call failed for %s: %s", ts, exc)
            return None


# ── Main correction pipeline ──────────────────────────────────────────────────

async def _correct_all_async(
    df: pd.DataFrame,
    anomalies: List[Dict],
    column_map: Dict[str, str],
) -> Tuple[pd.DataFrame, List[Dict]]:

    corrected_df = df.copy()
    corrected_df["irradiance"] = corrected_df["irradiance"].astype(float)
    col_loc = corrected_df.columns.get_loc("irradiance")

    ts_to_pos = {ts: pos for pos, ts in enumerate(df.index)}
    correction_log: List[Dict] = []
    ai_queue: List[Dict] = []
    dismissed = 0

    # ── Pre-classify every anomaly ────────────────────────────────────────────
    for anomaly in anomalies:
        ts        = anomaly.get("timestamp")
        bad_value = float(anomaly.get("value", 0))
        method    = anomaly.get("method", "unknown")
        severity  = anomaly.get("severity", "medium")

        ts_parsed = pd.Timestamp(ts)
        i = ts_to_pos.get(ts_parsed)
        if i is None:
            logger.warning("Timestamp %s not in DataFrame — skipping", ts)
            continue

        # 1. Nighttime
        if _is_nighttime(ts):
            corrected_df.iloc[i, col_loc] = 0.0
            correction_log.append({
                "timestamp": str(ts), "original_value": bad_value,
                "corrected_value": 0.0,
                "reasoning": "Nighttime — irradiance is physically zero.",
                "confidence": "high", "method_flagged_by": method,
                "severity": severity, "correction_source": "physics_rule",
            })
            continue

        # 2. Plausibility check
        if _is_physically_plausible(bad_value, ts, df, i):
            dismissed += 1
            continue

        # 3. Low severity — interpolate immediately
        if severity == "low":
            val = _interpolate(corrected_df, i)
            corrected_df.iloc[i, col_loc] = val
            correction_log.append({
                "timestamp": str(ts), "original_value": bad_value,
                "corrected_value": round(val, 2),
                "reasoning": "Low severity — interpolated from neighbours.",
                "confidence": "medium", "method_flagged_by": method,
                "severity": severity, "correction_source": "interpolation_fallback",
            })
            continue

        # 4. Queue for Claude
        prev_val = _interpolate(corrected_df, i)  # reuse as neighbour estimate
        col_pos  = df.columns.get_loc("irradiance")
        pv = float(df.iloc[i - 1, col_pos]) if i > 0 else prev_val
        nv = float(df.iloc[i + 1, col_pos]) if i < len(df) - 1 else prev_val
        ai_queue.append({
            "i": i, "ts": str(ts), "bad_value": bad_value,
            "prev_val": pv, "next_val": nv,
            "hour": ts_parsed.hour,
            "method": method, "severity": severity,
        })

    logger.info(
        "%d anomalies → %d nighttime, %d dismissed, %d interpolated, %d queued for AI",
        len(anomalies),
        sum(1 for c in correction_log if c["correction_source"] == "physics_rule"),
        dismissed,
        sum(1 for c in correction_log if c["correction_source"] == "interpolation_fallback"),
        len(ai_queue),
    )

    # ── Fire AI calls concurrently ────────────────────────────────────────────
    if ai_queue:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        results = await asyncio.gather(*[
            _call_claude(q["ts"], q["bad_value"], q["prev_val"], q["next_val"], q["hour"], semaphore)
            for q in ai_queue
        ])

        for q, corrected_value in zip(ai_queue, results):
            i = q["i"]
            if corrected_value is not None:
                corrected_value = max(GHI_MIN, min(GHI_MAX, corrected_value))
                corrected_df.iloc[i, col_loc] = corrected_value
                correction_log.append({
                    "timestamp": q["ts"], "original_value": q["bad_value"],
                    "corrected_value": round(corrected_value, 2),
                    "reasoning": "AI-corrected value.",
                    "confidence": "high", "method_flagged_by": q["method"],
                    "severity": q["severity"], "correction_source": "ai",
                })
            else:
                # Claude failed — fall back to interpolation
                val = _interpolate(corrected_df, i)
                corrected_df.iloc[i, col_loc] = val
                correction_log.append({
                    "timestamp": q["ts"], "original_value": q["bad_value"],
                    "corrected_value": round(val, 2),
                    "reasoning": "AI unavailable — interpolated from neighbours.",
                    "confidence": "low", "method_flagged_by": q["method"],
                    "severity": q["severity"], "correction_source": "interpolation_fallback",
                })

    ai_count     = sum(1 for c in correction_log if c["correction_source"] == "ai")
    interp_count = sum(1 for c in correction_log if "fallback" in c["correction_source"])
    logger.info(
        "Correction complete: %d total (%d AI, %d interpolation, %d dismissed as plausible)",
        len(correction_log), ai_count, interp_count, dismissed,
    )
    return corrected_df, correction_log


def correct_anomalies_with_ai(
    df: pd.DataFrame,
    anomalies: List[Dict],
    column_map: Dict[str, str],
) -> Tuple[pd.DataFrame, List[Dict]]:
    return asyncio.run(_correct_all_async(df, anomalies, column_map))


def compute_correction_stats(correction_log: List[Dict]) -> Dict:
    if not correction_log:
        return {}
    deltas      = [abs(c["corrected_value"] - c["original_value"]) for c in correction_log]
    sources     = [c["correction_source"] for c in correction_log]
    confidences = [c["confidence"] for c in correction_log]
    return {
        "total_corrected":          len(correction_log),
        "ai_corrections":           sources.count("ai"),
        "physics_rule_corrections": sources.count("physics_rule"),
        "interpolation_fallbacks":  sum(1 for s in sources if "fallback" in s),
        "avg_correction_delta":     round(float(np.mean(deltas)), 4),
        "max_correction_delta":     round(float(np.max(deltas)), 4),
        "high_confidence":          confidences.count("high"),
        "medium_confidence":        confidences.count("medium"),
        "low_confidence":           confidences.count("low"),
    }
