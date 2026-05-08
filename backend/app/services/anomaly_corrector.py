"""
anomaly_corrector.py — AI-powered anomaly correction using Claude Sonnet.

Pipeline per anomaly:
  1. Nighttime rule      — set to 0.0, no API call
  2. Plausibility check  — dismiss values close to neighbours
  3. Low severity        — linear interpolation, no API call
  4. High / medium       — batched Claude Sonnet call (10 per batch, 5 concurrent)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import anthropic

logger = logging.getLogger(__name__)
client = anthropic.AsyncAnthropic()

GHI_MIN       = 0.0
GHI_MAX       = 1200.0
NIGHT_HOURS   = set(range(0, 5)) | set(range(21, 24))

BATCH_SIZE    = 10   # anomalies per Sonnet call — reliable JSON at this size
MAX_CONCURRENT = 5   # concurrent batches: 5 × 10 × ~85 tokens ≈ 4,250 OTPM (Tier 2 = 32k)
_MODEL        = "claude-sonnet-4-6"
_MAX_TOKENS   = 1200  # 10 items × ~100 output tokens + buffer


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


# ── Batch Claude call ─────────────────────────────────────────────────────────

def _build_batch_prompt(batch: List[Dict]) -> str:
    items = json.dumps([
        {
            "id":         i,
            "timestamp":  q["ts"],
            "hour":       q["hour"],
            "bad_value":  q["bad_value"],
            "prev_hour":  q["prev_val"],
            "next_hour":  q["next_val"],
            "severity":   q["severity"],
        }
        for i, q in enumerate(batch)
    ], default=str)

    return (
        f"Fix {len(batch)} solar irradiance anomalies. "
        f"Valid range: 0–1200 W/m². Night hours (0–4, 21–23) must be 0.\n"
        f"Use prev_hour and next_hour as context for what a plausible value is.\n\n"
        f"Data: {items}\n\n"
        f"Reply with a JSON array of exactly {len(batch)} objects in the same order:\n"
        f'[{{"id":0,"corrected_value":<float>,"reasoning":"<one sentence>","confidence":"high|medium|low"}},...]\n'
        f"JSON only, no markdown."
    )


def _parse_batch_response(raw: str, batch_size: int) -> List[Dict | None]:
    """Parse Claude's JSON array response. Returns list of dicts (or None on failure)."""
    # Strip markdown fences
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        raw = max((p for p in parts if "[" in p), key=len, default=raw)
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]

    start, end = raw.find("["), raw.rfind("]")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]

    try:
        results = json.loads(raw)
        if isinstance(results, list):
            # Index by id field for safe ordering
            by_id = {r["id"]: r for r in results if isinstance(r, dict) and "id" in r}
            return [by_id.get(i) for i in range(batch_size)]
    except (json.JSONDecodeError, KeyError):
        pass

    # Object-by-object fallback
    recovered = []
    depth, start_idx, in_str, prev = 0, None, False, ""
    for i, ch in enumerate(raw):
        if ch == '"' and prev != "\\":
            in_str = not in_str
        if not in_str:
            if ch == "{":
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start_idx is not None:
                    try:
                        recovered.append(json.loads(raw[start_idx:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start_idx = None
        prev = ch

    if recovered:
        by_id = {r["id"]: r for r in recovered if isinstance(r, dict) and "id" in r}
        return [by_id.get(i) for i in range(batch_size)]

    logger.warning("Could not parse Claude response — returning all None")
    return [None] * batch_size


async def _call_claude_batch(
    batch: List[Dict],
    semaphore: asyncio.Semaphore,
) -> List[Dict | None]:
    async with semaphore:
        prompt = _build_batch_prompt(batch)
        try:
            response = await client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            logger.debug("Sonnet batch(%d) raw (first 300): %s", len(batch), raw[:300])
            return _parse_batch_response(raw, len(batch))
        except Exception as exc:
            logger.error("Sonnet batch call failed: %s", exc)
            return [None] * len(batch)


# ── Main correction pipeline ──────────────────────────────────────────────────

async def _correct_all_async(
    df: pd.DataFrame,
    anomalies: List[Dict],
    column_map: Dict[str, str],
) -> Tuple[pd.DataFrame, List[Dict]]:

    corrected_df = df.copy()
    corrected_df["irradiance"] = corrected_df["irradiance"].astype(float)
    col_loc = corrected_df.columns.get_loc("irradiance")

    # O(1) timestamp lookup
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

        # 1. Nighttime physics rule
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

        # 2. Plausibility check — dismiss false positives
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

        # 4. Queue for Sonnet
        col_pos = df.columns.get_loc("irradiance")
        pv = float(df.iloc[i - 1, col_pos]) if i > 0 else 0.0
        nv = float(df.iloc[i + 1, col_pos]) if i < len(df) - 1 else 0.0
        ai_queue.append({
            "i": i, "ts": str(ts), "bad_value": bad_value,
            "prev_val": pv, "next_val": nv,
            "hour": ts_parsed.hour,
            "method": method, "severity": severity,
        })

    logger.info(
        "%d anomalies → %d nighttime, %d dismissed, %d interpolated, %d queued for Sonnet",
        len(anomalies),
        sum(1 for c in correction_log if c["correction_source"] == "physics_rule"),
        dismissed,
        sum(1 for c in correction_log if c["correction_source"] == "interpolation_fallback"),
        len(ai_queue),
    )

    # ── Split into batches and fire concurrently ──────────────────────────────
    if ai_queue:
        batches = [
            ai_queue[s: s + BATCH_SIZE]
            for s in range(0, len(ai_queue), BATCH_SIZE)
        ]
        logger.info(
            "Firing %d Sonnet batch(es) — %d concurrent, %d anomalies total",
            len(batches), MAX_CONCURRENT, len(ai_queue),
        )

        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        batch_results = await asyncio.gather(
            *[_call_claude_batch(b, semaphore) for b in batches],
            return_exceptions=True,
        )

        for batch, results in zip(batches, batch_results):
            if isinstance(results, Exception):
                logger.error("Batch exception: %s", results)
                results = [None] * len(batch)

            for q, result in zip(batch, results):
                i = q["i"]
                if result and isinstance(result, dict) and "corrected_value" in result:
                    cv = max(GHI_MIN, min(GHI_MAX, float(result["corrected_value"])))
                    corrected_df.iloc[i, col_loc] = cv
                    correction_log.append({
                        "timestamp": q["ts"], "original_value": q["bad_value"],
                        "corrected_value": round(cv, 2),
                        "reasoning": result.get("reasoning", "AI corrected."),
                        "confidence": result.get("confidence", "medium"),
                        "method_flagged_by": q["method"],
                        "severity": q["severity"], "correction_source": "ai",
                    })
                else:
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
        "Correction complete: %d total (%d AI, %d interpolation, %d dismissed)",
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
