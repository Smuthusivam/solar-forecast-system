"""
anomaly_corrector.py
AI-powered anomaly correction using Claude API.

Pipeline:
  1. Physical validation  — dismiss anomalies that are actually plausible values
  2. Nighttime rule       — set to 0, no API call
  3. Low severity         — linear interpolation, no API call
  4. High/medium          — sent to Claude in batches of BATCH_SIZE (one API call per batch)
"""

import json
import asyncio
import logging
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import anthropic

logger = logging.getLogger(__name__)
client = anthropic.AsyncAnthropic()

# Physical bounds for solar irradiance (W/m²)
GHI_MIN = 0.0
GHI_MAX = 1200.0

# Nighttime hours — irradiance must be 0
NIGHT_HOURS = set(range(0, 5)) | set(range(21, 24))

# Anomalies per Claude call — smaller batches = more accurate JSON from Claude
BATCH_SIZE = 10

# Max simultaneous Claude API calls — avoids rate-limit 429s
MAX_CONCURRENT = 15


# ─────────────────────────────────────────────────────────────────────────────
# Physical validation — dismiss anomalies that are actually plausible
# ─────────────────────────────────────────────────────────────────────────────

def _is_physically_plausible(value: float, ts, df: pd.DataFrame, idx: int) -> bool:
    """
    Return True if the flagged value is actually reasonable given:
      - hard physical bounds (0–1200 W/m²)
      - time of day
      - neighbourhood median (within 20% of local median → probably fine)
    """
    hour = pd.Timestamp(ts).hour

    # Absolute bounds
    if value < GHI_MIN or value > GHI_MAX:
        return False

    # Nighttime must be 0
    if hour in NIGHT_HOURS and value > 10:
        return False

    # Daytime: compare to local neighbourhood median (±6 rows)
    col = "irradiance"
    if col not in df.columns:
        return True

    window_start = max(0, idx - 6)
    window_end   = min(len(df), idx + 7)
    # Exclude the anomaly point itself so it doesn't skew the median
    neighbour_idx = [j for j in range(window_start, window_end) if j != idx]
    neighbours = df[col].iloc[neighbour_idx].dropna().values

    if len(neighbours) < 3:
        return True  # not enough context to judge

    median = float(np.median(neighbours))
    if median == 0:
        return value == 0

    deviation_pct = abs(value - median) / median
    # Only dismiss if very close to neighbours (within 20%) — let Claude judge everything else
    return deviation_pct < 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_nighttime(ts) -> bool:
    try:
        return pd.Timestamp(ts).hour in NIGHT_HOURS
    except Exception:
        return False


def _interpolate_fallback(df: pd.DataFrame, idx: int, col: str) -> float:
    """Linear interpolation from nearest valid neighbours."""
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


def _get_context_rows(df: pd.DataFrame, idx: int, context_cols: List[str]) -> List[Dict]:
    """Return ±1 row window as a list of dicts (3 rows total — enough context, minimal tokens)."""
    existing = [c for c in context_cols if c in df.columns]
    return df.iloc[max(0, idx - 1): min(len(df), idx + 2)][existing].to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────────────────
# Batch prompt — one Claude call handles up to BATCH_SIZE anomalies
# ─────────────────────────────────────────────────────────────────────────────

def _build_batch_prompt(batch: List[Dict], column_map: Dict[str, str]) -> str:
    items_json = json.dumps(batch, default=str)  # compact, no indent — fewer tokens
    weather_keys = list(column_map.keys())
    return f"""Solar irradiance sensor anomaly correction. Fix {len(batch)} flagged readings.
Weather columns available: {weather_keys}

Rules:
- Night (hours 0-4, 21-23): corrected_value=0.0, is_genuine=true
- If value matches neighbours within 30%: is_genuine=false, corrected_value=original_value
- Bounds: 0-1200 W/m²
- Use context rows and weather to estimate plausible daytime values

Data:
{items_json}

Respond with a JSON array only, one object per input in order:
{{"timestamp":"<same>","is_genuine":<bool>,"corrected_value":<float>,"reasoning":"<one sentence, no quotes>","confidence":"<high|medium|low>"}}"""


def _sanitise_raw_json(raw: str) -> str:
    """
    Clean up common issues in Claude's JSON output before parsing:
    - Strip markdown fences
    - Collapse literal newlines inside string values (the main parse-error cause)
    - Trim leading/trailing whitespace
    """
    # Strip markdown fences
    if "```" in raw:
        parts = raw.split("```")
        # Take the part most likely to be JSON (longest part containing '[')
        raw = max((p for p in parts if "[" in p), key=len, default=raw)
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]

    # Find the actual array boundaries and discard anything outside
    start = raw.find("[")
    end   = raw.rfind("]")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]

    # Replace literal newlines/tabs inside the JSON string with spaces.
    # We do this character-by-character so we only touch chars inside string values.
    result   = []
    in_str   = False
    prev     = ""
    for ch in raw:
        if ch == '"' and prev != "\\":
            in_str = not in_str
        if in_str and ch in ("\n", "\r", "\t"):
            result.append(" ")
        else:
            result.append(ch)
        prev = ch

    return "".join(result).strip()


async def _call_claude_batch(
    batch_items: List[Dict],
    column_map: Dict[str, str],
    semaphore: asyncio.Semaphore,
) -> List[Dict]:
    """Send one batch to Claude and return a parsed list of correction results."""
    async with semaphore:
        prompt = _build_batch_prompt(batch_items, column_map)

        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

    raw = response.content[0].text.strip()

    raw = _sanitise_raw_json(raw)
    logger.debug("Sanitised response (first 300 chars): %s", raw[:300])

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed after sanitise: %s — trying object-by-object repair", str(e)[:120])

        # Object-by-object extraction as last resort
        result    = []
        depth     = 0
        start_idx = None
        in_str    = False
        prev      = ""

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
                            obj = json.loads(_sanitise_raw_json(raw[start_idx:i + 1]))
                            result.append(obj)
                        except json.JSONDecodeError:
                            pass
                        start_idx = None
            prev = ch

        if result:
            logger.info("Repaired JSON: recovered %d / %d objects", len(result), len(batch_items))
            return result

        logger.error("All parse attempts failed. Raw (first 500): %s", raw[:500])
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Main async correction pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def _correct_all_async(
    df: pd.DataFrame,
    anomalies: List[Dict],
    column_map: Dict[str, str],
) -> Tuple[pd.DataFrame, List[Dict]]:

    corrected_df = df.copy()
    irradiance_col = "irradiance"
    corrected_df[irradiance_col] = corrected_df[irradiance_col].astype(float)
    col_loc = corrected_df.columns.get_loc(irradiance_col)

    weather_context_cols = [
        col for key, col in column_map.items()
        if key != "irradiance" and col in df.columns
    ]
    context_cols = [irradiance_col] + weather_context_cols

    # Pre-build O(1) timestamp → integer position map — avoids O(n) scan per anomaly
    ts_to_pos = {ts: pos for pos, ts in enumerate(df.index)}

    correction_log: List[Dict] = []
    ai_batch: List[Dict] = []
    ai_batch_meta: List[Dict] = []

    dismissed = 0

    for anomaly in anomalies:
        ts        = anomaly.get("timestamp")
        bad_value = float(anomaly.get("value", 0))
        method    = anomaly.get("method", "unknown")
        severity  = anomaly.get("severity", "medium")

        # O(1) row lookup
        ts_parsed = pd.Timestamp(ts)
        i = ts_to_pos.get(ts_parsed)
        if i is None:
            logger.warning("Timestamp %s not found — skipping", ts)
            continue

        # ── 1. Nighttime physics rule ─────────────────────────────────────
        if _is_nighttime(ts):
            corrected_df.iloc[i, col_loc] = 0.0
            correction_log.append({
                "timestamp": str(ts),
                "original_value": bad_value,
                "corrected_value": 0.0,
                "reasoning": "Nighttime — irradiance is physically zero.",
                "confidence": "high",
                "method_flagged_by": method,
                "severity": severity,
                "correction_source": "physics_rule",
            })
            continue

        # ── 2. Physical plausibility check — dismiss false positives ──────
        if _is_physically_plausible(bad_value, ts, df, i):
            dismissed += 1
            logger.debug("Dismissed plausible value %.2f at %s", bad_value, ts)
            continue

        # ── 3. Low severity — interpolate immediately ─────────────────────
        if severity == "low":
            fallback = _interpolate_fallback(corrected_df, i, irradiance_col)
            corrected_df.iloc[i, col_loc] = fallback
            correction_log.append({
                "timestamp": str(ts),
                "original_value": bad_value,
                "corrected_value": round(fallback, 4),
                "reasoning": "Low severity — linear interpolation applied.",
                "confidence": "medium",
                "method_flagged_by": method,
                "severity": severity,
                "correction_source": "interpolation_fallback",
            })
            continue

        # ── 4. Queue for Claude batch ─────────────────────────────────────
        context_rows = _get_context_rows(df, i, context_cols)
        ai_batch.append({
            "timestamp": str(ts),
            "original_value": bad_value,
            "method": method,
            "severity": severity,
            "context": context_rows,
        })
        ai_batch_meta.append({"i": i, "bad_value": bad_value, "method": method, "severity": severity, "ts": str(ts)})

    logger.info(
        "%d anomalies: %d nighttime, %d dismissed as plausible, %d low-severity interpolated, %d queued for AI",
        len(anomalies),
        sum(1 for c in correction_log if c["correction_source"] == "physics_rule"),
        dismissed,
        sum(1 for c in correction_log if c["correction_source"] == "interpolation_fallback"),
        len(ai_batch),
    )

    # ── Split into batches and fire all concurrently ─────────────────────────
    batches = [
        (ai_batch[s: s + BATCH_SIZE], ai_batch_meta[s: s + BATCH_SIZE])
        for s in range(0, len(ai_batch), BATCH_SIZE)
    ]
    logger.info(
        "Firing %d Claude batch(es) concurrently (max %d at a time) — %d anomalies total",
        len(batches), MAX_CONCURRENT, len(ai_batch),
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    batch_responses = await asyncio.gather(
        *[_call_claude_batch(items, column_map, semaphore) for items, _ in batches],
        return_exceptions=True,
    )

    for batch_idx, ((batch_items, batch_meta), results) in enumerate(zip(batches, batch_responses)):
        if isinstance(results, Exception):
            logger.error("Batch %d/%d failed: %s", batch_idx + 1, len(batches), results, exc_info=results)
            for meta in batch_meta:
                fallback = _interpolate_fallback(corrected_df, meta["i"], irradiance_col)
                corrected_df.iloc[meta["i"], col_loc] = fallback
                correction_log.append({
                    "timestamp": meta["ts"],
                    "original_value": meta["bad_value"],
                    "corrected_value": round(fallback, 4),
                    "reasoning": f"AI batch failed — interpolation used. ({str(results)[:80]})",
                    "confidence": "low",
                    "method_flagged_by": meta["method"],
                    "severity": meta["severity"],
                    "correction_source": "interpolation_fallback",
                })
            continue

        if len(results) != len(batch_meta):
            logger.warning(
                "Batch %d/%d: Claude returned %d results for %d inputs — padding missing with interpolation",
                batch_idx + 1, len(batches), len(results), len(batch_meta),
            )
            # Pad with None so zip still works
            results = list(results) + [None] * (len(batch_meta) - len(results))

        logger.info("Batch %d/%d: %d results received from Claude", batch_idx + 1, len(batches), len(results))

        for meta, result in zip(batch_meta, results):
            i         = meta["i"]
            bad_value = meta["bad_value"]

            if result is None or not isinstance(result, dict):
                fallback = _interpolate_fallback(corrected_df, i, irradiance_col)
                corrected_df.iloc[i, col_loc] = fallback
                correction_log.append({
                    "timestamp": meta["ts"],
                    "original_value": bad_value,
                    "corrected_value": round(fallback, 4),
                    "reasoning": "Missing result from AI batch — interpolation used.",
                    "confidence": "low",
                    "method_flagged_by": meta["method"],
                    "severity": meta["severity"],
                    "correction_source": "interpolation_fallback",
                })
                continue

            if not result.get("is_genuine", True):
                logger.debug("Claude: value at %s is plausible, skipping correction", meta["ts"])
                continue

            corrected_value = max(GHI_MIN, min(GHI_MAX, float(result["corrected_value"])))
            corrected_df.iloc[i, col_loc] = corrected_value
            correction_log.append({
                "timestamp": meta["ts"],
                "original_value": bad_value,
                "corrected_value": round(corrected_value, 4),
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", "medium"),
                "method_flagged_by": meta["method"],
                "severity": meta["severity"],
                "correction_source": "ai",
            })

    ai_count      = sum(1 for c in correction_log if c["correction_source"] == "ai")
    physics_count = sum(1 for c in correction_log if c["correction_source"] == "physics_rule")
    interp_count  = sum(1 for c in correction_log if "fallback" in c["correction_source"])
    logger.info(
        "Correction complete: %d corrected (%d AI, %d physics, %d interpolation), %d dismissed as plausible",
        len(correction_log), ai_count, physics_count, interp_count, dismissed,
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
        "total_corrected":            len(correction_log),
        "ai_corrections":             sources.count("ai"),
        "physics_rule_corrections":   sources.count("physics_rule"),
        "interpolation_fallbacks":    sum(1 for s in sources if "fallback" in s),
        "avg_correction_delta":       round(float(np.mean(deltas)), 4),
        "max_correction_delta":       round(float(np.max(deltas)), 4),
        "high_confidence":            confidences.count("high"),
        "medium_confidence":          confidences.count("medium"),
        "low_confidence":             confidences.count("low"),
    }
