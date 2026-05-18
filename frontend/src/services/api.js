import axios from "axios";

// All API calls go to FastAPI; override with VITE_API_BASE_URL for deployments.
const API_BASE_URL = import.meta?.env?.VITE_API_BASE_URL || "";

const API = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000, // 10 min — AI correction + dual pipeline runs can be slow
});

const ANOMALY_REPORT_CACHE_PREFIX = "solar-forecast-anomaly-report";

function getAnomalyReportCacheKey(sessionId) {
  return `${ANOMALY_REPORT_CACHE_PREFIX}:${sessionId}`;
}

function readJson(value) {
  if (!value) return null;

  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

export function readAnomalyReportCache(sessionId) {
  if (typeof window === "undefined" || !sessionId) return null;

  return readJson(window.localStorage.getItem(getAnomalyReportCacheKey(sessionId)));
}

export function writeAnomalyReportCache(sessionId, patch) {
  if (typeof window === "undefined" || !sessionId) return null;

  const nextValue = {
    ...(readAnomalyReportCache(sessionId) || {}),
    ...patch,
    sessionId,
    updatedAt: Date.now(),
  };

  window.localStorage.setItem(getAnomalyReportCacheKey(sessionId), JSON.stringify(nextValue));
  return nextValue;
}

// ─────────────────────────────────────────
// 1. UPLOAD CSV
// ─────────────────────────────────────────
export async function uploadCSV(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await API.post("/api/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
}

// ─────────────────────────────────────────
// 2. RUN FORECAST
// ─────────────────────────────────────────
export async function runForecast(sessionId, horizonHours = 24, trainSize = 80, skipFuture = false) {
  const response = await API.post("/api/forecast/run", {
    session_id:  sessionId,
    horizon:     horizonHours,
    train_size:  trainSize,
    skip_future: skipFuture,
  });
  return response.data;
}

// ─────────────────────────────────────────
// 3. GET ANOMALIES
// ─────────────────────────────────────────
export async function getAnomalies(sessionId) {
  const response = await API.get(`/api/anomaly?session_id=${sessionId}`);
  return response.data;
}

// ─────────────────────────────────────────
// 4. GET HISTORY
// ─────────────────────────────────────────
export async function getHistory() {
  const response = await API.get("/api/history");
  return response.data;
}

// ─────────────────────────────────────────
// 5. EXPORT CSV
// ─────────────────────────────────────────
export async function exportCSV(sessionId) {
  const response = await API.get(`/api/export/csv?session_id=${sessionId}`, {
    responseType: "blob",
  });

  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", "preprocessed_data.csv");
  document.body.appendChild(link);
  link.click();
  link.remove();
}

export async function exportFeaturesCSV(sessionId) {
  const response = await API.get(`/api/export/features?session_id=${sessionId}`, {
    responseType: "blob",
  });

  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", "features.csv");
  document.body.appendChild(link);
  link.click();
  link.remove();
}

// Build and download a correction log CSV from the correction result.
export function downloadComparisonCSV(correctionResult) {
  const { correction_log = [] } = correctionResult;
  if (!correction_log.length) return;

  const header = [
    "timestamp",
    "original_value",
    "corrected_value",
    "delta",
    "correction_source",
    "confidence",
    "reasoning",
  ];

  const rows = correction_log.map((c) => {
    const delta = (c.corrected_value != null && c.original_value != null)
      ? (c.corrected_value - c.original_value).toFixed(4)
      : "";
    return [
      c.timestamp,
      c.original_value  ?? "",
      c.corrected_value ?? "",
      delta,
      c.correction_source ?? "",
      c.confidence        ?? "",
      (c.reasoning ?? "").replace(/,/g, ";"),
    ].join(",");
  });

  const csv  = [header.join(","), ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", "correction_log.csv");
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

// ─────────────────────────────────────────
// 7. RUN AI CORRECTION  ← NEW
// Detects anomalies + AI-corrects them +
// re-runs models on both versions for comparison
// ─────────────────────────────────────────
export async function runCorrection(sessionId) {
  const response = await API.post(`/api/correction/run/${sessionId}`);
  return response.data;
}

// ─────────────────────────────────────────
// 8. RUN FORECAST ON CORRECTED DATA
// Uses the AI-corrected dataframe from a
// completed correction session
// ─────────────────────────────────────────
// ─────────────────────────────────────────
// 9. GET STORED FORECAST POINTS
// Retrieves predicted vs actual points
// saved to DB — used to replay chart from history
// ─────────────────────────────────────────
export async function getForecastPoints(runId) {
  const response = await API.get(`/api/history/${runId}/points`);
  return response.data;
}

export async function runForecastFromCorrected(correctionSessionId, horizonHours = 24, trainSize = 80) {
  const response = await API.post(
    `/api/correction/forecast/${correctionSessionId}`,
    null,
    { params: { horizon: horizonHours, train_size: trainSize } }
  );
  return response.data;
}
