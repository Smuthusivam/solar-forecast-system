import axios from "axios";

// All API calls go to FastAPI running on port 8000
const API = axios.create({
  baseURL: "http://localhost:8000",
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
export async function runForecast(sessionId, horizonHours = 24, trainSize = 80) {
  const response = await API.post("/api/forecast/run", {
    session_id: sessionId,
    horizon: horizonHours,
    train_size: trainSize,
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
    responseType: "blob", // important for file downloads
  });

  // Trigger browser download automatically
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", "forecast_results.csv");
  document.body.appendChild(link);
  link.click();
  link.remove();
}

// ─────────────────────────────────────────
// 6. EXPORT PDF
// ─────────────────────────────────────────
export async function exportPDF(sessionId) {
  const response = await API.get(`/api/export/pdf?session_id=${sessionId}`, {
    responseType: "blob",
  });

  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", "forecast_report.pdf");
  document.body.appendChild(link);
  link.click();
  link.remove();
}

export function getCorrectedCSVUrl(correctionId) {
  return `http://localhost:8000/api/correction/export/${correctionId}`;
}

// ─────────────────────────────────────────
// 7. RUN AI CORRECTION  ← NEW
// Detects anomalies + AI-corrects them +
// re-runs models on both versions for comparison
// ─────────────────────────────────────────
export async function runCorrection(sessionId) {
  const response = await API.post(`/api/correction/run/${sessionId}`);
  return response.data;
  // Returns: { correction_id, anomaly_count, anomalies_corrected,
  //            stats, correction_log, metrics_comparison,
  //            model_comparison, forecasts }
}
