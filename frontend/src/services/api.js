import axios from "axios";

// All API calls go to FastAPI running on port 8000
const API = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 120000, // 2 min — AI correction can be slow on large datasets
});

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
// 3. GET AVAILABLE MODELS + METRICS
// ─────────────────────────────────────────
export async function getModels() {
  const response = await API.get("/api/forecast/models");
  return response.data;
}

// ─────────────────────────────────────────
// 4. GET ANOMALIES
// ─────────────────────────────────────────
export async function getAnomalies(sessionId) {
  const response = await API.get(`/api/anomaly?session_id=${sessionId}`);
  return response.data;
}

// ─────────────────────────────────────────
// 5. GET HISTORY
// ─────────────────────────────────────────
export async function getHistory() {
  const response = await API.get("/api/history");
  return response.data;
}

// ─────────────────────────────────────────
// 6. EXPORT CSV
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
// 7. EXPORT PDF
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
  return `/api/correction/export/${correctionId}`;
}

// ─────────────────────────────────────────
// 8. RUN AI CORRECTION  ← NEW
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

// ─────────────────────────────────────────
// 9. GET CORRECTION LOG  ← NEW
// Fetch the full log for a finished correction
// ─────────────────────────────────────────
export async function getCorrectionLog(correctionId) {
  const response = await API.get(`/api/correction/log/${correctionId}`);
  return response.data;
  // Returns: { correction_id, session_id, total, corrections[] }
}

// ─────────────────────────────────────────
// 10. DOWNLOAD CORRECTED CSV  ← NEW
// Triggers a browser download of the
// AI-corrected dataset as a .csv file
// ─────────────────────────────────────────
export async function exportCorrectedCSV(correctionId) {
  const response = await API.get(`/api/correction/export/${correctionId}`, {
    responseType: "blob",
  });

  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", `corrected_${correctionId}.csv`);
  document.body.appendChild(link);
  link.click();
  link.remove();
}