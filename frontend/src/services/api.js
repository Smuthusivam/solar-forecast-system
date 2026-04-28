import axios from "axios";

// All API calls go to FastAPI running on port 8000
const API = axios.create({
  baseURL: "http://localhost:8000",
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
    horizon_hours: horizonHours,
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