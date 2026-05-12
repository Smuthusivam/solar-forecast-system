// Utility to manage forecast state across pages using localStorage
// This allows NavBar navigation to work properly without losing context

const STORAGE_KEY = "solar-forecast-state";

export function saveForecastState(forecast, result, filename = "", trainSize) {
  try {
    const existing = loadForecastState() || {};
    const state = {
      ...existing,
      forecast,
      result,
      filename,
      trainSize: typeof trainSize === "number" ? trainSize : existing.trainSize ?? 80,
      timestamp: Date.now(),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (err) {
    console.error("Failed to save forecast state:", err);
  }
}

export function loadForecastState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const state = JSON.parse(raw);
    return state;
  } catch (err) {
    console.error("Failed to load forecast state:", err);
    return null;
  }
}

export function clearForecastState() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (err) {
    console.error("Failed to clear forecast state:", err);
  }
}
