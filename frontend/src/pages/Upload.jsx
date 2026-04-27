import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadCSV, runForecast } from "../services/api";

function Upload() {
  const navigate = useNavigate();

  // ── State ────────────────────────────────────────
  const [file, setFile]           = useState(null);
  const [dragging, setDragging]   = useState(false);
  const [uploading, setUploading] = useState(false);
  const [running, setRunning]     = useState(false);
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);

  // ── File picked from input ───────────────────────
  function handleFileChange(e) {
    const picked = e.target.files[0];
    if (picked) {
      setFile(picked);
      setResult(null);
      setError(null);
    }
  }

  // ── Drag and drop handlers ───────────────────────
  function handleDragOver(e) {
    e.preventDefault();
    setDragging(true);
  }

  function handleDragLeave() {
    setDragging(false);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.name.endsWith(".csv")) {
      setFile(dropped);
      setResult(null);
      setError(null);
    } else {
      setError("Please drop a CSV file only.");
    }
  }

  // ── Upload CSV to FastAPI ────────────────────────
  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setError(null);

    try {
      const data = await uploadCSV(file);  // calls api.js
      setResult(data);                     // store response
    } catch (err) {
      setError("Upload failed. Make sure the backend is running.");
    } finally {
      setUploading(false);
    }
  }

  // ── Run Forecast → go to Dashboard ──────────────
  async function handleRunForecast() {
    if (!result?.session_id) return;
    setRunning(true);

    try {
      const forecast = await runForecast(result.session_id, 24);
      // Pass forecast data to Dashboard via navigation state
      navigate("/dashboard", { state: { forecast, result } });
    } catch (err) {
      setError("Forecast failed. Please try again.");
    } finally {
      setRunning(false);
    }
  }

  // ── UI ───────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-50 px-4 py-8 sm:px-6 lg:px-10">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8">

        {/* Title */}
        <div className="max-w-3xl">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Solar Irradiance Forecasting
          </h1>
          <p className="text-gray-500">
            Upload any solar or weather CSV to get started
          </p>
        </div>

        {/* Drop Zone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`w-full border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors
            ${dragging
              ? "border-blue-500 bg-blue-50"
              : "border-gray-300 bg-white hover:border-blue-400"
            }`}
        >
          <p className="text-4xl mb-3">☁️</p>
          <p className="text-gray-600 font-medium">
            Drag & drop your CSV here
          </p>
          <p className="text-gray-400 text-sm mb-4">or</p>

          {/* File input */}
          <label className="cursor-pointer bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700 transition">
            Browse File
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>

          {/* Show selected file name */}
          {file && (
            <p className="mt-4 text-sm text-green-600 font-medium">
              ✅ {file.name}
            </p>
          )}
        </div>

        {/* Upload Button */}
        {file && !result && (
          <div className="flex justify-start">
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 transition"
            >
              {uploading ? "Uploading..." : "Upload & Detect Columns"}
            </button>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <p className="text-red-500 font-medium">{error}</p>
        )}

        {/* Detection Result Card */}
        {result && (
          <div className="w-full bg-white rounded-xl shadow p-6">

            {/* Mode badge */}
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-gray-800">
                Detection Result
              </h2>
              <span className={`text-xs font-semibold px-3 py-1 rounded-full
                ${result.mode === "direct"
                  ? "bg-green-100 text-green-700"
                  : "bg-yellow-100 text-yellow-700"
                }`}>
                {result.mode === "direct" ? "✅ Direct GHI" : "⚡ GHI Estimated"}
              </span>
            </div>

            {/* Detected columns */}
            <div className="mb-4">
              <p className="text-sm text-gray-500 mb-2">Detected Columns:</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(result.detected_columns || {}).map(([key, val]) => (
                  <span
                    key={key}
                    className="bg-blue-50 text-blue-700 text-xs px-3 py-1 rounded-full font-medium"
                  >
                    {key}: <strong>{val}</strong>
                  </span>
                ))}
              </div>
            </div>

            {/* Preview Table */}
            {result.preview && result.preview.length > 0 && (
              <div className="mb-4 overflow-x-auto">
                <p className="text-sm text-gray-500 mb-2">Data Preview (first 5 rows):</p>
                <table className="text-xs w-full border-collapse">
                  <thead>
                    <tr>
                      {Object.keys(result.preview[0]).map((col) => (
                        <th key={col} className="border border-gray-200 bg-gray-50 px-2 py-1 text-left">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.preview.map((row, i) => (
                      <tr key={i}>
                        {Object.values(row).map((val, j) => (
                          <td key={j} className="border border-gray-200 px-2 py-1">
                            {val}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Run Forecast Button */}
            <button
              onClick={handleRunForecast}
              disabled={running}
              className="w-full bg-green-600 text-white py-3 rounded-lg font-semibold hover:bg-green-700 disabled:opacity-50 transition"
            >
              {running ? "Running Models..." : "🚀 Run Forecast"}
            </button>

          </div>
        )}

      </div>
    </div>
  );
}

export default Upload;