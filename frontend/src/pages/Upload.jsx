import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadCSV, runForecast } from "../services/api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, ReferenceLine
} from "recharts";

// ─────────────────────────────────────────────────────────────────────────
// Reusable bits
// ─────────────────────────────────────────────────────────────────────────
function StatCard({ label, value, unit, color, sub }) {
  return (
    <div className="bg-white rounded-xl shadow p-4">
      <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${color}`}>
        {value}
        {unit && <span className="text-sm font-normal text-gray-400 ml-1">{unit}</span>}
      </p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  );
}

function Card({ title, subtitle, children, right }) {
  return (
    <div className="bg-white rounded-xl shadow p-6 mb-6">
      <div className="flex items-start justify-between mb-2">
        <div>
          <h2 className="text-lg font-semibold text-gray-700">{title}</h2>
          {subtitle && <p className="text-sm text-gray-400">{subtitle}</p>}
        </div>
        {right}
      </div>
      <div className="mt-3">{children}</div>
    </div>
  );
}

const HEATMAP_COLORS = [
  "#f8fafc", "#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa",
  "#3b82f6", "#2563eb", "#1d4ed8", "#1e40af", "#1e3a8a"
];

function getHeatColor(value, max) {
  if (max === 0) return HEATMAP_COLORS[0];
  const idx = Math.min(
    HEATMAP_COLORS.length - 1,
    Math.floor((value / max) * HEATMAP_COLORS.length)
  );
  return HEATMAP_COLORS[idx];
}

function Upload() {
  const navigate = useNavigate();
  const storageKey = "solar-forecast-upload-state";

  // ── State ────────────────────────────────────────
  const [file, setFile]           = useState(null);
  const [dragging, setDragging]   = useState(false);
  const [uploading, setUploading] = useState(false);
  const [running, setRunning]     = useState(false);
  const [result, setResult]       = useState(null);
  const [forecast, setForecast]   = useState(null);
  const [error, setError]         = useState(null);
  const [trainSize, setTrainSize] = useState(80);
  const [activeTab, setActiveTab] = useState("data");
  const [fileName, setFileName]   = useState("");

  useEffect(() => {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return;
    try {
      const saved = JSON.parse(raw);
      if (saved?.result) setResult(saved.result);
      if (saved?.forecast) setForecast(saved.forecast);
      if (typeof saved?.trainSize === "number") setTrainSize(saved.trainSize);
      if (typeof saved?.activeTab === "string") setActiveTab(saved.activeTab);
      if (typeof saved?.fileName === "string") setFileName(saved.fileName);
    } catch {
      localStorage.removeItem(storageKey);
    }
  }, []);

  useEffect(() => {
    const payload = {
      result,
      forecast,
      trainSize,
      activeTab,
      fileName,
    };
    localStorage.setItem(storageKey, JSON.stringify(payload));
  }, [result, forecast, trainSize, activeTab, fileName]);

  function clearSession() {
    setFile(null);
    setResult(null);
    setForecast(null);
    setError(null);
    setFileName("");
    localStorage.removeItem(storageKey);
  }

  // ── File picked from input ───────────────────────
  function handleFileChange(e) {
    const picked = e.target.files[0];
    if (picked) {
      setFile(picked);
      setFileName(picked.name);
      setResult(null);
      setForecast(null);
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
      setFileName(dropped.name);
      setResult(null);
      setForecast(null);
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
      setForecast(null);
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
      const forecastData = await runForecast(result.session_id, 24, trainSize);
      setForecast(forecastData);
      setActiveTab("data");
    } catch (err) {
      setError("Forecast failed. Please try again.");
    } finally {
      setRunning(false);
    }
  }

  const points = forecast?.forecast ?? [];

  // ── Computed dataset stats ─────────────────────────────────────────────
  const stats = useMemo(() => {
    if (!forecast || points.length === 0) return null;

    const actuals = points.map(p => p.actual).filter(v => v != null);
    const residuals = points
      .filter(p => p.actual != null)
      .map(p => p.actual - p.predicted);

    const mean = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    const std = arr => {
      if (!arr.length) return 0;
      const m = mean(arr);
      return Math.sqrt(mean(arr.map(v => (v - m) ** 2)));
    };

    const sortedActuals = [...actuals].sort((a, b) => a - b);
    const median = sortedActuals.length
      ? sortedActuals[Math.floor(sortedActuals.length / 2)]
      : 0;

    const testRatio = Math.max(0.01, 1 - trainSize / 100);
    const trainRatio = 1 - testRatio;
    const trainCount = Math.floor(points.length / testRatio * trainRatio);
    const totalRows = trainCount + points.length;

    const dataStats = result?.data_stats;
    const meanActuals = mean(actuals);
    const minActuals = actuals.length ? Math.min(...actuals) : 0;
    const maxActuals = actuals.length ? Math.max(...actuals) : 0;

    const residualMin = residuals.length ? Math.min(...residuals) : 0;
    const residualMax = residuals.length ? Math.max(...residuals) : 0;

    return {
      totalPoints: points.length,
      trainRows: trainCount,
      testRows: points.length,
      featureCount: forecast.feature_importance
        ? Object.keys(forecast.feature_importance).length
        : 22,
      irradiance: {
        mean: dataStats?.irradiance_mean ?? meanActuals,
        median,
        std: std(actuals),
        min: dataStats?.irradiance_min ?? minActuals,
        max: dataStats?.irradiance_max ?? maxActuals,
      },
      residuals: {
        mean: mean(residuals),
        std: std(residuals),
        min: residualMin,
        max: residualMax,
      },
      dateRange: {
        start: points[0]?.timestamp,
        end: points[points.length - 1]?.timestamp,
      },
      totalRows,
    };
  }, [forecast, points, result, trainSize]);

  // ── Forecast chart data (sampled) ──────────────────────────────────────
  const chartData = useMemo(() => {
    if (!points.length) return [];
    const step = Math.max(1, Math.floor(points.length / 200));
    return points
      .filter((_, i) => i % step === 0)
      .map(p => ({
        time: p.timestamp.substring(5, 16).replace("T", " "),
        Predicted: parseFloat(p.predicted?.toFixed(1)),
        Actual: p.actual != null ? parseFloat(p.actual?.toFixed(1)) : null,
      }));
  }, [points]);

  // ── Residual histogram ─────────────────────────────────────────────────
  const histogramData = useMemo(() => {
    const residuals = points
      .filter(p => p.actual != null)
      .map(p => p.actual - p.predicted);

    if (residuals.length === 0) return [];

    const min = Math.min(...residuals);
    const max = Math.max(...residuals);
    const bins = 25;
    const width = (max - min) / bins || 1;

    const counts = Array(bins).fill(0);
    residuals.forEach(r => {
      const idx = Math.min(bins - 1, Math.floor((r - min) / width));
      counts[idx]++;
    });

    return counts.map((count, i) => ({
      bin: (min + i * width).toFixed(0),
      count,
    }));
  }, [points]);

  // ── Hourly average irradiance ──────────────────────────────────────────
  const hourlyAvg = useMemo(() => {
    const buckets = Array.from({ length: 24 }, () => ({ sum: 0, count: 0 }));
    points.forEach(p => {
      if (p.actual == null) return;
      const h = new Date(p.timestamp).getHours();
      buckets[h].sum += p.actual;
      buckets[h].count += 1;
    });
    return buckets.map((b, h) => ({
      hour: `${h}:00`,
      avg: b.count ? parseFloat((b.sum / b.count).toFixed(1)) : 0,
    }));
  }, [points]);

  // ── Hour × Day heatmap ─────────────────────────────────────────────────
  const heatmap = useMemo(() => {
    const grid = Array.from({ length: 7 }, () =>
      Array.from({ length: 24 }, () => ({ sum: 0, count: 0 }))
    );

    points.forEach(p => {
      if (p.actual == null) return;
      const d = new Date(p.timestamp);
      const day = d.getDay();
      const hour = d.getHours();
      grid[day][hour].sum += p.actual;
      grid[day][hour].count += 1;
    });

    let max = 0;
    const data = grid.map(row =>
      row.map(cell => {
        const avg = cell.count ? cell.sum / cell.count : 0;
        if (avg > max) max = avg;
        return avg;
      })
    );

    return { data, max };
  }, [points]);

  const dayLabels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

  // ── UI ───────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-50 px-4 py-8 sm:px-6 lg:px-10">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8">

        {/* Title */}
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div className="max-w-3xl">
            <h1 className="text-3xl font-bold text-gray-800 mb-2">
              Solar Irradiance Forecasting
            </h1>
            <p className="text-gray-500">
              Upload any solar or weather CSV to get started
            </p>
          </div>

          <button
            onClick={() => navigate("/history")}
            className="self-start rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-100"
          >
            View History
          </button>
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
              {file.name}
            </p>
          )}
          {!file && fileName && (
            <p className="mt-4 text-xs text-gray-500">
              Last file: {fileName} (select again to re-upload)
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

        {(result || forecast) && (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs text-gray-400">
              Restored session{fileName ? ` for ${fileName}` : ""}
            </p>
            <button
              onClick={clearSession}
              className="text-xs text-red-600 underline underline-offset-2"
            >
              Clear uploaded data
            </button>
          </div>
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
                {result.mode === "direct" ? "Direct GHI" : "GHI Estimated"}
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

            {/* Train / Test Split Selector */}
            <div className="mb-5">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-gray-700">Train / Test Split</p>
                <p className="text-xs text-gray-400">
                  Train <span className="font-semibold text-gray-600">{trainSize}%</span>
                  {" "}/ Test <span className="font-semibold text-gray-600">{100 - trainSize}%</span>
                </p>
              </div>
              <div className="flex gap-2">
                {[70, 75, 80, 85, 90].map((pct) => (
                  <button
                    key={pct}
                    onClick={() => setTrainSize(pct)}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium border transition
                      ${trainSize === pct
                        ? "bg-blue-600 text-white border-blue-600"
                        : "bg-white text-gray-600 border-gray-300 hover:border-blue-400 hover:text-blue-600"
                      }`}
                  >
                    {pct}%
                  </button>
                ))}
              </div>
              <p className="text-xs text-gray-400 mt-2">
                Higher training % gives the model more data to learn from; lower gives a larger test set for evaluation.
              </p>
            </div>

            {/* Run Forecast Button */}
            <button
              onClick={handleRunForecast}
              disabled={running}
              className="w-full bg-green-600 text-white py-3 rounded-lg font-semibold hover:bg-green-700 disabled:opacity-50 transition"
            >
              {running ? "Running Models..." : "Run Forecast"}
            </button>

          </div>
        )}

        {/* EDA Section */}
        {forecast && stats && (
          <div className="w-full">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between mb-4">
              <div>
                <h2 className="text-xl font-semibold text-gray-800">Data Overview & EDA</h2>
                <p className="text-sm text-gray-400">Review dataset stats, patterns, and errors before the forecast view</p>
              </div>
              <button
                onClick={() => navigate("/dashboard", { state: { forecast, result } })}
                className="self-start rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-100"
              >
                Open Forecast Dashboard
              </button>
            </div>

            {/* EDA Tabs */}
            <div className="flex gap-1 mb-6 border-b border-gray-200">
              {[
                { id: "data", label: "Data Overview" },
                { id: "patterns", label: "Patterns" },
                { id: "errors", label: "Error Analysis" }
              ].map(t => (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  className={`px-4 py-2 text-sm font-medium transition border-b-2
                    ${activeTab === t.id
                      ? "text-blue-600 border-blue-600"
                      : "text-gray-500 border-transparent hover:text-gray-800"
                    }`}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {/* ── Data Overview ───────────────────────────── */}
            {activeTab === "data" && (
              <>
                <Card title="Dataset Overview"
                  subtitle="Summary statistics of the input data and ML pipeline">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <StatCard label="Total Rows"
                      value={(stats.totalRows).toLocaleString()}
                      color="text-gray-700" sub="raw data points" />
                    <StatCard label="Features Engineered"
                      value={stats.featureCount}
                      color="text-blue-600" sub="time + lag + weather" />
                    <StatCard label="Date Range"
                      value={`${Math.round((new Date(stats.dateRange.end) - new Date(stats.dateRange.start)) / (1000*60*60*24))}`}
                      unit="days"
                      color="text-purple-600" sub="span of test data" />
                    <StatCard label="Detection Mode"
                      value={forecast.detection_mode}
                      color="text-green-600" sub="GHI source" />
                  </div>
                </Card>

                <Card title="Train / Test Split"
                  subtitle={`Time-aware ${trainSize}/${100 - trainSize} split — no shuffling to preserve temporal order`}>
                  <div className="flex items-stretch h-12 rounded-lg overflow-hidden mb-3">
                    <div className="bg-blue-500 flex items-center justify-center text-white font-medium"
                      style={{ width: `${trainSize}%` }}>
                      Train: {stats.trainRows.toLocaleString()} rows ({trainSize}%)
                    </div>
                    <div className="bg-orange-500 flex items-center justify-center text-white font-medium"
                      style={{ width: `${100 - trainSize}%` }}>
                      Test: {stats.testRows.toLocaleString()} ({100 - trainSize}%)
                    </div>
                  </div>
                  <p className="text-xs text-gray-400">
                    Time series MUST NOT be shuffled. The model is trained on earlier data
                    and evaluated on the most recent slice to simulate real forecasting.
                  </p>
                </Card>

                <Card title="Irradiance Statistics"
                  subtitle="Distribution and range of the target variable">
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    <StatCard label="Mean" value={stats.irradiance.mean.toFixed(1)} unit="W/m²" color="text-blue-600" />
                    <StatCard label="Median" value={stats.irradiance.median.toFixed(1)} unit="W/m²" color="text-indigo-600" />
                    <StatCard label="Std Dev" value={stats.irradiance.std.toFixed(1)} unit="W/m²" color="text-purple-600" />
                    <StatCard label="Min" value={stats.irradiance.min.toFixed(1)} unit="W/m²" color="text-gray-500" />
                    <StatCard label="Max" value={stats.irradiance.max.toFixed(1)} unit="W/m²" color="text-orange-600" />
                  </div>
                </Card>

                <Card title="Test Period Coverage"
                  subtitle="The exact time window evaluated by the models">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-xs text-gray-500 mb-1">Start</p>
                      <p className="text-lg font-mono text-gray-700">
                        {stats.dateRange.start?.substring(0, 19).replace("T", " ")}
                      </p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-xs text-gray-500 mb-1">End</p>
                      <p className="text-lg font-mono text-gray-700">
                        {stats.dateRange.end?.substring(0, 19).replace("T", " ")}
                      </p>
                    </div>
                  </div>
                </Card>
              </>
            )}

            {/* ── Patterns ───────────────────────────────── */}
            {activeTab === "patterns" && (
              <>
                <Card title="Average Irradiance by Hour of Day"
                  subtitle="Solar bell curve — peak at noon, zero at night">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={hourlyAvg}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="hour" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 11 }}
                        label={{ value: "W/m²", angle: -90, position: "insideLeft" }} />
                      <Tooltip />
                      <Bar dataKey="avg" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>

                <Card title="Day of Week × Hour Heatmap"
                  subtitle="Average irradiance pattern — darker = higher irradiance">
                  <div className="overflow-x-auto">
                    <table className="text-xs">
                      <thead>
                        <tr>
                          <th className="p-1"></th>
                          {Array.from({ length: 24 }).map((_, h) => (
                            <th key={h} className="p-1 text-gray-400 font-normal w-7">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {heatmap.data.map((row, d) => (
                          <tr key={d}>
                            <td className="p-1 text-gray-500 font-medium pr-2">{dayLabels[d]}</td>
                            {row.map((val, h) => (
                              <td key={h}
                                title={`${dayLabels[d]} ${h}:00 — ${val.toFixed(0)} W/m²`}
                                className="p-0">
                                <div
                                  className="w-7 h-7 rounded"
                                  style={{ backgroundColor: getHeatColor(val, heatmap.max) }}
                                />
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div className="flex items-center gap-2 mt-3 text-xs text-gray-400">
                      <span>Low</span>
                      {HEATMAP_COLORS.map((c, i) => (
                        <div key={i} className="w-5 h-3 rounded"
                          style={{ backgroundColor: c }} />
                      ))}
                      <span>High ({heatmap.max.toFixed(0)} W/m²)</span>
                    </div>
                  </div>
                </Card>
              </>
            )}

            {/* ── Error Analysis ─────────────────────────── */}
            {activeTab === "errors" && (
              <>
                <Card title="Residual Statistics"
                  subtitle="Distribution of prediction errors (Actual − Predicted)">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <StatCard label="Mean Residual" value={stats.residuals.mean.toFixed(2)}
                      unit="W/m²"
                      color={Math.abs(stats.residuals.mean) < 5 ? "text-green-600" : "text-orange-500"}
                      sub="ideal: near 0" />
                    <StatCard label="Std Deviation" value={stats.residuals.std.toFixed(2)}
                      unit="W/m²" color="text-purple-600" sub="error spread" />
                    <StatCard label="Min Error" value={stats.residuals.min.toFixed(1)}
                      unit="W/m²" color="text-gray-500" sub="largest under-pred" />
                    <StatCard label="Max Error" value={stats.residuals.max.toFixed(1)}
                      unit="W/m²" color="text-gray-500" sub="largest over-pred" />
                  </div>
                </Card>

                <Card title="Residual Histogram"
                  subtitle="A bell shape centred on zero indicates a well-calibrated model">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={histogramData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="bin" tick={{ fontSize: 10 }}
                        label={{ value: "Residual (W/m²)", position: "insideBottom", offset: -5 }} />
                      <YAxis tick={{ fontSize: 11 }} label={{ value: "Frequency", angle: -90, position: "insideLeft" }} />
                      <Tooltip />
                      <ReferenceLine x="0" stroke="#10b981" strokeDasharray="4 2" />
                      <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>

                <Card title="Residuals Over Time"
                  subtitle="Look for patterns — random scatter is good, trends indicate model bias">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={chartData.map(d => ({
                      time: d.time,
                      Residual: d.Actual != null ? parseFloat((d.Actual - d.Predicted).toFixed(1)) : null,
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="time" tick={{ fontSize: 10 }}
                        interval={Math.floor(chartData.length / 8)} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip />
                      <ReferenceLine y={0} stroke="#10b981" strokeDasharray="4 2" />
                      <Line type="monotone" dataKey="Residual"
                        stroke="#ef4444" strokeWidth={1} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
              </>
            )}
          </div>
        )}

      </div>
    </div>
  );
}

export default Upload;