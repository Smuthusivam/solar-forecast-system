import { useState, useMemo } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, Cell, AreaChart, Area,
  ReferenceLine, ComposedChart
} from "recharts";
import { exportCSV, exportPDF } from "../services/api";

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

// ─────────────────────────────────────────────────────────────────────────
// Main Dashboard
// ─────────────────────────────────────────────────────────────────────────
function Dashboard() {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecast, result, historyRun } = location.state || {};

  const [activeTab, setActiveTab] = useState("forecast");
  const [exporting, setExporting] = useState(null);

  // ── Saved history summary mode (no full forecast arrays in DB) ─────────
  if (!forecast && historyRun) {
    const modeClass = historyRun.detection_mode === "direct"
      ? "bg-green-100 text-green-700"
      : "bg-yellow-100 text-yellow-700";

    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-2xl font-bold text-gray-800">Saved Run Dashboard</h1>
              <p className="text-sm text-gray-400 mt-1">
                {historyRun.filename} · Run #{historyRun.run_id}
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => navigate("/history")}
                className="border border-gray-300 text-gray-700 px-4 py-2 rounded-lg text-sm hover:bg-gray-50"
              >
                ← History
              </button>
              <button
                onClick={() => navigate("/")}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700"
              >
                New Forecast
              </button>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow p-6 mb-6">
            <div className="flex items-center gap-3 mb-4">
              <span className={`text-xs font-semibold px-3 py-1 rounded-full ${modeClass}`}>
                {historyRun.detection_mode}
              </span>
              <span className="text-xs text-gray-400">Horizon: {historyRun.horizon}h</span>
              <span className="text-xs text-gray-400">Rows: {historyRun.rows_processed.toLocaleString()}</span>
              <span className="text-xs text-gray-400">Anomalies: {historyRun.anomaly_count}</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <StatCard
                label="RMSE"
                value={historyRun.ensemble_rmse.toFixed(2)}
                unit="W/m²"
                color="text-blue-600"
                sub="saved ensemble metric"
              />
              <StatCard
                label="MAE"
                value={historyRun.ensemble_mae.toFixed(2)}
                unit="W/m²"
                color="text-purple-600"
                sub="saved ensemble metric"
              />
              <StatCard
                label="R²"
                value={historyRun.ensemble_r2.toFixed(4)}
                color={historyRun.ensemble_r2 > 0.85 ? "text-green-600" : "text-orange-500"}
                sub="saved ensemble metric"
              />
            </div>
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-800">
            This view shows saved summary metrics from history. Full forecast charts are not stored in the database, so rerun the forecast from upload to see detailed plots.
          </div>
        </div>
      </div>
    );
  }

  // ── No data guard ──────────────────────────────────────────────────────
  if (!forecast) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center gap-4">
        <p className="text-gray-500 text-lg">No forecast data found.</p>
        <button onClick={() => navigate("/")}
          className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700">
          Go Back to Upload
        </button>
      </div>
    );
  }

  const em = forecast.ensemble_metrics;
  const points = forecast.forecast;

  // ── Computed dataset stats ─────────────────────────────────────────────
  const stats = useMemo(() => {
    const actuals = points.map(p => p.actual).filter(v => v != null);
    const preds   = points.map(p => p.predicted);
    const residuals = points
      .filter(p => p.actual != null)
      .map(p => p.actual - p.predicted);

    const mean = arr => arr.reduce((a, b) => a + b, 0) / (arr.length || 1);
    const std  = arr => {
      const m = mean(arr);
      return Math.sqrt(mean(arr.map(v => (v - m) ** 2)));
    };

    const sortedActuals = [...actuals].sort((a, b) => a - b);
    const median = sortedActuals[Math.floor(sortedActuals.length / 2)] ?? 0;

    const trainCount = Math.floor(points.length / 0.2 * 0.8);
    const totalRows  = trainCount + points.length;

    const dataStats = result?.data_stats;
    const meanActuals = mean(actuals);
    const minActuals = actuals.length ? Math.min(...actuals) : 0;
    const maxActuals = actuals.length ? Math.max(...actuals) : 0;

    return {
      totalPoints: points.length,
      trainRows:   trainCount,
      testRows:    points.length,
      featureCount: forecast.feature_importance
        ? Object.keys(forecast.feature_importance).length
        : 22,
      irradiance: {
        mean:   dataStats?.irradiance_mean ?? meanActuals,
        median,
        std:    std(actuals),
        min:    dataStats?.irradiance_min ?? minActuals,
        max:    dataStats?.irradiance_max ?? maxActuals,
      },
      residuals: {
        mean: mean(residuals),
        std:  std(residuals),
        min:  Math.min(...residuals),
        max:  Math.max(...residuals),
      },
      dateRange: {
        start: points[0]?.timestamp,
        end:   points[points.length - 1]?.timestamp,
      },
      totalRows,
    };
  }, [points, forecast]);

  // ── Forecast chart data (sampled) ──────────────────────────────────────
  const step = Math.max(1, Math.floor(points.length / 200));
  const chartData = points
    .filter((_, i) => i % step === 0)
    .map(p => ({
      time:      p.timestamp.substring(5, 16).replace("T", " "),
      Predicted: parseFloat(p.predicted?.toFixed(1)),
      Actual:    p.actual != null ? parseFloat(p.actual?.toFixed(1)) : null,
      Lower:     p.lower  != null ? parseFloat(p.lower?.toFixed(1))  : null,
      Upper:     p.upper  != null ? parseFloat(p.upper?.toFixed(1))  : null,
    }));

  // ── Residual histogram ─────────────────────────────────────────────────
  const histogramData = useMemo(() => {
    const residuals = points
      .filter(p => p.actual != null)
      .map(p => p.actual - p.predicted);

    if (residuals.length === 0) return [];

    const min = Math.min(...residuals);
    const max = Math.max(...residuals);
    const bins = 25;
    const width = (max - min) / bins;

    const counts = Array(bins).fill(0);
    residuals.forEach(r => {
      const idx = Math.min(bins - 1, Math.floor((r - min) / width));
      counts[idx]++;
    });

    return counts.map((count, i) => ({
      bin:   (min + i * width).toFixed(0),
      count,
    }));
  }, [points]);

  // ── Hourly average irradiance ──────────────────────────────────────────
  const hourlyAvg = useMemo(() => {
    const buckets = Array.from({ length: 24 }, () => ({ sum: 0, count: 0 }));
    points.forEach(p => {
      if (p.actual == null) return;
      const h = new Date(p.timestamp).getHours();
      buckets[h].sum   += p.actual;
      buckets[h].count += 1;
    });
    return buckets.map((b, h) => ({
      hour: `${h}:00`,
      avg:  b.count ? parseFloat((b.sum / b.count).toFixed(1)) : 0,
    }));
  }, [points]);

  // ── Hour × Day heatmap ─────────────────────────────────────────────────
  const heatmap = useMemo(() => {
    // Build [day_of_week][hour] = avg irradiance
    const grid = Array.from({ length: 7 }, () =>
      Array.from({ length: 24 }, () => ({ sum: 0, count: 0 }))
    );

    points.forEach(p => {
      if (p.actual == null) return;
      const d = new Date(p.timestamp);
      const day  = d.getDay();
      const hour = d.getHours();
      grid[day][hour].sum   += p.actual;
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

  // ── Train/test split visual ────────────────────────────────────────────
  const splitData = [
    { name: "Train (80%)", value: stats.trainRows, fill: "#3b82f6" },
    { name: "Test  (20%)", value: stats.testRows,  fill: "#f59e0b" },
  ];

  // ── Export handlers ────────────────────────────────────────────────────
  async function handleExport(type) {
    setExporting(type);
    try {
      if (type === "csv") await exportCSV(result.session_id);
      else                await exportPDF(result.session_id);
    } catch (err) {
      alert("Export failed: " + (err?.message || ""));
    } finally {
      setExporting(null);
    }
  }

  // ── Tabs ───────────────────────────────────────────────────────────────
  const tabs = [
    { id: "forecast", label: "Forecast"      },
    { id: "data",     label: "Data Overview" },
    { id: "patterns", label: "Patterns"      },
    { id: "errors",   label: "Error Analysis"},
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-6">

      {/* ─── Header ─────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Forecast Dashboard</h1>
          <p className="text-sm text-gray-400 mt-1">
            {result?.filename} — Run #{forecast.run_id} —
            <span className="ml-1">
              {forecast.detection_mode === "direct"
                ? "Direct GHI" : "Estimated GHI"}
            </span>
          </p>
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => handleExport("csv")}
            disabled={exporting !== null}
            className="border border-gray-300 text-gray-600 px-3 py-2 rounded-lg text-sm hover:bg-gray-50 disabled:opacity-50">
            {exporting === "csv" ? "..." : "CSV"}
          </button>
          <button
            onClick={() => handleExport("pdf")}
            disabled={exporting !== null}
            className="border border-gray-300 text-gray-600 px-3 py-2 rounded-lg text-sm hover:bg-gray-50 disabled:opacity-50">
            {exporting === "pdf" ? "..." : "PDF"}
          </button>
          <button onClick={() => navigate("/forecast", { state: { sessionId: result.session_id, filename: result.filename } })}
            className="border border-green-300 text-green-600 px-4 py-2 rounded-lg text-sm hover:bg-green-50">
            Forecast →
          </button>
          <button onClick={() => navigate("/compare", { state: { forecast, result } })}
            className="border border-blue-300 text-blue-600 px-4 py-2 rounded-lg text-sm hover:bg-blue-50">
            Models →
          </button>
          <button onClick={() => navigate(`/anomalies?session_id=${result.session_id}`, { state: { forecast, result } })}
            className="border border-orange-300 text-orange-600 px-4 py-2 rounded-lg text-sm hover:bg-orange-50">
            Anomalies →
          </button>
          <button onClick={() => navigate("/")}
            className="border border-gray-300 text-gray-600 px-4 py-2 rounded-lg text-sm hover:bg-gray-50">
            ← Upload
          </button>
        </div>
      </div>

      {/* ─── Top metrics row ────────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard label="RMSE" value={em.rmse.toFixed(2)} unit="W/m²"
          color="text-blue-600" sub="lower is better" />
        <StatCard label="MAE" value={em.mae.toFixed(2)} unit="W/m²"
          color="text-purple-600" sub="lower is better" />
        <StatCard label="R²" value={em.r2.toFixed(4)}
          color={em.r2 > 0.85 ? "text-green-600" : "text-orange-500"}
          sub={em.r2 > 0.85 ? "excellent fit" : "moderate fit"} />
        <StatCard label="MAPE" value={em.mape.toFixed(2)} unit="%"
          color="text-indigo-600" sub="daytime only" />
      </div>

      {/* ─── Tabs ─────────────────────────────────────────────────── */}
      <div className="flex gap-1 mb-6 border-b border-gray-200">
        {tabs.map(t => (
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

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* TAB 1 — FORECAST                                                 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      {activeTab === "forecast" && (
        <>
          <Card title="Predicted vs Actual"
            subtitle="Ensemble forecast on the held-out test set with confidence intervals">
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="time" tick={{ fontSize: 10 }}
                  interval={Math.floor(chartData.length / 8)} />
                <YAxis tick={{ fontSize: 11 }}
                  label={{ value: "W/m²", angle: -90, position: "insideLeft" }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <Legend />
                <Area type="monotone" dataKey="Upper" stroke="none" fill="#dbeafe" fillOpacity={0.6} />
                <Area type="monotone" dataKey="Lower" stroke="none" fill="#ffffff" />
                <Line type="monotone" dataKey="Actual"    stroke="#10b981" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Predicted" stroke="#3b82f6" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Model Weight Distribution"
            subtitle="Weights computed inversely from validation RMSE — lower error = higher weight">
            <div className="grid grid-cols-2 gap-4">
              {forecast.per_model.map(m => (
                <div key={m.model_name} className="bg-gray-50 rounded-lg p-4 text-center">
                  <p className="text-sm text-gray-500 mb-1">{m.model_name}</p>
                  <p className="text-3xl font-bold text-blue-600">
                    {(m.weight * 100).toFixed(1)}%
                  </p>
                  <div className="mt-2 text-xs text-gray-400 space-y-0.5">
                    <p>RMSE: {m.metrics.rmse.toFixed(2)}</p>
                    <p>R²: {m.metrics.r2.toFixed(4)}</p>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {forecast.feature_importance && (
            <Card title="Top 10 Feature Importances"
              subtitle="Average gain importance from XGBoost + LightGBM">
              <div className="space-y-2">
                {Object.entries(forecast.feature_importance)
                  .slice(0, 10)
                  .map(([feat, score]) => (
                    <div key={feat} className="flex items-center gap-3">
                      <span className="text-sm text-gray-600 w-40 truncate">{feat}</span>
                      <div className="flex-1 bg-gray-100 rounded-full h-3">
                        <div className="bg-blue-500 h-3 rounded-full"
                          style={{ width: `${(score * 100).toFixed(1)}%` }} />
                      </div>
                      <span className="text-xs text-gray-400 w-12 text-right">
                        {(score * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
              </div>
            </Card>
          )}
        </>
      )}


      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* TAB 3 — DATA OVERVIEW                                            */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      {activeTab === "data" && (
        <>
          {/* Dataset overview */}
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

          {/* Train/test split visual */}
          <Card title="Train / Test Split"
            subtitle="Time-aware 80/20 split — no shuffling to preserve temporal order">
            <div className="flex items-stretch h-12 rounded-lg overflow-hidden mb-3">
              <div className="bg-blue-500 flex items-center justify-center text-white font-medium"
                style={{ width: "80%" }}>
                Train: {stats.trainRows.toLocaleString()} rows (80%)
              </div>
              <div className="bg-orange-500 flex items-center justify-center text-white font-medium"
                style={{ width: "20%" }}>
                Test: {stats.testRows.toLocaleString()} (20%)
              </div>
            </div>
            <p className="text-xs text-gray-400">
              Time series MUST NOT be shuffled. The model is trained on earlier data
              and evaluated on the most recent 20% to simulate real forecasting.
            </p>
          </Card>

          {/* Irradiance statistics */}
          <Card title="Irradiance Statistics"
            subtitle="Distribution and range of the target variable">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <StatCard label="Mean"   value={stats.irradiance.mean.toFixed(1)}   unit="W/m²" color="text-blue-600" />
              <StatCard label="Median" value={stats.irradiance.median.toFixed(1)} unit="W/m²" color="text-indigo-600" />
              <StatCard label="Std Dev" value={stats.irradiance.std.toFixed(1)}   unit="W/m²" color="text-purple-600" />
              <StatCard label="Min"    value={stats.irradiance.min.toFixed(1)}    unit="W/m²" color="text-gray-500" />
              <StatCard label="Max"    value={stats.irradiance.max.toFixed(1)}    unit="W/m²" color="text-orange-600" />
            </div>
          </Card>

          {/* Date range card */}
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

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* TAB 3 — PATTERNS                                                 */}
      {/* ═══════════════════════════════════════════════════════════════ */}
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

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* TAB 4 — ERROR ANALYSIS                                           */}
      {/* ═══════════════════════════════════════════════════════════════ */}
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
              <StatCard label="Min Error"     value={stats.residuals.min.toFixed(1)}
                unit="W/m²" color="text-gray-500" sub="largest under-pred" />
              <StatCard label="Max Error"     value={stats.residuals.max.toFixed(1)}
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
  );
}

export default Dashboard;