import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Area, ComposedChart
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

// ─────────────────────────────────────────────────────────────────────────
// Main Dashboard
// ─────────────────────────────────────────────────────────────────────────
function Dashboard() {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecast, result, historyRun } = location.state || {};

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

    </div>
  );
}

export default Dashboard;