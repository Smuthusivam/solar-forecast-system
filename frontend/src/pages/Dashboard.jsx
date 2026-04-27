import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from "recharts";

function MetricCard({ label, value, unit, color }) {
  return (
    <div className="bg-white rounded-xl shadow p-5 flex flex-col gap-1">
      <p className="text-sm text-gray-500">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>
        {value}
        <span className="text-sm font-normal text-gray-400 ml-1">{unit}</span>
      </p>
    </div>
  );
}

function Dashboard() {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecast, result } = location.state || {};

  const [horizon, setHorizon] = useState(24);

  // ── No data guard ─────────────────────────────────────────────────────────
  if (!forecast) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center gap-4">
        <p className="text-gray-500 text-lg">No forecast data found.</p>
        <button
          onClick={() => navigate("/")}
          className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
        >
          Go Back to Upload
        </button>
      </div>
    );
  }

  const em = forecast.ensemble_metrics;

  // ── Prepare chart data ────────────────────────────────────────────────────
  // Show only every Nth point to keep chart readable
  const step   = Math.max(1, Math.floor(forecast.forecast.length / 200));
  const points = forecast.forecast
    .filter((_, i) => i % step === 0)
    .map((p) => ({
      time:      p.timestamp.substring(5, 16).replace("T", " "),
      Predicted: parseFloat(p.predicted?.toFixed(1)),
      Actual:    p.actual != null ? parseFloat(p.actual?.toFixed(1)) : null,
      Lower:     p.lower  != null ? parseFloat(p.lower?.toFixed(1))  : null,
      Upper:     p.upper  != null ? parseFloat(p.upper?.toFixed(1))  : null,
    }));

  return (
    <div className="min-h-screen bg-gray-50 p-6">

      {/* ── Header ── */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">
            Forecast Dashboard
          </h1>
          <p className="text-gray-500 text-sm mt-1">
            {result?.filename} — {forecast.forecast.length} forecast points
          </p>
        </div>

        {/* Navigation */}
        <div className="flex gap-3">
          <button
            onClick={() => navigate("/")}
            className="border border-gray-300 text-gray-600 px-4 py-2 rounded-lg text-sm hover:bg-gray-100"
          >
            ← Upload New
          </button>
          <button
            onClick={() => navigate("/compare", { state: { forecast, result } })}
            className="border border-blue-300 text-blue-600 px-4 py-2 rounded-lg text-sm hover:bg-blue-50"
          >
            Model Comparison →
          </button>
          <button
            onClick={() => navigate("/anomalies", { state: { forecast, result } })}
            className="border border-orange-300 text-orange-600 px-4 py-2 rounded-lg text-sm hover:bg-orange-50"
          >
            Anomaly Report →
          </button>
        </div>
      </div>

      {/* ── Detection mode badge ── */}
      <div className="mb-6">
        <span className={`text-xs font-semibold px-3 py-1 rounded-full
          ${forecast.detection_mode === "direct"
            ? "bg-green-100 text-green-700"
            : "bg-yellow-100 text-yellow-700"
          }`}>
          {forecast.detection_mode === "direct" ? "✅ Direct GHI" : "⚡ GHI Estimated"}
        </span>
        <span className="ml-3 text-xs text-gray-400">
          Run ID: {forecast.run_id}
        </span>
      </div>

      {/* ── Metrics cards ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard
          label="RMSE"
          value={em.rmse.toFixed(2)}
          unit="W/m²"
          color="text-blue-600"
        />
        <MetricCard
          label="MAE"
          value={em.mae.toFixed(2)}
          unit="W/m²"
          color="text-purple-600"
        />
        <MetricCard
          label="R²"
          value={em.r2.toFixed(4)}
          unit=""
          color={em.r2 > 0.85 ? "text-green-600" : "text-orange-500"}
        />
        <MetricCard
          label="MAPE"
          value={em.mape.toFixed(2)}
          unit="%"
          color="text-indigo-600"
        />
      </div>

      {/* ── Horizon selector ── */}
      <div className="flex gap-2 mb-4">
        <span className="text-sm text-gray-500 self-center">Horizon:</span>
        {[24, 48, 72].map((h) => (
          <button
            key={h}
            onClick={() => setHorizon(h)}
            className={`px-4 py-1 rounded-full text-sm font-medium transition
              ${horizon === h
                ? "bg-blue-600 text-white"
                : "bg-white text-gray-600 border border-gray-300 hover:bg-gray-50"
              }`}
          >
            {h}h
          </button>
        ))}
      </div>

      {/* ── Forecast chart ── */}
      <div className="bg-white rounded-xl shadow p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-700 mb-4">
          Predicted vs Actual Irradiance
        </h2>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={points} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 11 }}
              interval={Math.floor(points.length / 8)}
            />
            <YAxis
              tick={{ fontSize: 11 }}
              label={{ value: "W/m²", angle: -90, position: "insideLeft", offset: 10 }}
            />
            <Tooltip
              contentStyle={{ fontSize: 12 }}
              formatter={(val) => [`${val} W/m²`]}
            />
            <Legend />

            {/* Confidence interval band (Prophet upper/lower) */}
            <Line
              type="monotone"
              dataKey="Upper"
              stroke="#bfdbfe"
              strokeWidth={1}
              dot={false}
              name="Upper CI"
            />
            <Line
              type="monotone"
              dataKey="Lower"
              stroke="#bfdbfe"
              strokeWidth={1}
              dot={false}
              name="Lower CI"
            />

            {/* Actual values */}
            <Line
              type="monotone"
              dataKey="Actual"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              name="Actual"
            />

            {/* Ensemble predictions */}
            <Line
              type="monotone"
              dataKey="Predicted"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="Predicted"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── Per model weights ── */}
      <div className="bg-white rounded-xl shadow p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-700 mb-4">
          Ensemble Weights
        </h2>
        <div className="flex gap-4">
          {forecast.per_model.map((m) => (
            <div key={m.model_name} className="flex-1 bg-gray-50 rounded-lg p-4 text-center">
              <p className="text-sm text-gray-500 mb-1">{m.model_name}</p>
              <p className="text-2xl font-bold text-blue-600">
                {(m.weight * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-gray-400 mt-1">
                RMSE: {m.metrics.rmse.toFixed(2)}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* ── Feature importance ── */}
      {forecast.feature_importance && (
        <div className="bg-white rounded-xl shadow p-6">
          <h2 className="text-lg font-semibold text-gray-700 mb-4">
            Top Feature Importances
          </h2>
          <div className="space-y-2">
            {Object.entries(forecast.feature_importance)
              .slice(0, 10)
              .map(([feat, score]) => (
                <div key={feat} className="flex items-center gap-3">
                  <span className="text-sm text-gray-600 w-36 truncate">{feat}</span>
                  <div className="flex-1 bg-gray-100 rounded-full h-3">
                    <div
                      className="bg-blue-500 h-3 rounded-full"
                      style={{ width: `${(score * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-400 w-12 text-right">
                    {(score * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}

    </div>
  );
}

export default Dashboard;