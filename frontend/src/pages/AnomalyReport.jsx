// AnomalyReport.jsx — Full anomaly detection + AI correction + model comparison page

import { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine, BarChart, Bar,
} from "recharts";
import { getAnomalies, runCorrection, getCorrectedCSVUrl } from "../services/api";

// ── Severity badge ──────────────────────────────────────────────────────────
const SeverityBadge = ({ severity }) => {
  const colors = {
    high: "bg-red-100 text-red-700 border border-red-300",
    medium: "bg-yellow-100 text-yellow-700 border border-yellow-300",
    low: "bg-blue-100 text-blue-700 border border-blue-300",
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${colors[severity] || colors.low}`}>
      {severity}
    </span>
  );
};

// ── Confidence badge ────────────────────────────────────────────────────────
const ConfidenceBadge = ({ confidence }) => {
  const colors = {
    high: "bg-green-100 text-green-700",
    medium: "bg-yellow-100 text-yellow-700",
    low: "bg-red-100 text-red-700",
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors[confidence] || ""}`}>
      {confidence} confidence
    </span>
  );
};

// ── Source badge ────────────────────────────────────────────────────────────
const SourceBadge = ({ source }) => {
  const map = {
    ai: { label: "🤖 AI", cls: "bg-purple-100 text-purple-700" },
    physics_rule: { label: "⚡ Physics", cls: "bg-cyan-100 text-cyan-700" },
    interpolation_fallback: { label: "📐 Interpolation", cls: "bg-gray-100 text-gray-600" },
  };
  const { label, cls } = map[source] || { label: source, cls: "bg-gray-100 text-gray-600" };
  return <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${cls}`}>{label}</span>;
};

// ── Metric delta card ────────────────────────────────────────────────────────
const MetricDeltaCard = ({ label, data }) => {
  if (!data) return null;
  const improved = data.delta < 0; // lower RMSE/MAE = better
  const r2Metric = label === "R²";
  const isImproved = r2Metric ? data.delta > 0 : data.delta < 0;

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 flex flex-col gap-1">
      <div className="text-xs text-gray-500 font-medium uppercase tracking-wide">{label}</div>
      <div className="flex items-end gap-2">
        <span className="text-2xl font-bold text-gray-800">
          {data.corrected.toFixed(4)}
        </span>
        <span className="text-sm text-gray-400 mb-0.5 line-through">{data.original.toFixed(4)}</span>
      </div>
      <div className={`text-sm font-semibold ${isImproved ? "text-green-600" : "text-red-500"}`}>
        {isImproved ? "▼" : "▲"} {Math.abs(data.improvement_pct).toFixed(2)}% {isImproved ? "improvement" : "degraded"}
      </div>
    </div>
  );
};

// ── Main component ──────────────────────────────────────────────────────────
export default function AnomalyReport({ datasetId }) {
  const location = useLocation();
  const routeSessionId = location.state?.result?.session_id || new URLSearchParams(location.search).get("session_id");
  const sessionId = datasetId || routeSessionId;

  const [anomalies, setAnomalies] = useState([]);
  const [loadingAnomalies, setLoadingAnomalies] = useState(true);
  const [errorAnomalies, setErrorAnomalies] = useState(null);

  const [correctionResult, setCorrectionResult] = useState(null);
  const [loadingCorrection, setLoadingCorrection] = useState(false);
  const [correctionError, setCorrectionError] = useState(null);

  const [activeTab, setActiveTab] = useState("detection"); // "detection" | "correction" | "comparison"

  // Load anomalies on mount
  useEffect(() => {
    if (!sessionId) {
      setLoadingAnomalies(false);
      setErrorAnomalies("No session found. Run a forecast first, then open Anomalies from the results page.");
      return;
    }

    setLoadingAnomalies(true);
    getAnomalies(sessionId)
      .then((res) => setAnomalies(res.anomalies || []))
      .catch((e) => setErrorAnomalies(e.message))
      .finally(() => setLoadingAnomalies(false));
  }, [sessionId]);

  const handleRunCorrection = async () => {
    setLoadingCorrection(true);
    setCorrectionError(null);
    try {
      const res = await runCorrection(sessionId);
      setCorrectionResult(res);
      setActiveTab("correction");
    } catch (e) {
      setCorrectionError(e.response?.data?.detail || e.message);
    } finally {
      setLoadingCorrection(false);
    }
  };

  // Build chart data for anomaly timeline
  const timelineData = anomalies.slice(0, 60).map((a) => ({
    time: new Date(a.timestamp).toLocaleString(),
    value: a.value,
    severity: a.severity,
  }));

  // Build chart data for before/after forecast comparison
  const comparisonChartData = (() => {
    if (!correctionResult?.forecasts) return [];
    const { original = [], corrected = [], timestamps = [], actuals = [] } = correctionResult.forecasts;
    return timestamps.map((ts, i) => ({
      time: new Date(ts).toLocaleString(),
      "Original Forecast": original[i] ?? null,
      "Corrected Forecast": corrected[i] ?? null,
      "Actual": actuals[i] ?? null,
    }));
  })();

  // Build correction log chart — delta per correction
  const correctionDeltaData = correctionResult?.correction_log?.map((c, i) => ({
    name: `#${i + 1}`,
    original: c.original_value,
    corrected: c.corrected_value,
    delta: Math.abs(c.corrected_value - c.original_value),
  })) || [];

  const tabs = [
    { key: "detection", label: "🔍 Detection" },
    { key: "correction", label: "🤖 AI Correction", disabled: !correctionResult },
    { key: "comparison", label: "📊 Before vs After", disabled: !correctionResult },
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-6 space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Anomaly Report</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            AI-powered anomaly detection and intelligent correction for dataset{" "}
            <code className="bg-gray-100 px-1 rounded">{datasetId}</code>
          </p>
        </div>

        <div className="flex gap-3 flex-wrap">
          {correctionResult && (
            <a
              href={getCorrectedCSVUrl(correctionResult.session_id)}
              className="px-4 py-2 rounded-lg bg-green-600 text-white text-sm font-semibold hover:bg-green-700 transition flex items-center gap-2"
            >
              ⬇ Download Corrected CSV
            </a>
          )}
          <button
            onClick={handleRunCorrection}
            disabled={loadingCorrection || loadingAnomalies}
            className="px-4 py-2 rounded-lg bg-purple-600 text-white text-sm font-semibold hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-2"
          >
            {loadingCorrection ? (
              <>
                <span className="animate-spin">⚙</span> Running AI Correction…
              </>
            ) : (
              "🤖 Run AI Correction"
            )}
          </button>
        </div>
      </div>

      {correctionError && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
          ⚠ Correction failed: {correctionError}
        </div>
      )}

      {/* Stats bar */}
      {correctionResult && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { label: "Anomalies Found", value: correctionResult.anomaly_count, color: "text-orange-600" },
            { label: "AI Corrected", value: correctionResult.stats.ai_corrections, color: "text-purple-600" },
            { label: "Physics Rule", value: correctionResult.stats.physics_rule_corrections, color: "text-cyan-600" },
            { label: "Interpolated", value: correctionResult.stats.interpolation_fallbacks, color: "text-gray-600" },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-white rounded-xl border border-gray-200 p-4 text-center">
              <div className={`text-3xl font-bold ${color}`}>{value}</div>
              <div className="text-xs text-gray-500 mt-1">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-200">
        {tabs.map(({ key, label, disabled }) => (
          <button
            key={key}
            onClick={() => !disabled && setActiveTab(key)}
            disabled={disabled}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition
              ${activeTab === key
                ? "bg-white border border-b-white border-gray-200 text-purple-700 -mb-px"
                : disabled
                  ? "text-gray-300 cursor-not-allowed"
                  : "text-gray-500 hover:text-gray-700"
              }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ── Tab: Detection ──────────────────────────────────────────────────── */}
      {activeTab === "detection" && (
        <div className="space-y-6">
          {loadingAnomalies ? (
            <div className="flex items-center justify-center h-40 text-gray-400">Loading anomalies…</div>
          ) : errorAnomalies ? (
            <div className="text-red-500 text-sm">Error: {errorAnomalies}</div>
          ) : anomalies.length === 0 ? (
            <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-6 rounded-xl text-center">
              ✅ No anomalies detected in this dataset!
            </div>
          ) : (
            <>
              {/* Summary chips */}
              <div className="flex gap-3 flex-wrap">
                {["high", "medium", "low"].map((s) => (
                  <div key={s} className="bg-white border border-gray-200 rounded-full px-4 py-1.5 text-sm">
                    <SeverityBadge severity={s} />{" "}
                    <span className="ml-1 font-semibold">{anomalies.filter((a) => a.severity === s).length}</span>
                  </div>
                ))}
              </div>

              {/* Timeline chart */}
              <div className="bg-white rounded-xl border border-gray-200 p-4">
                <h3 className="font-semibold text-gray-800 mb-4">Anomaly Timeline</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={timelineData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ fontSize: 12, borderRadius: 8 }}
                      formatter={(v) => [`${v} W/m²`, "Value"]}
                    />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke="#f97316"
                      strokeWidth={2}
                      dot={{ r: 4, fill: "#f97316" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Anomaly table */}
              <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
                <div className="px-4 py-3 border-b border-gray-100">
                  <h3 className="font-semibold text-gray-800">Detected Anomalies ({anomalies.length})</h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
                      <tr>
                        {["Timestamp", "Value (W/m²)", "Method", "Severity"].map((h) => (
                          <th key={h} className="px-4 py-2 text-left">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {anomalies.map((a, i) => (
                        <tr key={i} className="hover:bg-gray-50">
                          <td className="px-4 py-2 font-mono text-xs">{new Date(a.timestamp).toLocaleString()}</td>
                          <td className="px-4 py-2 font-semibold text-orange-600">{a.value.toFixed(2)}</td>
                          <td className="px-4 py-2">
                            <span className="px-2 py-0.5 bg-gray-100 rounded text-xs font-mono">{a.method}</span>
                          </td>
                          <td className="px-4 py-2"><SeverityBadge severity={a.severity} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* ── Tab: AI Correction ──────────────────────────────────────────────── */}
      {activeTab === "correction" && correctionResult && (
        <div className="space-y-6">

          {/* Avg delta chart */}
          {correctionDeltaData.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <h3 className="font-semibold text-gray-800 mb-4">
                Correction Magnitude — Original vs Corrected Values
              </h3>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={correctionDeltaData} barCategoryGap="30%">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} unit=" W/m²" />
                  <Tooltip formatter={(v) => [`${v.toFixed(2)} W/m²`]} />
                  <Legend />
                  <Bar dataKey="original" name="Original" fill="#f97316" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="corrected" name="Corrected" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Correction log table */}
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
              <h3 className="font-semibold text-gray-800">
                AI Correction Log ({correctionResult.correction_log.length} corrections)
              </h3>
              <div className="text-xs text-gray-400">
                Avg delta: {correctionResult.stats.avg_correction_delta?.toFixed(2)} W/m²
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
                  <tr>
                    {["Timestamp", "Original", "Corrected", "Δ", "Source", "Confidence", "Reasoning"].map((h) => (
                      <th key={h} className="px-4 py-2 text-left whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {correctionResult.correction_log.map((c, i) => {
                    const delta = c.corrected_value - c.original_value;
                    return (
                      <tr key={i} className="hover:bg-gray-50 align-top">
                        <td className="px-4 py-2 font-mono text-xs whitespace-nowrap">
                          {new Date(c.timestamp).toLocaleString()}
                        </td>
                        <td className="px-4 py-2 font-semibold text-orange-600">
                          {c.original_value.toFixed(2)}
                        </td>
                        <td className="px-4 py-2 font-semibold text-purple-600">
                          {c.corrected_value.toFixed(2)}
                        </td>
                        <td className={`px-4 py-2 font-semibold text-xs ${delta < 0 ? "text-green-600" : "text-red-500"}`}>
                          {delta > 0 ? "+" : ""}{delta.toFixed(2)}
                        </td>
                        <td className="px-4 py-2">
                          <SourceBadge source={c.correction_source} />
                        </td>
                        <td className="px-4 py-2">
                          <ConfidenceBadge confidence={c.confidence} />
                        </td>
                        <td className="px-4 py-2 text-xs text-gray-600 max-w-xs">
                          {c.reasoning}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab: Before vs After Comparison ────────────────────────────────── */}
      {activeTab === "comparison" && correctionResult && (
        <div className="space-y-6">

          {/* Metric cards */}
          {correctionResult.metrics_comparison && Object.keys(correctionResult.metrics_comparison).length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {Object.entries(correctionResult.metrics_comparison).map(([key, data]) => (
                <MetricDeltaCard key={key} label={key.toUpperCase()} data={data} />
              ))}
            </div>
          )}

          {/* Forecast comparison chart */}
          {comparisonChartData.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <h3 className="font-semibold text-gray-800 mb-1">
                Forecast: Original Data vs AI-Corrected Data
              </h3>
              <p className="text-xs text-gray-400 mb-4">
                Compare how ML predictions change after anomaly correction
              </p>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={comparisonChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 11 }} unit=" W/m²" />
                  <Tooltip
                    contentStyle={{ fontSize: 12, borderRadius: 8 }}
                    formatter={(v) => [`${v?.toFixed(2)} W/m²`]}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="Actual"
                    stroke="#22c55e"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="Original Forecast"
                    stroke="#f97316"
                    strokeWidth={2}
                    dot={false}
                    strokeDasharray="5 3"
                  />
                  <Line
                    type="monotone"
                    dataKey="Corrected Forecast"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Per-model comparison table */}
          {correctionResult.model_comparison?.original && (
            <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-100">
                <h3 className="font-semibold text-gray-800">Per-Model Metrics: Before vs After Correction</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
                    <tr>
                      <th className="px-4 py-2 text-left">Model</th>
                      {["RMSE", "MAE", "R²", "MAPE"].map((m) => (
                        <>
                          <th key={`${m}-orig`} className="px-3 py-2 text-center text-orange-500">{m} (Orig)</th>
                          <th key={`${m}-corr`} className="px-3 py-2 text-center text-purple-500">{m} (Corr)</th>
                        </>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {Object.keys(correctionResult.model_comparison.original).map((model) => {
                      const orig = correctionResult.model_comparison.original[model];
                      const corr = correctionResult.model_comparison.corrected?.[model] || {};
                      return (
                        <tr key={model} className="hover:bg-gray-50">
                          <td className="px-4 py-2 font-semibold capitalize">{model}</td>
                          {["rmse", "mae", "r2", "mape"].map((metric) => (
                            <>
                              <td key={`${model}-${metric}-orig`} className="px-3 py-2 text-center text-orange-600">
                                {orig[metric]?.toFixed(4) ?? "—"}
                              </td>
                              <td
                                key={`${model}-${metric}-corr`}
                                className={`px-3 py-2 text-center font-semibold ${
                                  metric === "r2"
                                    ? (corr[metric] > orig[metric] ? "text-green-600" : "text-red-500")
                                    : (corr[metric] < orig[metric] ? "text-green-600" : "text-red-500")
                                }`}
                              >
                                {corr[metric]?.toFixed(4) ?? "—"}
                              </td>
                            </>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}