// AnomalyReport.jsx — Full anomaly detection + AI correction + model comparison page

import { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, BarChart, Bar, RadarChart,
  Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ScatterChart,
  Scatter, ZAxis, Cell,
} from "recharts";
import {
  getAnomalies,
  runCorrection,
  getCorrectedCSVUrl,
  downloadComparisonCSV,
  readAnomalyReportCache,
  writeAnomalyReportCache,
} from "../services/api";
import {
  Section, TabNav, MetricCard, AlertBox,
  SeverityBadge, ConfidenceBadge, SourceBadge, fmtWm2,
} from "../components/ui";

// ── Main component ───────────────────────────────────────────────────────────
export default function AnomalyReport({ datasetId }) {
  const location = useLocation();
  const routeSessionId = location.state?.result?.session_id || new URLSearchParams(location.search).get("session_id");
  const sessionId = datasetId || routeSessionId;

  const [anomalies,        setAnomalies]        = useState([]);
  const [loadingAnomalies, setLoadingAnomalies] = useState(true);
  const [errorAnomalies,   setErrorAnomalies]   = useState(null);
  const [correctionResult, setCorrectionResult] = useState(null);
  const [loadingCorrection,setLoadingCorrection]= useState(false);
  const [correctionError,  setCorrectionError]  = useState(null);
  const [activeTab,        setActiveTab]        = useState("detection");
  const [correctionPageSize, setCorrectionPageSize] = useState(25);

  useEffect(() => {
    if (!sessionId) {
      setLoadingAnomalies(false);
      setErrorAnomalies("No session found.");
      return;
    }

    setErrorAnomalies(null);
    setCorrectionError(null);

    const cached = readAnomalyReportCache(sessionId);
    if (cached?.anomalies) {
      setAnomalies(cached.anomalies);
      setLoadingAnomalies(false);
    } else {
      setLoadingAnomalies(true);
      getAnomalies(sessionId)
        .then((res) => {
          const nextAnomalies = res.anomalies || [];
          setAnomalies(nextAnomalies);
          writeAnomalyReportCache(sessionId, { anomalies: nextAnomalies });
        })
        .catch((e) => setErrorAnomalies(e.message))
        .finally(() => setLoadingAnomalies(false));
    }

    if (cached?.correctionResult) {
      setCorrectionResult(cached.correctionResult);
      setActiveTab(cached.activeTab || "comparison");
    } else {
      setCorrectionResult(null);
      setActiveTab("detection");
    }
  }, [sessionId]);

  const handleRunCorrection = async () => {
    setLoadingCorrection(true); setCorrectionError(null);
    try {
      const res = await runCorrection(sessionId);
      setCorrectionResult(res);
      writeAnomalyReportCache(sessionId, {
        correctionResult: res,
        activeTab: "comparison",
      });
      setActiveTab("comparison");
    } catch (e) {
      setCorrectionError(e.response?.data?.detail || e.message);
    } finally { setLoadingCorrection(false); }
  };

  // ── Derived chart data ────────────────────────────────────────────────────

  const timelineData = anomalies.slice(0, 80).map((a) => ({
    time:  new Date(a.timestamp).toLocaleDateString(),
    value: a.value,
  }));

  const severityBreakdown = ["high", "medium", "low"].map((s) => ({
    severity: s,
    count:    anomalies.filter((a) => a.severity === s).length,
  }));

  const methodBreakdown = [...new Set(anomalies.map((a) => a.method))].map((m) => ({
    method: m,
    count:  anomalies.filter((a) => a.method === m).length,
  }));

  const { forecasts, metrics_comparison: mc, model_comparison: modelComp, correction_log, stats } = correctionResult || {};

  const forecastChartData = (() => {
    if (!forecasts) return [];
    const { original = [], corrected = [], timestamps = [], actuals = [] } = forecasts;
    return timestamps.map((ts, i) => ({
      time:               new Date(ts).toLocaleDateString(),
      "Actual":           actuals[i]   ?? null,
      "Original Forecast":original[i]  ?? null,
      "Corrected Forecast":corrected[i] ?? null,
    }));
  })();

  // Error between actual and each forecast
  const errorChartData = (() => {
    if (!forecasts) return [];
    const { original = [], corrected = [], timestamps = [], actuals = [] } = forecasts;
    return timestamps.map((ts, i) => ({
      time:             new Date(ts).toLocaleDateString(),
      "Original Error": actuals[i] != null ? Math.abs(actuals[i] - (original[i] ?? 0)) : null,
      "Corrected Error":actuals[i] != null ? Math.abs(actuals[i] - (corrected[i] ?? 0)) : null,
    }));
  })();

  // Scatter: original prediction vs actual, corrected prediction vs actual
  const scatterOrig = forecasts
    ? forecasts.actuals.map((a, i) => ({ actual: a, predicted: forecasts.original[i] })).filter(p => p.actual != null)
    : [];
  const scatterCorr = forecasts
    ? forecasts.actuals.map((a, i) => ({ actual: a, predicted: forecasts.corrected[i] })).filter(p => p.actual != null)
    : [];

  // Per-model bar data
  const modelBarData = (() => {
    if (!modelComp?.original) return [];
    return Object.keys(modelComp.original).map((m) => ({
      model:              m,
      "RMSE (Original)":  modelComp.original[m]?.rmse,
      "RMSE (Corrected)": modelComp.corrected?.[m]?.rmse,
      "MAE (Original)":   modelComp.original[m]?.mae,
      "MAE (Corrected)":  modelComp.corrected?.[m]?.mae,
    }));
  })();

  // Radar data — R² scores per model
  const radarData = (() => {
    if (!modelComp?.original) return [];
    return Object.keys(modelComp.original).map((m) => ({
      model:     m,
      "R² Before": +(modelComp.original[m]?.r2 * 100).toFixed(2),
      "R² After":  +(modelComp.corrected?.[m]?.r2 * 100).toFixed(2),
    }));
  })();

  // Correction delta scatter
  const correctionDeltaData = (correction_log || []).map((c, i) => ({
    name:      `#${i + 1}`,
    original:  c.original_value,
    corrected: c.corrected_value,
    delta:     Math.abs(c.corrected_value - c.original_value),
  }));

  // Source breakdown for bar chart
  const sourceBreakdown = [
    { source: "AI",           count: stats?.ai_corrections             || 0, fill: "#8b5cf6" },
    { source: "Physics Rule", count: stats?.physics_rule_corrections   || 0, fill: "#06b6d4" },
    { source: "Interpolation",count: stats?.interpolation_fallbacks    || 0, fill: "#94a3b8" },
  ];

  const tabs = [
    { key: "detection",  label: "Detection" },
    { key: "correction", label: "AI Correction", disabled: !correctionResult },
    { key: "comparison", label: "Before vs After", disabled: !correctionResult },
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-6 space-y-6">

      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Anomaly Report</h1>
          <p className="text-sm text-gray-500 mt-0.5">AI-powered detection, validation &amp; correction</p>
        </div>
        <div className="flex gap-3 flex-wrap">
          {correctionResult && (
            <>
              <a
                href={getCorrectedCSVUrl(correctionResult.session_id)}
                className="px-4 py-2 rounded-lg bg-green-600 text-white text-sm font-semibold hover:bg-green-700 transition"
              >
                Download Corrected CSV
              </a>
              <button
                onClick={() => downloadComparisonCSV(correctionResult)}
                className="px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm font-semibold hover:bg-indigo-700 transition"
              >
                Download Comparison CSV
              </button>
            </>
          )}
          <button
            onClick={handleRunCorrection}
            disabled={loadingCorrection || loadingAnomalies}
            className="px-4 py-2 rounded-lg bg-purple-600 text-white text-sm font-semibold hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {loadingCorrection ? "Running AI Correction..." : "Run AI Correction"}
          </button>
        </div>
      </div>

      {correctionError && (
        <AlertBox>Correction failed: {correctionError}</AlertBox>
      )}

      {/* Top stats */}
      {correctionResult && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { label: "Anomalies Found",  value: correctionResult.anomaly_count,          color: "text-orange-600" },
            { label: "AI Corrected",     value: stats?.ai_corrections,                   color: "text-purple-600" },
            { label: "Physics Rule",     value: stats?.physics_rule_corrections,          color: "text-cyan-600"   },
            { label: "Interpolated",     value: stats?.interpolation_fallbacks,           color: "text-slate-500"  },
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
                : disabled ? "text-gray-300 cursor-not-allowed" : "text-gray-500 hover:text-gray-700"}`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ══════════════════════════════════════════════════════════════════════
          TAB: Detection
      ══════════════════════════════════════════════════════════════════════ */}
      {activeTab === "detection" && (
        <div className="space-y-6">
          {loadingAnomalies ? (
            <div className="flex items-center justify-center h-40 text-gray-400">Loading anomalies…</div>
          ) : errorAnomalies ? (
            <div className="text-red-500 text-sm">Error: {errorAnomalies}</div>
          ) : anomalies.length === 0 ? (
            <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-6 rounded-xl text-center">
              No anomalies detected. The dataset is clean.
            </div>
          ) : (
            <>
              {/* Severity chips */}
              <div className="flex gap-3 flex-wrap">
                {["high", "medium", "low"].map((s) => (
                  <div key={s} className="bg-white border border-gray-200 rounded-full px-4 py-1.5 text-sm flex items-center gap-2">
                    <SeverityBadge severity={s} />
                    <span className="font-semibold">{anomalies.filter((a) => a.severity === s).length}</span>
                  </div>
                ))}
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Anomaly timeline */}
                <Section title="Anomaly Value Timeline" subtitle="Flagged irradiance values over time">
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={timelineData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                      <YAxis tick={{ fontSize: 11 }} unit=" W/m²" />
                      <Tooltip formatter={fmtWm2} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                      <Line type="monotone" dataKey="value" stroke="#f97316" strokeWidth={2} dot={{ r: 3 }} name="Value" />
                    </LineChart>
                  </ResponsiveContainer>
                </Section>

                {/* Severity breakdown */}
                <Section title="Severity Breakdown" subtitle="Count of anomalies by severity level">
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={severityBreakdown} barCategoryGap="35%">
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="severity" tick={{ fontSize: 12 }} />
                      <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                      <Tooltip />
                      <Bar dataKey="count" name="Anomalies" radius={[6, 6, 0, 0]}>
                        {severityBreakdown.map((entry) => (
                          <Cell key={entry.severity} fill={entry.severity === "high" ? "#ef4444" : entry.severity === "medium" ? "#f59e0b" : "#3b82f6"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Section>

                {/* Detection method breakdown */}
                <Section title="Detection Method Breakdown" subtitle="How each anomaly was flagged">
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={methodBreakdown} barCategoryGap="35%">
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="method" tick={{ fontSize: 12 }} />
                      <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                      <Tooltip />
                      <Bar dataKey="count" name="Count" fill="#6366f1" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Section>

                {/* Anomaly value distribution */}
                <Section title="Anomaly Value Distribution" subtitle="Scatter of flagged values by index">
                  <ResponsiveContainer width="100%" height={220}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="index" name="Index" tick={{ fontSize: 10 }} label={{ value: "Anomaly #", position: "insideBottom", offset: -4, fontSize: 11 }} />
                      <YAxis dataKey="value" name="Value" unit=" W/m²" tick={{ fontSize: 11 }} />
                      <ZAxis range={[30, 30]} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v) => [`${Number(v).toFixed(2)} W/m²`]} />
                      <Scatter
                        name="Anomaly"
                        data={anomalies.map((a, i) => ({ index: i + 1, value: a.value }))}
                        fill="#f97316"
                        fillOpacity={0.7}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </Section>
              </div>

              {/* Anomaly table */}
              <Section title={`Detected Anomalies (${anomalies.length})`}>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
                      <tr>
                        {["#", "Timestamp", "Value (W/m²)", "Expected", "Deviation", "Method", "Severity"].map((h) => (
                          <th key={h} className="px-4 py-2 text-left">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {anomalies.map((a, i) => (
                        <tr key={i} className="hover:bg-gray-50">
                          <td className="px-4 py-2 text-gray-400 text-xs">{i + 1}</td>
                          <td className="px-4 py-2 font-mono text-xs">{new Date(a.timestamp).toLocaleString()}</td>
                          <td className="px-4 py-2 font-semibold text-orange-600">{a.value?.toFixed(2)}</td>
                          <td className="px-4 py-2 text-gray-600">{a.expected?.toFixed(2) ?? "—"}</td>
                          <td className="px-4 py-2 text-red-500">{a.deviation?.toFixed(2) ?? "—"}</td>
                          <td className="px-4 py-2"><span className="px-2 py-0.5 bg-gray-100 rounded text-xs font-mono">{a.method}</span></td>
                          <td className="px-4 py-2"><SeverityBadge severity={a.severity} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Section>
            </>
          )}
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════════
          TAB: AI Correction
      ══════════════════════════════════════════════════════════════════════ */}
      {activeTab === "correction" && correctionResult && (
        <div className="space-y-6">

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Correction source breakdown */}
            <Section title="Correction Method Breakdown" subtitle="How each anomaly was corrected">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={sourceBreakdown} barCategoryGap="35%">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="source" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" name="Count" radius={[6, 6, 0, 0]}>
                    {sourceBreakdown.map((s) => <Cell key={s.source} fill={s.fill} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Section>

            {/* Delta scatter — magnitude of each correction */}
            <Section title="Correction Magnitude per Point" subtitle="How much each value was shifted">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={correctionDeltaData} barCategoryGap="15%">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={Math.floor(correctionDeltaData.length / 10)} />
                  <YAxis tick={{ fontSize: 11 }} unit=" W/m²" />
                  <Tooltip formatter={fmtWm2} />
                  <Bar dataKey="delta" name="Delta" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Section>

            {/* Original vs corrected values */}
            <Section title="Original vs Corrected Values" subtitle="Side-by-side comparison per correction" className="lg:col-span-2">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={correctionDeltaData} barCategoryGap="20%">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={Math.floor(correctionDeltaData.length / 10)} />
                  <YAxis tick={{ fontSize: 11 }} unit=" W/m²" />
                  <Tooltip formatter={fmtWm2} />
                  <Legend />
                  <Bar dataKey="original"  name="Original"  fill="#f97316" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="corrected" name="Corrected" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Section>
          </div>

          {/* Correction log table */}
          <Section
            title={`Correction Log (${correction_log?.length ?? 0} entries)`}
            subtitle={`Avg delta: ${stats?.avg_correction_delta?.toFixed(2)} W/m²  |  Max delta: ${stats?.max_correction_delta?.toFixed(2)} W/m²  |  AI corrections: ${stats?.ai_corrections ?? 0}`}
          >
            <div className="space-y-4">
              {/* Source filter legend */}
              <div className="flex flex-wrap gap-4 text-xs p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-purple-500 rounded"></div>
                  <span>AI Corrected (<strong>{stats?.ai_corrections ?? 0}</strong>)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-cyan-500 rounded"></div>
                  <span>Physics Rule (<strong>{stats?.physics_rule_corrections ?? 0}</strong>)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-gray-500 rounded"></div>
                  <span>Interpolation (<strong>{stats?.interpolation_fallbacks ?? 0}</strong>)</span>
                </div>
              </div>

              {/* Correction table */}
              <div className="overflow-x-auto border border-gray-200 rounded-lg">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 text-gray-500 text-xs uppercase sticky top-0">
                    <tr>
                      {["#", "Timestamp", "Original", "Corrected", "Δ", "Source", "Confidence", "Reasoning"].map((h) => (
                        <th key={h} className="px-4 py-2 text-left whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {correction_log?.map((c, i) => {
                      const delta = c.corrected_value - c.original_value;
                      const sourceColor = c.correction_source === "ai" ? "bg-purple-50" : c.correction_source === "physics_rule" ? "bg-cyan-50" : "bg-gray-50";
                      const sourceBorder = c.correction_source === "ai" ? "border-l-4 border-l-purple-500" : c.correction_source === "physics_rule" ? "border-l-4 border-l-cyan-500" : "border-l-4 border-l-gray-400";
                      return (
                        <tr key={i} className={`hover:bg-yellow-50 align-top ${sourceColor} ${sourceBorder}`}>
                          <td className="px-4 py-2 text-gray-400 text-xs">{i + 1}</td>
                          <td className="px-4 py-2 font-mono text-xs whitespace-nowrap">{new Date(c.timestamp).toLocaleString()}</td>
                          <td className="px-4 py-2 font-semibold text-orange-600">{c.original_value.toFixed(2)}</td>
                          <td className={`px-4 py-2 font-bold text-lg ${c.correction_source === "ai" ? "text-purple-700" : c.correction_source === "physics_rule" ? "text-cyan-700" : "text-gray-700"}`}>
                            {c.corrected_value.toFixed(2)}
                          </td>
                          <td className={`px-4 py-2 font-semibold text-xs ${delta < 0 ? "text-green-600" : "text-red-500"}`}>
                            {delta > 0 ? "+" : ""}{delta.toFixed(2)}
                          </td>
                          <td className="px-4 py-2"><SourceBadge source={c.correction_source} /></td>
                          <td className="px-4 py-2"><ConfidenceBadge confidence={c.confidence} /></td>
                          <td className="px-4 py-2 text-xs text-gray-600 max-w-xs">{c.reasoning}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Pagination info */}
              <div className="text-xs text-gray-500 text-center">
                Showing {correction_log?.length ?? 0} entries total
              </div>
            </div>
          </Section>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════════
          TAB: Before vs After
      ══════════════════════════════════════════════════════════════════════ */}
      {activeTab === "comparison" && correctionResult && (
        <div className="space-y-6">

          {/* ── Ensemble metric cards ── */}
          {mc && Object.keys(mc).length > 0 && (
            <div>
              <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">Model Metrics</h2>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {mc.rmse  && <MetricCard label="RMSE"  original={mc.rmse.original}  corrected={mc.rmse.corrected}  />}
                {mc.mae   && <MetricCard label="MAE"   original={mc.mae.original}   corrected={mc.mae.corrected}   />}
                {mc.r2    && <MetricCard label="R²"    original={mc.r2.original}     corrected={mc.r2.corrected}    higherIsBetter />}
                {mc.mape  && <MetricCard label="MAPE"  original={mc.mape?.original}  corrected={mc.mape?.corrected} />}
              </div>
            </div>
          )}

          {/* ── Chart 1: Forecast overlay ── */}
          {forecastChartData.length > 0 && (
            <Section
              title="Forecast Comparison — Original vs Corrected vs Actual"
              subtitle="How model predictions shift after AI correction"
            >
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={forecastChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 11 }} unit=" W/m²" />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={fmtWm2} />
                  <Legend />
                  <Line type="monotone" dataKey="Actual"             stroke="#22c55e" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="Original Forecast"  stroke="#f97316" strokeWidth={2} dot={false} strokeDasharray="6 3" />
                  <Line type="monotone" dataKey="Corrected Forecast" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Section>
          )}

          {/* ── Chart 2: Absolute error over time ── */}
          {errorChartData.length > 0 && (
            <Section
              title="Absolute Prediction Error Over Time"
              subtitle="Lower is better — gap between forecast and actual readings"
            >
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={errorChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 11 }} unit=" W/m²" />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={fmtWm2} />
                  <Legend />
                  <Line type="monotone" dataKey="Original Error"  stroke="#f97316" strokeWidth={2} dot={false} strokeDasharray="6 3" />
                  <Line type="monotone" dataKey="Corrected Error" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Section>
          )}

          {/* ── Chart 3 + 4: Scatter — predicted vs actual ── */}
          {scatterOrig.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Section title="Original: Predicted vs Actual" subtitle="Points close to the diagonal = accurate">
                <ResponsiveContainer width="100%" height={260}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="actual"    name="Actual"    unit=" W/m²" tick={{ fontSize: 10 }} label={{ value: "Actual", position: "insideBottom", offset: -4, fontSize: 11 }} />
                    <YAxis dataKey="predicted" name="Predicted" unit=" W/m²" tick={{ fontSize: 10 }} label={{ value: "Predicted", angle: -90, position: "insideLeft", fontSize: 11 }} />
                    <ZAxis range={[20, 20]} />
                    <Tooltip formatter={(v) => [`${Number(v).toFixed(2)} W/m²`]} />
                    <Scatter data={scatterOrig} fill="#f97316" fillOpacity={0.5} name="Original" />
                  </ScatterChart>
                </ResponsiveContainer>
              </Section>

              <Section title="Corrected: Predicted vs Actual" subtitle="Points close to the diagonal = accurate">
                <ResponsiveContainer width="100%" height={260}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="actual"    name="Actual"    unit=" W/m²" tick={{ fontSize: 10 }} label={{ value: "Actual", position: "insideBottom", offset: -4, fontSize: 11 }} />
                    <YAxis dataKey="predicted" name="Predicted" unit=" W/m²" tick={{ fontSize: 10 }} label={{ value: "Predicted", angle: -90, position: "insideLeft", fontSize: 11 }} />
                    <ZAxis range={[20, 20]} />
                    <Tooltip formatter={(v) => [`${Number(v).toFixed(2)} W/m²`]} />
                    <Scatter data={scatterCorr} fill="#8b5cf6" fillOpacity={0.5} name="Corrected" />
                  </ScatterChart>
                </ResponsiveContainer>
              </Section>
            </div>
          )}

          {/* ── Chart 5: Per-model RMSE + MAE grouped bar ── */}
          {modelBarData.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Section title="Per-Model RMSE: Before vs After" subtitle="Lower RMSE = better">
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={modelBarData} barCategoryGap="30%">
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="model" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 11 }} unit=" W/m²" />
                    <Tooltip formatter={fmtWm2} />
                    <Legend />
                    <Bar dataKey="RMSE (Original)"  fill="#f97316" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="RMSE (Corrected)" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Section>

              <Section title="Per-Model MAE: Before vs After" subtitle="Lower MAE = better">
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={modelBarData} barCategoryGap="30%">
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="model" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 11 }} unit=" W/m²" />
                    <Tooltip formatter={fmtWm2} />
                    <Legend />
                    <Bar dataKey="MAE (Original)"  fill="#fb923c" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="MAE (Corrected)" fill="#a78bfa" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Section>
            </div>
          )}

          {/* ── Chart 6: Radar — R² per model ── */}
          {radarData.length > 0 && (
            <Section title="Model R² Score Comparison" subtitle="Higher = better fit — before vs after correction">
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                  <PolarGrid />
                  <PolarAngleAxis dataKey="model" tick={{ fontSize: 12 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} unit="%" />
                  <Radar name="R² Before" dataKey="R² Before" stroke="#f97316" fill="#f97316" fillOpacity={0.25} />
                  <Radar name="R² After"  dataKey="R² After"  stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.25} />
                  <Legend />
                  <Tooltip formatter={(v) => [`${v}%`]} />
                </RadarChart>
              </ResponsiveContainer>
            </Section>
          )}

          {/* ── Per-model full metrics table ── */}
          {modelComp?.original && (
            <Section title="Full Per-Model Metrics Table" subtitle="All metrics before and after correction">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
                    <tr>
                      <th className="px-4 py-2 text-left">Model</th>
                      {["RMSE", "MAE", "R²", "MAPE"].map((m) => (
                        <th key={m} colSpan={2} className="px-3 py-2 text-center border-l border-gray-200">{m}</th>
                      ))}
                    </tr>
                    <tr className="border-t border-gray-100">
                      <th className="px-4 py-1" />
                      {["RMSE", "MAE", "R²", "MAPE"].map((m) => (
                        <>
                          <th key={`${m}-o`} className="px-3 py-1 text-center text-orange-500 font-normal border-l border-gray-200">Before</th>
                          <th key={`${m}-c`} className="px-3 py-1 text-center text-purple-500 font-normal">After</th>
                        </>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {Object.keys(modelComp.original).map((model) => {
                      const orig = modelComp.original[model];
                      const corr = modelComp.corrected?.[model] || {};
                      return (
                        <tr key={model} className="hover:bg-gray-50">
                          <td className="px-4 py-2 font-semibold capitalize">{model}</td>
                          {["rmse", "mae", "r2", "mape"].map((metric) => {
                            const higherBetter = metric === "r2";
                            const improved = higherBetter ? corr[metric] > orig[metric] : corr[metric] < orig[metric];
                            return (
                              <>
                                <td key={`${model}-${metric}-o`} className="px-3 py-2 text-center text-orange-600 border-l border-gray-100">
                                  {orig[metric]?.toFixed(4) ?? "—"}
                                </td>
                                <td key={`${model}-${metric}-c`} className={`px-3 py-2 text-center font-semibold ${improved ? "text-green-600" : "text-red-500"}`}>
                                  {corr[metric]?.toFixed(4) ?? "—"}
                                </td>
                              </>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Section>
          )}
        </div>
      )}
    </div>
  );
}
