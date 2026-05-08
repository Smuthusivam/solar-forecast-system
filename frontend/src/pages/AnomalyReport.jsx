// AnomalyReport.jsx — Full anomaly detection + AI correction + model comparison page

import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, BarChart, Bar, ScatterChart,
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
  saveForecastState, loadForecastState
} from "../services/forecastState";
import {
  Section, TabNav, MetricCard, AlertBox,
  SeverityBadge, ConfidenceBadge, SourceBadge, fmtWm2,
  PageHeader,
} from "../components/ui";

// ── Main component ───────────────────────────────────────────────────────────
export default function AnomalyReport({ datasetId }) {
  const location = useLocation();
  const navigate = useNavigate();
  const savedState = loadForecastState();
  
  const routeSessionId = location.state?.result?.session_id || new URLSearchParams(location.search).get("session_id");
  const sessionId = datasetId || routeSessionId || savedState?.result?.session_id;
  const forecast = location.state?.forecast || savedState?.forecast;
  const result   = location.state?.result || savedState?.result;

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

  // Save state when it changes
  useEffect(() => {
    if (forecast && result) {
      saveForecastState(forecast, result, result.filename);
    }
  }, [forecast, result]);

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

  const { correction_log, stats } = correctionResult || {};

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
    { key: "correction", label: "Correction", disabled: !correctionResult },
    { key: "comparison", label: "Before & After", disabled: !correctionResult },
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-6 space-y-6">

      <PageHeader
        title="Anomaly Report"
        subtitle="Detection, validation & AI-assisted correction"
        actions={[
          <button
            key="compare"
            onClick={() => navigate("/compare", { state: { forecast, result } })}
            className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:bg-slate-50"
          >
            Model Comparison
          </button>,
          correctionResult && (
            <a
              key="download-csv"
              href={getCorrectedCSVUrl(correctionResult.session_id)}
              className="rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm font-medium text-emerald-700 transition hover:bg-emerald-100"
            >
              Corrected CSV
            </a>
          ),
          correctionResult && (
            <button
              key="download-compare"
              onClick={() => downloadComparisonCSV(correctionResult)}
              className="rounded-full border border-indigo-200 bg-indigo-50 px-4 py-2 text-sm font-medium text-indigo-700 transition hover:bg-indigo-100"
            >
              Comparison CSV
            </button>
          ),
          correctionResult && (
            <button
              key="forecast"
              onClick={() => navigate("/forecast", {
                state: {
                  correctionSessionId: correctionResult.session_id,
                  sessionId,
                  filename: result?.filename || "corrected dataset",
                  fromCorrection: true,
                }
              })}
              className="rounded-full bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800"
            >
              Open Forecast
            </button>
          ),
          <button
            key="run"
            onClick={handleRunCorrection}
            disabled={loadingCorrection || loadingAnomalies}
            className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loadingCorrection ? "Running correction..." : "Run correction"}
          </button>,
        ].filter(Boolean)}
      />

      {correctionError && (
        <AlertBox>Correction failed: {correctionError}</AlertBox>
      )}

      {/* Total anomalies — always visible once loaded */}
      {!loadingAnomalies && !errorAnomalies && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="bg-white rounded-xl border border-gray-200 p-4 text-center">
            <div className="text-3xl font-bold text-orange-600">{anomalies.length}</div>
            <div className="text-xs text-gray-500 mt-1">Total Anomalies</div>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-4 text-center">
            <div className="text-3xl font-bold text-red-500">{anomalies.filter(a => a.severity === "high").length}</div>
            <div className="text-xs text-gray-500 mt-1">High Severity</div>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-4 text-center">
            <div className="text-3xl font-bold text-yellow-500">{anomalies.filter(a => a.severity === "medium").length}</div>
            <div className="text-xs text-gray-500 mt-1">Medium Severity</div>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-4 text-center">
            <div className="text-3xl font-bold text-blue-500">{anomalies.filter(a => a.severity === "low").length}</div>
            <div className="text-xs text-gray-500 mt-1">Low Severity</div>
          </div>
        </div>
      )}

      {/* Correction stats — visible after correction runs */}
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
                            {["No.", "Timestamp", "Value (W/m²)", "Expected", "Deviation", "Detection method", "Severity"].map((h) => (
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
      {activeTab === "comparison" && correctionResult && (() => {
        const log = correction_log || [];
        // Exclude trivial nighttime physics-rule corrections (value was already 0)
        const meaningful = log.filter(c => c.correction_source !== "physics_rule");

        const timelineCompare = meaningful.map((c, i) => ({
          name:      `#${i + 1}`,
          Original:  parseFloat(c.original_value.toFixed(2)),
          Corrected: parseFloat(c.corrected_value.toFixed(2)),
          delta:     parseFloat((c.corrected_value - c.original_value).toFixed(2)),
        }));

        return (
          <div className="space-y-6">

            {/* Summary stat cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                { label: "AI Corrections",  value: stats?.ai_corrections,          color: "text-purple-600", sub: "Claude corrected"   },
                { label: "Physics Rules",   value: stats?.physics_rule_corrections, color: "text-cyan-600",   sub: "nighttime = 0"     },
                { label: "Interpolated",    value: stats?.interpolation_fallbacks,  color: "text-slate-500",  sub: "neighbour average" },
                { label: "Avg Change",      value: `${(stats?.avg_correction_delta || 0).toFixed(1)} W/m²`,
                                            color: "text-orange-600", sub: `max ${(stats?.max_correction_delta || 0).toFixed(1)} W/m²` },
              ].map(({ label, value, color, sub }) => (
                <div key={label} className="bg-white rounded-xl border border-gray-200 p-4 text-center">
                  <div className={`text-2xl font-bold ${color}`}>{value}</div>
                  <div className="text-xs text-gray-500 mt-1">{label}</div>
                  <div className="text-xs text-gray-400">{sub}</div>
                </div>
              ))}
            </div>

            {meaningful.length === 0 ? (
              <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-8 rounded-xl text-center">
                <p className="text-lg font-semibold">All corrections were nighttime physics rules</p>
                <p className="text-sm mt-1">No AI or interpolation corrections were needed.</p>
              </div>
            ) : (
              <>
                {/* Chart 1: Original vs Corrected side-by-side */}
                <Section
                  title="Original vs Corrected Irradiance Values"
                  subtitle="Each corrected point — before (orange) and after (purple) AI correction"
                >
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={timelineCompare} barCategoryGap="20%">
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={Math.max(0, Math.floor(timelineCompare.length / 12) - 1)} />
                      <YAxis tick={{ fontSize: 11 }} unit=" W/m²" />
                      <Tooltip formatter={fmtWm2} />
                      <Legend />
                      <Bar dataKey="Original"  fill="#f97316" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Corrected" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Section>

                {/* Chart 2: Net delta — green=decreased (fixed), red=increased */}
                <Section
                  title="Correction Delta per Point"
                  subtitle="Green = value reduced (spike fixed), red = value increased (dip fixed)"
                >
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={timelineCompare} barCategoryGap="15%">
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={Math.max(0, Math.floor(timelineCompare.length / 12) - 1)} />
                      <YAxis tick={{ fontSize: 11 }} unit=" W/m²" />
                      <Tooltip formatter={fmtWm2} />
                      <Bar dataKey="delta" name="Δ Value" radius={[4, 4, 0, 0]}>
                        {timelineCompare.map((entry, i) => (
                          <Cell key={i} fill={entry.delta < 0 ? "#22c55e" : "#ef4444"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Section>

                {/* Chart 3: Scatter original vs corrected */}
                <Section
                  title="Original vs Corrected — Scatter"
                  subtitle="Points on the diagonal = unchanged. Deviation = correction applied."
                >
                  <ResponsiveContainer width="100%" height={280}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="Original"  name="Original"  unit=" W/m²" tick={{ fontSize: 10 }} label={{ value: "Original (W/m²)",  position: "insideBottom", offset: -4, fontSize: 11 }} />
                      <YAxis dataKey="Corrected" name="Corrected" unit=" W/m²" tick={{ fontSize: 10 }} label={{ value: "Corrected (W/m²)", angle: -90, position: "insideLeft", fontSize: 11 }} />
                      <ZAxis range={[25, 25]} />
                      <Tooltip formatter={(v) => [`${Number(v).toFixed(2)} W/m²`]} />
                      <Scatter data={timelineCompare} fill="#8b5cf6" fillOpacity={0.6} name="Correction" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </Section>
              </>
            )}

            <div className="pt-4 flex justify-end">
              <button
                onClick={() => navigate("/compare", { state: { forecast, result } })}
                className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:bg-slate-50"
              >
                Back to Model Comparison
              </button>
            </div>
          </div>
        );
      })()}
        </div>
      )}
    </div>
  );
}
