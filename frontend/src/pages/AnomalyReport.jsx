// AnomalyReport.jsx -- Full anomaly detection + AI correction + model comparison page

import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, BarChart, Bar, Cell,
  ScatterChart, Scatter, ZAxis,
} from "recharts";
import {
  getAnomalies,
  runCorrection,
  runForecastFromCorrected,
  getCorrectedCSVUrl,
  downloadComparisonCSV,
  readAnomalyReportCache,
  writeAnomalyReportCache,
} from "../services/api";
import {
  saveForecastState, loadForecastState
} from "../services/forecastState";
import {
  Section, SeverityBadge, ConfidenceBadge, SourceBadge, fmtWm2,
  PageHeader, AlertBox,
} from "../components/ui";

// -- Main component -----------------------------------------------------------
export default function AnomalyReport({ datasetId }) {
  const location = useLocation();
  const navigate = useNavigate();
  const savedState = loadForecastState();

  const routeSessionId = location.state?.result?.session_id || new URLSearchParams(location.search).get("session_id");
  const sessionId = datasetId || routeSessionId || savedState?.result?.session_id;
  const forecast = location.state?.forecast || savedState?.forecast;
  const result   = location.state?.result || savedState?.result;

  const [anomalies,         setAnomalies]         = useState([]);
  const [loadingAnomalies,  setLoadingAnomalies]  = useState(true);
  const [errorAnomalies,    setErrorAnomalies]     = useState(null);
  const [correctionResult,  setCorrectionResult]  = useState(null);
  const [loadingCorrection, setLoadingCorrection] = useState(false);
  const [correctionError,   setCorrectionError]   = useState(null);
  const [comparisonResult,  setComparisonResult]  = useState(null);
  const [loadingComparison, setLoadingComparison] = useState(false);
  const [activeTab,         setActiveTab]         = useState("detection");

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
      setComparisonResult(cached.comparisonResult || null);
      setActiveTab(cached.activeTab || "comparison");
    } else {
      setCorrectionResult(null);
      setComparisonResult(null);
      setActiveTab("detection");
    }
  }, [sessionId]);

  useEffect(() => {
    if (forecast && result) {
      saveForecastState(forecast, result, result.filename);
    }
  }, [forecast, result]);

  const handleRunCorrection = async () => {
    setLoadingCorrection(true);
    setCorrectionError(null);
    setComparisonResult(null);
    try {
      const res = await runCorrection(sessionId);
      setCorrectionResult(res);

      // Immediately run ML on corrected data so Before & After tab has comparison data
      setLoadingComparison(true);
      let cmpResult = null;
      try {
        cmpResult = await runForecastFromCorrected(res.session_id, 24, 80);
        setComparisonResult(cmpResult);
      } catch (_) {
        // comparison is best-effort — don't block the rest of the UI
      } finally {
        setLoadingComparison(false);
      }

      writeAnomalyReportCache(sessionId, {
        anomalies,
        correctionResult: res,
        comparisonResult: cmpResult,
        activeTab: "comparison",
      });
      setActiveTab("comparison");
    } catch (e) {
      setCorrectionError(e.response?.data?.detail || e.message);
    } finally {
      setLoadingCorrection(false);
    }
  };

  // -- Derived chart data ------------------------------------------------------

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

  const correctionDeltaData = (correction_log || []).map((c, i) => ({
    name:      `#${i + 1}`,
    original:  c.original_value,
    corrected: c.corrected_value,
    delta:     Math.abs(c.corrected_value - c.original_value),
  }));

  const sourceBreakdown = [
    { source: "AI",           count: stats?.ai_corrections           || 0, fill: "#8b5cf6" },
    { source: "Physics Rule", count: stats?.physics_rule_corrections || 0, fill: "#06b6d4" },
    { source: "Interpolation",count: stats?.interpolation_fallbacks  || 0, fill: "#94a3b8" },
  ];

  // -- Tab config --------------------------------------------------------------

  const TABS = [
    { id: "detection",  label: "Detection" },
    { id: "correction", label: "Correction",   disabled: !correctionResult },
    { id: "comparison", label: "Before & After", disabled: !correctionResult },
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
            {loadingCorrection ? "Running correction..." : "Run Correction"}
          </button>,
        ].filter(Boolean)}
      />

      {correctionError && (
        <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          Correction failed: {correctionError}
        </div>
      )}

      {/* Summary stat cards -- always visible once loaded */}
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

      {/* Correction stats -- visible after correction runs */}
      {correctionResult && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { label: "Anomalies Found", value: correctionResult.anomaly_count,        color: "text-orange-600" },
            { label: "AI Corrected",    value: stats?.ai_corrections,                 color: "text-purple-600" },
            { label: "Physics Rule",    value: stats?.physics_rule_corrections,        color: "text-cyan-600"   },
            { label: "Interpolated",    value: stats?.interpolation_fallbacks,         color: "text-slate-500"  },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-white rounded-xl border border-gray-200 p-4 text-center">
              <div className={`text-3xl font-bold ${color}`}>{value ?? 0}</div>
              <div className="text-xs text-gray-500 mt-1">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-gray-200">
        {TABS.map(({ id, label, disabled }) => (
          <button
            key={id}
            onClick={() => !disabled && setActiveTab(id)}
            disabled={disabled}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition
              ${activeTab === id
                ? "bg-white border border-b-white border-gray-200 text-purple-700 -mb-px"
                : disabled
                  ? "text-gray-300 cursor-not-allowed"
                  : "text-gray-500 hover:text-gray-700"}`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ===================================================================
          TAB: Detection
      =================================================================== */}
      {activeTab === "detection" && (
        <div className="space-y-6">
          {loadingAnomalies ? (
            <div className="flex items-center justify-center h-40 text-gray-400">Loading anomalies...</div>
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
                      <YAxis tick={{ fontSize: 11 }} unit=" W/m2" />
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
                          <Cell key={entry.severity} fill={
                            entry.severity === "high" ? "#ef4444" :
                            entry.severity === "medium" ? "#f59e0b" : "#3b82f6"
                          } />
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

                {/* Anomaly value scatter */}
                <Section title="Anomaly Value Distribution" subtitle="Scatter of flagged values by index">
                  <ResponsiveContainer width="100%" height={220}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="index" name="Index" tick={{ fontSize: 10 }}
                        label={{ value: "Anomaly #", position: "insideBottom", offset: -4, fontSize: 11 }} />
                      <YAxis dataKey="value" name="Value" unit=" W/m2" tick={{ fontSize: 11 }} />
                      <ZAxis range={[30, 30]} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }}
                        formatter={(v) => [`${Number(v).toFixed(2)} W/m2`]} />
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
                        {["No.", "Timestamp", "Value (W/m2)", "Expected", "Deviation", "Detection Method", "Severity"].map((h) => (
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
                          <td className="px-4 py-2 text-gray-600">{a.expected?.toFixed(2) ?? "--"}</td>
                          <td className="px-4 py-2 text-red-500">{a.deviation?.toFixed(2) ?? "--"}</td>
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

      {/* ===================================================================
          TAB: AI Correction
      =================================================================== */}
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

            {/* Delta bar -- magnitude of each correction */}
            <Section title="Correction Magnitude per Point" subtitle="How much each value was shifted">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={correctionDeltaData} barCategoryGap="15%">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }}
                    interval={Math.max(0, Math.floor(correctionDeltaData.length / 10) - 1)} />
                  <YAxis tick={{ fontSize: 11 }} unit=" W/m2" />
                  <Tooltip formatter={fmtWm2} />
                  <Bar dataKey="delta" name="Delta" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Section>

            {/* Original vs corrected */}
            <Section title="Original vs Corrected Values" subtitle="Side-by-side comparison per correction" className="lg:col-span-2">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={correctionDeltaData} barCategoryGap="20%">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }}
                    interval={Math.max(0, Math.floor(correctionDeltaData.length / 10) - 1)} />
                  <YAxis tick={{ fontSize: 11 }} unit=" W/m2" />
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
            subtitle={`Avg delta: ${stats?.avg_correction_delta?.toFixed(2)} W/m2  |  Max delta: ${stats?.max_correction_delta?.toFixed(2)} W/m2  |  AI corrections: ${stats?.ai_corrections ?? 0}`}
          >
            <div className="space-y-4">
              <div className="flex flex-wrap gap-4 text-xs p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-purple-500 rounded" />
                  <span>AI Corrected (<strong>{stats?.ai_corrections ?? 0}</strong>)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-cyan-500 rounded" />
                  <span>Physics Rule (<strong>{stats?.physics_rule_corrections ?? 0}</strong>)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-gray-500 rounded" />
                  <span>Interpolation (<strong>{stats?.interpolation_fallbacks ?? 0}</strong>)</span>
                </div>
              </div>

              <div className="overflow-x-auto border border-gray-200 rounded-lg">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 text-gray-500 text-xs uppercase sticky top-0">
                    <tr>
                      {["#", "Timestamp", "Original", "Corrected", "Delta", "Source", "Confidence", "Reasoning"].map((h) => (
                        <th key={h} className="px-4 py-2 text-left whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {correction_log?.map((c, i) => {
                      const delta = c.corrected_value - c.original_value;
                      const sourceColor  = c.correction_source === "ai" ? "bg-purple-50" : c.correction_source === "physics_rule" ? "bg-cyan-50" : "bg-gray-50";
                      const sourceBorder = c.correction_source === "ai" ? "border-l-4 border-l-purple-500" : c.correction_source === "physics_rule" ? "border-l-4 border-l-cyan-500" : "border-l-4 border-l-gray-400";
                      return (
                        <tr key={i} className={`hover:bg-yellow-50 align-top ${sourceColor} ${sourceBorder}`}>
                          <td className="px-4 py-2 text-gray-400 text-xs">{i + 1}</td>
                          <td className="px-4 py-2 font-mono text-xs whitespace-nowrap">{new Date(c.timestamp).toLocaleString()}</td>
                          <td className="px-4 py-2 font-semibold text-orange-600">{c.original_value.toFixed(2)}</td>
                          <td className={`px-4 py-2 font-bold ${c.correction_source === "ai" ? "text-purple-700" : c.correction_source === "physics_rule" ? "text-cyan-700" : "text-gray-700"}`}>
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

              <div className="text-xs text-gray-500 text-center">
                Showing {correction_log?.length ?? 0} entries total
              </div>
            </div>
          </Section>
        </div>
      )}

      {/* ===================================================================
          TAB: Before & After
      =================================================================== */}
      {activeTab === "comparison" && correctionResult && (() => {
        // Original forecast metrics come from location.state / savedState
        const origMetrics   = forecast?.metrics;
        const origModels    = forecast?.models_info || [];
        const origPoints    = forecast?.forecast    || [];

        // Corrected forecast metrics come from the pipeline run after correction
        const cmpMetrics    = comparisonResult?.metrics;
        const cmpModels     = comparisonResult?.models_info || [];
        const cmpPoints     = comparisonResult?.forecast    || [];

        // Model fitness status label from R² value
        const getFitStatus = (r2) => {
          if (r2 == null) return null;
          if (r2 >= 0.95) return { label: "Excellent fit",  cls: "bg-green-100 text-green-700 border-green-300" };
          if (r2 >= 0.85) return { label: "Good fit",       cls: "bg-blue-100 text-blue-700 border-blue-300"   };
          if (r2 >= 0.75) return { label: "Acceptable fit", cls: "bg-yellow-100 text-yellow-700 border-yellow-300" };
          if (r2 >= 0.50) return { label: "Weak fit",       cls: "bg-orange-100 text-orange-700 border-orange-300" };
          if (r2 >= 0)    return { label: "Poor fit",       cls: "bg-red-100 text-red-600 border-red-300"      };
          return               { label: "Failing",          cls: "bg-red-200 text-red-800 border-red-400"      };
        };

        // Build metric improvement cards
        const metricCards = origMetrics && cmpMetrics
          ? [
              { label: "RMSE",  orig: origMetrics.rmse,  corr: cmpMetrics.rmse,  lowerBetter: true  },
              { label: "MAE",   orig: origMetrics.mae,   corr: cmpMetrics.mae,   lowerBetter: true  },
              { label: "R²",    orig: origMetrics.r2,    corr: cmpMetrics.r2,    lowerBetter: false },
            ].filter(m => m.orig != null && m.corr != null)
          : [];

        const step = Math.max(1, Math.floor(Math.max(origPoints.length, cmpPoints.length) / 200));

        // Per-model chart data: before predictions come from origModels, after from cmpModels
        const modelChartData = (() => {
          if (!origModels.length || !origPoints.length) return {};
          const out = {};
          origModels.forEach(om => {
            const cm = cmpModels.find(c => c.model_name === om.model_name);
            out[om.model_name] = origPoints
              .filter((_, i) => i % step === 0)
              .map((p, i) => ({
                time:   p.timestamp.substring(5, 16).replace("T", " "),
                Actual: p.actual != null ? parseFloat(p.actual.toFixed(1)) : null,
                Before: om.predictions?.[i * step] != null
                  ? parseFloat(om.predictions[i * step].toFixed(1)) : null,
                After:  cm?.predictions?.[i * step] != null
                  ? parseFloat(cm.predictions[i * step].toFixed(1)) : null,
              }));
          });
          return out;
        })();

        return (
          <div className="space-y-6">

            {loadingComparison && (
              <AlertBox variant="info">
                Running model on corrected data for comparison...
              </AlertBox>
            )}

            {!origMetrics && !loadingComparison && (
              <AlertBox variant="warning">
                Original forecast metrics not available. Run a forecast from the Upload page first.
              </AlertBox>
            )}

            {/* Metric improvement cards */}
            {metricCards.length > 0 && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {metricCards.map(({ label, orig, corr, lowerBetter }) => {
                  const improved = lowerBetter ? corr < orig : corr > orig;
                  const pct = orig !== 0 ? Math.abs((corr - orig) / orig * 100).toFixed(2) : null;
                  const origStatus = label === "R²" ? getFitStatus(orig) : null;
                  const corrStatus = label === "R²" ? getFitStatus(corr) : null;
                  return (
                    <div key={label} className="bg-white rounded-xl border border-gray-200 p-4 space-y-2">
                      <div className="text-xs font-semibold text-gray-500 uppercase">{label}</div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-gray-400 w-14">Before</span>
                          <span className="font-mono text-gray-600">{orig.toFixed(4)}</span>
                        </div>
                        {origStatus && (
                          <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${origStatus.cls}`}>
                            {origStatus.label}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-gray-400 w-14">After</span>
                          <span className={`font-mono font-semibold ${improved ? "text-green-600" : "text-red-500"}`}>
                            {corr.toFixed(4)}
                          </span>
                        </div>
                        {corrStatus && (
                          <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${corrStatus.cls}`}>
                            {corrStatus.label}
                          </span>
                        )}
                      </div>
                      {pct && (
                        <div className={`pt-1 border-t border-gray-100 text-xs font-semibold flex items-center gap-1 ${improved ? "text-green-600" : "text-red-500"}`}>
                          {improved ? "▼" : "▲"} {pct}% {improved ? "improvement" : "degradation"}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {/* Per-model individual charts — Before vs After vs Actual */}
            {Object.keys(modelChartData).length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {origModels.map(om => {
                  const cm      = cmpModels.find(c => c.model_name === om.model_name);
                  const data    = modelChartData[om.model_name] || [];
                  const color   = om.model_name === "XGBoost" ? "#3b82f6" : "#10b981";
                  const origSt  = getFitStatus(om.metrics.r2);
                  const corrSt  = getFitStatus(cm?.metrics.r2);
                  return (
                    <Section
                      key={om.model_name}
                      title={om.model_name}
                      subtitle="Predicted vs Actual — before and after AI correction"
                    >
                      {/* Status badges */}
                      <div className="flex items-center gap-2 mb-3 flex-wrap">
                        {origSt && (
                          <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${origSt.cls}`}>
                            Before: {origSt.label}
                          </span>
                        )}
                        {corrSt && cm && (
                          <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${corrSt.cls}`}>
                            After: {corrSt.label}
                          </span>
                        )}
                        {om.is_best && (
                          <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 border border-blue-200 font-medium">
                            Best model
                          </span>
                        )}
                      </div>

                      {/* Metrics row */}
                      <div className="grid grid-cols-3 gap-2 mb-3 text-center text-xs">
                        {["rmse", "mae", "r2"].map(k => {
                          const improved = k === "r2"
                            ? (cm?.metrics[k] ?? 0) > om.metrics[k]
                            : (cm?.metrics[k] ?? Infinity) < om.metrics[k];
                          return (
                            <div key={k} className="bg-gray-50 rounded-lg p-2">
                              <div className="text-gray-400 uppercase font-semibold mb-0.5">{k.toUpperCase()}</div>
                              <div className="text-gray-500 line-through text-[11px]">{om.metrics[k].toFixed(3)}</div>
                              <div className={`font-bold ${improved ? "text-green-600" : "text-red-500"}`}>
                                {cm ? cm.metrics[k].toFixed(3) : "—"}
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {/* Chart */}
                      <ResponsiveContainer width="100%" height={220}>
                        <LineChart data={data}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="time" tick={{ fontSize: 9 }}
                            interval={Math.floor(data.length / 6)} />
                          <YAxis tick={{ fontSize: 10 }} unit=" W/m²" />
                          <Tooltip contentStyle={{ fontSize: 11 }}
                            formatter={(v) => v != null ? [`${v} W/m²`] : ["—"]} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Line type="monotone" dataKey="Actual" stroke="#6b7280" strokeWidth={1.5} dot={false} />
                          <Line type="monotone" dataKey="Before" stroke="#f97316" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                          <Line type="monotone" dataKey="After"  stroke={color}   strokeWidth={1.5} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </Section>
                  );
                })}
              </div>
            )}

            {/* Full per-model metrics table */}
            {origModels.length > 0 && cmpModels.length > 0 && (
              <Section title="Full Per-Model Metrics Table" subtitle="All metrics before and after correction">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
                      <tr>
                        <th className="px-4 py-2 text-left" rowSpan={2}>Model</th>
                        {["RMSE", "MAE", "R²"].map(m => (
                          <th key={m} colSpan={2} className="px-4 py-2 text-center border-l border-gray-200">{m}</th>
                        ))}
                      </tr>
                      <tr>
                        {["RMSE", "MAE", "R²"].map(m => (
                          <>
                            <th key={`${m}-b`} className="px-3 py-1 text-center text-gray-400 border-l border-gray-200 font-normal">Before</th>
                            <th key={`${m}-a`} className="px-3 py-1 text-center text-purple-600 font-normal">After</th>
                          </>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {origModels.map((om, i) => {
                        const cm = cmpModels.find(c => c.model_name === om.model_name);
                        const cell = (orig, corr, lowerBetter) => {
                          const improved = corr != null && (lowerBetter ? corr < orig : corr > orig);
                          return (
                            <>
                              <td className="px-3 py-2 text-center text-gray-500 border-l border-gray-100 font-mono text-xs">{orig?.toFixed(3) ?? "—"}</td>
                              <td className={`px-3 py-2 text-center font-semibold font-mono text-xs ${improved ? "text-green-600" : corr != null ? "text-red-500" : "text-gray-400"}`}>
                                {corr?.toFixed(3) ?? "—"}
                              </td>
                            </>
                          );
                        };
                        return (
                          <tr key={i} className="hover:bg-gray-50">
                            <td className="px-4 py-2 font-medium">
                              <div className="flex items-center gap-2">
                                <span className={`w-2 h-2 rounded-full ${om.model_name === "XGBoost" ? "bg-blue-500" : "bg-emerald-500"}`} />
                                {om.model_name}
                                {om.is_best && <span className="text-xs bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded-full">best</span>}
                              </div>
                            </td>
                            {cell(om.metrics.rmse, cm?.metrics.rmse, true)}
                            {cell(om.metrics.mae,  cm?.metrics.mae,  true)}
                            {cell(om.metrics.r2,   cm?.metrics.r2,   false)}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Section>
            )}

            <div className="pt-2 flex justify-end">
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
  );
}
