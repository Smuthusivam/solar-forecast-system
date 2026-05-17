// Forecast.jsx — User-configurable future forecast page

import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import { runForecast, runForecastFromCorrected } from "../services/api";
import { saveForecastState, loadForecastState } from "../services/forecastState";
import { StatCard, TabNav, PageHeader } from "../components/ui";

// ── Preset horizon options ────────────────────────────────────────────────────
const PRESETS = [
  { label: "24 h",   value: 24,  desc: "Next day" },
  { label: "48 h",   value: 48,  desc: "2 days"   },
  { label: "72 h",   value: 72,  desc: "3 days"   },
  { label: "1 week", value: 168, desc: "7 days"   },
];

export default function Forecast() {
  const location = useLocation();
  const navigate = useNavigate();
  const savedState = loadForecastState();
  
  const sessionId           = location.state?.sessionId || location.state?.result?.session_id || savedState?.result?.session_id;
  const correctionSessionId = location.state?.correctionSessionId ?? savedState?.correctionSessionId ?? null;
  const fromCorrection      = location.state?.fromCorrection || false;
  const filename            = location.state?.filename || location.state?.result?.filename || savedState?.filename || "dataset";

  const [selectedPreset, setSelectedPreset] = useState(24);
  const [loading,        setLoading]        = useState(false);
  const [error,          setError]          = useState(null);
  const [result,         setResult]         = useState(null);
  const [activeTab,      setActiveTab]      = useState("chart");
  const trainSize = location.state?.trainSize ?? savedState?.trainSize ?? 80;

  // Persist correctionSessionId to localStorage so it survives navigation
  useEffect(() => {
    if (correctionSessionId) {
      saveForecastState(
        savedState?.forecast,
        savedState?.result,
        savedState?.filename,
        trainSize,
        correctionSessionId,
      );
    }
  }, [correctionSessionId]);

  const horizon = selectedPreset;

  const handleRun = async () => {
    if (!correctionSessionId) { setError("No corrected dataset found. Please run AI correction first from the Anomaly page."); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await runForecastFromCorrected(correctionSessionId, horizon, trainSize);
      setResult(res);
      setActiveTab("chart");
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  // ── Derived data ────────────────────────────────────────────────────────────
  const futurePoints = result?.future_forecast || [];
  const testPoints   = result?.forecast        || [];

  const chartData = futurePoints.map((p) => ({
    time:      p.timestamp.substring(5, 16).replace("T", " "),
    Forecast:  parseFloat(p.predicted?.toFixed(1)),
  }));

  const peakPoint  = futurePoints.reduce((best, p) => p.predicted > (best?.predicted ?? -1) ? p : best, null);
  const nightHours = futurePoints.filter(p => p.predicted === 0).length;
  const dayHours    = futurePoints.length - nightHours;

  // Hourly average from test set for the background reference curve
  const hourlyAvg = (() => {
    const buckets = Array.from({ length: 24 }, () => ({ sum: 0, n: 0 }));
    testPoints.forEach(p => {
      if (p.actual == null) return;
      const h = new Date(p.timestamp).getHours();
      buckets[h].sum += p.actual; buckets[h].n += 1;
    });
    return buckets.map((b, h) => ({ hour: h, avg: b.n ? +(b.sum / b.n).toFixed(1) : 0 }));
  })();

  const tabs = [
    { id: "chart", label: "Forecast Chart" },
    { id: "table", label: "Hourly Table"   },
  ];


  return (
    <div className="min-h-screen bg-gray-50 p-6 space-y-6">

      <PageHeader
        title="Future Forecast"
        subtitle={
          <>
            {filename} — predict irradiance beyond your data
            {fromCorrection && (
              <span className="ml-2 inline-flex rounded-full bg-violet-100 px-2 py-0.5 text-xs font-semibold text-violet-700">
                Using AI-corrected data
              </span>
            )}
          </>
        }
        actions={[
          <button
            key="history"
            onClick={() => navigate("/history")}
            className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:bg-slate-50"
          >
            View History
          </button>,
          <button
            key="anomaly"
            onClick={() => navigate("/anomalies", { state: { result: location.state?.result || { session_id: sessionId, filename }, forecast: location.state?.forecast } })}
            className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:bg-slate-50"
          >
            Open Anomaly
          </button>,
        ]}
      />

      {fromCorrection && (
        <div className="bg-purple-50 border border-purple-200 rounded-xl px-4 py-3 text-sm text-purple-800">
          Forecasting on <strong>AI-corrected dataset</strong> — anomalies have been detected and corrected before training.
        </div>
      )}

      {/* ── Config panel ────────────────────────────────────────────────────── */}
      <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 space-y-5">
        <h2 className="font-semibold text-gray-800">Forecast Settings</h2>

        {/* Horizon presets */}
        <div>
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide block mb-2">
            Forecast Horizon
          </label>
          <div className="flex gap-2 flex-wrap">
            {PRESETS.map(p => (
              <button
                key={p.value}
                onClick={() => setSelectedPreset(p.value)}
                className={`px-4 py-2 rounded-full text-sm font-medium border transition
                  ${selectedPreset === p.value
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-300 bg-white text-slate-700 hover:border-slate-400 hover:bg-slate-50"}`}
              >
                <span>{p.label}</span>
                <span className="ml-1 text-xs opacity-70">({p.desc})</span>
              </button>
            ))}
          </div>

        </div>

        {/* Run button */}
        <button
          onClick={handleRun}
          disabled={loading || !correctionSessionId}
          className="flex w-full items-center justify-center gap-2 rounded-full bg-slate-900 py-3 text-sm font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? (
            <><span className="animate-spin inline-block">⚙</span> Running forecast…</>
          ) : (
            `Run ${horizon}h Forecast`
          )}
        </button>

        {!correctionSessionId && (
          <p className="text-center text-xs text-rose-600">
            No corrected dataset found. Please run AI correction from the{" "}
            <button onClick={() => navigate("/anomalies")} className="font-medium underline underline-offset-2">Anomaly page</button>{" "}
            first.
          </p>
        )}

        {error && (
          <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            ⚠ {error}
          </div>
        )}
      </div>

      {/* ── Results ─────────────────────────────────────────────────────────── */}
      {result && (
        <>
          {/* Summary stats */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <StatCard label="Horizon" value={horizon} unit="h" color="text-blue-600" sub="hours forecasted" />
            <StatCard label="Peak GHI" value={peakPoint?.predicted?.toFixed(1) ?? "—"} unit="W/m²"
              color="text-orange-500"
              sub={peakPoint ? new Date(peakPoint.timestamp).toLocaleString() : "—"} />
            <StatCard label="Daylight Hours" value={dayHours} unit={`/ ${futurePoints.length}`}
              color="text-purple-600" sub="non-zero predicted hours" />
          </div>

          {/* Best model results */}
          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5">
            <p className="text-sm font-semibold text-gray-700 mb-3">
              Best Model: <span className="text-blue-600">{result.best_model}</span>
            </p>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-xl font-bold text-blue-600">{result.metrics.rmse.toFixed(2)}</div>
                <div className="text-xs text-gray-400 mt-0.5">RMSE W/m²</div>
              </div>
              <div>
                <div className="text-xl font-bold text-purple-600">{result.metrics.mae.toFixed(2)}</div>
                <div className="text-xs text-gray-400 mt-0.5">MAE W/m²</div>
              </div>
              <div>
                <div className={`text-xl font-bold ${result.metrics.r2 > 0.85 ? "text-green-600" : "text-orange-500"}`}>
                  {result.metrics.r2.toFixed(4)}
                </div>
                <div className="text-xs text-gray-400 mt-0.5">R² Score</div>
              </div>
            </div>
          </div>

          <TabNav tabs={tabs} active={activeTab} onChange={setActiveTab} />

          {/* ── Tab: Forecast Chart ── */}
          {activeTab === "chart" && (
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 space-y-4">
              <div>
                <h3 className="font-semibold text-gray-900">
                  {horizon}h Solar Irradiance Forecast
                </h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  Predicted GHI beyond your last data point · trained on {trainSize}% of data
                </p>
              </div>
              <ResponsiveContainer width="100%" height={380}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}    />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }}
                    interval={Math.max(1, Math.floor(chartData.length / 12))} />
                  <YAxis tick={{ fontSize: 11 }}
                    label={{ value: "W/m²", angle: -90, position: "insideLeft" }} />
                  <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }}
                    formatter={(v) => [`${v} W/m²`, "Forecast"]} />
                  <ReferenceLine y={0} stroke="#e5e7eb" />
                  <Area type="monotone" dataKey="Forecast" stroke="#3b82f6"
                    strokeWidth={2} fill="url(#grad)" dot={false} />
                </AreaChart>
              </ResponsiveContainer>

              {/* Hourly avg reference */}
              {hourlyAvg.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-600 mb-2">
                    Historical Hourly Average (reference)
                  </h4>
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart data={hourlyAvg} barCategoryGap="20%">
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="hour" tick={{ fontSize: 10 }}
                        tickFormatter={h => `${h}h`} />
                      <YAxis tick={{ fontSize: 10 }} unit=" W/m²" />
                      <Tooltip formatter={(v) => [`${v} W/m²`, "Avg"]} />
                      <Bar dataKey="avg" fill="#f59e0b" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}

          {/* ── Tab: Hourly Table ── */}
          {activeTab === "table" && (
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-5 py-4 border-b border-gray-100">
                <h3 className="font-semibold text-gray-900">Hourly Forecast ({futurePoints.length} hours)</h3>
                <p className="text-xs text-gray-400 mt-0.5">Full hour-by-hour predicted irradiance</p>
              </div>
              <div className="overflow-x-auto max-h-150 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 text-gray-500 text-xs uppercase sticky top-0">
                    <tr>
                      <th className="px-4 py-2 text-left">#</th>
                      <th className="px-4 py-2 text-left">Timestamp</th>
                      <th className="px-4 py-2 text-right">GHI (W/m²)</th>
                      <th className="px-4 py-2 text-right">Energy (Wh/m²)</th>
                      <th className="px-4 py-2 text-left">Level</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {futurePoints.map((p, i) => {
                      const v = p.predicted;
                      const level = v > 600 ? { label: "High",   cls: "text-orange-600 bg-orange-50" }
                        : v > 200            ? { label: "Medium", cls: "text-yellow-600 bg-yellow-50" }
                        : v > 0              ? { label: "Low",    cls: "text-blue-600 bg-blue-50"   }
                        :                      { label: "Night",  cls: "text-gray-400 bg-gray-50"   };
                      return (
                        <tr key={i} className="hover:bg-gray-50">
                          <td className="px-4 py-2 text-gray-400 text-xs">{i + 1}</td>
                          <td className="px-4 py-2 font-mono text-xs">
                            {new Date(p.timestamp).toLocaleString()}
                          </td>
                          <td className="px-4 py-2 text-right font-semibold">{v.toFixed(1)}</td>
                          <td className="px-4 py-2 text-right text-gray-500">{v.toFixed(1)}</td>
                          <td className="px-4 py-2">
                            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${level.cls}`}>
                              {level.label}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
