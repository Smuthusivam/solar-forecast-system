// Forecast.jsx — User-configurable future forecast page

import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";
import { runForecast, runForecastFromCorrected } from "../services/api";
import { StatCard, TabNav } from "../components/ui";

// ── Preset horizon options ────────────────────────────────────────────────────
const PRESETS = [
  { label: "24 h",   value: 24,  desc: "Next day"  },
  { label: "48 h",   value: 48,  desc: "2 days"    },
  { label: "72 h",   value: 72,  desc: "3 days"    },
  { label: "1 week", value: 168, desc: "7 days"    },
  { label: "Custom", value: 0,   desc: "Set hours"  },
];

export default function Forecast() {
  const location = useLocation();
  const navigate = useNavigate();
  const sessionId           = location.state?.sessionId || location.state?.result?.session_id;
  const correctionSessionId = location.state?.correctionSessionId;
  const fromCorrection      = location.state?.fromCorrection || false;
  const filename            = location.state?.filename || location.state?.result?.filename || "dataset";

  const [selectedPreset, setSelectedPreset] = useState(24);
  const [customHours,    setCustomHours]    = useState(96);
  const [trainSize,      setTrainSize]      = useState(80);
  const [loading,        setLoading]        = useState(false);
  const [error,          setError]          = useState(null);
  const [result,         setResult]         = useState(null);
  const [activeTab,      setActiveTab]      = useState("chart");

  const horizon = selectedPreset === 0 ? customHours : selectedPreset;

  const handleRun = async () => {
    if (!sessionId && !correctionSessionId) { setError("No session found. Upload a CSV first."); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const res = correctionSessionId
        ? await runForecastFromCorrected(correctionSessionId, horizon, trainSize)
        : await runForecast(sessionId, horizon, trainSize);
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

  const peakPoint   = futurePoints.reduce((best, p) => p.predicted > (best?.predicted ?? -1) ? p : best, null);
  const totalEnergy = futurePoints.reduce((s, p) => s + (p.predicted || 0), 0);
  const nightHours  = futurePoints.filter(p => p.predicted === 0).length;
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

  // Daily energy totals from the future forecast
  const dailyEnergy = (() => {
    const days = {};
    futurePoints.forEach(p => {
      const day = p.timestamp.substring(0, 10);
      days[day] = (days[day] || 0) + p.predicted;
    });
    return Object.entries(days).map(([date, total]) => ({
      date: date.substring(5),
      energy: +(total / 1000).toFixed(3),
    }));
  })();

  // Per-model comparison on test set
  const modelData = (result?.models_info || []).map(m => ({
    model:  m.model_name,
    RMSE:   +m.metrics.rmse.toFixed(2),
    MAE:    +m.metrics.mae.toFixed(2),
    R2:     +m.metrics.r2.toFixed(4),
    is_best: m.is_best,
  }));

  const tabs = [
    { id: "chart",  label: "Forecast Chart" },
    { id: "daily",  label: "Daily Energy"   },
    { id: "models", label: "Models"         },
    { id: "table",  label: "Hourly Table"   },
  ];


  return (
    <div className="min-h-screen bg-gray-50 p-6 space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Future Forecast</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            {filename} — predict irradiance beyond your data
            {fromCorrection && (
              <span className="ml-2 px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-semibold rounded-full">
                Using AI-corrected data
              </span>
            )}
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => navigate("/history")}
            className="text-sm text-gray-500 border border-gray-300 px-3 py-1.5 rounded-lg hover:bg-gray-50">
            View History
          </button>
          <button onClick={() => navigate("/anomalies", { state: { result: location.state?.result || { session_id: sessionId, filename }, forecast: location.state?.forecast } })}
            className="text-sm text-gray-500 border border-gray-300 px-3 py-1.5 rounded-lg hover:bg-gray-50">
            Open Anomaly
          </button>
          <button onClick={() => navigate(-1)}
            className="text-sm text-gray-500 border border-gray-300 px-3 py-1.5 rounded-lg hover:bg-gray-50">
            ← Back
          </button>
        </div>
      </div>

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
                className={`px-4 py-2 rounded-lg text-sm font-medium border transition
                  ${selectedPreset === p.value
                    ? "bg-blue-600 text-white border-blue-600"
                    : "bg-white text-gray-700 border-gray-300 hover:border-blue-400"}`}
              >
                <span>{p.label}</span>
                <span className="ml-1 text-xs opacity-70">({p.desc})</span>
              </button>
            ))}
          </div>

          {/* Custom hours input */}
          {selectedPreset === 0 && (
            <div className="mt-3 flex items-center gap-3">
              <input
                type="number"
                min={1} max={8760}
                value={customHours}
                onChange={e => setCustomHours(Math.min(8760, Math.max(1, +e.target.value)))}
                className="w-28 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
              />
              <span className="text-sm text-gray-500">hours (max 8760 = 1 year)</span>
            </div>
          )}
        </div>

        {/* Train size slider */}
        <div>
          <label className="text-xs font-medium text-gray-500 uppercase tracking-wide block mb-2">
            Training Data — <span className="text-blue-600 font-semibold">{trainSize}%</span>
            <span className="text-gray-400 ml-2 font-normal normal-case">
              (more training = better model, less test data)
            </span>
          </label>
          <div className="flex items-center gap-4">
            <input
              type="range" min={50} max={95} step={5}
              value={trainSize}
              onChange={e => setTrainSize(+e.target.value)}
              className="flex-1 accent-blue-600"
            />
            <div className="flex gap-2 text-xs text-gray-500">
              <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded">Train {trainSize}%</span>
              <span className="bg-orange-100 text-orange-700 px-2 py-0.5 rounded">Test {100 - trainSize}%</span>
            </div>
          </div>
        </div>

        {/* Run button */}
        <button
          onClick={handleRun}
          disabled={loading || (!sessionId && !correctionSessionId)}
          className="w-full py-3 rounded-xl bg-blue-600 text-white font-semibold text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
        >
          {loading
            ? <><span className="animate-spin inline-block">⚙</span> Running forecast…</>
            : `Run ${horizon}h Forecast`}
        </button>

        {!sessionId && !correctionSessionId && (
          <p className="text-xs text-red-500 text-center">
            No dataset session found.{" "}
            <button onClick={() => navigate("/")} className="underline">Upload a CSV first</button>.
          </p>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-3 text-sm">
            ⚠ {error}
          </div>
        )}
      </div>

      {/* ── Results ─────────────────────────────────────────────────────────── */}
      {result && (
        <>
          {/* Summary stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard label="Horizon" value={horizon} unit="h" color="text-blue-600" sub="hours forecasted" />
            <StatCard label="Peak GHI" value={peakPoint?.predicted?.toFixed(1)} unit="W/m²"
              color="text-orange-500"
              sub={peakPoint ? new Date(peakPoint.timestamp).toLocaleString() : "—"} />
            <StatCard label="Total Energy" value={(totalEnergy / 1000).toFixed(2)} unit="kWh/m²"
              color="text-green-600" sub="integrated irradiance" />
            <StatCard label="Daylight Hours" value={dayHours} unit={`/ ${futurePoints.length}`}
              color="text-purple-600" sub="non-zero predicted hours" />
          </div>

          {/* Model accuracy on test set */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard label="RMSE" value={result.metrics.rmse.toFixed(2)} unit="W/m²"
              color="text-blue-600" sub={`${result.best_model} · test set`} />
            <StatCard label="MAE"  value={result.metrics.mae.toFixed(2)}  unit="W/m²"
              color="text-purple-600" sub={`${result.best_model} · test set`} />
            <StatCard label="R²"   value={result.metrics.r2.toFixed(4)}
              color={result.metrics.r2 > 0.85 ? "text-green-600" : "text-orange-500"}
              sub={result.metrics.r2 > 0.85 ? "excellent fit" : "moderate fit"} />
            <StatCard label="Rows Used" value={result.rows_processed?.toLocaleString()}
              color="text-gray-700" sub="training data points" />
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

          {/* ── Tab: Daily Energy ── */}
          {activeTab === "daily" && (
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 space-y-4">
              <div>
                <h3 className="font-semibold text-gray-900">Daily Energy Yield</h3>
                <p className="text-xs text-gray-400 mt-0.5">Integrated irradiance per day (kWh/m²)</p>
              </div>
              {dailyEnergy.length === 0 ? (
                <p className="text-gray-400 text-sm">Not enough data for daily breakdown.</p>
              ) : (
                <>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={dailyEnergy} barCategoryGap="30%">
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} unit=" kWh/m²" />
                      <Tooltip formatter={(v) => [`${v} kWh/m²`, "Energy"]} />
                      <Bar dataKey="energy" radius={[6, 6, 0, 0]}>
                        {dailyEnergy.map((_, i) => (
                          <Cell key={i} fill={i % 2 === 0 ? "#3b82f6" : "#60a5fa"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>

                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
                        <tr>
                          <th className="px-4 py-2 text-left">Date</th>
                          <th className="px-4 py-2 text-right">Energy (kWh/m²)</th>
                          <th className="px-4 py-2 text-left">Bar</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {dailyEnergy.map((d, i) => {
                          const maxE = Math.max(...dailyEnergy.map(x => x.energy));
                          return (
                            <tr key={i} className="hover:bg-gray-50">
                              <td className="px-4 py-2 font-mono text-xs">{d.date}</td>
                              <td className="px-4 py-2 text-right font-semibold text-blue-600">{d.energy}</td>
                              <td className="px-4 py-2">
                                <div className="bg-gray-100 rounded-full h-2 w-40">
                                  <div className="bg-blue-500 h-2 rounded-full"
                                    style={{ width: `${(d.energy / maxE) * 100}%` }} />
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          )}

          {/* ── Tab: Models ── */}
          {activeTab === "models" && (
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 space-y-4">
              <div>
                <h3 className="font-semibold text-gray-900">Model Performance on Test Set</h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  Evaluated on the held-out {100 - trainSize}% test split before generating future forecast
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {modelData.map(m => (
                  <div key={m.model}
                    className={`rounded-xl p-5 space-y-3 ${m.is_best ? "border-2 border-blue-400 bg-blue-50" : "border border-gray-200"}`}>
                    <div className="flex items-center justify-between">
                      <h4 className="font-semibold text-gray-800">{m.model}</h4>
                      {m.is_best && (
                        <span className="text-sm font-bold text-white bg-blue-600 px-3 py-1 rounded-full">
                          Best
                        </span>
                      )}
                    </div>
                    <div className="grid grid-cols-3 gap-3 text-center">
                      <div>
                        <div className="text-lg font-bold text-blue-600">{m.RMSE}</div>
                        <div className="text-xs text-gray-400">RMSE W/m²</div>
                      </div>
                      <div>
                        <div className="text-lg font-bold text-purple-600">{m.MAE}</div>
                        <div className="text-xs text-gray-400">MAE W/m²</div>
                      </div>
                      <div>
                        <div className={`text-lg font-bold ${m.R2 > 0.85 ? "text-green-600" : "text-orange-500"}`}>{m.R2}</div>
                        <div className="text-xs text-gray-400">R²</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {result.feature_importance && (
                <div>
                  <h4 className="text-sm font-medium text-gray-600 mb-3">Top Feature Importances</h4>
                  <div className="space-y-2">
                    {Object.entries(result.feature_importance).slice(0, 10).map(([feat, score]) => (
                      <div key={feat} className="flex items-center gap-3">
                        <span className="text-sm text-gray-600 w-36 truncate">{feat}</span>
                        <div className="flex-1 bg-gray-100 rounded-full h-2.5">
                          <div className="bg-blue-500 h-2.5 rounded-full"
                            style={{ width: `${(score * 100).toFixed(1)}%` }} />
                        </div>
                        <span className="text-xs text-gray-400 w-10 text-right">
                          {(score * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
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
