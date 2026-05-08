import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from "recharts";
import { getHistory, getForecastPoints } from "../services/api";
import { R2Badge, AlertBox, PageHeader } from "../components/ui";

function History() {
  const navigate = useNavigate();

  const [runs,         setRuns]         = useState([]);
  const [loading,      setLoading]      = useState(true);
  const [error,        setError]        = useState(null);
  const [search,       setSearch]       = useState("");
  const [sortBy,       setSortBy]       = useState("created_at");
  const [sortDir,      setSortDir]      = useState("desc");
  const [expandedId,   setExpandedId]   = useState(null);
  const [expandedData, setExpandedData] = useState(null);
  const [loadingId,    setLoadingId]    = useState(null);

  useEffect(() => {
    getHistory()
      .then(data => { setRuns(data.runs || []); setLoading(false); })
      .catch(err  => { setError("Failed to load history: " + (err?.message || "")); setLoading(false); });
  }, []);

  const filtered = runs
    .filter(r => r.filename.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      let av = a[sortBy], bv = b[sortBy];
      if (typeof av === "string") av = av.toLowerCase();
      if (typeof bv === "string") bv = bv.toLowerCase();
      return sortDir === "asc" ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
    });

  function toggleSort(col) {
    if (sortBy === col) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortBy(col); setSortDir("desc"); }
  }

  function SortIcon({ col }) {
    if (sortBy !== col) return <span className="text-gray-300 ml-1">↕</span>;
    return <span className="text-blue-500 ml-1">{sortDir === "asc" ? "↑" : "↓"}</span>;
  }

  async function handleRowClick(run) {
    // Collapse if already open
    if (expandedId === run.run_id) {
      setExpandedId(null);
      setExpandedData(null);
      return;
    }

    setExpandedId(run.run_id);
    setExpandedData(null);
    setLoadingId(run.run_id);

    try {
      const pointsData = await getForecastPoints(run.run_id);
      setExpandedData({ run, points: pointsData.points || [] });
    } catch {
      setExpandedData({ run, points: [] });
    } finally {
      setLoadingId(null);
    }
  }

  // Summary stats
  const totalRuns = runs.length;
  const avgR2     = runs.length ? runs.reduce((a, r) => a + r.r2, 0) / runs.length : 0;
  const bestRun   = runs.length ? [...runs].sort((a, b) => b.r2 - a.r2)[0] : null;
  const totalRows = runs.reduce((a, r) => a + r.rows_processed, 0);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="mb-4 text-4xl animate-spin">📋</div>
          <p className="text-slate-500">Loading forecast history...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6 space-y-6">

      <PageHeader
        title="Forecast History"
        subtitle="All past forecast runs stored in the database"
        actions={[
          <button key="new" onClick={() => navigate("/")}
            className="rounded-full bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800">
            + New Forecast
          </button>,
        ]}
      />

      {error && <AlertBox>{error}</AlertBox>}

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Total Runs",         value: totalRuns,                   sub: "forecast sessions",    color: "text-slate-900" },
          { label: "Avg R²",             value: avgR2.toFixed(4),            sub: "across all runs",      color: avgR2 > 0.9 ? "text-emerald-600" : "text-amber-600" },
          { label: "Best R²",            value: bestRun ? bestRun.r2.toFixed(4) : "—", sub: bestRun ? bestRun.filename.slice(0, 18) : "no runs yet", color: "text-slate-900" },
          { label: "Total Rows",         value: totalRows.toLocaleString(),  sub: "processed",            color: "text-slate-900" },
        ].map(({ label, value, sub, color }) => (
          <div key={label} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <p className="text-[11px] font-semibold uppercase tracking-widest text-slate-400">{label}</p>
            <p className={`mt-2 text-2xl font-semibold ${color}`}>{value}</p>
            <p className="mt-1 text-xs text-slate-500">{sub}</p>
          </div>
        ))}
      </div>

      {/* Empty state */}
      {runs.length === 0 && !error && (
        <div className="rounded-2xl border border-slate-200 bg-white p-12 text-center shadow-sm">
          <p className="text-5xl mb-4">📭</p>
          <p className="mb-2 text-lg text-slate-500">No forecast runs yet</p>
          <p className="mb-6 text-sm text-slate-400">Upload a CSV and run a forecast to see it appear here</p>
          <button onClick={() => navigate("/")}
            className="rounded-full bg-slate-900 px-6 py-2 text-white transition hover:bg-slate-800">
            Upload a File
          </button>
        </div>
      )}

      {/* Table + inline detail */}
      {runs.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">

          {/* Search */}
          <div className="border-b border-slate-100 p-4">
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search by filename..."
              className="w-full rounded-xl border border-slate-300 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-slate-400"
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-slate-50 text-slate-500">
                  {[
                    { label: "Run #",      col: "run_id",         align: "text-left"   },
                    { label: "File",       col: "filename",       align: "text-left"   },
                    { label: "Model",      col: "best_model",     align: "text-left"   },
                    { label: "Horizon",    col: "horizon",        align: "text-center" },
                    { label: "RMSE",       col: "rmse",           align: "text-right"  },
                    { label: "MAE",        col: "mae",            align: "text-right"  },
                    { label: "R²",         col: "r2",             align: "text-right"  },
                    { label: "Rows",       col: "rows_processed", align: "text-right"  },
                    { label: "Anomalies",  col: "anomaly_count",  align: "text-right"  },
                    { label: "Date",       col: "created_at",     align: "text-right"  },
                  ].map(({ label, col, align }) => (
                    <th key={col}
                      className={`cursor-pointer p-3 ${align} hover:text-slate-900`}
                      onClick={() => toggleSort(col)}>
                      {label} <SortIcon col={col} />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((run, i) => (
                  <>
                    <tr
                      key={run.run_id}
                      onClick={() => handleRowClick(run)}
                      className={`border-b cursor-pointer transition
                        ${expandedId === run.run_id
                          ? "bg-slate-100 border-b-0"
                          : i % 2 === 0 ? "bg-white hover:bg-slate-50" : "bg-slate-50/40 hover:bg-slate-50"}
                        ${loadingId === run.run_id ? "opacity-60" : ""}`}
                    >
                      <td className="p-3 font-mono text-gray-400">
                        {loadingId === run.run_id
                          ? <span className="text-blue-500 animate-pulse">Loading...</span>
                          : <span className="flex items-center gap-1">
                              <span className={`text-xs transition-transform ${expandedId === run.run_id ? "rotate-90" : ""}`}>▶</span>
                              #{run.run_id}
                            </span>
                        }
                      </td>
                      <td className="p-3">
                        <div className="font-medium text-gray-700 truncate max-w-40" title={run.filename}>
                          {run.filename}
                        </div>
                      </td>
                      <td className="p-3 text-gray-600 text-xs">{run.best_model || "—"}</td>
                      <td className="p-3 text-center text-gray-600">{run.horizon}h</td>
                      <td className="p-3 text-right font-mono">{run.rmse.toFixed(2)}</td>
                      <td className="p-3 text-right font-mono">{run.mae.toFixed(2)}</td>
                      <td className="p-3 text-right"><R2Badge value={run.r2} /></td>
                      <td className="p-3 text-right text-gray-600">{run.rows_processed.toLocaleString()}</td>
                      <td className="p-3 text-right">
                        <span className={run.anomaly_count > 200 ? "text-orange-500 font-medium" : "text-gray-600"}>
                          {run.anomaly_count}
                        </span>
                      </td>
                      <td className="p-3 text-right text-gray-400 text-xs font-mono">
                        {new Date(run.created_at).toLocaleString()}
                      </td>
                    </tr>

                    {/* Inline detail panel */}
                    {expandedId === run.run_id && (
                      <tr key={`detail-${run.run_id}`}>
                        <td colSpan={10} className="bg-slate-50 border-b border-slate-200 px-6 py-5">
                          {!expandedData ? (
                            <div className="text-center text-slate-400 py-4">Loading run details...</div>
                          ) : (
                            <RunDetail run={expandedData.run} points={expandedData.points} navigate={navigate} />
                          )}
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>

          <div className="p-4 border-t border-gray-100 flex items-center justify-between">
            <p className="text-xs text-gray-400">
              Showing {filtered.length} of {runs.length} runs
              {search && ` matching "${search}"`}
            </p>
            <p className="text-xs text-gray-400">Click any row to expand run details</p>
          </div>
        </div>
      )}
    </div>
  );
}


// -- Inline detail panel -------------------------------------------------------

function RunDetail({ run, points, navigate }) {
  const testPoints   = points.filter(p => !p.is_future);
  const futurePoints = points.filter(p =>  p.is_future);

  const step = Math.max(1, Math.floor(testPoints.length / 150));
  const chartData = testPoints
    .filter((_, i) => i % step === 0)
    .map(p => ({
      time:      p.timestamp.substring(5, 16).replace("T", " "),
      Predicted: p.predicted != null ? parseFloat(p.predicted.toFixed(1)) : null,
      Actual:    p.actual    != null ? parseFloat(p.actual.toFixed(1))    : null,
    }));

  const futureStep = Math.max(1, Math.floor(futurePoints.length / 100));
  const futureChartData = futurePoints
    .filter((_, i) => i % futureStep === 0)
    .map(p => ({
      time:     p.timestamp.substring(5, 16).replace("T", " "),
      Forecast: p.predicted != null ? parseFloat(p.predicted.toFixed(1)) : null,
    }));

  return (
    <div className="space-y-5">

      {/* Metric + info row */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        {[
          { label: "RMSE",      value: `${run.rmse.toFixed(2)} W/m²`, color: "text-blue-600"   },
          { label: "MAE",       value: `${run.mae.toFixed(2)} W/m²`,  color: "text-purple-600" },
          { label: "R²",        value: run.r2.toFixed(4),             color: run.r2 > 0.85 ? "text-green-600" : "text-orange-500" },
          { label: "Rows",      value: run.rows_processed.toLocaleString(), color: "text-slate-700" },
          { label: "Anomalies", value: run.anomaly_count,             color: run.anomaly_count > 200 ? "text-orange-500" : "text-slate-700" },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-white rounded-xl border border-gray-200 p-3 text-center">
            <div className={`text-xl font-bold ${color}`}>{value}</div>
            <div className="text-xs text-gray-400 mt-0.5">{label}</div>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-2 text-xs text-gray-500">
        <span className="bg-white border border-gray-200 rounded-full px-3 py-1">
          Best model: <strong className="text-gray-700">{run.best_model || "—"}</strong>
        </span>
        <span className="bg-white border border-gray-200 rounded-full px-3 py-1">
          Horizon: <strong className="text-gray-700">{run.horizon}h</strong>
        </span>
        <span className="bg-white border border-gray-200 rounded-full px-3 py-1">
          Saved: <strong className="text-gray-700">{new Date(run.created_at).toLocaleString()}</strong>
        </span>
        <span className="bg-white border border-gray-200 rounded-full px-3 py-1">
          File: <strong className="text-gray-700">{run.filename}</strong>
        </span>
      </div>

      {/* Predicted vs Actual chart */}
      {chartData.length > 0 ? (
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <p className="text-sm font-semibold text-gray-700 mb-3">Predicted vs Actual — Test Set</p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" tick={{ fontSize: 9 }}
                interval={Math.floor(chartData.length / 8)} />
              <YAxis tick={{ fontSize: 10 }} unit=" W/m²" />
              <Tooltip contentStyle={{ fontSize: 11 }}
                formatter={(v) => v != null ? [`${v} W/m²`] : ["—"]} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="Actual"    stroke="#10b981" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="Predicted" stroke="#3b82f6" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-4 text-sm text-amber-800">
          No chart data stored for this run. Upload the same CSV again to get the full chart.
        </div>
      )}

      {/* Future forecast chart */}
      {futureChartData.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <p className="text-sm font-semibold text-gray-700 mb-3">Future Forecast ({futurePoints.length}h ahead)</p>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={futureChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" tick={{ fontSize: 9 }}
                interval={Math.floor(futureChartData.length / 8)} />
              <YAxis tick={{ fontSize: 10 }} unit=" W/m²" />
              <Tooltip contentStyle={{ fontSize: 11 }}
                formatter={(v) => v != null ? [`${v} W/m²`] : ["—"]} />
              <Line type="monotone" dataKey="Forecast" stroke="#8b5cf6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

    </div>
  );
}

export default History;
