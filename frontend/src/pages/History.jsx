import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { getHistory, getForecastPoints } from "../services/api";
import { R2Badge, AlertBox, PageHeader } from "../components/ui";

const MODE_COLORS = {
  direct:    { bg: "bg-green-100",  text: "text-green-700"  },
  estimated: { bg: "bg-yellow-100", text: "text-yellow-700" },
};

function History() {
  const navigate = useNavigate();

  const [runs,        setRuns]        = useState([]);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState(null);
  const [search,      setSearch]      = useState("");
  const [sortBy,      setSortBy]      = useState("created_at");
  const [sortDir,     setSortDir]     = useState("desc");
  const [loadingRunId, setLoadingRunId] = useState(null);

  // ── Fetch all runs on mount ─────────────────────────────────────────
  useEffect(() => {
    getHistory()
      .then(data => {
        setRuns(data.runs || []);
        setLoading(false);
      })
      .catch(err => {
        setError("Failed to load history: " + (err?.message || ""));
        setLoading(false);
      });
  }, []);

  // ── Sort + filter ───────────────────────────────────────────────────
  const filtered = runs
    .filter(r =>
      r.filename.toLowerCase().includes(search.toLowerCase()) ||
      r.detection_mode.toLowerCase().includes(search.toLowerCase())
    )
    .sort((a, b) => {
      let aVal = a[sortBy];
      let bVal = b[sortBy];
      if (typeof aVal === "string") aVal = aVal.toLowerCase();
      if (typeof bVal === "string") bVal = bVal.toLowerCase();
      return sortDir === "asc" ? (aVal > bVal ? 1 : -1) : (aVal < bVal ? 1 : -1);
    });

  function toggleSort(col) {
    if (sortBy === col) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortBy(col); setSortDir("desc"); }
  }

  function SortIcon({ col }) {
    if (sortBy !== col) return <span className="text-gray-300 ml-1">↕</span>;
    return <span className="text-blue-500 ml-1">{sortDir === "asc" ? "↑" : "↓"}</span>;
  }

  // ── Summary stats ───────────────────────────────────────────────────
  const totalRuns   = runs.length;
  const avgR2       = runs.length
    ? runs.reduce((a, r) => a + r.r2, 0) / runs.length
    : 0;
  const bestRun     = runs.length
    ? [...runs].sort((a, b) => b.r2 - a.r2)[0]
    : null;
  const totalRows   = runs.reduce((a, r) => a + r.rows_processed, 0);

  // ── Loading ─────────────────────────────────────────────────────────
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
    <div className="min-h-screen bg-gray-50 p-6">

      <PageHeader
        title="Forecast History"
        subtitle="All past forecast runs stored in the database"
        actions={[
          <button
            key="new-forecast"
            onClick={() => navigate("/")}
            className="rounded-full bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800"
          >
            + New Forecast
          </button>,
        ]}
      />

      {error && <AlertBox>{error}</AlertBox>}

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Total Runs</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{totalRuns}</p>
          <p className="mt-1 text-xs text-slate-500">forecast sessions</p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Avg R²</p>
          <p className={`mt-2 text-2xl font-semibold ${avgR2 > 0.9 ? "text-emerald-600" : "text-amber-600"}`}>
            {avgR2.toFixed(4)}
          </p>
          <p className="mt-1 text-xs text-slate-500">across all runs</p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Best Run</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">
            {bestRun ? bestRun.r2.toFixed(4) : "—"}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {bestRun ? bestRun.filename.slice(0, 20) : "no runs yet"}
          </p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Total Rows Processed</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">
            {totalRows.toLocaleString()}
          </p>
          <p className="mt-1 text-xs text-slate-500">across all runs</p>
        </div>
      </div>

      {/* Empty state */}
      {runs.length === 0 && !error && (
        <div className="rounded-2xl border border-slate-200 bg-white p-12 text-center shadow-sm">
          <p className="text-5xl mb-4">📭</p>
          <p className="mb-2 text-lg text-slate-500">No forecast runs yet</p>
          <p className="mb-6 text-sm text-slate-400">
            Upload a CSV and run a forecast to see it appear here
          </p>
          <button onClick={() => navigate("/")}
            className="rounded-full bg-slate-900 px-6 py-2 text-white transition hover:bg-slate-800">
            Upload a File
          </button>
        </div>
      )}

      {/* Table */}
      {runs.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">

          {/* Search bar */}
          <div className="border-b border-slate-100 p-4">
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search by filename or detection mode..."
              className="w-full rounded-xl border border-slate-300 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-slate-400"
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-slate-50 text-slate-500">
                  <th className="cursor-pointer p-3 text-left hover:text-slate-900"
                    onClick={() => toggleSort("run_id")}>
                    Run # <SortIcon col="run_id" />
                  </th>
                  <th className="cursor-pointer p-3 text-left hover:text-slate-900"
                    onClick={() => toggleSort("filename")}>
                    File <SortIcon col="filename" />
                  </th>
                  <th className="p-3 text-center">Mode</th>
                  <th className="cursor-pointer p-3 text-center hover:text-slate-900"
                    onClick={() => toggleSort("horizon")}>
                    Horizon <SortIcon col="horizon" />
                  </th>
                  <th className="cursor-pointer p-3 text-right hover:text-slate-900"
                    onClick={() => toggleSort("rmse")}>
                    RMSE <SortIcon col="rmse" />
                  </th>
                  <th className="cursor-pointer p-3 text-right hover:text-slate-900"
                    onClick={() => toggleSort("mae")}>
                    MAE <SortIcon col="mae" />
                  </th>
                  <th className="cursor-pointer p-3 text-right hover:text-slate-900"
                    onClick={() => toggleSort("r2")}>
                    R² <SortIcon col="r2" />
                  </th>
                  <th className="cursor-pointer p-3 text-right hover:text-slate-900"
                    onClick={() => toggleSort("rows_processed")}>
                    Rows <SortIcon col="rows_processed" />
                  </th>
                  <th className="cursor-pointer p-3 text-right hover:text-slate-900"
                    onClick={() => toggleSort("anomaly_count")}>
                    Anomalies <SortIcon col="anomaly_count" />
                  </th>
                  <th className="cursor-pointer p-3 text-right hover:text-slate-900"
                    onClick={() => toggleSort("created_at")}>
                    Date <SortIcon col="created_at" />
                  </th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((run, i) => {
                  const modeStyle = MODE_COLORS[run.detection_mode] || MODE_COLORS.direct;
                  return (
                    <tr key={run.run_id}
                      className={`border-b hover:bg-slate-50 cursor-pointer transition
                        ${i % 2 === 0 ? "bg-white" : "bg-slate-50/30"}
                        ${loadingRunId === run.run_id ? "opacity-60" : ""}`}
                      onClick={async () => {
                        if (loadingRunId) return;
                        setLoadingRunId(run.run_id);
                        try {
                          const pointsData = await getForecastPoints(run.run_id);
                          if (pointsData.count > 0) {
                            // Reconstruct forecast object so Dashboard can render the full chart
                            const forecast = {
                              run_id:         run.run_id,
                              detection_mode: run.detection_mode,
                              best_model:     run.best_model || "best model",
                              metrics:        { rmse: run.rmse, mae: run.mae, r2: run.r2 },
                              forecast:       pointsData.points.filter(p => !p.is_future),
                              future_forecast: pointsData.points.filter(p => p.is_future),
                              models_info:    [],
                              feature_importance: null,
                            };
                            const result = { session_id: run.session_id, filename: run.filename };
                            navigate("/dashboard", { state: { forecast, result, historyRun: run } });
                          } else {
                            // No points stored (old run) — fall back to summary-only view
                            navigate("/dashboard", { state: { historyRun: run } });
                          }
                        } catch {
                          navigate("/dashboard", { state: { historyRun: run } });
                        } finally {
                          setLoadingRunId(null);
                        }
                      }}
                    >
                      <td className="p-3 font-mono text-gray-400">
                        {loadingRunId === run.run_id ? (
                          <span className="text-blue-500 animate-pulse">Loading…</span>
                        ) : (
                          `#${run.run_id}`
                        )}
                      </td>
                      <td className="p-3">
                        <div className="font-medium text-gray-700 truncate max-w-40"
                          title={run.filename}>
                          {run.filename}
                        </div>
                      </td>
                      <td className="p-3 text-center">
                        <span className={`text-xs font-semibold px-2 py-1 rounded-full ${modeStyle.bg} ${modeStyle.text}`}>
                          {run.detection_mode}
                        </span>
                      </td>
                      <td className="p-3 text-center text-gray-600">{run.horizon}h</td>
                      <td className="p-3 text-right font-mono">{run.rmse.toFixed(2)}</td>
                      <td className="p-3 text-right font-mono">{run.mae.toFixed(2)}</td>
                      <td className="p-3 text-right">
                        <R2Badge value={run.r2} />
                      </td>
                      <td className="p-3 text-right text-gray-600">
                        {run.rows_processed.toLocaleString()}
                      </td>
                      <td className="p-3 text-right">
                        <span className={`font-medium ${run.anomaly_count > 200 ? "text-orange-500" : "text-gray-600"}`}>
                          {run.anomaly_count}
                        </span>
                      </td>
                      <td className="p-3 text-right text-gray-400 text-xs font-mono">
                        {new Date(run.created_at).toLocaleString()}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-gray-100 flex items-center justify-between">
            <p className="text-xs text-gray-400">
              Showing {filtered.length} of {runs.length} runs
              {search && ` matching "${search}"`}
            </p>
            <p className="text-xs text-gray-400">
              Click any row to open the saved run summary dashboard
            </p>
          </div>
        </div>
      )}

    </div>
  );
}

export default History;