import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { getHistory, getForecastPoints } from "../services/api";
import { R2Badge, AlertBox } from "../components/ui";

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
          <div className="text-4xl mb-4 animate-spin">📋</div>
          <p className="text-gray-500">Loading forecast history...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Forecast History</h1>
          <p className="text-sm text-gray-400 mt-1">
            All past forecast runs stored in the database
          </p>
        </div>
        <button
          onClick={() => navigate("/")}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700"
        >
          + New Forecast
        </button>
      </div>

      {error && <AlertBox>{error}</AlertBox>}

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-xl shadow p-4">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Total Runs</p>
          <p className="text-2xl font-bold text-blue-600 mt-1">{totalRuns}</p>
          <p className="text-xs text-gray-400 mt-1">forecast sessions</p>
        </div>
        <div className="bg-white rounded-xl shadow p-4">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Avg R²</p>
          <p className={`text-2xl font-bold mt-1 ${avgR2 > 0.9 ? "text-green-600" : "text-orange-500"}`}>
            {avgR2.toFixed(4)}
          </p>
          <p className="text-xs text-gray-400 mt-1">across all runs</p>
        </div>
        <div className="bg-white rounded-xl shadow p-4">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Best Run</p>
          <p className="text-2xl font-bold text-purple-600 mt-1">
            {bestRun ? bestRun.r2.toFixed(4) : "—"}
          </p>
          <p className="text-xs text-gray-400 mt-1">
            {bestRun ? bestRun.filename.slice(0, 20) : "no runs yet"}
          </p>
        </div>
        <div className="bg-white rounded-xl shadow p-4">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Total Rows Processed</p>
          <p className="text-2xl font-bold text-indigo-600 mt-1">
            {totalRows.toLocaleString()}
          </p>
          <p className="text-xs text-gray-400 mt-1">across all runs</p>
        </div>
      </div>

      {/* Empty state */}
      {runs.length === 0 && !error && (
        <div className="bg-white rounded-xl shadow p-12 text-center">
          <p className="text-5xl mb-4">📭</p>
          <p className="text-lg text-gray-500 mb-2">No forecast runs yet</p>
          <p className="text-sm text-gray-400 mb-6">
            Upload a CSV and run a forecast to see it appear here
          </p>
          <button onClick={() => navigate("/")}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700">
            Upload a File
          </button>
        </div>
      )}

      {/* Table */}
      {runs.length > 0 && (
        <div className="bg-white rounded-xl shadow">

          {/* Search bar */}
          <div className="p-4 border-b border-gray-100">
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search by filename or detection mode..."
              className="w-full border border-gray-300 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b text-gray-500">
                  <th className="text-left p-3 cursor-pointer hover:text-blue-600"
                    onClick={() => toggleSort("run_id")}>
                    Run # <SortIcon col="run_id" />
                  </th>
                  <th className="text-left p-3 cursor-pointer hover:text-blue-600"
                    onClick={() => toggleSort("filename")}>
                    File <SortIcon col="filename" />
                  </th>
                  <th className="text-center p-3">Mode</th>
                  <th className="text-center p-3 cursor-pointer hover:text-blue-600"
                    onClick={() => toggleSort("horizon")}>
                    Horizon <SortIcon col="horizon" />
                  </th>
                  <th className="text-right p-3 cursor-pointer hover:text-blue-600"
                    onClick={() => toggleSort("rmse")}>
                    RMSE <SortIcon col="rmse" />
                  </th>
                  <th className="text-right p-3 cursor-pointer hover:text-blue-600"
                    onClick={() => toggleSort("mae")}>
                    MAE <SortIcon col="mae" />
                  </th>
                  <th className="text-right p-3 cursor-pointer hover:text-blue-600"
                    onClick={() => toggleSort("r2")}>
                    R² <SortIcon col="r2" />
                  </th>
                  <th className="text-right p-3 cursor-pointer hover:text-blue-600"
                    onClick={() => toggleSort("rows_processed")}>
                    Rows <SortIcon col="rows_processed" />
                  </th>
                  <th className="text-right p-3 cursor-pointer hover:text-blue-600"
                    onClick={() => toggleSort("anomaly_count")}>
                    Anomalies <SortIcon col="anomaly_count" />
                  </th>
                  <th className="text-right p-3 cursor-pointer hover:text-blue-600"
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
                      className={`border-b hover:bg-blue-50 cursor-pointer transition
                        ${i % 2 === 0 ? "bg-white" : "bg-gray-50/30"}
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