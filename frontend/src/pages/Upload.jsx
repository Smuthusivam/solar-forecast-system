import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadCSV } from "../services/api";
import { saveForecastState } from "../services/forecastState";
import { useForecastJobPolling } from "../hooks/useForecastJobPolling";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar
} from "recharts";
import { StatCard, Card, TabNav, PageHeader } from "../components/ui";

function Upload() {
  const navigate = useNavigate();
  const storageKey = "solar-forecast-upload-state";

  // ── State ────────────────────────────────────────
  const [file, setFile]           = useState(null);
  const [dragging, setDragging]   = useState(false);
  const [uploading, setUploading] = useState(false);
  const [running, setRunning]     = useState(false);
  const [result, setResult]       = useState(null);
  const [forecast, setForecast]   = useState(null);
  const [error, setError]         = useState(null);
  const [trainSize, setTrainSize] = useState(80);
  const [activeTab, setActiveTab] = useState("data");
  const [fileName, setFileName]   = useState("");
  const {
    status: forecastJobStatus,
    error: forecastJobError,
    isBusy: isForecastJobBusy,
    runForecastAsync,
  } = useForecastJobPolling();

  useEffect(() => {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return;
    try {
      const saved = JSON.parse(raw);
      if (saved?.result) setResult(saved.result);
      if (saved?.forecast) setForecast(saved.forecast);
      if (typeof saved?.trainSize === "number") setTrainSize(saved.trainSize);
      if (typeof saved?.activeTab === "string") {
        const allowedTabs = new Set(["data", "quality", "patterns"]);
        setActiveTab(allowedTabs.has(saved.activeTab) ? saved.activeTab : "data");
      }
      if (typeof saved?.fileName === "string") setFileName(saved.fileName);
    } catch {
      localStorage.removeItem(storageKey);
    }
  }, []);

  useEffect(() => {
    const payload = { result, forecast, trainSize, activeTab, fileName };
    localStorage.setItem(storageKey, JSON.stringify(payload));
  }, [result, forecast, trainSize, activeTab, fileName]);

  function clearSession() {
    setFile(null);
    setResult(null);
    setForecast(null);
    setError(null);
    setFileName("");
    localStorage.removeItem(storageKey);
  }

  // ── File picked from input ───────────────────────
  function handleFileChange(e) {
    const picked = e.target.files[0];
    if (!picked) return;
    if (!picked.name.endsWith(".csv")) {
      setError("Wrong file format. Please upload a CSV file.");
      e.target.value = "";
      return;
    }
    setFile(picked);
    setFileName(picked.name);
    setResult(null);
    setForecast(null);
    setError(null);
  }

  // ── Drag and drop handlers ───────────────────────
  function handleDragOver(e) {
    e.preventDefault();
    setDragging(true);
  }

  function handleDragLeave() {
    setDragging(false);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.name.endsWith(".csv")) {
      setFile(dropped);
      setFileName(dropped.name);
      setResult(null);
      setForecast(null);
      setError(null);
    } else {
      setError("Wrong file format. Please upload a CSV file.");
    }
  }

  // ── Upload CSV to FastAPI ────────────────────────
  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setError(null);

    try {
      const data = await uploadCSV(file);
      setResult(data);
      setForecast(null);
    } catch (err) {
      const detail = err?.response?.data?.detail;
      if (detail?.code === "invalid_file_type") {
        setError("Wrong file format. Please upload a CSV file.");
      } else if (detail?.code === "irrelevant_dataset") {
        setError(detail.message);
      } else if (detail?.code === "preprocessing_failed") {
        setError(detail.message);
      } else {
        setError("Upload failed. Make sure the backend is running.");
      }
    } finally {
      setUploading(false);
    }
  }

  // ── Run Models → evaluate on test set, then go to Dashboard ──────────────
  async function handleRunForecast() {
    if (!result?.session_id) return;
    setRunning(true);
    setError(null);

    try {
      const forecastData = await runForecastAsync({
        sessionId: result.session_id,
        horizonHours: 24,
        trainSize,
        skipFuture: true,
        pollIntervalMs: 2000,
      });
      setForecast(forecastData);
      // Save to shared state for NavBar navigation
      saveForecastState(forecastData, result, fileName || result.filename);
      navigate("/dashboard", { state: { forecast: forecastData, result } });
    } catch (err) {
      setError(err?.message || "Model run failed. Please try again.");
    } finally {
      setRunning(false);
    }
  }

  const points = forecast?.forecast ?? [];
  const uploadStats = result?.data_stats ?? null;

  // ── Upload-level pattern aggregates ───────────────────────────────────
  const hourlyAvg = useMemo(() => {
    if (!uploadStats?.hourly_avg?.length) return [];
    return uploadStats.hourly_avg.map((avg, h) => ({
      hour: `${h}:00`,
      avg: parseFloat(Number(avg || 0).toFixed(1)),
    }));
  }, [uploadStats]);

  const weekdayAvg = useMemo(() => {
    if (!uploadStats?.weekday_avg?.length) return [];
    const labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    return uploadStats.weekday_avg.map((avg, i) => ({
      day: labels[i],
      avg: parseFloat(Number(avg || 0).toFixed(1)),
    }));
  }, [uploadStats]);

  const monthlyAvg = useMemo(() => {
    if (!uploadStats?.monthly_avg?.length) return [];
    const labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    return uploadStats.monthly_avg.map((avg, i) => ({
      month: labels[i],
      avg: parseFloat(Number(avg || 0).toFixed(1)),
    }));
  }, [uploadStats]);

  const dailyAvg = useMemo(() => {
    if (!uploadStats?.daily_avg?.length) return [];
    const rows = uploadStats.daily_avg.map((row) => ({
      day: row.date,
      avg: parseFloat(Number(row.avg || 0).toFixed(1)),
    }));
    if (rows.length <= 120) return rows;
    const step = Math.ceil(rows.length / 120);
    return rows.filter((_, i) => i % step === 0);
  }, [uploadStats]);

  // ── UI ───────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-50 px-4 py-8 sm:px-6 lg:px-10">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8">

        <PageHeader
          title="Solar Irradiance Forecasting"
          subtitle="Upload a solar or weather CSV to begin the forecast workflow"
          actions={[
            <button
              key="history"
              onClick={() => navigate("/history")}
              className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:bg-slate-50"
            >
              View History
            </button>,
          ]}
        />

        {/* Drop Zone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`w-full border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors
            ${dragging
              ? "border-blue-500 bg-blue-50"
              : "border-gray-300 bg-white hover:border-blue-400"
            }`}
        >
          <p className="text-gray-600 font-medium">
            Drag & drop your CSV here
          </p>
          <p className="text-gray-400 text-sm mb-4">or</p>

          {/* File input */}
          <label className="cursor-pointer rounded-full border border-slate-300 bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800">
            Browse File
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>

          {/* Show selected file name */}
          {file && (
            <p className="mt-4 text-sm text-green-600 font-medium">
              {file.name}
            </p>
          )}
          {!file && fileName && (
            <p className="mt-4 text-xs text-gray-500">
              Last file: {fileName} (select again to re-upload)
            </p>
          )}
        </div>

        {/* Upload Button */}
        {file && !result && (
          <div className="flex justify-start">
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="rounded-full bg-slate-900 px-8 py-3 font-medium text-white transition hover:bg-slate-800 disabled:opacity-50"
            >
              {uploading ? "Uploading..." : "Upload"}
            </button>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-300 rounded-xl p-4 flex items-start gap-3">
            <span className="text-red-500 text-xl mt-0.5">✕</span>
            <div>
              <p className="font-semibold text-red-700">Upload failed</p>
              <p className="text-sm text-red-600 mt-1">{error}</p>
            </div>
          </div>
        )}

        {forecastJobError && !error && (
          <div className="bg-red-50 border border-red-300 rounded-xl p-4 flex items-start gap-3">
            <span className="text-red-500 text-xl mt-0.5">✕</span>
            <div>
              <p className="font-semibold text-red-700">Forecast failed</p>
              <p className="text-sm text-red-600 mt-1">{forecastJobError}</p>
            </div>
          </div>
        )}

        {(result || forecast) && (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs text-gray-400">
              Restored session{fileName ? ` for ${fileName}` : ""}
            </p>
            <button
              onClick={clearSession}
              className="text-xs text-red-600 underline underline-offset-2"
            >
              Clear uploaded data
            </button>
          </div>
        )}

        {/* Detection Result Card */}
        {result && (
          result.confidence < 0.70 ? (
            /* ── Low confidence — invalid dataset, block everything ── */
            <div className="w-full bg-white rounded-xl shadow p-6 border-2 border-red-200">
              <div className="flex items-start gap-4">
                <div className="text-4xl">⛔</div>
                <div className="flex-1">
                  <h2 className="text-lg font-bold text-red-700 mb-1">Invalid Dataset</h2>
                  <p className="text-sm text-red-600 mb-3">
                    Column detection confidence is too low ({(result.confidence * 100).toFixed(0)}% — minimum 70% required).
                    The system could not reliably identify the required solar irradiance column in your CSV.
                  </p>
                  <ul className="text-sm text-red-700 space-y-1 list-disc list-inside mb-4">
                    <li>Make sure your CSV contains a GHI or irradiance column</li>
                    <li>Check that column headers are clearly named (e.g. "GHI", "irradiance", "solar_radiation")</li>
                    <li>Remove extra metadata rows at the top of the file</li>
                  </ul>
                  <button
                    onClick={clearSession}
                    className="rounded-full bg-red-600 px-5 py-2 text-sm font-medium text-white transition hover:bg-red-700"
                  >
                    Upload a different file
                  </button>
                </div>
              </div>
            </div>
          ) : (
            /* ── Normal flow ── */
            <div className="w-full bg-white rounded-xl shadow p-6">

              {/* Mode badge */}
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-bold text-gray-800">
                  Detection Result
                </h2>
                <span className="text-xs font-semibold px-3 py-1 rounded-full bg-green-100 text-green-700">
                  Direct GHI
                </span>
              </div>

              {/* Preview Table */}
              {result.preview && result.preview.length > 0 && (
                <div className="mb-4 overflow-x-auto">
                  <p className="text-sm text-gray-500 mb-2">Data Preview (first 5 rows):</p>
                  <table className="text-xs w-full border-collapse">
                    <thead>
                      <tr>
                        {Object.keys(result.preview[0]).map((col) => (
                          <th key={col} className="border border-gray-200 bg-gray-50 px-2 py-1 text-left">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.preview.map((row, i) => (
                        <tr key={i}>
                          {Object.values(row).map((val, j) => (
                            <td key={j} className="border border-gray-200 px-2 py-1">
                              {val}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Train / Test Split Selector */}
              <div className="mb-5">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-gray-700">Train / Test Split</p>
                  <p className="text-xs text-gray-400">
                    Train <span className="font-semibold text-gray-600">{trainSize}%</span>
                    {" "}/ Test <span className="font-semibold text-gray-600">{100 - trainSize}%</span>
                  </p>
                </div>
                <div className="flex gap-2">
                  {[70, 75, 80, 85, 90].map((pct) => (
                    <button
                      key={pct}
                      onClick={() => setTrainSize(pct)}
                      className={`flex-1 rounded-full border px-4 py-2 text-sm font-medium transition
                        ${trainSize === pct
                          ? "border-slate-900 bg-slate-900 text-white"
                          : "border-slate-300 bg-white text-slate-600 hover:border-slate-400 hover:text-slate-900"
                        }`}
                    >
                      {pct}%
                    </button>
                  ))}
                </div>
                <p className="text-xs text-gray-400 mt-2">
                  Higher training % gives the model more data to learn from; lower gives a larger test set for evaluation.
                </p>
              </div>

              {/* Run Models Button */}
              <button
                onClick={handleRunForecast}
                disabled={running || isForecastJobBusy}
                className="w-full rounded-full bg-slate-900 py-3 font-medium text-white transition hover:bg-slate-800 disabled:opacity-50"
              >
                {(running || isForecastJobBusy)
                  ? (forecastJobStatus === "processing" ? "Running Models... (processing)" : "Running Models...")
                  : "Run Models"}
              </button>

              {(running || isForecastJobBusy) && (
                <p className="mt-2 text-xs text-gray-500 text-center">
                  Forecast job status: {forecastJobStatus}. Polling every 2 seconds...
                </p>
              )}

            </div>
          )
        )}

        {/* EDA Section — only shown when confidence is sufficient */}
        {result && uploadStats && result.confidence >= 0.70 && (
          <div className="w-full">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between mb-4">
              <div>
                <h2 className="text-xl font-semibold text-gray-800">Data Overview & EDA</h2>
                <p className="text-sm text-gray-400">Review dataset stats and quality right after upload</p>
              </div>
              {forecast && (
                <button
                  onClick={() => navigate("/dashboard", { state: { forecast, result } })}
                  className="self-start rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-100"
                >
                  Open Forecast Dashboard
                </button>
              )}
            </div>

            <TabNav
              tabs={[
                { id: "data",     label: "Overview"     },
                { id: "columns",  label: "Columns"      },
                { id: "quality",  label: "Data Quality" },
                { id: "patterns", label: "Patterns"     },
              ]}
              active={activeTab}
              onChange={setActiveTab}
            />

            {/* ── Overview ───────────────────────────── */}
            {activeTab === "data" && (
              <>
                <Card title="Dataset Overview"
                  subtitle="Summary statistics of the raw upload and cleaned data">
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <StatCard label="Rows (raw)"
                      value={uploadStats.rows_raw.toLocaleString()}
                      color="text-gray-700" sub="before cleaning" />
                    <StatCard label="Rows (clean)"
                      value={uploadStats.rows_clean.toLocaleString()}
                      color="text-blue-600" sub="after cleaning" />
                    <StatCard label="Clean %"
                      value={uploadStats.pct_clean.toFixed(1)}
                      unit="%"
                      color="text-green-600" sub="rows kept" />
                  </div>
                </Card>

                <Card title="Date Range"
                  subtitle="The full span of the uploaded dataset">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-xs text-gray-500 mb-1">Start</p>
                      <p className="text-lg font-mono text-gray-700">
                        {uploadStats.date_start?.substring(0, 19).replace("T", " ")}
                      </p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-xs text-gray-500 mb-1">End</p>
                      <p className="text-lg font-mono text-gray-700">
                        {uploadStats.date_end?.substring(0, 19).replace("T", " ")}
                      </p>
                    </div>
                  </div>
                  <p className="text-xs text-gray-400 mt-2">
                    Span: {uploadStats.date_range_days} days
                  </p>
                </Card>

                <Card title="Irradiance Summary"
                  subtitle="Basic stats on the target variable">
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    <StatCard label="Mean" value={uploadStats.irradiance_mean.toFixed(1)} unit="W/m²" color="text-blue-600" />
                    <StatCard label="Min" value={uploadStats.irradiance_min.toFixed(1)} unit="W/m²" color="text-gray-500" />
                    <StatCard label="Max" value={uploadStats.irradiance_max.toFixed(1)} unit="W/m²" color="text-orange-600" />
                    <StatCard label="Detection"
                      value="direct"
                      color="text-green-600" sub="GHI source" />
                    <StatCard label="Confidence"
                      value={(result.confidence * 100).toFixed(0)}
                      unit="%"
                      color="text-indigo-600" sub="column mapping" />
                  </div>
                </Card>

                <Card title="Planned Train / Test Split"
                  subtitle={`Time-aware ${trainSize}/${100 - trainSize} split — no shuffling to preserve temporal order`}>
                  <div className="flex items-stretch h-12 rounded-lg overflow-hidden mb-3">
                    <div className="bg-blue-500 flex items-center justify-center text-white font-medium"
                      style={{ width: `${trainSize}%` }}>
                      Train: {Math.round(uploadStats.rows_clean * (trainSize / 100)).toLocaleString()} rows ({trainSize}%)
                    </div>
                    <div className="bg-orange-500 flex items-center justify-center text-white font-medium"
                      style={{ width: `${100 - trainSize}%` }}>
                      Test: {Math.round(uploadStats.rows_clean * ((100 - trainSize) / 100)).toLocaleString()} ({100 - trainSize}%)
                    </div>
                  </div>
                  <p className="text-xs text-gray-400">
                    Time series MUST NOT be shuffled. The model is trained on earlier data
                    and evaluated on the most recent slice to simulate real forecasting.
                  </p>
                </Card>
              </>
            )}

            {/* ── Columns ───────────────────────────── */}
            {activeTab === "columns" && (
              <>
                <Card title="Detected Column Mapping"
                  subtitle="How Claude mapped your CSV columns to physical variables">
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(result.detected_columns || {}).map(([key, val]) => {
                      // Timestamp null means NSRDB split date columns — handled automatically
                      const display = !val && key === "timestamp"
                        ? "auto (Year/Month/Day/Hour)"
                        : val || "not found";
                      const found = !!val || key === "timestamp";
                      return (
                        <span key={key}
                          className={`text-xs px-3 py-1 rounded-full font-medium border
                            ${found
                              ? "bg-blue-50 text-blue-700 border-blue-200"
                              : "bg-gray-100 text-gray-400 border-gray-200"}`}>
                          {key}: <strong>{display}</strong>
                        </span>
                      );
                    })}
                  </div>
                </Card>

                <Card title="All Available Columns"
                  subtitle="All columns present in the dataset after preprocessing">
                  <div className="flex flex-wrap gap-2">
                    {(uploadStats?.columns_available || []).map((col) => (
                      <span key={col}
                        className="bg-gray-100 text-gray-700 text-xs px-3 py-1 rounded-full">
                        {col}
                      </span>
                    ))}
                  </div>
                </Card>

                <Card title="Detection Summary"
                  subtitle="Column detection confidence and mode">
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <StatCard label="Mode"
                      value="direct"
                      color="text-green-600"
                      sub="GHI column found" />
                    <StatCard label="Confidence"
                      value={(result.confidence * 100).toFixed(0)}
                      unit="%"
                      color="text-indigo-600" sub="column mapping certainty" />
                    <StatCard label="Warnings"
                      value={result.warnings?.length || 0}
                      color="text-gray-600" sub="non-fatal issues" />
                  </div>
                  {result.warnings?.length > 0 && (
                    <ul className="mt-4 list-disc pl-5 text-sm text-gray-600 space-y-1">
                      {result.warnings.map((w, i) => <li key={i}>{w}</li>)}
                    </ul>
                  )}
                </Card>
              </>
            )}

            {/* ── Data Quality ───────────────────────── */}
            {activeTab === "quality" && (
              <>
                <Card title="Cleaning Summary"
                  subtitle="What changed during preprocessing">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <StatCard label="Rows Dropped"
                      value={(uploadStats.rows_raw - uploadStats.rows_clean).toLocaleString()}
                      color="text-orange-600" sub="invalid timestamps / gaps" />
                    <StatCard label="Clean %"
                      value={uploadStats.pct_clean.toFixed(1)}
                      unit="%"
                      color="text-green-600" sub="rows kept" />
                    <StatCard label="Detection Mode"
                      value="direct"
                      color="text-indigo-600" sub="GHI column" />
                    <StatCard label="Warnings"
                      value={result.warnings?.length || 0}
                      color="text-gray-600" sub="non-fatal" />
                  </div>
                </Card>

                {result.warnings?.length > 0 && (
                  <Card title="Warnings"
                    subtitle="Issues found during column detection">
                    <ul className="list-disc pl-5 text-sm text-gray-600 space-y-1">
                      {result.warnings.map((warn, idx) => (
                        <li key={idx}>{warn}</li>
                      ))}
                    </ul>
                  </Card>
                )}
              </>
            )}

            {/* ── Patterns ───────────────────────────────── */}
            {activeTab === "patterns" && (
              <>
                {!uploadStats && (
                  <Card title="Patterns"
                    subtitle="Upload a dataset to compute time-based patterns">
                    <p className="text-sm text-gray-500">Patterns are computed during preprocessing. Upload a CSV first.</p>
                  </Card>
                )}
                {uploadStats && (
                  <>
                    <Card title="Average Irradiance by Hour of Day"
                      subtitle="Solar bell curve — peak at noon, zero at night">
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={hourlyAvg}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="hour" tick={{ fontSize: 10 }} />
                          <YAxis tick={{ fontSize: 11 }}
                            label={{ value: "W/m²", angle: -90, position: "insideLeft" }} />
                          <Tooltip />
                          <Bar dataKey="avg" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </Card>

                    <Card title="Average Irradiance by Day of Week"
                      subtitle="Weekday pattern — highlights operational differences">
                      <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={weekdayAvg}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="day" />
                          <YAxis tick={{ fontSize: 11 }}
                            label={{ value: "W/m²", angle: -90, position: "insideLeft" }} />
                          <Tooltip />
                          <Bar dataKey="avg" fill="#10b981" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </Card>

                    <Card title="Monthly Average Irradiance"
                      subtitle="Seasonality view across months">
                      <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={monthlyAvg}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="month" />
                          <YAxis tick={{ fontSize: 11 }}
                            label={{ value: "W/m²", angle: -90, position: "insideLeft" }} />
                          <Tooltip />
                          <Bar dataKey="avg" fill="#6366f1" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </Card>

                    <Card title="Daily Average Trend"
                      subtitle="Smoothed daily averages (sampled)">
                      <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={dailyAvg}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="day" tick={{ fontSize: 10 }} />
                          <YAxis tick={{ fontSize: 11 }}
                            label={{ value: "W/m²", angle: -90, position: "insideLeft" }} />
                          <Tooltip />
                          <Line type="monotone" dataKey="avg" stroke="#f97316" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </Card>
                  </>
                )}
              </>
            )}

          </div>
        )}

      </div>
    </div>
  );
}

export default Upload;