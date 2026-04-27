import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ScatterChart, Scatter,
  BarChart, Bar, Cell
} from "recharts";
import { getAnomalies } from "../services/api";

const SEVERITY_COLORS = {
  high:   "#ef4444",
  medium: "#f59e0b",
  low:    "#3b82f6",
};

const METHOD_COLORS = {
  zscore:  "#8b5cf6",
  iqr:     "#10b981",
  rolling: "#f59e0b",
};

function Badge({ text, color }) {
  return (
    <span className="text-xs font-semibold px-2 py-0.5 rounded-full"
      style={{ backgroundColor: color + "20", color }}>
      {text}
    </span>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div className="bg-white rounded-xl shadow p-5">
      <p className="text-sm text-gray-500">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${color}`}>{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  );
}

function AnomalyReport() {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecast, result } = location.state || {};

  const [anomalyData, setAnomalyData]   = useState(null);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState(null);
  const [severityFilter, setSeverity]   = useState("all");
  const [methodFilter, setMethod]       = useState("all");

  const sessionId = result?.session_id;

  // ── Fetch anomalies on mount ──────────────────────────────────────────────
  useEffect(() => {
    if (!sessionId) {
      setError("No session found. Please upload a file first.");
      setLoading(false);
      return;
    }

    getAnomalies(sessionId)
      .then(data => {
        setAnomalyData(data);
        setLoading(false);
      })
      .catch(err => {
        setError("Failed to load anomalies. " + (err?.message || ""));
        setLoading(false);
      });
  }, [sessionId]);

  // ── Loading state ─────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4 animate-spin">⚙️</div>
          <p className="text-gray-500">Running anomaly detection...</p>
          <p className="text-xs text-gray-400 mt-2">Z-score + IQR + Rolling window</p>
        </div>
      </div>
    );
  }

  // ── Error state ───────────────────────────────────────────────────────────
  if (error || !anomalyData) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center gap-4">
        <p className="text-red-500">{error || "No data found."}</p>
        <button onClick={() => navigate("/")}
          className="bg-blue-600 text-white px-6 py-2 rounded-lg">
          Go Back to Upload
        </button>
      </div>
    );
  }

  const { anomalies, total_points, anomaly_count, anomaly_rate } = anomalyData;

  // ── Filter anomalies ──────────────────────────────────────────────────────
  const filtered = anomalies.filter(a => {
    const matchSeverity = severityFilter === "all" || a.severity === severityFilter;
    const matchMethod   = methodFilter   === "all" || a.method   === methodFilter;
    return matchSeverity && matchMethod;
  });

  // ── Severity breakdown ────────────────────────────────────────────────────
  const severityCounts = {
    high:   anomalies.filter(a => a.severity === "high").length,
    medium: anomalies.filter(a => a.severity === "medium").length,
    low:    anomalies.filter(a => a.severity === "low").length,
  };

  const methodCounts = {
    zscore:  anomalies.filter(a => a.method === "zscore").length,
    iqr:     anomalies.filter(a => a.method === "iqr").length,
    rolling: anomalies.filter(a => a.method === "rolling").length,
  };

  // ── Timeline chart data ───────────────────────────────────────────────────
  // Combine forecast points + anomaly markers
  const step = Math.max(1, Math.floor((forecast?.forecast?.length || 1) / 200));
  const timelineData = (forecast?.forecast || [])
    .filter((_, i) => i % step === 0)
    .map(p => {
      const ts = p.timestamp;
      const anomaly = anomalies.find(a => a.timestamp.substring(0, 16) === ts.substring(0, 16));
      return {
        time:      ts.substring(5, 16).replace("T", " "),
        actual:    p.actual != null ? parseFloat(p.actual.toFixed(1)) : null,
        predicted: parseFloat(p.predicted.toFixed(1)),
        anomaly:   anomaly ? parseFloat(anomaly.value.toFixed(1)) : null,
        severity:  anomaly?.severity || null,
      };
    });

  // ── Severity bar chart data ───────────────────────────────────────────────
  const severityBarData = Object.entries(severityCounts).map(([s, count]) => ({
    severity: s.charAt(0).toUpperCase() + s.slice(1),
    count,
    fill: SEVERITY_COLORS[s],
  }));

  // ── Method bar chart data ─────────────────────────────────────────────────
  const methodBarData = Object.entries(methodCounts).map(([m, count]) => ({
    method: m.charAt(0).toUpperCase() + m.slice(1),
    count,
    fill: METHOD_COLORS[m],
  }));

  // ── Deviation scatter data ────────────────────────────────────────────────
  const scatterData = anomalies.map(a => ({
    expected:  parseFloat(a.expected.toFixed(1)),
    value:     parseFloat(a.value.toFixed(1)),
    deviation: parseFloat(a.deviation.toFixed(1)),
    severity:  a.severity,
    fill:      SEVERITY_COLORS[a.severity],
  }));

  return (
    <div className="min-h-screen bg-gray-50 p-6">

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Anomaly Report</h1>
          <p className="text-sm text-gray-400 mt-1">
            {result?.filename} — Z-score + IQR + Rolling window detection
          </p>
        </div>
        <div className="flex gap-3">
          <button onClick={() => navigate("/dashboard", { state: { forecast, result } })}
            className="border border-gray-300 text-gray-600 px-4 py-2 rounded-lg text-sm hover:bg-gray-50">
            ← Dashboard
          </button>
          <button onClick={() => navigate("/compare", { state: { forecast, result } })}
            className="border border-blue-300 text-blue-600 px-4 py-2 rounded-lg text-sm hover:bg-blue-50">
            Model Comparison →
          </button>
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard
          label="Total Points Scanned"
          value={total_points.toLocaleString()}
          sub="full irradiance series"
          color="text-gray-700"
        />
        <StatCard
          label="Anomalies Detected"
          value={anomaly_count}
          sub={`${(anomaly_rate * 100).toFixed(2)}% of data`}
          color={anomaly_count > 50 ? "text-red-500" : "text-orange-500"}
        />
        <StatCard
          label="High Severity"
          value={severityCounts.high}
          sub="critical anomalies"
          color="text-red-500"
        />
        <StatCard
          label="Most Used Method"
          value={Object.entries(methodCounts).sort((a,b) => b[1]-a[1])[0][0]}
          sub="primary detector"
          color="text-purple-600"
        />
      </div>

      {/* Anomaly timeline */}
      <div className="bg-white rounded-xl shadow p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-700 mb-1">
          Anomaly Timeline
        </h2>
        <p className="text-sm text-gray-400 mb-4">
          Red dots = detected anomalies overlaid on the irradiance series
        </p>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={timelineData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="time" tick={{ fontSize: 10 }}
              interval={Math.floor(timelineData.length / 8)} />
            <YAxis tick={{ fontSize: 11 }}
              label={{ value: "W/m²", angle: -90, position: "insideLeft" }} />
            <Tooltip contentStyle={{ fontSize: 11 }} />
            <Legend />
            <Line type="monotone" dataKey="actual" stroke="#10b981"
              strokeWidth={1.5} dot={false} name="Actual" />
            <Line type="monotone" dataKey="predicted" stroke="#3b82f6"
              strokeWidth={1.5} dot={false} name="Predicted" />
            <Line type="monotone" dataKey="anomaly" stroke="#ef4444"
              strokeWidth={0} dot={{ fill: "#ef4444", r: 4 }} name="Anomaly" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Severity + Method charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">

        {/* Severity breakdown */}
        <div className="bg-white rounded-xl shadow p-6">
          <h2 className="text-lg font-semibold text-gray-700 mb-1">
            Severity Breakdown
          </h2>
          <p className="text-sm text-gray-400 mb-4">
            How many anomalies fall in each severity category
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={severityBarData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="severity" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {severityBarData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Detection method breakdown */}
        <div className="bg-white rounded-xl shadow p-6">
          <h2 className="text-lg font-semibold text-gray-700 mb-1">
            Detection Method
          </h2>
          <p className="text-sm text-gray-400 mb-4">
            Which method flagged the most anomalies
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={methodBarData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="method" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {methodBarData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Deviation scatter */}
      <div className="bg-white rounded-xl shadow p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-700 mb-1">
          Expected vs Observed (Anomalies Only)
        </h2>
        <p className="text-sm text-gray-400 mb-4">
          Points far from the diagonal line = large deviations.
          Color = severity (red=high, orange=medium, blue=low)
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="expected" name="Expected"
              label={{ value: "Expected W/m²", position: "insideBottom", offset: -5 }}
              tick={{ fontSize: 11 }} />
            <YAxis dataKey="value" name="Observed"
              label={{ value: "Observed W/m²", angle: -90, position: "insideLeft" }}
              tick={{ fontSize: 11 }} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }}
              content={({ payload }) => {
                if (!payload?.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="bg-white border border-gray-200 rounded p-2 text-xs shadow">
                    <p>Expected: {d.expected} W/m²</p>
                    <p>Observed: {d.value} W/m²</p>
                    <p>Deviation: {d.deviation} W/m²</p>
                    <p>Severity: {d.severity}</p>
                  </div>
                );
              }}
            />
            <Scatter data={scatterData} fill="#ef4444">
              {scatterData.map((entry, i) => (
                <Cell key={i} fill={entry.fill} fillOpacity={0.7} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Anomaly table */}
      <div className="bg-white rounded-xl shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold text-gray-700">
              Anomaly Records
            </h2>
            <p className="text-sm text-gray-400">
              Showing {filtered.length} of {anomaly_count} anomalies
            </p>
          </div>

          {/* Filters */}
          <div className="flex gap-3">
            <select
              value={severityFilter}
              onChange={e => setSeverity(e.target.value)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5"
            >
              <option value="all">All Severities</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
            <select
              value={methodFilter}
              onChange={e => setMethod(e.target.value)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5"
            >
              <option value="all">All Methods</option>
              <option value="zscore">Z-score</option>
              <option value="iqr">IQR</option>
              <option value="rolling">Rolling</option>
            </select>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b">
                <th className="text-left p-3 text-gray-500">#</th>
                <th className="text-left p-3 text-gray-500">Timestamp</th>
                <th className="text-right p-3 text-gray-500">Observed (W/m²)</th>
                <th className="text-right p-3 text-gray-500">Expected (W/m²)</th>
                <th className="text-right p-3 text-gray-500">Deviation</th>
                <th className="text-center p-3 text-gray-500">Severity</th>
                <th className="text-center p-3 text-gray-500">Method</th>
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, 50).map((a, i) => (
                <tr key={i} className="border-b hover:bg-gray-50">
                  <td className="p-3 text-gray-400">{i + 1}</td>
                  <td className="p-3 font-mono text-xs text-gray-600">
                    {a.timestamp.substring(0, 16).replace("T", " ")}
                  </td>
                  <td className="p-3 text-right">{a.value.toFixed(1)}</td>
                  <td className="p-3 text-right text-gray-400">{a.expected.toFixed(1)}</td>
                  <td className="p-3 text-right font-medium text-red-500">
                    {a.deviation.toFixed(1)}
                  </td>
                  <td className="p-3 text-center">
                    <Badge text={a.severity} color={SEVERITY_COLORS[a.severity]} />
                  </td>
                  <td className="p-3 text-center">
                    <Badge text={a.method} color={METHOD_COLORS[a.method]} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filtered.length > 50 && (
            <p className="text-center text-xs text-gray-400 mt-3">
              Showing first 50 of {filtered.length} records
            </p>
          )}
        </div>
      </div>

    </div>
  );
}

export default AnomalyReport;