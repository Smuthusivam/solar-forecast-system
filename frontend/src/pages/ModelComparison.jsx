import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ReferenceLine
} from "recharts";

// ─── Reusable bits ─────────────────────────────────────────────────────
function Card({ title, subtitle, children }) {
  return (
    <div className="bg-white rounded-xl shadow p-6 mb-6">
      <h2 className="text-lg font-semibold text-gray-700">{title}</h2>
      {subtitle && <p className="text-sm text-gray-400 mb-4">{subtitle}</p>}
      <div className="mt-3">{children}</div>
    </div>
  );
}

const MODEL_COLORS = {
  XGBoost:  "#3b82f6",
  LightGBM: "#10b981",
  Prophet:  "#f59e0b",
  Ensemble: "#8b5cf6",
  Actual:   "#6b7280",
};

const METRIC_INFO = {
  rmse: { label: "RMSE",  unit: "W/m²", lower: true,  desc: "Root Mean Square Error — penalises large errors" },
  mae:  { label: "MAE",   unit: "W/m²", lower: true,  desc: "Mean Absolute Error" },
  r2:   { label: "R²",    unit: "",     lower: false, desc: "Coefficient of Determination — 1.0 = perfect" },
  mape: { label: "MAPE",  unit: "%",    lower: true,  desc: "Mean Absolute Percentage Error" },
};

// Strict model health check — considers both absolute test quality AND train/test gap.
// Priority: absolute quality failures are caught first, then overfitting/underfitting.
function getModelStatus(trainR2, testR2) {
  if (trainR2 == null || testR2 == null) return null;
  const gap = trainR2 - testR2;

  // Absolute quality — catches bad models that happen to have a small gap
  if (testR2 < 0)    return { label: "Failing",      badge: "bg-red-100 text-red-700 border-red-300",    dot: "bg-red-600" };
  if (testR2 < 0.50) return { label: "Poor",         badge: "bg-red-50 text-red-500 border-red-200",     dot: "bg-red-400" };
  if (testR2 < 0.75) return { label: "Weak",         badge: "bg-orange-50 text-orange-600 border-orange-200", dot: "bg-orange-400" };

  // Gap-based overfitting — only meaningful when the model isn't already failing
  if (gap > 0.15)    return { label: "Overfitting",  badge: "bg-red-50 text-red-500 border-red-200",     dot: "bg-red-400" };
  if (gap > 0.05)    return { label: "Mild overfit", badge: "bg-orange-50 text-orange-500 border-orange-200", dot: "bg-orange-400" };
  if (gap < -0.05)   return { label: "Underfit",     badge: "bg-yellow-50 text-yellow-700 border-yellow-200", dot: "bg-yellow-500" };

  // Good — differentiate by how good
  if (testR2 >= 0.95) return { label: "Excellent",   badge: "bg-green-100 text-green-700 border-green-300", dot: "bg-green-600" };
  if (testR2 >= 0.85) return { label: "Good",        badge: "bg-green-50 text-green-600 border-green-200",  dot: "bg-green-500" };
  return                     { label: "Acceptable",  badge: "bg-blue-50 text-blue-600 border-blue-200",    dot: "bg-blue-400" };
}


function ModelComparison() {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecast, result } = location.state || {};

  const [view, setView] = useState("test");   // "train" | "test" | "both"

  if (!forecast) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center gap-4">
        <p className="text-gray-500">No forecast data found.</p>
        <button onClick={() => navigate("/")}
          className="bg-blue-600 text-white px-6 py-2 rounded-lg">
          Go Back to Upload
        </button>
      </div>
    );
  }

  const { per_model, forecast: points, ensemble_metrics, feature_importance } = forecast;

  // ── Best model on test ──────────────────────────────────────────────
  const bestModel = [...per_model].sort((a, b) => a.metrics.rmse - b.metrics.rmse)[0];

  // ── Build comparison rows including ensemble ────────────────────────
  const allModels = [
    ...per_model,
    { model_name: "Ensemble", metrics: ensemble_metrics, train_metrics: null, weight: 1.0 }
  ];

  // ── Time series chart data ──────────────────────────────────────────
  const step = Math.max(1, Math.floor(points.length / 150));
  const perModelChartData = points
    .filter((_, i) => i % step === 0)
    .map((p, i) => {
      const row = {
        time:   p.timestamp.substring(5, 16).replace("T", " "),
        Actual: p.actual != null ? parseFloat(p.actual.toFixed(1)) : null,
      };
      per_model.forEach(m => {
        row[m.model_name] = m.predictions[i * step] != null
          ? parseFloat(m.predictions[i * step].toFixed(1))
          : null;
      });
      row["Ensemble"] = parseFloat(p.predicted?.toFixed(1));
      return row;
    });

  // ── Residuals ─────────────────────────────────────────────────────────
  const residualData = points
    .filter((_, i) => i % step === 0)
    .map((p, i) => {
      const row = { time: p.timestamp.substring(5, 16).replace("T", " ") };
      per_model.forEach(m => {
        const pred = m.predictions[i * step];
        row[m.model_name] = pred != null && p.actual != null
          ? parseFloat((p.actual - pred).toFixed(1))
          : null;
      });
      row["Ensemble"] = p.actual != null
        ? parseFloat((p.actual - p.predicted).toFixed(1))
        : null;
      return row;
    });

  // ── Radar chart data — uses TEST metrics ────────────────────────────
  const maxRMSE = Math.max(...allModels.map(m => m.metrics.rmse));
  const maxMAE  = Math.max(...allModels.map(m => m.metrics.mae));
  const maxMAPE = Math.max(...allModels.map(m => m.metrics.mape || 0));

  const radarData = [
    { metric: "R²",          ...Object.fromEntries(allModels.map(m => [m.model_name, parseFloat((m.metrics.r2 * 100).toFixed(1))])) },
    { metric: "RMSE (inv)",  ...Object.fromEntries(allModels.map(m => [m.model_name, parseFloat(((1 - m.metrics.rmse / maxRMSE) * 100).toFixed(1))])) },
    { metric: "MAE (inv)",   ...Object.fromEntries(allModels.map(m => [m.model_name, parseFloat(((1 - m.metrics.mae  / maxMAE)  * 100).toFixed(1))])) },
    { metric: "MAPE (inv)",  ...Object.fromEntries(allModels.map(m => [m.model_name, parseFloat(((1 - (m.metrics.mape || 0) / (maxMAPE || 1)) * 100).toFixed(1))])) },
  ];

  // ── Feature importance ────────────────────────────────────────────────
  const featData = feature_importance
    ? Object.entries(feature_importance)
        .slice(0, 15)
        .map(([name, score]) => ({
          name,
          score: parseFloat((score * 100).toFixed(2)),
        }))
    : [];

  // ── Train vs Test gap chart data ─────────────────────────────────────
  const gapData = per_model.map(m => ({
    name:     m.model_name,
    "Train R²": m.train_metrics?.r2 ?? 0,
    "Test R²":  m.metrics.r2,
    fill:     MODEL_COLORS[m.model_name],
  }));

  return (
    <div className="min-h-screen bg-gray-50 p-6">

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Model Comparison</h1>
          <p className="text-sm text-gray-400 mt-1">
            XGBoost vs LightGBM vs Prophet vs Ensemble — train + test comparison
          </p>
        </div>
        <div className="flex gap-3">
          <button onClick={() => navigate("/dashboard", { state: { forecast, result } })}
            className="border border-gray-300 text-gray-600 px-4 py-2 rounded-lg text-sm hover:bg-gray-50">
            ← Dashboard
          </button>
          <button onClick={() => navigate("/anomalies", { state: { forecast, result } })}
            className="border border-orange-300 text-orange-600 px-4 py-2 rounded-lg text-sm hover:bg-orange-50">
            Anomalies →
          </button>
        </div>
      </div>

      {/* Best model banner */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-6 flex items-center gap-4">
        <div>
          <p className="font-semibold text-blue-800">
            Best Individual Model on Test Set: {bestModel.model_name}
          </p>
          <p className="text-sm text-blue-600">
            Test RMSE={bestModel.metrics.rmse.toFixed(2)} W/m² |
            Test R²={bestModel.metrics.r2.toFixed(4)} |
            {bestModel.train_metrics &&
              ` Train R²=${bestModel.train_metrics.r2.toFixed(4)}`}
          </p>
        </div>
      </div>

      {/* View selector */}
      <div className="flex gap-2 mb-6">
        <span className="text-sm text-gray-500 self-center">View metrics:</span>
        {[
          { id: "test",  label: "Test Only" },
          { id: "train", label: "Train Only" },
          { id: "both",  label: "Train + Test" },
        ].map(v => (
          <button key={v.id}
            onClick={() => setView(v.id)}
            className={`px-4 py-1 rounded-full text-sm font-medium transition
              ${view === v.id
                ? "bg-blue-600 text-white"
                : "bg-white text-gray-600 border border-gray-300 hover:bg-gray-50"}`}>
            {v.label}
          </button>
        ))}
      </div>

      {/* Train vs Test metrics table */}
      <Card title="Train vs Test Metrics"
        subtitle="Compare model performance on training data (what it has seen) vs test data (held-out 20%). A large gap suggests overfitting.">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b">
                <th rowSpan={2} className="text-left p-3 text-gray-600 align-bottom">Model</th>
                {(view === "train" || view === "both") &&
                  <th colSpan={4} className="text-center p-2 text-gray-600 bg-blue-50">TRAIN SET</th>}
                {(view === "test" || view === "both") &&
                  <th colSpan={4} className="text-center p-2 text-gray-600 bg-orange-50">TEST SET</th>}
                <th rowSpan={2} className="text-right p-3 text-gray-600 align-bottom">Status</th>
              </tr>
              <tr className="bg-gray-50 border-b">
                {(view === "train" || view === "both") && (
                  <>
                    <th className="text-right p-2 text-gray-500 bg-blue-50">RMSE</th>
                    <th className="text-right p-2 text-gray-500 bg-blue-50">MAE</th>
                    <th className="text-right p-2 text-gray-500 bg-blue-50">R²</th>
                    <th className="text-right p-2 text-gray-500 bg-blue-50">MAPE</th>
                  </>
                )}
                {(view === "test" || view === "both") && (
                  <>
                    <th className="text-right p-2 text-gray-500 bg-orange-50">RMSE</th>
                    <th className="text-right p-2 text-gray-500 bg-orange-50">MAE</th>
                    <th className="text-right p-2 text-gray-500 bg-orange-50">R²</th>
                    <th className="text-right p-2 text-gray-500 bg-orange-50">MAPE</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {per_model.map((m, i) => {
                const status = getModelStatus(m.train_metrics?.r2, m.metrics.r2);
                const isBest = m.model_name === bestModel.model_name;
                return (
                  <tr key={i} className="border-b hover:bg-gray-50">
                    <td className="p-3 flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full inline-block"
                        style={{ backgroundColor: MODEL_COLORS[m.model_name] }} />
                      {m.model_name}
                      {isBest &&
                        <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">best</span>}
                    </td>
                    {(view === "train" || view === "both") && (
                      <>
                        <td className="p-2 text-right">{m.train_metrics?.rmse?.toFixed(2) ?? "—"}</td>
                        <td className="p-2 text-right">{m.train_metrics?.mae?.toFixed(2) ?? "—"}</td>
                        <td className={`p-2 text-right font-medium ${(m.train_metrics?.r2 || 0) > 0.9 ? "text-green-600" : "text-orange-500"}`}>
                          {m.train_metrics?.r2?.toFixed(4) ?? "—"}
                        </td>
                        <td className="p-2 text-right">{m.train_metrics?.mape?.toFixed(2) ?? "—"}</td>
                      </>
                    )}
                    {(view === "test" || view === "both") && (
                      <>
                        <td className="p-2 text-right">{m.metrics.rmse.toFixed(2)}</td>
                        <td className="p-2 text-right">{m.metrics.mae.toFixed(2)}</td>
                        <td className={`p-2 text-right font-medium ${m.metrics.r2 > 0.9 ? "text-green-600" : m.metrics.r2 > 0 ? "text-yellow-600" : "text-red-500"}`}>
                          {m.metrics.r2.toFixed(4)}
                        </td>
                        <td className="p-2 text-right">{m.metrics.mape?.toFixed(2) ?? "—"}</td>
                      </>
                    )}
                    <td className="p-3 text-right">
                      {status ? (
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${status.badge}`}>
                          <span className={`w-1.5 h-1.5 rounded-full ${status.dot}`} />
                          {status.label}
                        </span>
                      ) : "—"}
                    </td>
                  </tr>
                );
              })}
              <tr className="border-b bg-purple-50 font-semibold">
                <td className="p-3 flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full inline-block"
                    style={{ backgroundColor: MODEL_COLORS.Ensemble }} />
                  Ensemble
                </td>
                {(view === "train" || view === "both") && (
                  <>
                    <td className="p-2 text-right text-gray-400">—</td>
                    <td className="p-2 text-right text-gray-400">—</td>
                    <td className="p-2 text-right text-gray-400">—</td>
                    <td className="p-2 text-right text-gray-400">—</td>
                  </>
                )}
                {(view === "test" || view === "both") && (
                  <>
                    <td className="p-2 text-right">{ensemble_metrics.rmse.toFixed(2)}</td>
                    <td className="p-2 text-right">{ensemble_metrics.mae.toFixed(2)}</td>
                    <td className="p-2 text-right text-green-600">{ensemble_metrics.r2.toFixed(4)}</td>
                    <td className="p-2 text-right">{ensemble_metrics.mape?.toFixed(2) ?? "—"}</td>
                  </>
                )}
                <td className="p-3 text-right text-xs text-gray-400">weighted blend</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>

      {/* Train R² vs Test R² gap chart */}
      <Card title="Train R² vs Test R² — Generalization Check"
        subtitle="Smaller gap = better generalization. Large train>test gap means the model memorized training data instead of learning patterns.">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={gapData} barGap={4}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="Train R²" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Test R²"  fill="#f59e0b" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Bar charts of all metrics */}
      <Card title="Test Set Metric Bar Charts"
        subtitle="Visual comparison of each metric across all four models on the held-out test set">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(METRIC_INFO).map(([key, info]) => {
            const barData = allModels.map(m => ({
              name:  m.model_name,
              value: parseFloat((m.metrics[key] ?? 0).toFixed(4)),
              fill:  MODEL_COLORS[m.model_name],
            }));
            return (
              <div key={key}>
                <p className="text-sm font-medium text-gray-600 mb-2">
                  {info.label}
                  <span className="text-xs text-gray-400 ml-2">{info.desc}</span>
                </p>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={barData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip formatter={v => [`${v} ${info.unit}`]} />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {barData.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Individual Predicted vs Actual — one chart per model */}
      <Card title="Predicted vs Actual — Per Model"
        subtitle="Each model's predictions (colored) against ground truth (gray) on the held-out test set">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[...per_model, { model_name: "Ensemble", metrics: ensemble_metrics, train_metrics: null }].map(m => {
            const color = MODEL_COLORS[m.model_name];
            const chartData = perModelChartData.map(row => ({
              time:      row.time,
              Actual:    row.Actual,
              Predicted: row[m.model_name],
            }));
            return (
              <div key={m.model_name} className="border border-gray-100 rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full inline-block" style={{ backgroundColor: color }} />
                    <span className="font-semibold text-gray-700 text-sm">{m.model_name}</span>
                  </div>
                  <div className="flex gap-3 text-xs text-gray-500">
                    <span>RMSE <strong className="text-gray-700">{m.metrics.rmse.toFixed(2)}</strong></span>
                    <span>R² <strong className={m.metrics.r2 > 0.9 ? "text-green-600" : m.metrics.r2 > 0 ? "text-yellow-600" : "text-red-500"}>
                      {m.metrics.r2.toFixed(4)}
                    </strong></span>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="time" tick={{ fontSize: 9 }}
                      interval={Math.floor(chartData.length / 6)} />
                    <YAxis tick={{ fontSize: 10 }}
                      label={{ value: "W/m²", angle: -90, position: "insideLeft", style: { fontSize: 10 } }} />
                    <Tooltip contentStyle={{ fontSize: 11 }}
                      formatter={(val, name) => [val != null ? `${val} W/m²` : "—", name]} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line type="monotone" dataKey="Actual"    stroke={MODEL_COLORS.Actual} strokeWidth={1.5} dot={false} />
                    <Line type="monotone" dataKey="Predicted" stroke={color}               strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Predicted vs Actual — All Models overlaid */}
      <Card title="Predicted vs Actual — All Models"
        subtitle="Time series comparison on the test set. Solid gray = ground truth, dashed lines = individual models, solid purple = ensemble">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={perModelChartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="time" tick={{ fontSize: 10 }}
              interval={Math.floor(perModelChartData.length / 8)} />
            <YAxis tick={{ fontSize: 11 }}
              label={{ value: "W/m²", angle: -90, position: "insideLeft" }} />
            <Tooltip contentStyle={{ fontSize: 11 }} />
            <Legend />
            <Line type="monotone" dataKey="Actual"   stroke={MODEL_COLORS.Actual}   strokeWidth={2} dot={false} />
            {per_model.map(m => (
              <Line key={m.model_name} type="monotone" dataKey={m.model_name}
                stroke={MODEL_COLORS[m.model_name]} strokeWidth={1.5} dot={false}
                strokeDasharray="4 2" />
            ))}
            <Line type="monotone" dataKey="Ensemble" stroke={MODEL_COLORS.Ensemble} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Residuals over time */}
      <Card title="Residual Analysis (Actual − Predicted)"
        subtitle="Positive = under-predicted, Negative = over-predicted. Ideal: residuals scattered around 0">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={residualData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="time" tick={{ fontSize: 10 }}
              interval={Math.floor(residualData.length / 8)} />
            <YAxis tick={{ fontSize: 11 }}
              label={{ value: "Residual W/m²", angle: -90, position: "insideLeft" }} />
            <Tooltip contentStyle={{ fontSize: 11 }} />
            <Legend />
            <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="4 2" />
            {allModels.map(m => (
              <Line key={m.model_name} type="monotone" dataKey={m.model_name}
                stroke={MODEL_COLORS[m.model_name]} strokeWidth={1.5} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Radar chart */}
      <Card title="Model Performance Radar"
        subtitle="Overall view across all metrics. Larger filled area = better model. RMSE/MAE/MAPE inverted so all metrics point outward for the best">
        <ResponsiveContainer width="100%" height={340}>
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
            <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10 }} />
            {allModels.map(m => (
              <Radar key={m.model_name} name={m.model_name} dataKey={m.model_name}
                stroke={MODEL_COLORS[m.model_name]} fill={MODEL_COLORS[m.model_name]}
                fillOpacity={0.15} strokeWidth={2} />
            ))}
            <Legend />
            <Tooltip />
          </RadarChart>
        </ResponsiveContainer>
      </Card>

      {/* Feature importance */}
      {featData.length > 0 && (
        <Card title="Feature Importance (XGBoost + LightGBM Average)"
          subtitle="Top 15 features by averaged gain importance — these are what the tree models rely on most">
          <ResponsiveContainer width="100%" height={380}>
            <BarChart data={featData} layout="vertical" margin={{ left: 120, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis type="number" tick={{ fontSize: 11 }} unit="%" />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={115} />
              <Tooltip formatter={v => [`${v}%`]} />
              <Bar dataKey="score" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Ensemble weights */}
      <Card title="Ensemble Weight Distribution"
        subtitle="Weights computed from validation RMSE — best individual model gets the highest weight automatically">
        <div className="flex gap-4">
          {per_model.map(m => (
            <div key={m.model_name} className="flex-1 rounded-xl p-5 text-center"
              style={{ backgroundColor: MODEL_COLORS[m.model_name] + "18", border: `1px solid ${MODEL_COLORS[m.model_name]}40` }}>
              <p className="text-sm text-gray-500 mb-1">{m.model_name}</p>
              <p className="text-3xl font-bold" style={{ color: MODEL_COLORS[m.model_name] }}>
                {(m.weight * 100).toFixed(1)}%
              </p>
              <div className="mt-2 text-xs text-gray-400 space-y-0.5">
                <p>Train R²: {m.train_metrics?.r2?.toFixed(4) ?? "—"}</p>
                <p>Test R²:  {m.metrics.r2.toFixed(4)}</p>
                <p>Test RMSE: {m.metrics.rmse.toFixed(2)}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

    </div>
  );
}

export default ModelComparison;