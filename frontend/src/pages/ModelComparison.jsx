import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, Cell,
  ReferenceLine
} from "recharts";
import { saveForecastState, loadForecastState } from "../services/forecastState";
import { Card, PageHeader } from "../components/ui";

const MODEL_COLORS = {
  XGBoost:  "#3b82f6",
  LightGBM: "#10b981",
  Actual:   "#6b7280",
};

const METRIC_INFO = {
  rmse: { label: "RMSE",  unit: "W/m²", lower: true,  desc: "Root Mean Square Error — penalises large errors" },
  mae:  { label: "MAE",   unit: "W/m²", lower: true,  desc: "Mean Absolute Error" },
  r2:   { label: "R²",    unit: "",     lower: false, desc: "Coefficient of Determination — 1.0 = perfect" },
};

// Strict model health check — considers absolute test quality, R² gap, AND RMSE ratio.
// R² gap alone is misleading: a model can have train R²=0.9999 / test R²=0.98 (tiny gap)
// while its RMSE explodes 4-5× on test — that is overfitting and must be flagged.
function getModelStatus(trainR2, testR2, trainRmse, testRmse) {
  if (trainR2 == null || testR2 == null) return null;

  const r2Gap    = trainR2 - testR2;
  const rmseRatio = (trainRmse > 0 && testRmse != null) ? testRmse / trainRmse : 1;

  // 1. Absolute quality failures — catch bad models regardless of gap
  if (testR2 < 0)    return { label: "Failing",      badge: "bg-red-100 text-red-700 border-red-300",         dot: "bg-red-600" };
  if (testR2 < 0.50) return { label: "Poor",         badge: "bg-red-50 text-red-500 border-red-200",          dot: "bg-red-400" };
  if (testR2 < 0.75) return { label: "Weak",         badge: "bg-orange-50 text-orange-600 border-orange-200", dot: "bg-orange-400" };

  // 2. Overfitting — flag if EITHER R² gap is large OR RMSE ratio is high
  //    RMSE ratio catches cases where R² gap looks small but error blows up on test
  if (r2Gap > 0.15 || rmseRatio > 3.5) return { label: "Overfitting",  badge: "bg-red-50 text-red-500 border-red-200",          dot: "bg-red-400" };
  if (r2Gap > 0.05 || rmseRatio > 2.0) return { label: "Mild overfit", badge: "bg-orange-50 text-orange-500 border-orange-200", dot: "bg-orange-400" };

  // 3. Underfitting
  if (r2Gap < -0.05) return { label: "Underfit",    badge: "bg-yellow-50 text-yellow-700 border-yellow-200", dot: "bg-yellow-500" };

  // 4. Genuinely good — model generalizes well
  if (testR2 >= 0.95) return { label: "Excellent",  badge: "bg-green-100 text-green-700 border-green-300",   dot: "bg-green-600" };
  if (testR2 >= 0.85) return { label: "Good",       badge: "bg-green-50 text-green-600 border-green-200",    dot: "bg-green-500" };
  return                     { label: "Acceptable", badge: "bg-blue-50 text-blue-600 border-blue-200",       dot: "bg-blue-400" };
}


function ModelComparison() {
  const location = useLocation();
  const navigate = useNavigate();
  const locationState = location.state || {};
  const savedState = loadForecastState();
  
  // Use location state if available, otherwise fall back to saved state
  const { forecast, result } = locationState.forecast ? locationState : (savedState || {});
  const cachedTrainSize = locationState.trainSize ?? savedState?.trainSize ?? 80;

  const [view, setView] = useState("test");   // "train" | "test" | "both"

  // Save state when it changes
  useEffect(() => {
    if (forecast && result) {
      saveForecastState(
        forecast,
        result,
        result.filename,
        cachedTrainSize,
      );
    }
  }, [forecast, result, cachedTrainSize]);

  if (!forecast) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center gap-4">
        <p className="text-slate-500">No forecast data found.</p>
        <button onClick={() => navigate("/")}
          className="rounded-full bg-slate-900 px-6 py-2 text-white transition hover:bg-slate-800">
          Go Back to Upload
        </button>
      </div>
    );
  }

  const { models_info, forecast: points, metrics, feature_importance, best_model: bestModelName } = forecast;

  const bestModel = (models_info || []).find(m => m.is_best) || models_info?.[0];

  const step = Math.max(1, Math.floor(points.length / 150));
  const perModelChartData = points
    .filter((_, i) => i % step === 0)
    .map((p, i) => {
      const row = {
        time:   p.timestamp.substring(5, 16).replace("T", " "),
        Actual: p.actual != null ? parseFloat(p.actual.toFixed(1)) : null,
      };
      (models_info || []).forEach(m => {
        row[m.model_name] = m.predictions[i * step] != null
          ? parseFloat(m.predictions[i * step].toFixed(1))
          : null;
      });
      return row;
    });

  // Residuals
  const residualData = points
    .filter((_, i) => i % step === 0)
    .map((p, i) => {
      const row = { time: p.timestamp.substring(5, 16).replace("T", " ") };
      (models_info || []).forEach(m => {
        const pred = m.predictions[i * step];
        row[m.model_name] = pred != null && p.actual != null
          ? parseFloat((p.actual - pred).toFixed(1))
          : null;
      });
      return row;
    });

  // Feature importance
  const featData = feature_importance
    ? Object.entries(feature_importance)
        .slice(0, 15)
        .map(([name, score]) => ({
          name,
          score: parseFloat((score * 100).toFixed(2)),
        }))
    : [];

  // Train vs Test gap chart data
  const gapData = (models_info || []).map(m => ({
    name:     m.model_name,
    "Train R²": m.train_metrics?.r2 ?? 0,
    "Test R²":  m.metrics.r2,
    fill:     MODEL_COLORS[m.model_name],
  }));

  return (
    <div className="min-h-screen bg-gray-50 p-6">

      <PageHeader
        title="Model Comparison"
        subtitle="XGBoost vs LightGBM — train + test comparison"
        actions={[
          <button
            key="dashboard"
            onClick={() => navigate("/dashboard", { state: { forecast, result } })}
            className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-400 hover:bg-slate-50"
          >
            Dashboard
          </button>,
          <button
            key="anomalies"
            onClick={() => navigate(`/anomalies?session_id=${result.session_id}`, { state: { forecast, result } })}
            className="rounded-full bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800"
          >
            Anomalies
          </button>,
        ]}
      />


      {/* View selector */}
      <div className="mb-6 flex gap-2">
        <span className="self-center text-sm text-slate-500">View metrics:</span>
        {[
          { id: "test",  label: "Test Only" },
          { id: "train", label: "Train Only" },
          { id: "both",  label: "Train + Test" },
        ].map(v => (
          <button key={v.id}
            onClick={() => setView(v.id)}
            className={`rounded-full px-4 py-2 text-sm font-medium transition
              ${view === v.id
                ? "bg-slate-900 text-white"
                : "border border-slate-300 bg-white text-slate-600 hover:border-slate-400 hover:bg-slate-50"}`}>
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
                  <th colSpan={3} className="text-center p-2 text-gray-600 bg-blue-50">TRAIN SET</th>}
                {(view === "test" || view === "both") &&
                  <th colSpan={3} className="text-center p-2 text-gray-600 bg-orange-50">TEST SET</th>}
                <th rowSpan={2} className="text-right p-3 text-gray-600 align-bottom">Status</th>
              </tr>
              <tr className="bg-gray-50 border-b">
                {(view === "train" || view === "both") && (
                  <>
                    <th className="text-right p-2 text-gray-500 bg-blue-50">RMSE</th>
                    <th className="text-right p-2 text-gray-500 bg-blue-50">MAE</th>
                    <th className="text-right p-2 text-gray-500 bg-blue-50">R²</th>
                  </>
                )}
                {(view === "test" || view === "both") && (
                  <>
                    <th className="text-right p-2 text-gray-500 bg-orange-50">RMSE</th>
                    <th className="text-right p-2 text-gray-500 bg-orange-50">MAE</th>
                    <th className="text-right p-2 text-gray-500 bg-orange-50">R²</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {(models_info || []).map((m, i) => {
                const status = getModelStatus(m.train_metrics?.r2, m.metrics.r2, m.train_metrics?.rmse, m.metrics.rmse);
                return (
                  <tr key={i} className="border-b hover:bg-gray-50">
                    <td className="p-3 flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full inline-block"
                        style={{ backgroundColor: MODEL_COLORS[m.model_name] }} />
                      {m.model_name}
                      {m.is_best &&
                        <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">best</span>}
                    </td>
                    {(view === "train" || view === "both") && (
                      <>
                        <td className="p-2 text-right">{m.train_metrics?.rmse?.toFixed(2) ?? "—"}</td>
                        <td className="p-2 text-right">{m.train_metrics?.mae?.toFixed(2) ?? "—"}</td>
                        <td className={`p-2 text-right font-medium ${(m.train_metrics?.r2 || 0) > 0.9 ? "text-green-600" : "text-orange-500"}`}>
                          {m.train_metrics?.r2?.toFixed(4) ?? "—"}
                        </td>
                      </>
                    )}
                    {(view === "test" || view === "both") && (
                      <>
                        <td className="p-2 text-right">{m.metrics.rmse.toFixed(2)}</td>
                        <td className="p-2 text-right">{m.metrics.mae.toFixed(2)}</td>
                        <td className={`p-2 text-right font-medium ${m.metrics.r2 > 0.9 ? "text-green-600" : m.metrics.r2 > 0 ? "text-yellow-600" : "text-red-500"}`}>
                          {m.metrics.r2.toFixed(4)}
                        </td>
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
        subtitle="Visual comparison of each metric across models on the held-out test set">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(METRIC_INFO).map(([key, info]) => {
            const barData = (models_info || []).map(m => ({
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
          {(models_info || []).map(m => {
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
        subtitle="Time series comparison on the test set. Solid gray = ground truth, dashed lines = individual models">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={perModelChartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="time" tick={{ fontSize: 10 }}
              interval={Math.floor(perModelChartData.length / 8)} />
            <YAxis tick={{ fontSize: 11 }}
              label={{ value: "W/m²", angle: -90, position: "insideLeft" }} />
            <Tooltip contentStyle={{ fontSize: 11 }} />
            <Legend />
            <Line type="monotone" dataKey="Actual" stroke={MODEL_COLORS.Actual} strokeWidth={2} dot={false} />
            {(models_info || []).map(m => (
              <Line key={m.model_name} type="monotone" dataKey={m.model_name}
                stroke={MODEL_COLORS[m.model_name]} strokeWidth={1.5} dot={false}
                strokeDasharray="4 2" />
            ))}
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
            {(models_info || []).map(m => (
              <Line key={m.model_name} type="monotone" dataKey={m.model_name}
                stroke={MODEL_COLORS[m.model_name]} strokeWidth={1.5} dot={false} />
            ))}
          </LineChart>
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

      {/* Model selection summary */}
      <Card title="Model Selection"
        subtitle="The model with the lowest test RMSE is automatically selected for forecasting">
        <div className="flex gap-4">
          {(models_info || []).map(m => (
            <div key={m.model_name} className="flex-1 rounded-xl p-5 text-center"
              style={{ backgroundColor: MODEL_COLORS[m.model_name] + "18", border: `2px solid ${m.is_best ? MODEL_COLORS[m.model_name] : MODEL_COLORS[m.model_name] + "40"}` }}>
              <div className="flex items-center justify-center gap-2 mb-1">
                <p className="text-sm text-gray-500">{m.model_name}</p>
                {m.is_best && <span className="text-xs bg-blue-600 text-white px-2 py-0.5 rounded-full">Selected</span>}
              </div>
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