import { useLocation, useNavigate } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter,
  ReferenceLine
} from "recharts";

// ── Reusable section card ─────────────────────────────────────────────────
function Card({ title, subtitle, children }) {
  return (
    <div className="bg-white rounded-xl shadow p-6 mb-6">
      <h2 className="text-lg font-semibold text-gray-700">{title}</h2>
      {subtitle && <p className="text-sm text-gray-400 mb-4">{subtitle}</p>}
      <div className="mt-3">{children}</div>
    </div>
  );
}

// ── Color map per model ───────────────────────────────────────────────────
const MODEL_COLORS = {
  XGBoost:  "#3b82f6",
  LightGBM: "#10b981",
  Prophet:  "#f59e0b",
  Ensemble: "#8b5cf6",
  Actual:   "#6b7280",
};

const METRIC_INFO = {
  rmse: { label: "RMSE",  unit: "W/m²", lower: true,  desc: "Root Mean Square Error — penalises large errors" },
  mae:  { label: "MAE",   unit: "W/m²", lower: true,  desc: "Mean Absolute Error — average magnitude of errors" },
  r2:   { label: "R²",    unit: "",     lower: false, desc: "Coefficient of Determination — 1.0 = perfect" },
  mape: { label: "MAPE",  unit: "%",    lower: true,  desc: "Mean Absolute Percentage Error" },
};

function ModelComparison() {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecast, result } = location.state || {};

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

  const { per_model, forecast: points, ensemble_metrics, ensemble_train_metrics, feature_importance } = forecast;

  // ── Build comparison table rows ───────────────────────────────────────────
  const allModels = [
    ...per_model,
    { model_name: "Ensemble", metrics: ensemble_metrics, weight: 1.0 }
  ];

  // ── Chart data: predicted vs actual per model (sampled) ──────────────────
  const step = Math.max(1, Math.floor(points.length / 150));
  const actuals = points.filter((_, i) => i % step === 0).map(p => p.actual);

  const perModelChartData = points
    .filter((_, i) => i % step === 0)
    .map((p, i) => {
      const row = {
        time:    p.timestamp.substring(5, 16).replace("T", " "),
        Actual:  p.actual != null ? parseFloat(p.actual.toFixed(1)) : null,
      };
      per_model.forEach(m => {
        row[m.model_name] = m.predictions[i * step] != null
          ? parseFloat(m.predictions[i * step].toFixed(1))
          : null;
      });
      row["Ensemble"] = parseFloat(p.predicted?.toFixed(1));
      return row;
    });

  // ── Residuals per model ───────────────────────────────────────────────────
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

  // ── Scatter: predicted vs actual ─────────────────────────────────────────
  const scatterData = per_model.map(m => ({
    name: m.model_name,
    color: MODEL_COLORS[m.model_name],
    data: points
      .filter((_, i) => i % step === 0)
      .map((p, i) => ({
        actual:    p.actual,
        predicted: m.predictions[i * step],
      }))
      .filter(d => d.actual != null && d.predicted != null),
  }));

  // ── Feature importance ────────────────────────────────────────────────────
  const featData = feature_importance
    ? Object.entries(feature_importance)
        .slice(0, 15)
        .map(([name, score]) => ({
          name,
          score: parseFloat((score * 100).toFixed(2)),
        }))
    : [];

  // ── Best model ────────────────────────────────────────────────────────────
  const bestModel = [...per_model].sort((a, b) => a.metrics.rmse - b.metrics.rmse)[0];

  return (
    <div className="min-h-screen bg-gray-50 p-6">

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Model Comparison</h1>
          <p className="text-sm text-gray-400 mt-1">
            XGBoost vs LightGBM vs Prophet vs Ensemble — {points.length} test points
          </p>
        </div>
        <div className="flex gap-3">
          <button onClick={() => navigate("/dashboard", { state: { forecast, result } })}
            className="border border-gray-300 text-gray-600 px-4 py-2 rounded-lg text-sm hover:bg-gray-50">
            ← Dashboard
          </button>
          <button onClick={() => navigate("/anomalies", { state: { forecast, result } })}
            className="border border-orange-300 text-orange-600 px-4 py-2 rounded-lg text-sm hover:bg-orange-50">
            Anomaly Report →
          </button>
        </div>
      </div>

      {/* Best model banner */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-6 flex items-center gap-4">
        <span className="text-2xl">🏆</span>
        <div>
          <p className="font-semibold text-blue-800">
            Best Individual Model: {bestModel.model_name}
          </p>
          <p className="text-sm text-blue-600">
            RMSE={bestModel.metrics.rmse.toFixed(2)} W/m²  |
            R²={bestModel.metrics.r2.toFixed(4)}  |
            MAE={bestModel.metrics.mae.toFixed(2)} W/m²
          </p>
        </div>
      </div>

      {/* Metrics comparison table */}
      <Card title="Metrics Comparison Table"
        subtitle="All four metrics across all models on the held-out test set (last 20% of data)">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b">
                <th className="text-left p-3 text-gray-600">Model</th>
                <th className="text-right p-3 text-gray-600">RMSE (W/m²) ↓</th>
                <th className="text-right p-3 text-gray-600">MAE (W/m²) ↓</th>
                <th className="text-right p-3 text-gray-600">R² ↑</th>
                <th className="text-right p-3 text-gray-600">MAPE (%) ↓</th>
                <th className="text-right p-3 text-gray-600">Weight</th>
              </tr>
            </thead>
            <tbody>
              {allModels.map((m, i) => {
                const isEnsemble = m.model_name === "Ensemble";
                const isBest = m.model_name === bestModel.model_name;
                return (
                  <tr key={i}
                    className={`border-b ${isEnsemble ? "bg-purple-50 font-semibold" : "hover:bg-gray-50"}`}>
                    <td className="p-3 flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full inline-block"
                        style={{ backgroundColor: MODEL_COLORS[m.model_name] }} />
                      {m.model_name}
                      {isBest && !isEnsemble &&
                        <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full ml-1">best</span>}
                    </td>
                    <td className="p-3 text-right">{m.metrics.rmse.toFixed(2)}</td>
                    <td className="p-3 text-right">{m.metrics.mae.toFixed(2)}</td>
                    <td className={`p-3 text-right font-medium ${m.metrics.r2 > 0.9 ? "text-green-600" : m.metrics.r2 > 0 ? "text-yellow-600" : "text-red-500"}`}>
                      {m.metrics.r2.toFixed(4)}
                    </td>
                    <td className="p-3 text-right">{m.metrics.mape?.toFixed(2) ?? "—"}</td>
                    <td className="p-3 text-right text-gray-400">
                      {m.model_name === "Ensemble" ? "—" : `${(m.weight * 100).toFixed(1)}%`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Metrics bar charts */}
      <Card title="Metric Bar Charts"
        subtitle="Visual comparison of RMSE, MAE, R², MAPE across all models">
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
                  <BarChart data={barData} margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip formatter={v => [`${v} ${info.unit}`]} />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {barData.map((entry, i) => (
                        <rect key={i} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Predicted vs Actual — all models */}
      <Card title="Predicted vs Actual — All Models"
        subtitle="Time series comparison of each model's predictions against ground truth">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={perModelChartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="time" tick={{ fontSize: 10 }} interval={Math.floor(perModelChartData.length / 8)} />
            <YAxis tick={{ fontSize: 11 }} label={{ value: "W/m²", angle: -90, position: "insideLeft" }} />
            <Tooltip contentStyle={{ fontSize: 11 }} />
            <Legend />
            <Line type="monotone" dataKey="Actual"   stroke={MODEL_COLORS.Actual}   strokeWidth={2} dot={false} />
            {per_model.map(m => (
              <Line key={m.model_name} type="monotone" dataKey={m.model_name}
                stroke={MODEL_COLORS[m.model_name]} strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
            ))}
            <Line type="monotone" dataKey="Ensemble" stroke={MODEL_COLORS.Ensemble} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Individual Predicted vs Actual charts */}
      <Card title="Individual Model Predictions"
        subtitle="Each model is shown separately against the actual values">
        <div className="flex flex-col gap-6">
          {per_model.map((m) => (
            <div
              key={m.model_name}
              className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm"
            >
              <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                  <h3 className="text-base font-semibold text-gray-800">
                    {m.model_name}
                  </h3>
                  <p className="text-xs text-gray-500">
                      Train RMSE {m.train_metrics?.rmse.toFixed(2) ?? "—"} W/m² | Predict RMSE {m.metrics.rmse.toFixed(2)} W/m²
                    </p>
                    <p className="text-xs text-gray-400">
                      Train MAE {m.train_metrics?.mae.toFixed(2) ?? "—"} W/m² | Predict MAE {m.metrics.mae.toFixed(2)} W/m²
                    </p>
                </div>
                <span
                  className="h-3 w-3 rounded-full"
                  style={{ backgroundColor: MODEL_COLORS[m.model_name] }}
                />
              </div>

              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={perModelChartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    dataKey="time"
                    tick={{ fontSize: 9 }}
                    interval={Math.floor(perModelChartData.length / 6)}
                  />
                  <YAxis
                    tick={{ fontSize: 10 }}
                    label={{ value: "W/m²", angle: -90, position: "insideLeft" }}
                  />
                  <Tooltip contentStyle={{ fontSize: 10 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line
                    type="monotone"
                    dataKey="Actual"
                    stroke={MODEL_COLORS.Actual}
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey={m.model_name}
                    stroke={MODEL_COLORS[m.model_name]}
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ))}

          <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <h3 className="text-base font-semibold text-gray-800">
                  Ensemble
                </h3>
                <p className="text-xs text-gray-500">
                  Train RMSE {ensemble_train_metrics?.rmse.toFixed(2) ?? "—"} W/m² | Predict RMSE {ensemble_metrics.rmse.toFixed(2)} W/m²
                </p>
                <p className="text-xs text-gray-400">
                  Train MAE {ensemble_train_metrics?.mae.toFixed(2) ?? "—"} W/m² | Predict MAE {ensemble_metrics.mae.toFixed(2)} W/m²
                </p>
              </div>
              <span
                className="h-3 w-3 rounded-full"
                style={{ backgroundColor: MODEL_COLORS.Ensemble }}
              />
            </div>

            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={perModelChartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 9 }}
                  interval={Math.floor(perModelChartData.length / 6)}
                />
                <YAxis
                  tick={{ fontSize: 10 }}
                  label={{ value: "W/m²", angle: -90, position: "insideLeft" }}
                />
                <Tooltip contentStyle={{ fontSize: 10 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  type="monotone"
                  dataKey="Actual"
                  stroke={MODEL_COLORS.Actual}
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="Ensemble"
                  stroke={MODEL_COLORS.Ensemble}
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </Card>

      {/* Residual plot */}
      <Card title="Residual Analysis (Actual − Predicted)"
        subtitle="Positive = under-predicted, Negative = over-predicted. Ideal: residuals centred around 0">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={residualData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="time" tick={{ fontSize: 10 }} interval={Math.floor(residualData.length / 8)} />
            <YAxis tick={{ fontSize: 11 }} label={{ value: "Residual W/m²", angle: -90, position: "insideLeft" }} />
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

      {/* Feature importance */}
      {featData.length > 0 && (
        <Card title="Feature Importance (XGBoost + LightGBM average)"
          subtitle="Top 15 features by averaged gain importance from both tree models">
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
        subtitle="Weights computed inversely from validation RMSE — best model gets highest weight automatically">
        <div className="flex gap-4">
          {per_model.map(m => (
            <div key={m.model_name} className="flex-1 rounded-xl p-5 text-center"
              style={{ backgroundColor: MODEL_COLORS[m.model_name] + "18", border: `1px solid ${MODEL_COLORS[m.model_name]}40` }}>
              <p className="text-sm text-gray-500 mb-1">{m.model_name}</p>
              <p className="text-3xl font-bold" style={{ color: MODEL_COLORS[m.model_name] }}>
                {(m.weight * 100).toFixed(1)}%
              </p>
              <div className="mt-2 text-xs text-gray-400 space-y-0.5">
                <p>RMSE: {m.metrics.rmse.toFixed(2)}</p>
                <p>R²: {m.metrics.r2.toFixed(4)}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

    </div>
  );
}

export default ModelComparison;