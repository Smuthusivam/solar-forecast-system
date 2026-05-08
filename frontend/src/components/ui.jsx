// Shared UI primitives used across all pages.

// ── StatCard ──────────────────────────────────────────────────────────────────
export function StatCard({ label, value, unit, color = "text-gray-800", sub }) {
  return (
    <div className="bg-white rounded-xl shadow p-4">
      <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${color}`}>
        {value}
        {unit && <span className="text-sm font-normal text-gray-400 ml-1">{unit}</span>}
      </p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  );
}

// ── Card ──────────────────────────────────────────────────────────────────────
export function Card({ title, subtitle, children, right }) {
  return (
    <div className="bg-white rounded-xl shadow p-6 mb-6">
      <div className="flex items-start justify-between mb-2">
        <div>
          <h2 className="text-lg font-semibold text-gray-700">{title}</h2>
          {subtitle && <p className="text-sm text-gray-400">{subtitle}</p>}
        </div>
        {right}
      </div>
      <div className="mt-3">{children}</div>
    </div>
  );
}

// ── Section (border variant used in AnomalyReport / Forecast) ─────────────────
export function Section({ title, subtitle, children }) {
  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden mb-6">
      <div className="px-5 py-4 border-b border-gray-100">
        <h3 className="font-semibold text-gray-900 text-base">{title}</h3>
        {subtitle && <p className="text-xs text-gray-400 mt-0.5">{subtitle}</p>}
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}

// ── TabNav ────────────────────────────────────────────────────────────────────
export function TabNav({ tabs, active, onChange }) {
  return (
    <div className="flex gap-1 border-b border-gray-200 mb-6">
      {tabs.map(t => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={`px-4 py-2 text-sm font-medium transition border-b-2
            ${active === t.id
              ? "text-blue-600 border-blue-600"
              : "text-gray-500 border-transparent hover:text-gray-800"}`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

// ── AlertBox ──────────────────────────────────────────────────────────────────
export function AlertBox({ children, variant = "error" }) {
  const styles = {
    error:   "bg-red-50 border-red-200 text-red-600",
    warning: "bg-amber-50 border-amber-200 text-amber-800",
    info:    "bg-blue-50 border-blue-200 text-blue-700",
  };
  return (
    <div className={`border rounded-xl p-4 mb-6 text-sm ${styles[variant]}`}>
      {children}
    </div>
  );
}

// ── PageHeader ────────────────────────────────────────────────────────────────
export function PageHeader({ title, subtitle, actions }) {
  return (
    <div className="flex items-center justify-between mb-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-800">{title}</h1>
        {subtitle && <p className="text-sm text-gray-400 mt-1">{subtitle}</p>}
      </div>
      {actions && <div className="flex gap-2">{actions}</div>}
    </div>
  );
}

// ── Badges ────────────────────────────────────────────────────────────────────
export function SeverityBadge({ severity }) {
  const colors = {
    high:   "bg-red-100 text-red-700 border border-red-300",
    medium: "bg-yellow-100 text-yellow-700 border border-yellow-300",
    low:    "bg-blue-100 text-blue-700 border border-blue-300",
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${colors[severity] || colors.low}`}>
      {severity}
    </span>
  );
}

export function ConfidenceBadge({ confidence }) {
  const colors = {
    high:   "bg-green-100 text-green-700",
    medium: "bg-yellow-100 text-yellow-700",
    low:    "bg-red-100 text-red-700",
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors[confidence] || ""}`}>
      {confidence} confidence
    </span>
  );
}

export function SourceBadge({ source }) {
  const map = {
    ai:                     { label: "AI",            cls: "bg-purple-100 text-purple-700" },
    physics_rule:           { label: "Physics",       cls: "bg-cyan-100 text-cyan-700" },
    interpolation_fallback: { label: "Interpolation", cls: "bg-gray-100 text-gray-600" },
  };
  const { label, cls } = map[source] || { label: source, cls: "bg-gray-100 text-gray-600" };
  return <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${cls}`}>{label}</span>;
}

export function R2Badge({ value }) {
  if (value > 0.95) return <span className="text-green-600 font-bold">{value.toFixed(4)}</span>;
  if (value > 0.85) return <span className="text-blue-600 font-bold">{value.toFixed(4)}</span>;
  if (value > 0)    return <span className="text-orange-500 font-bold">{value.toFixed(4)}</span>;
  return <span className="text-red-500 font-bold">{value.toFixed(4)}</span>;
}

// ── MetricCard (before/after comparison) ─────────────────────────────────────
export function MetricCard({ label, original, corrected, higherIsBetter = false }) {
  const isImproved = higherIsBetter ? corrected > original : corrected < original;
  const pct = original !== 0 ? Math.abs((corrected - original) / original * 100) : 0;
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 space-y-2">
      <div className="text-xs text-gray-500 font-medium uppercase tracking-wide">{label}</div>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold text-gray-900">{corrected?.toFixed(3)}</span>
        <span className="text-sm text-gray-400 line-through">{original?.toFixed(3)}</span>
      </div>
      <div className={`flex items-center gap-1 text-sm font-semibold ${isImproved ? "text-green-600" : "text-red-500"}`}>
        <span>{isImproved ? "▼" : "▲"}</span>
        <span>{pct.toFixed(2)}% {isImproved ? "improvement" : "degraded"}</span>
      </div>
    </div>
  );
}

// ── Formatters ────────────────────────────────────────────────────────────────
export const fmtWm2 = (v) => v != null ? [`${Number(v).toFixed(2)} W/m²`] : ["—"];
// ── NavBar ───────────────────────────────────────────────────────────────────
import { NavLink } from "react-router-dom";

export function NavBar() {
  const items = [
    { to: "/", label: "Upload" },
    { to: "/dashboard", label: "Dashboard" },
    { to: "/compare", label: "Model Comparison" },
    { to: "/anomalies", label: "Anomaly" },
    { to: "/forecast", label: "Forecast" },
  ];

  return (
    <nav className="w-full border-b border-gray-100 mb-4">
      <div className="max-w-6xl mx-auto flex gap-2 px-2">
        {items.map((it) => (
          <NavLink
            key={it.to}
            to={it.to}
            className={({ isActive }) => `px-3 py-1 text-sm font-medium transition ${isActive ? "text-blue-600" : "text-gray-600 hover:text-gray-800"}`}
          >
            {it.label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
