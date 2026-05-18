// Shared UI primitives used across all pages.

import { NavLink } from "react-router-dom";

// StatCard
export function StatCard({ label, value, unit, color = "text-gray-800", sub }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">{label}</p>
      <p className={`mt-2 text-2xl font-semibold ${color}`}>
        {value}
        {unit && <span className="ml-1 text-sm font-normal text-slate-400">{unit}</span>}
      </p>
      {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
    </div>
  );
}

// Card
export function Card({ title, subtitle, children, right }) {
  return (
    <div className="mb-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-3 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
          {subtitle && <p className="text-sm text-slate-500">{subtitle}</p>}
        </div>
        {right}
      </div>
      <div>{children}</div>
    </div>
  );
}

// Section (border variant used in AnomalyReport / Forecast)
export function Section({ title, subtitle, children, className = "" }) {
  return (
    <div className={`mb-6 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm ${className}`}>
      <div className="border-b border-slate-100 px-5 py-4">
        <h3 className="text-base font-semibold text-slate-900">{title}</h3>
        {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}

// TabNav
export function TabNav({ tabs, active, onChange }) {
  return (
    <div className="mb-6 flex gap-1 border-b border-slate-200">
      {tabs.map(t => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={`border-b-2 px-4 py-2 text-sm font-medium transition
            ${active === t.id
              ? "border-slate-900 text-slate-900"
              : "border-transparent text-slate-500 hover:text-slate-900"}`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

// AlertBox
export function AlertBox({ children, variant = "error" }) {
  const styles = {
    error:   "border-red-200 bg-red-50 text-red-700",
    warning: "border-amber-200 bg-amber-50 text-amber-900",
    info:    "border-slate-200 bg-slate-50 text-slate-700",
  };
  return (
    <div className={`mb-6 rounded-2xl border p-4 text-sm shadow-sm ${styles[variant]}`}>
      {children}
    </div>
  );
}

// PageHeader
export function PageHeader({ title, subtitle, actions }) {
  return (
    <div className="mb-6 rounded-2xl border border-slate-200 bg-white/90 px-5 py-4 shadow-sm backdrop-blur">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Machine Learning Approach to Solar Irradiance Prediction</p>
          <h1 className="mt-1 text-2xl font-semibold text-slate-900">{title}</h1>
          {subtitle && <p className="mt-1 text-sm text-slate-500">{subtitle}</p>}
        </div>
        {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
      </div>
    </div>
  );
}

// Badges
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

// MetricCard (before/after comparison)
export function MetricCard({ label, original, corrected, higherIsBetter = false }) {
  const isImproved = higherIsBetter ? corrected > original : corrected < original;
  const pct = original !== 0 ? Math.abs((corrected - original) / original * 100) : 0;
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm space-y-2">
      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">{label}</div>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-semibold text-slate-900">{corrected?.toFixed(3)}</span>
        <span className="text-sm text-slate-400 line-through">{original?.toFixed(3)}</span>
      </div>
      <div className={`flex items-center gap-1 text-sm font-semibold ${isImproved ? "text-emerald-600" : "text-rose-500"}`}>
        <span>{isImproved ? "▼" : "▲"}</span>
        <span>{pct.toFixed(2)}% {isImproved ? "improvement" : "degraded"}</span>
      </div>
    </div>
  );
}

export const fmtWm2 = (v) => v != null ? [`${Number(v).toFixed(2)} W/m²`] : ["—"];

// NavBar
export function NavBar() {
  const items = [
    { to: "/", label: "Upload" },
    { to: "/dashboard", label: "Dashboard" },
    { to: "/compare", label: "Model Comparison" },
    { to: "/anomalies", label: "Anomaly" },
    { to: "/forecast", label: "Forecast" },
    { to: "/history", label: "View History" },
  ];

  return (
    <nav className="sticky top-0 z-50 border-b border-slate-200/80 bg-white/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <NavLink to="/" className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-900 text-sm font-bold text-white shadow-sm">
            SF
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold text-slate-900">Machine Learning Approach to Solar Irradiance Prediction</div>
            <div className="text-xs text-slate-500">Upload to forecast workflow</div>
          </div>
        </NavLink>

        <div className="hidden items-center gap-1 rounded-full bg-slate-100 p-1 md:flex">
          {items.map((it) => (
            <NavLink
              key={it.to}
              to={it.to}
              className={({ isActive }) => `rounded-full px-4 py-2 text-sm font-medium transition ${isActive ? "bg-slate-900 text-white shadow-sm" : "text-slate-600 hover:bg-white hover:text-slate-900"}`}
            >
              {it.label}
            </NavLink>
          ))}
        </div>

        <div className="md:hidden">
          <select
            defaultValue=""
            onChange={(e) => e.target.value && (window.location.href = e.target.value)}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm"
            aria-label="Navigate to page"
          >
            <option value="" disabled>
              Navigate
            </option>
            {items.map((it) => (
              <option key={it.to} value={it.to}>
                {it.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </nav>
  );
}
