/*
  Shared UI primitives
  ====================
  Chart-adjacent pieces follow the data-viz rules deliberately:
   - values and labels wear text tokens, never the series color; a small swatch
     beside them carries identity instead
   - a legend is present whenever two or more series share a plot
   - tooltips enhance but never gate: every charted value is also reachable in a
     table view
*/

import { fmt } from '../api'

export const COLORS = {
  s1: '#3987e5',   // blue     — slot 1
  s2: '#d95926',   // orange   — slot 2
  s3: '#199e70',   // aqua     — slot 3
  s4: '#c98500',   // yellow   — slot 4
  s8: '#e66767',   // red      — slot 8
  good: '#0ca30c',
  warning: '#fab219',
  critical: '#d03b3b',
  grid: '#2c2c2a',
  axis: '#383835',
  muted: '#898781',
  secondary: '#c3c2b7',
  surface: '#1a1a19',
}

export function Card({ title, note, right, children, style, className = '' }) {
  return (
    <section className={`card ${className}`} style={style}>
      {(title || right) && (
        <header className="card-head">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {note && <div className="card-note">{note}</div>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  )
}

export function Stat({ label, value, sub, tone, small }) {
  const toneClass = tone ? ` stat-${tone}` : ''
  return (
    <div className="stat">
      <div className="stat-label" title={label}>{label}</div>
      <div className={`stat-value${small ? ' sm' : ''}${toneClass}`}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}

export function Badge({ kind, children }) {
  return <span className={`badge badge-${kind}`}>{children}</span>
}

export function RiskBadge({ level }) {
  if (level === 'HIGH') return <Badge kind="high">HIGH</Badge>
  if (level === 'MEDIUM') return <Badge kind="med">MED</Badge>
  return <Badge kind="low">LOW</Badge>
}

/** Identity legend. Required whenever a plot carries two or more series. */
export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span className="legend-item" key={it.label}>
          <span className="legend-swatch" style={{ background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  )
}

/**
 * Recharts tooltip. Labels stay in ink tokens; only the swatch carries the
 * series color, so text contrast never depends on the palette.
 */
export function ChartTooltip({ active, payload, label, labelKey, format }) {
  if (!active || !payload?.length) return null
  return (
    <div className="tooltip-card">
      {label !== undefined && (
        <div style={{ color: COLORS.secondary, marginBottom: 5 }}>
          {labelKey ? `${labelKey} ${label}` : label}
        </div>
      )}
      {payload.map((p) => (
        <div className="t-row" key={p.dataKey ?? p.name}>
          <span className="t-key">
            <span
              className="legend-swatch"
              style={{ background: p.color ?? p.stroke ?? p.fill, marginRight: 6 }}
            />
            {p.name}
          </span>
          <span className="t-val">
            {format ? format(p.value, p.dataKey) : fmt.dec(p.value, 3)}
          </span>
        </div>
      ))}
    </div>
  )
}

export function Empty({ icon = '○', title, hint }) {
  return (
    <div className="feed-empty">
      <div style={{ fontSize: 24, opacity: 0.5 }}>{icon}</div>
      <div>{title}</div>
      {hint && <div className="small muted">{hint}</div>}
    </div>
  )
}

export function Loading({ label = 'Loading…' }) {
  return <div className="feed-empty"><span className="muted">{label}</span></div>
}

export function ErrorBox({ message }) {
  return (
    <div className="banner warn">
      <span aria-hidden="true">⚠</span>
      <div>{message}</div>
    </div>
  )
}

/** Shared axis styling so every chart's chrome recedes identically. */
export const axisProps = {
  stroke: COLORS.axis,
  tick: { fill: COLORS.muted, fontSize: 11 },
  tickLine: false,
}

export const gridProps = {
  stroke: COLORS.grid,
  strokeDasharray: '0',   // solid hairlines; dashes read as "threshold", not "grid"
  vertical: false,
}
