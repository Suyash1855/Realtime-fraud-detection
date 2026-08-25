/*
  Transaction detail — why the model scored this one the way it did.

  The contribution chart is a DIVERGING encoding, which is the correct form here:
  each feature pushes the prediction toward fraud or away from it, and zero means
  "this feature made no difference". Blue and red are warm/cool opposites with a
  neutral gray axis at the midpoint — never a rainbow, never a hue at zero.

  Values come from XGBoost's exact tree SHAP (pred_contribs), summed in log-odds
  space, which is why base + contributions reconstructs the logit exactly.
*/

import { useEffect } from 'react'
import { api, fmt, useAsync } from '../api'
import { Badge, ErrorBox, Loading, RiskBadge, COLORS } from './ui'

export default function TransactionDrawer({ txId, onClose }) {
  const { data, loading, error } = useAsync(() => api.transaction(txId), [txId])

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
    }
  }, [onClose])

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Transaction ${txId} detail`}
      style={{
        position: 'fixed', inset: 0, zIndex: 50,
        background: 'rgba(0,0,0,0.62)',
        display: 'flex', justifyContent: 'flex-end',
      }}
      onClick={onClose}
    >
      <aside
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(560px, 100%)', height: '100%', overflowY: 'auto',
          background: 'var(--surface-1)', borderLeft: '1px solid var(--border)',
          padding: '18px 20px 40px',
        }}
      >
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
          <div>
            <div className="stat-label">Transaction</div>
            <div className="mono" style={{ fontSize: 19 }}>{txId}</div>
          </div>
          <button className="btn" onClick={onClose} aria-label="Close">✕</button>
        </header>

        <hr className="divider" />

        {loading && <Loading label="Computing SHAP contributions…" />}
        {error && <ErrorBox message={error} />}
        {data && <Detail d={data} />}
      </aside>
    </div>
  )
}

function Detail({ d }) {
  const exp = d.explanation
  const verdictCorrect = d.true_label === null ? null : (d.is_flagged === (d.true_label === 1))

  return (
    <>
      <div className="grid grid-2" style={{ gap: 10 }}>
        <div className="stat">
          <div className="stat-label">Fraud score</div>
          <div className="stat-value sm" style={{ color: d.is_flagged ? COLORS.critical : COLORS.good }}>
            {fmt.dec(d.score, 4)}
          </div>
          <div className="stat-sub"><RiskBadge level={d.risk_level} /></div>
        </div>
        <div className="stat">
          <div className="stat-label">Amount</div>
          <div className="stat-value sm">{fmt.moneyPrecise(d.amount)}</div>
          <div className="stat-sub">
            {d.true_label === null ? (
              <span className="muted">no ground truth</span>
            ) : verdictCorrect ? (
              <Badge kind="tp">✓ model was right</Badge>
            ) : (
              <Badge kind="fp">✗ model was wrong</Badge>
            )}
          </div>
        </div>
      </div>

      <h3 className="card-title" style={{ margin: '22px 0 4px' }}>
        Why this score
      </h3>
      <p className="small muted" style={{ marginTop: 0 }}>
        Top {exp.top_features.length} of {exp.total_features} features by absolute
        contribution, in log-odds. Bars right of the axis pushed toward fraud.
      </p>

      <Waterfall features={exp.top_features} />

      <div className="legend" style={{ marginTop: 12 }}>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: COLORS.s8 }} />
          pushes toward fraud
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: COLORS.s1 }} />
          pushes toward legitimate
        </span>
      </div>

      <dl className="kv" style={{ marginTop: 18 }}>
        <dt>Base (log-odds)</dt><dd>{fmt.dec(exp.base_value, 4)}</dd>
        <dt>Final (log-odds)</dt><dd>{fmt.dec(exp.logit, 4)}</dd>
        <dt>Probability</dt><dd>{fmt.dec(exp.probability, 4)}</dd>
      </dl>

      <h3 className="card-title" style={{ margin: '22px 0 8px' }}>Attributes</h3>
      <div className="table-wrap">
        <table>
          <tbody>
            {Object.entries(d.attributes).map(([k, v]) => (
              <tr key={k}>
                <td className="muted">{k}</td>
                <td className="mono">{v ?? <span className="muted">missing</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Table view: the same contribution values, readable without the chart. */}
      <h3 className="card-title" style={{ margin: '22px 0 8px' }}>Contributions (table view)</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              <th className="num">Value</th>
              <th className="num">Contribution</th>
            </tr>
          </thead>
          <tbody>
            {exp.top_features.map((f) => (
              <tr key={f.feature}>
                <td className="mono">{f.feature}</td>
                <td className="num mono muted">
                  {f.value === null ? 'missing' : fmt.dec(f.value, 2)}
                </td>
                <td className="num mono" style={{ color: f.contribution > 0 ? COLORS.s8 : COLORS.s1 }}>
                  {f.contribution > 0 ? '+' : ''}{fmt.dec(f.contribution, 4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

/**
 * Diverging bar chart, hand-drawn in SVG rather than pulled from a chart library:
 * the zero axis must sit at a fixed x for every row, and each bar's 4px rounded
 * end must be on the outer edge only. That is fiddly to coax out of a generic
 * bar component and trivial to state directly.
 */
function Waterfall({ features }) {
  const max = Math.max(...features.map((f) => Math.abs(f.contribution)), 1e-6)
  const ROW = 26
  const AXIS = 0.5          // zero line at the horizontal midpoint
  const height = features.length * ROW + 8

  return (
    <figure style={{ margin: '14px 0 0' }}>
      <svg width="100%" height={height} role="img"
           aria-label="Feature contributions to this prediction">
        {/* Neutral midpoint rule — the diverging axis */}
        <line
          x1={`${AXIS * 100}%`} x2={`${AXIS * 100}%`} y1={0} y2={height}
          stroke={COLORS.axis} strokeWidth={1}
        />
        {features.map((f, i) => {
          const y = i * ROW + 4
          const frac = (Math.abs(f.contribution) / max) * (AXIS * 0.92)
          const positive = f.contribution > 0
          const color = positive ? COLORS.s8 : COLORS.s1
          const x = positive ? AXIS : AXIS - frac
          return (
            <g key={f.feature}>
              <rect
                x={`${x * 100}%`} y={y} width={`${frac * 100}%`} height={12}
                fill={color} rx={3}
              />
              <text
                x={positive ? `${AXIS * 100 - 1}%` : `${AXIS * 100 + 1}%`}
                y={y + 10}
                textAnchor={positive ? 'end' : 'start'}
                fill={COLORS.secondary} fontSize={11}
                fontFamily="ui-monospace, monospace"
              >
                {f.feature}
              </text>
              <text
                x={positive ? `${(x + frac) * 100 + 1}%` : `${x * 100 - 1}%`}
                y={y + 10}
                textAnchor={positive ? 'start' : 'end'}
                fill={COLORS.muted} fontSize={10}
                fontFamily="ui-monospace, monospace"
              >
                {positive ? '+' : ''}{f.contribution.toFixed(3)}
              </text>
            </g>
          )
        })}
      </svg>
      <figcaption className="small muted" style={{ marginTop: 6 }}>
        Bar length is the magnitude of each feature's contribution to the log-odds.
      </figcaption>
    </figure>
  )
}
