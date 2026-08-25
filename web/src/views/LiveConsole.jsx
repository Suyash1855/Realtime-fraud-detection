/*
  Live console — the streaming scoring screen.

  Layout intent: the numbers that change fastest sit highest, the alert queue
  fills the main column, and the threshold control sits directly beneath the
  metrics it moves so cause and effect are visible in one glance.

  Throughput and latency get SEPARATE sparklines on purpose. They have unrelated
  scales, and putting them on one plot with two y-axes would invent a correlation
  that isn't in the data.
*/

import { useEffect, useState } from 'react'
import {
  Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, fmt, useDebounced } from '../api'
import {
  Card, ChartTooltip, Empty, Legend, RiskBadge, Stat, COLORS, axisProps,
} from '../components/ui'

const RATES = [50, 100, 250, 500, 1000]

export default function LiveConsole({ info, sim, onSelectTransaction }) {
  const defaultThreshold = info.model.fraud_threshold
  const [threshold, setThreshold] = useState(defaultThreshold)
  const [rate, setRate] = useState(250)
  const [loop, setLoop] = useState(false)

  const debouncedThreshold = useDebounced(threshold, 110)
  const [projected, setProjected] = useState(null)

  const { status, stats, feed, spark, progress, error, start, stop, retune } = sim
  const running = status === 'running'

  // Projected impact across the whole demo set at the current threshold. This is
  // pure arithmetic server-side over precomputed scores, so it stays responsive
  // while dragging.
  useEffect(() => {
    let alive = true
    api.thresholdMetrics(debouncedThreshold)
      .then((d) => alive && setProjected(d))
      .catch(() => {})
    return () => { alive = false }
  }, [debouncedThreshold])

  // Retune a running session so the slider affects live scoring too.
  useEffect(() => { retune(debouncedThreshold) }, [debouncedThreshold, retune])

  return (
    <div className="grid" style={{ gap: 14 }}>

      {/* ── Control bar ──────────────────────────────────────────────────── */}
      <Card>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 18, alignItems: 'center' }}>
          <div className="status-line" style={{ minWidth: 190 }}>
            <span className={`live-dot${running ? '' : ' idle'}`} />
            <strong style={{ color: 'var(--text-primary)' }}>
              {running ? 'LIVE' : status === 'complete' ? 'COMPLETE' : status === 'error' ? 'ERROR' : 'IDLE'}
            </strong>
            <span className="muted">
              {fmt.int(stats.processed)} scored · {fmt.int(stats.flagged)} flagged
            </span>
          </div>

          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span className="small muted">Rate</span>
            <div className="seg" role="group" aria-label="Transactions per second">
              {RATES.map((r) => (
                <button
                  key={r}
                  aria-pressed={rate === r}
                  onClick={() => setRate(r)}
                  disabled={running}
                >
                  {r}/s
                </button>
              ))}
            </div>
          </div>

          <label className="small muted" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input
              type="checkbox"
              checked={loop}
              onChange={(e) => setLoop(e.target.checked)}
              disabled={running}
            />
            Loop
          </label>

          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
            {running ? (
              <button className="btn btn-danger" onClick={stop}>■ Stop</button>
            ) : (
              <button
                className="btn btn-primary"
                onClick={() => start({ rate, threshold, loop, shuffle: true })}
              >
                ▶ {status === 'idle' ? 'Start simulation' : 'Restart'}
              </button>
            )}
          </div>
        </div>

        {running && (
          <div style={{ marginTop: 12, height: 3, background: COLORS.grid, borderRadius: 2 }}>
            <div
              style={{
                width: `${Math.min(progress * 100, 100)}%`, height: '100%',
                background: COLORS.s1, borderRadius: 2, transition: 'width .25s linear',
              }}
            />
          </div>
        )}

        {error && (
          <div className="banner warn" style={{ marginTop: 12 }}>
            <span aria-hidden="true">⚠</span><div>{error}</div>
          </div>
        )}
      </Card>

      {/* ── Live metrics ─────────────────────────────────────────────────── */}
      <div className="grid grid-4">
        <Stat
          label="Throughput"
          value={running || stats.processed ? `${fmt.int(Math.round(stats.throughput))}/s` : '—'}
          sub={`${fmt.int(stats.processed)} processed`}
        />
        <Stat
          label="Latency p95"
          value={stats.processed ? fmt.ms(stats.latency_p95_ms) : '—'}
          sub={stats.processed ? `p50 ${fmt.ms(stats.latency_p50_ms)} · p99 ${fmt.ms(stats.latency_p99_ms)}` : 'per transaction'}
          tone="accent"
        />
        <Stat
          label="Fraud caught"
          value={fmt.money(stats.fraud_amount_caught)}
          sub={`${fmt.int(stats.true_positives)} true positives`}
          tone="good"
        />
        <Stat
          label="Fraud missed"
          value={fmt.money(stats.fraud_amount_missed)}
          sub={`${fmt.int(stats.false_negatives)} false negatives`}
          tone="danger"
        />
      </div>

      <div className="grid grid-4">
        <Stat label="Live precision" value={stats.flagged ? fmt.dec(stats.precision, 3) : '—'}
              sub={`${fmt.int(stats.true_positives)} of ${fmt.int(stats.flagged)} alerts correct`} small />
        <Stat label="Live recall" value={stats.processed ? fmt.dec(stats.recall, 3) : '—'}
              sub="of fraud seen so far" small />
        <Stat label="Alert rate" value={stats.processed ? fmt.pct(stats.flag_rate, 2) : '—'}
              sub={`${fmt.int(stats.high_risk)} high risk`} small />
        <Stat label="False positives" value={fmt.int(stats.false_positives)}
              sub="good customers blocked" small tone={stats.false_positives > 0 ? 'warn' : undefined} />
      </div>

      {/* ── Threshold control ────────────────────────────────────────────── */}
      <Card
        title="Decision threshold"
        note="Drag to retune. Affects live scoring and projects across the full demo set."
        right={<span className="mono" style={{ fontSize: 19 }}>{fmt.dec(threshold, 3)}</span>}
      >
        <input
          type="range" min={0.05} max={0.99} step={0.005}
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          aria-label="Fraud decision threshold"
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 5 }}>
          <span className="small muted">0.05 — catch more, alert more</span>
          <button
            className="btn"
            style={{ padding: '3px 10px', fontSize: 11 }}
            onClick={() => setThreshold(defaultThreshold)}
          >
            reset to {fmt.dec(defaultThreshold, 3)}
          </button>
          <span className="small muted">0.99 — alert less, miss more</span>
        </div>

        {projected && (
          <>
            <hr className="divider" />
            <div className="card-note" style={{ marginBottom: 10 }}>
              Projected over all {fmt.int(info.dataset.n_transactions)} demo transactions
            </div>
            <div className="grid grid-4" style={{ gap: 10 }}>
              <Stat label="Precision" value={fmt.dec(projected.precision, 3)}
                    sub={`${fmt.int(projected.true_positives)} true positives`} small />
              <Stat label="Recall" value={fmt.dec(projected.recall, 3)}
                    sub={`${fmt.int(projected.false_negatives)} fraud missed`} small />
              <Stat label="Alerts raised" value={fmt.int(projected.alerts)}
                    sub={`${fmt.int(projected.false_positives)} false alarms`} small
                    tone={projected.false_positives > 500 ? 'warn' : undefined} />
              <Stat label="Fraud $ captured" value={fmt.money(projected.fraud_amount_caught)}
                    sub={`${fmt.pct(projected.capture_rate, 1)} of ${fmt.money(projected.fraud_amount_total)}`}
                    small tone="good" />
            </div>
          </>
        )}
      </Card>

      {/* ── Feed + sparklines ────────────────────────────────────────────── */}
      <div
        className="grid"
        style={{
          gridTemplateColumns: 'minmax(0, 2.1fr) minmax(0, 1fr)',
          gap: 14,
          alignItems: 'start',   // an empty feed card must not stretch to the sidebar's height
        }}
      >
        <Card
          title="Alert queue"
          note="Flagged transactions only · newest first · click a row to see why"
          right={<span className="small muted">{fmt.int(feed.length)} shown</span>}
        >
          {feed.length === 0 ? (
            <Empty
              icon={running ? '⋯' : '○'}
              title={running ? 'Watching for flagged transactions…' : 'No alerts yet'}
              hint={running ? 'Only transactions above the threshold appear here' : 'Start the simulation to populate the queue'}
            />
          ) : (
            <div className="feed table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Transaction</th>
                    <th className="num">Score</th>
                    <th>Risk</th>
                    <th className="num">Amount</th>
                    <th>Product</th>
                    <th>Email domain</th>
                    <th>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {feed.map((f, i) => (
                    <tr
                      key={`${f.transaction_id}-${i}`}
                      className={`clickable ${f.risk_level === 'HIGH' ? 'risk-high' : 'risk-med'}`}
                      onClick={() => onSelectTransaction(f.transaction_id)}
                    >
                      <td className="mono">{f.transaction_id}</td>
                      <td className="num mono">{fmt.dec(f.score, 3)}</td>
                      <td><RiskBadge level={f.risk_level} /></td>
                      <td className="num mono">{fmt.moneyPrecise(f.amount)}</td>
                      <td className="muted">{f.product ?? '—'}</td>
                      <td className="muted">{f.email_domain ?? '—'}</td>
                      <td>
                        {f.correct === null ? <span className="muted">—</span>
                          : f.correct
                            ? <span className="badge badge-tp">✓ fraud</span>
                            : <span className="badge badge-fp">✗ legit</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <div className="grid" style={{ gap: 14, alignContent: 'start' }}>
          <Sparkline
            title="Throughput"
            note="transactions per second"
            data={spark}
            dataKey="throughput"
            color={COLORS.s1}
            format={(v) => `${Math.round(v)}/s`}
          />
          <Sparkline
            title="Latency p95"
            note="milliseconds per transaction"
            data={spark}
            dataKey="latency"
            color={COLORS.s2}
            format={(v) => `${v.toFixed(2)}ms`}
          />
          <Card title="Session" note="current run">
            <dl className="kv">
              <dt>Elapsed</dt><dd>{stats.elapsed_seconds ?? 0}s</dd>
              <dt>Volume</dt><dd>{fmt.money(stats.amount_total)}</dd>
              <dt>Flagged $</dt><dd>{fmt.money(stats.amount_flagged)}</dd>
              <dt>Model</dt><dd>v{info.model.model_version}</dd>
            </dl>
          </Card>
        </div>
      </div>
    </div>
  )
}

/**
 * Single-series area sparkline. One series means no legend box is needed — the
 * card title names it — but the value format is shared with the tooltip so a
 * reader never has to guess units.
 */
function Sparkline({ title, note, data, dataKey, color, format }) {
  const hasData = data.length > 1
  return (
    <Card title={title} note={note}>
      <div style={{ height: 92 }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id={`fill-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.28} />
                  <stop offset="100%" stopColor={color} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <XAxis dataKey="t" hide />
              <YAxis hide domain={['dataMin', 'dataMax']} />
              <Tooltip
                content={<ChartTooltip format={(v) => format(v)} />}
                cursor={{ stroke: COLORS.axis, strokeWidth: 1 }}
              />
              <Area
                type="monotone"
                dataKey={dataKey}
                name={title}
                stroke={color}
                strokeWidth={2}
                fill={`url(#fill-${dataKey})`}
                dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="feed-empty" style={{ height: '100%' }}>
            <span className="small muted">waiting for data</span>
          </div>
        )}
      </div>
      {hasData && (
        <div className="mono" style={{ fontSize: 17, marginTop: 2 }}>
          {format(data[data.length - 1][dataKey])}
        </div>
      )}
    </Card>
  )
}
