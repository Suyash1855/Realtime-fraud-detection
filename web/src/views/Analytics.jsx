/*
  Analytics — the model's behaviour across every operating point.

  Chart choices worth stating:
   - Score distribution is normalised WITHIN each class. Raw counts would be
     useless here: legitimate transactions outnumber fraud ~29:1, so the fraud
     hump would be invisible. "% of that class" is the honest comparison, and the
     axis says so.
   - The cost sweep plots three dollar series on ONE axis. Never two y-scales.
   - Every chart has a table-view twin, so no value is reachable only by hovering.
*/

import { useState } from 'react'
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, fmt, useAsync, useDebounced } from '../api'
import {
  Card, ChartTooltip, ErrorBox, Legend, Loading, Stat, COLORS, axisProps, gridProps,
} from '../components/ui'

export default function Analytics({ info }) {
  const modelThreshold = info.model.fraud_threshold
  const [threshold, setThreshold] = useState(modelThreshold)
  const [fpCost, setFpCost] = useState(5)
  const [fnCost, setFnCost] = useState(100)
  const [showTables, setShowTables] = useState(false)

  const dThreshold = useDebounced(threshold, 110)
  const dFp = useDebounced(fpCost, 300)
  const dFn = useDebounced(fnCost, 300)

  const curves = useAsync(() => api.curves(140), [])
  const dist = useAsync(() => api.distribution(30), [])
  const metrics = useAsync(() => api.thresholdMetrics(dThreshold), [dThreshold])
  const cost = useAsync(() => api.cost(dFp, dFn), [dFp, dFn])

  return (
    <div className="grid" style={{ gap: 14 }}>

      <div className="banner">
        <span aria-hidden="true">ℹ</span>
        <div>
          Every figure below is computed on {fmt.int(info.dataset.n_transactions)} held-out
          transactions from the chronological test period — data the model never trained on.
          <button
            className="btn"
            style={{ marginLeft: 12, padding: '2px 10px', fontSize: 11 }}
            onClick={() => setShowTables((s) => !s)}
          >
            {showTables ? 'Hide' : 'Show'} table views
          </button>
        </div>
      </div>

      {/* ── Operating point ──────────────────────────────────────────────── */}
      <Card
        title="Operating point"
        note="Move the threshold to see the whole tradeoff move with it"
        right={<span className="mono" style={{ fontSize: 19 }}>{fmt.dec(threshold, 3)}</span>}
      >
        <input
          type="range" min={0.05} max={0.99} step={0.005}
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          aria-label="Decision threshold"
        />
        {metrics.data && <ConfusionAndImpact m={metrics.data} />}
      </Card>

      {/* ── PR curve + distribution ──────────────────────────────────────── */}
      <div className="grid grid-2">
        <Card
          title="Precision & recall vs threshold"
          note="The tradeoff, in full. The dashed rule marks the deployed threshold."
        >
          {curves.loading && <Loading />}
          {curves.error && <ErrorBox message={curves.error} />}
          {curves.data && (
            <>
              <div style={{ height: 268 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={curves.data.pr} margin={{ top: 22, right: 14, bottom: 20, left: -12 }}>
                    <CartesianGrid {...gridProps} />
                    <XAxis
                      dataKey="threshold" type="number" domain={[0, 1]}
                      tickFormatter={(v) => v.toFixed(1)} {...axisProps}
                      label={{ value: 'threshold', position: 'insideBottom', offset: -12,
                               fill: COLORS.muted, fontSize: 11 }}
                    />
                    <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} {...axisProps} />
                    <Tooltip
                      content={<ChartTooltip labelKey="threshold" format={(v) => fmt.dec(v, 3)} />}
                      cursor={{ stroke: COLORS.axis }}
                    />
                    <ReferenceLine
                      x={modelThreshold} stroke={COLORS.muted}
                      strokeDasharray="4 4"
                      label={{ value: 'deployed', fill: COLORS.muted, fontSize: 10, position: 'top' }}
                    />
                    <ReferenceLine x={threshold} stroke={COLORS.s4} strokeWidth={1.5} />
                    <Line type="monotone" dataKey="precision" name="Precision"
                          stroke={COLORS.s1} strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="recall" name="Recall"
                          stroke={COLORS.s2} strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <Legend items={[
                { label: 'Precision', color: COLORS.s1 },
                { label: 'Recall', color: COLORS.s2 },
              ]} />
            </>
          )}
        </Card>

        <Card
          title="Score separation"
          note="Normalised within each class — legitimate outnumbers fraud ~29:1"
        >
          {dist.loading && <Loading />}
          {dist.error && <ErrorBox message={dist.error} />}
          {dist.data && (
            <>
              <div style={{ height: 268 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={toDensity(dist.data)} margin={{ top: 22, right: 14, bottom: 20, left: -12 }}>
                    <CartesianGrid {...gridProps} />
                    <XAxis
                      dataKey="score" type="number" domain={[0, 1]}
                      tickFormatter={(v) => v.toFixed(1)} {...axisProps}
                      label={{ value: 'fraud score', position: 'insideBottom', offset: -12,
                               fill: COLORS.muted, fontSize: 11 }}
                    />
                    <YAxis tickFormatter={(v) => `${v.toFixed(0)}%`} {...axisProps} />
                    <Tooltip
                      content={<ChartTooltip labelKey="score" format={(v) => `${v.toFixed(1)}%`} />}
                      cursor={{ stroke: COLORS.axis }}
                    />
                    <ReferenceLine x={threshold} stroke={COLORS.s4} strokeWidth={1.5}
                                   label={{ value: 'threshold', fill: COLORS.s4, fontSize: 10, position: 'top' }} />
                    <Area type="monotone" dataKey="legitimate" name="Legitimate"
                          stroke={COLORS.s1} strokeWidth={2} fill={COLORS.s1} fillOpacity={0.14}
                          isAnimationActive={false} />
                    <Area type="monotone" dataKey="fraud" name="Fraud"
                          stroke={COLORS.s8} strokeWidth={2} fill={COLORS.s8} fillOpacity={0.14}
                          isAnimationActive={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <Legend items={[
                { label: 'Legitimate', color: COLORS.s1 },
                { label: 'Fraud', color: COLORS.s8 },
              ]} />
            </>
          )}
        </Card>
      </div>

      {/* ── Cost model ───────────────────────────────────────────────────── */}
      <Card
        title="Cost model"
        note="What is a false alarm worth against a missed fraud? Set the ratio and the optimal threshold follows."
      >
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 18, marginBottom: 16 }}>
          <div style={{ width: 190 }}>
            <label className="field" htmlFor="fp-cost">
              Cost of blocking a good customer
            </label>
            <input id="fp-cost" type="number" min={0} step={1} value={fpCost}
                   onChange={(e) => setFpCost(Math.max(0, parseFloat(e.target.value) || 0))} />
          </div>
          <div style={{ width: 190 }}>
            <label className="field" htmlFor="fn-cost">
              Cost of missing a fraud
            </label>
            <input id="fn-cost" type="number" min={0} step={5} value={fnCost}
                   onChange={(e) => setFnCost(Math.max(0, parseFloat(e.target.value) || 0))} />
          </div>
          {cost.data && (
            <div style={{ display: 'flex', gap: 14, alignItems: 'flex-end' }}>
              <Stat label="Cost-optimal threshold" value={fmt.dec(cost.data.optimal_threshold, 3)}
                    sub="minimises total cost" small tone="accent" />
              <Stat label="Total cost there" value={fmt.money(cost.data.optimal_cost)}
                    sub={`at ${fpCost}:${fnCost} ratio`} small />
            </div>
          )}
        </div>

        {cost.loading && <Loading />}
        {cost.error && <ErrorBox message={cost.error} />}
        {cost.data && (
          <>
            <div style={{ height: 262 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={cost.data.points} margin={{ top: 22, right: 14, bottom: 20, left: 4 }}>
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="threshold" type="number" domain={[0, 1]}
                         tickFormatter={(v) => v.toFixed(1)} {...axisProps}
                         label={{ value: 'threshold', position: 'insideBottom', offset: -12,
                                  fill: COLORS.muted, fontSize: 11 }} />
                  <YAxis tickFormatter={(v) => fmt.compact(v)} {...axisProps} />
                  <Tooltip
                    content={<ChartTooltip labelKey="threshold" format={(v) => fmt.money(v)} />}
                    cursor={{ stroke: COLORS.axis }}
                  />
                  <ReferenceLine x={cost.data.optimal_threshold} stroke={COLORS.s4}
                                 strokeWidth={1.5}
                                 label={{ value: 'optimum', fill: COLORS.s4, fontSize: 10, position: 'top' }} />
                  <Line type="monotone" dataKey="total_cost" name="Total cost"
                        stroke={COLORS.s1} strokeWidth={2.5} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="fp_cost" name="False alarms"
                        stroke={COLORS.s2} strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="fn_cost" name="Missed fraud"
                        stroke={COLORS.s3} strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <Legend items={[
              { label: 'Total cost', color: COLORS.s1 },
              { label: 'Cost of false alarms', color: COLORS.s2 },
              { label: 'Cost of missed fraud', color: COLORS.s3 },
            ]} />
            <p className="small muted" style={{ marginBottom: 0, marginTop: 12 }}>
              This is the argument for not leaving the threshold at 0.5. The optimum moves
              with the business, not with the model — raise the cost of a missed fraud and
              the threshold drops, accepting more false alarms to catch more fraud.
            </p>
          </>
        )}
      </Card>

      {/* ── Table views ──────────────────────────────────────────────────── */}
      {showTables && curves.data && (
        <Card title="Table view" note="Every charted value, readable without hovering">
          <div className="table-wrap" style={{ maxHeight: 400, overflowY: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th className="num">Threshold</th>
                  <th className="num">Precision</th>
                  <th className="num">Recall</th>
                  <th className="num">F1</th>
                  <th className="num">Alerts</th>
                  <th className="num">False positives</th>
                </tr>
              </thead>
              <tbody>
                {curves.data.pr.filter((_, i) => i % 3 === 0).map((p) => (
                  <tr key={p.threshold}>
                    <td className="num mono">{fmt.dec(p.threshold, 3)}</td>
                    <td className="num mono">{fmt.dec(p.precision, 3)}</td>
                    <td className="num mono">{fmt.dec(p.recall, 3)}</td>
                    <td className="num mono">{fmt.dec(p.f1, 3)}</td>
                    <td className="num mono">{fmt.int(p.alerts)}</td>
                    <td className="num mono">{fmt.int(p.false_positives)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}

/** Confusion matrix as a 2×2 grid — a table, not a chart. Plus dollar impact. */
function ConfusionAndImpact({ m }) {
  const cells = [
    { label: 'True positives', value: m.true_positives, hint: 'fraud caught', tone: 'good' },
    { label: 'False positives', value: m.false_positives, hint: 'good customers blocked', tone: 'warn' },
    { label: 'False negatives', value: m.false_negatives, hint: 'fraud missed', tone: 'danger' },
    { label: 'True negatives', value: m.true_negatives, hint: 'correctly allowed', tone: undefined },
  ]
  return (
    <>
      <hr className="divider" />
      <div className="grid grid-4" style={{ gap: 10 }}>
        {cells.map((c) => (
          <Stat key={c.label} label={c.label} value={fmt.int(c.value)} sub={c.hint} tone={c.tone} small />
        ))}
      </div>
      <div className="grid grid-4" style={{ gap: 10, marginTop: 10 }}>
        <Stat label="Precision" value={fmt.dec(m.precision, 3)} sub="of alerts, correct" small />
        <Stat label="Recall" value={fmt.dec(m.recall, 3)} sub="of fraud, caught" small />
        <Stat label="F1" value={fmt.dec(m.f1, 3)} sub="harmonic mean" small />
        <Stat label="Fraud $ captured" value={fmt.money(m.fraud_amount_caught)}
              sub={`${fmt.pct(m.capture_rate, 1)} of ${fmt.money(m.fraud_amount_total)}`} small tone="good" />
      </div>
    </>
  )
}

/**
 * Counts -> percentage within class. With a 29:1 class imbalance the raw fraud
 * histogram is invisible next to the legitimate one; normalising per class is
 * what makes separation readable.
 */
function toDensity(d) {
  const fraudTotal = d.fraud.reduce((a, b) => a + b, 0) || 1
  const legitTotal = d.legitimate.reduce((a, b) => a + b, 0) || 1
  return d.fraud.map((_, i) => ({
    score: (d.bin_edges[i] + d.bin_edges[i + 1]) / 2,
    fraud: (d.fraud[i] / fraudTotal) * 100,
    legitimate: (d.legitimate[i] / legitTotal) * 100,
  }))
}
