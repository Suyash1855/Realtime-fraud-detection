/*
  Landing — what this system is, and what its numbers actually mean.

  The leakage comparison is the centrepiece rather than a footnote. An earlier
  version of this pipeline reported AUC 0.9469 using a random train/test split on
  what is inherently time-series data. Splitting chronologically drops it to
  0.8932. That gap is future knowledge, not skill, and saying so is the honest
  framing. See PROJECT_ANALYSIS.md sections 3 and 9.
*/

import { Card, Stat, Legend, COLORS } from '../components/ui'
import { fmt } from '../api'

// Historical benchmark from registry v5 (random split). Kept as a constant
// because it describes a superseded model that the app no longer serves.
const RANDOM_SPLIT_AUC = 0.9469

export default function Landing({ info, onStart }) {
  const { model, dataset, architecture } = info
  const m = model.metrics || {}
  const chronoAuc = m.auc_roc ?? 0
  const gap = RANDOM_SPLIT_AUC - chronoAuc

  return (
    <div className="grid" style={{ gap: 18 }}>

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <Card>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 28, alignItems: 'flex-start' }}>
          <div style={{ flex: '1 1 440px', minWidth: 0 }}>
            <div className="tag" style={{ marginBottom: 12 }}>
              IEEE-CIS Fraud Detection · 590,540 transactions
            </div>
            <h1 style={{ fontSize: 28, lineHeight: 1.2, margin: '0 0 12px', letterSpacing: '-0.02em' }}>
              Real-time card fraud detection
            </h1>
            <p className="secondary" style={{ margin: '0 0 18px', maxWidth: 620 }}>
              A streaming pipeline that scores card transactions as they arrive. XGBoost
              served from an MLflow model registry, feature engineering shared with
              training, and a decision threshold chosen from a precision/recall
              tradeoff rather than left at 0.5.
            </p>
            <button className="btn btn-primary btn-lg" onClick={onStart}>
              ▶ Start real-time simulation
            </button>
            <div className="small muted" style={{ marginTop: 10 }}>
              Streams {fmt.int(dataset.n_transactions)} held-out transactions through the
              live model.
            </div>
          </div>

          <div style={{ flex: '0 1 300px' }}>
            <dl className="kv">
              <dt>Model</dt><dd>{model.model_name} v{model.model_version}</dd>
              <dt>Algorithm</dt><dd>XGBoost · {fmt.int(model.n_trees)} trees</dd>
              <dt>Features</dt><dd>{model.n_features}</dd>
              <dt>Validation</dt><dd>{model.split_strategy}</dd>
              <dt>Threshold</dt><dd>{fmt.dec(model.fraud_threshold, 4)}</dd>
              <dt>Fraud rate</dt><dd>{fmt.pct(dataset.fraud_rate, 2)}</dd>
            </dl>
          </div>
        </div>
      </Card>

      {/* ── Honest headline metrics ──────────────────────────────────────── */}
      <div className="grid grid-4">
        <Stat
          label="AUC-ROC (chronological)"
          value={fmt.dec(chronoAuc, 4)}
          sub="held-out future period"
          tone="accent"
        />
        <Stat
          label="Avg precision"
          value={fmt.dec(m.avg_precision, 4)}
          sub="area under PR curve"
        />
        <Stat
          label="Precision @ threshold"
          value={fmt.dec(m.precision, 3)}
          sub={`recall ${fmt.dec(m.recall, 3)}`}
        />
        <Stat
          label="Transactions in demo"
          value={fmt.int(dataset.n_transactions)}
          sub={`${fmt.int(dataset.n_fraud)} fraudulent`}
        />
      </div>

      {/* ── The leakage story ────────────────────────────────────────────── */}
      <Card
        title="Why this AUC is lower than it could be"
        note="The most important number on this page is the one that went down"
      >
        <p className="secondary" style={{ marginTop: 0 }}>
          <code className="mono">TransactionDT</code> is a timestamp offset, so this is a
          time-series problem. An earlier version of this pipeline split train/test
          randomly, which scatters the same cards and devices across both sides and
          lets the model see the future. It reported a much better number.
        </p>

        <div className="grid grid-2" style={{ gap: 12, marginTop: 4 }}>
          <div className="stat">
            <div className="stat-label">Random split (superseded)</div>
            <div className="stat-value sm">{fmt.dec(RANDOM_SPLIT_AUC, 4)}</div>
            <div className="stat-sub">optimistic — leaks future information</div>
          </div>
          <div className="stat">
            <div className="stat-label">Chronological split (serving now)</div>
            <div className="stat-value sm stat-accent">{fmt.dec(chronoAuc, 4)}</div>
            <div className="stat-sub">honest — trained on past, tested on future</div>
          </div>
        </div>

        <LeakageBar randomAuc={RANDOM_SPLIT_AUC} chronoAuc={chronoAuc} />

        <p className="small muted" style={{ marginBottom: 0 }}>
          The {fmt.dec(gap, 4)} gap is future knowledge, not model skill. The pipeline
          also carves validation out separately, so early stopping never sees the test
          labels, and categorical encoders are fit on the training period alone — the
          unseen-category rate on the test period is roughly 19× that of validation,
          which is real drift the model has to survive.
        </p>
      </Card>

      {/* ── What is real here ────────────────────────────────────────────── */}
      <div className="grid grid-2">
        <Card
          title="What is real in this demo"
          note="No smoke and mirrors — these run for every transaction you see"
        >
          <ul className="secondary" style={{ margin: 0, paddingLeft: 18, fontSize: 13 }}>
            {architecture.real.map((x) => <li key={x} style={{ marginBottom: 5 }}>{x}</li>)}
          </ul>
        </Card>

        <Card
          title="What is simulated"
          note="Swapped for deployability — the real path runs locally"
        >
          <ul className="secondary" style={{ margin: 0, paddingLeft: 18, fontSize: 13 }}>
            {architecture.simulated.map((x) => <li key={x} style={{ marginBottom: 5 }}>{x}</li>)}
          </ul>
          <hr className="divider" />
          <div className="small muted" style={{ marginBottom: 6 }}>
            Runs locally via <code className="mono">docker compose up</code>:
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {architecture.local_only.map((x) => (
              <span className="tag" key={x}>{x.split(' (')[0]}</span>
            ))}
          </div>
        </Card>
      </div>

      {/* ── Provenance ───────────────────────────────────────────────────── */}
      <Card title="Where the demo data comes from">
        <p className="secondary small" style={{ margin: 0 }}>{dataset.provenance}</p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 20, marginTop: 14 }}>
          <div><div className="stat-label">Rows</div><div className="mono">{fmt.int(dataset.n_transactions)}</div></div>
          <div><div className="stat-label">Fraud</div><div className="mono">{fmt.int(dataset.n_fraud)} ({fmt.pct(dataset.fraud_rate, 2)})</div></div>
          <div><div className="stat-label">Identity coverage</div><div className="mono">{fmt.pct(dataset.identity_coverage, 1)}</div></div>
          <div><div className="stat-label">Fraud exposure</div><div className="mono">{fmt.money(dataset.fraud_amount_total)}</div></div>
          <div><div className="stat-label">Total volume</div><div className="mono">{fmt.money(dataset.amount_total)}</div></div>
        </div>
      </Card>
    </div>
  )
}

/**
 * Two bars on one shared 0–1 scale. Deliberately not a dual-axis chart and not a
 * pie: the story is a comparison of two numbers, so bar length is the encoding
 * and the values are labelled directly.
 */
function LeakageBar({ randomAuc, chronoAuc }) {
  const rows = [
    { label: 'Random split', value: randomAuc, color: COLORS.s2 },
    { label: 'Chronological', value: chronoAuc, color: COLORS.s1 },
  ]
  const scaleMin = 0.5   // AUC 0.5 is chance; anchoring there keeps the gap readable

  return (
    <figure style={{ margin: '16px 0 14px' }}>
      {rows.map((r) => {
        const pct = ((r.value - scaleMin) / (1 - scaleMin)) * 100
        return (
          <div key={r.label} style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
              <span className="secondary">{r.label}</span>
              <span className="mono" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {fmt.dec(r.value, 4)}
              </span>
            </div>
            <div style={{ height: 8, background: COLORS.grid, borderRadius: 4, overflow: 'hidden' }}>
              <div
                style={{
                  width: `${pct}%`, height: '100%', background: r.color,
                  borderRadius: 4, transition: 'width .5s ease',
                }}
              />
            </div>
          </div>
        )
      })}
      <figcaption className="small muted" style={{ marginTop: 8 }}>
        Scale starts at AUC 0.50 (chance). Higher is better.
      </figcaption>
      <div style={{ marginTop: 10 }}>
        <Legend items={[
          { label: 'Random split (leaky)', color: COLORS.s2 },
          { label: 'Chronological (honest)', color: COLORS.s1 },
        ]} />
      </div>
    </figure>
  )
}
