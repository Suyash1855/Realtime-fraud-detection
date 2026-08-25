/*
  API layer
  =========
  The app is served by the same FastAPI process that owns /api, so every request
  is same-origin and needs no base URL or CORS handling.
*/

import { useCallback, useEffect, useRef, useState } from 'react'

async function get(path) {
  const res = await fetch(path)
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* response wasn't JSON; keep the status text */
    }
    throw new Error(`${res.status} — ${detail}`)
  }
  return res.json()
}

export const api = {
  info: () => get('/api/info'),
  thresholdMetrics: (t) => get(`/api/analytics/threshold?t=${t}`),
  curves: (points = 140) => get(`/api/analytics/pr-curve?points=${points}`),
  distribution: (bins = 40) => get(`/api/analytics/distribution?bins=${bins}`),
  cost: (fp, fn) => get(`/api/analytics/cost?fp_cost=${fp}&fn_cost=${fn}`),
  transaction: (id) => get(`/api/transaction/${id}`),
}

/** Fetch-on-mount with loading/error state. */
export function useAsync(fn, deps = []) {
  const [state, setState] = useState({ data: null, loading: true, error: null })

  useEffect(() => {
    let alive = true
    setState((s) => ({ ...s, loading: true, error: null }))
    fn()
      .then((data) => alive && setState({ data, loading: false, error: null }))
      .catch((error) => alive && setState({ data: null, loading: false, error: error.message }))
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return state
}

/**
 * Debounced value. The threshold slider fires continuously while dragging; the
 * local number updates every frame for responsiveness while the network call
 * trails behind it.
 */
export function useDebounced(value, delay = 120) {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(id)
  }, [value, delay])
  return debounced
}

const MAX_FEED = 120
const MAX_SPARK = 90

const emptyStats = {
  processed: 0, flagged: 0, high_risk: 0,
  true_positives: 0, false_positives: 0, false_negatives: 0,
  precision: 0, recall: 0, flag_rate: 0, throughput: 0,
  latency_p50_ms: 0, latency_p95_ms: 0, latency_p99_ms: 0,
  amount_total: 0, amount_flagged: 0,
  fraud_amount_caught: 0, fraud_amount_missed: 0,
  elapsed_seconds: 0,
}

/**
 * Drives the live simulation over Server-Sent Events.
 *
 * SSE rather than WebSockets: the flow is strictly server-to-client, it
 * reconnects on its own, and it survives proxies that mishandle upgrades.
 */
export function useSimulation() {
  const [status, setStatus] = useState('idle')   // idle | running | complete | error
  const [stats, setStats] = useState(emptyStats)
  const [feed, setFeed] = useState([])
  const [spark, setSpark] = useState([])
  const [progress, setProgress] = useState(0)
  const [sessionId, setSessionId] = useState(null)
  const [error, setError] = useState(null)

  const sourceRef = useRef(null)
  const tickRef = useRef(0)

  const stop = useCallback(() => {
    const id = sessionId
    if (sourceRef.current) {
      sourceRef.current.close()
      sourceRef.current = null
    }
    // Best-effort server-side cleanup; the stream also stops on disconnect.
    if (id != null) {
      fetch(`/api/stream/${id}/stop`, { method: 'POST' }).catch(() => {})
    }
    setStatus((s) => (s === 'running' ? 'complete' : s))
  }, [sessionId])

  const reset = useCallback(() => {
    setStats(emptyStats)
    setFeed([])
    setSpark([])
    setProgress(0)
    setError(null)
    tickRef.current = 0
  }, [])

  const start = useCallback(
    ({ rate = 100, threshold = null, loop = false, shuffle = false } = {}) => {
      if (sourceRef.current) sourceRef.current.close()
      reset()
      setStatus('running')

      const params = new URLSearchParams({ rate: String(rate), loop: String(loop), shuffle: String(shuffle) })
      if (threshold != null) params.set('threshold', String(threshold))

      const es = new EventSource(`/api/stream?${params}`)
      sourceRef.current = es

      es.addEventListener('started', (e) => {
        const d = JSON.parse(e.data)
        setSessionId(d.session_id)
      })

      es.addEventListener('batch', (e) => {
        const d = JSON.parse(e.data)
        setStats(d.stats)
        setProgress(d.progress)

        if (d.flagged?.length) {
          setFeed((prev) => [...d.flagged.slice().reverse(), ...prev].slice(0, MAX_FEED))
        }

        // One sparkline point per batch: throughput and tail latency over time.
        tickRef.current += 1
        setSpark((prev) =>
          [...prev, {
            t: tickRef.current,
            throughput: d.stats.throughput,
            latency: d.stats.latency_p95_ms,
            flagged: d.stats.flagged,
          }].slice(-MAX_SPARK)
        )
      })

      es.addEventListener('complete', (e) => {
        const d = JSON.parse(e.data)
        if (d.stats) setStats(d.stats)
        setStatus('complete')
        es.close()
        sourceRef.current = null
      })

      es.addEventListener('error', (e) => {
        // A server-sent `error` event carries a payload; a transport failure does not.
        if (e.data) {
          try {
            setError(JSON.parse(e.data).message)
          } catch {
            setError('Stream error')
          }
        } else if (es.readyState === EventSource.CLOSED) {
          setError('Connection to the scoring stream was lost')
        }
        setStatus('error')
        es.close()
        sourceRef.current = null
      })
    },
    [reset]
  )

  const retune = useCallback(
    (value) => {
      if (sessionId == null || status !== 'running') return
      fetch(`/api/stream/${sessionId}/threshold?value=${value}`, { method: 'POST' }).catch(() => {})
    },
    [sessionId, status]
  )

  // Close the stream if the component unmounts mid-run.
  useEffect(() => () => sourceRef.current?.close(), [])

  return { status, stats, feed, spark, progress, sessionId, error, start, stop, reset, retune }
}

/* ── formatting ───────────────────────────────────────────────────────────── */

export const fmt = {
  int: (n) => (n ?? 0).toLocaleString('en-US'),
  pct: (n, d = 1) => `${((n ?? 0) * 100).toFixed(d)}%`,
  dec: (n, d = 3) => (n ?? 0).toFixed(d),
  money: (n) =>
    `$${Math.round(n ?? 0).toLocaleString('en-US')}`,
  moneyPrecise: (n) =>
    (n ?? 0).toLocaleString('en-US', { style: 'currency', currency: 'USD' }),
  ms: (n) => `${(n ?? 0).toFixed(n < 10 ? 2 : 1)}ms`,
  compact: (n) => {
    const v = n ?? 0
    if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(1)}M`
    if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(1)}k`
    return String(Math.round(v))
  },
}
