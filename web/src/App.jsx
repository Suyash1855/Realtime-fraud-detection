import { useCallback, useEffect, useState } from 'react'
import { api, useAsync, useSimulation } from './api'
import { ErrorBox, Loading } from './components/ui'
import TransactionDrawer from './components/TransactionDrawer'
import Landing from './views/Landing'
import LiveConsole from './views/LiveConsole'
import Analytics from './views/Analytics'

const VIEWS = [
  { id: 'overview', label: 'Overview' },
  { id: 'live', label: 'Live console' },
  { id: 'analytics', label: 'Analytics' },
]

const IDS = VIEWS.map((v) => v.id)

/** Read the view from the URL hash so each screen is linkable and survives reload. */
function viewFromHash() {
  const id = window.location.hash.replace(/^#\/?/, '').split('?')[0]
  return IDS.includes(id) ? id : 'overview'
}

export default function App() {
  const [view, setView] = useState(viewFromHash)
  const [selectedTx, setSelectedTx] = useState(null)

  const { data: info, loading, error } = useAsync(() => api.info(), [])

  // The simulation lives here, not inside the console view, so switching to
  // Analytics mid-run does not tear down the stream.
  const sim = useSimulation()

  const go = useCallback((id) => {
    setView(id)
    if (viewFromHash() !== id) window.location.hash = `#/${id}`
  }, [])

  // Back/forward buttons should move between views.
  useEffect(() => {
    const onHash = () => setView(viewFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const startFromLanding = useCallback(() => {
    go('live')
    sim.start({ rate: 250, loop: true, shuffle: true })
  }, [go, sim])

  // Deep link straight into a running simulation: #/live?autostart=1
  const autostart = window.location.hash.includes('autostart=1')
  useEffect(() => {
    if (info && autostart && view === 'live' && sim.status === 'idle') {
      sim.start({ rate: 250, loop: true, shuffle: true })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [info, autostart, view])

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">🛡️</span>
          <span>Fraud Detection</span>
          <span className="brand-sub">
            {info ? `${info.model.model_name} v${info.model.model_version}` : 'loading…'}
          </span>
        </div>
        <nav className="nav" aria-label="Views">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              onClick={() => go(v.id)}
              aria-current={view === v.id ? 'page' : undefined}
            >
              {v.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="page">
        {loading && <Loading label="Loading model and demo data…" />}
        {error && (
          <ErrorBox
            message={
              `Could not reach the API (${error}). If you are running the frontend ` +
              `separately, make sure uvicorn is up on port 8000.`
            }
          />
        )}

        {info && view === 'overview' && (
          <Landing info={info} onStart={startFromLanding} />
        )}
        {info && view === 'live' && (
          <LiveConsole info={info} sim={sim} onSelectTransaction={setSelectedTx} />
        )}
        {info && view === 'analytics' && <Analytics info={info} />}
      </main>

      {selectedTx && (
        <TransactionDrawer txId={selectedTx} onClose={() => setSelectedTx(null)} />
      )}
    </div>
  )
}
