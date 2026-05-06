import { useEffect, useState, useRef } from 'react'
import { List, RefreshCw, Filter, Download } from 'lucide-react'
import { getEvents } from '../api/client'
import { format } from 'date-fns'
import clsx from 'clsx'

const MODULE_OPTS  = ['all', 'face', 'vehicle']
const TYPE_OPTS    = ['all', 'authorized', 'stranger', 'spoof', 'known', 'unknown']

function badgeClass(type) {
  return {
    authorized: 'badge-green',
    known:      'badge-green',
    stranger:   'badge-yellow',
    unknown:    'badge-yellow',
    spoof:      'badge-red',
  }[type] ?? 'badge-gray'
}

function confidenceBar(val) {
  if (val == null) return null
  const pct  = Math.round(val * 100)
  const color = pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-surface rounded-full overflow-hidden">
        <div className={clsx('h-full rounded-full', color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-gray-300">{pct}%</span>
    </div>
  )
}

export default function EventLog({ ws }) {
  const [events,    setEvents]    = useState([])
  const [loading,   setLoading]   = useState(false)
  const [module,    setModule]    = useState('all')
  const [evType,    setEvType]    = useState('all')
  const [sinceH,    setSinceH]    = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const tableRef = useRef(null)

  const load = (mod = module, type = evType, hours = sinceH) => {
    setLoading(true)
    getEvents({
      module:      mod   !== 'all' ? mod  : null,
      event_type:  type  !== 'all' ? type : null,
      since_hours: hours || null,
      limit:       200,
    })
      .then(rows => setEvents(rows))
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  // Prepend live events
  useEffect(() => {
    if (!ws?.events?.length) return
    const newest = ws.events[0]
    if (!newest) return
    setEvents(prev => {
      // avoid duplicate if already present
      if (prev[0]?.id === newest.id) return prev
      return [newest, ...prev.slice(0, 199)]
    })
  }, [ws?.events])

  // CSV export
  const handleExport = () => {
    const header = 'id,timestamp,module,event_type,identity,confidence,method,camera_id'
    const rows = events.map(e =>
      [e.id, e.timestamp, e.module, e.event_type, e.identity ?? '', e.confidence ?? '', e.method ?? '', e.camera_id].join(',')
    )
    const csv  = [header, ...rows].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = `intelligate_events_${Date.now()}.csv`
    a.click()
  }

  return (
    <div className="p-6 flex flex-col gap-4 h-full">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Event Log</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            {events.length} events loaded
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleExport} className="btn-ghost text-sm">
            <Download size={14} /> Export CSV
          </button>
          <button onClick={() => load()} className="btn-ghost text-sm">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="card flex flex-wrap items-center gap-3">
        <Filter size={14} className="text-gray-500" />
        <select
          value={module}
          onChange={e => { setModule(e.target.value); load(e.target.value) }}
          className="input text-xs py-1.5"
        >
          {MODULE_OPTS.map(o => <option key={o} value={o}>{o === 'all' ? 'All modules' : o}</option>)}
        </select>
        <select
          value={evType}
          onChange={e => { setEvType(e.target.value); load(module, e.target.value) }}
          className="input text-xs py-1.5"
        >
          {TYPE_OPTS.map(o => <option key={o} value={o}>{o === 'all' ? 'All types' : o}</option>)}
        </select>
        <select
          value={sinceH}
          onChange={e => { setSinceH(e.target.value); load(module, evType, e.target.value) }}
          className="input text-xs py-1.5"
        >
          <option value="">All time</option>
          <option value="1">Last 1h</option>
          <option value="6">Last 6h</option>
          <option value="24">Last 24h</option>
          <option value="168">Last 7d</option>
        </select>
        <label className="flex items-center gap-1.5 text-xs text-gray-400 ml-auto cursor-pointer">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={e => setAutoScroll(e.target.checked)}
            className="accent-brand-500"
          />
          Auto-scroll
        </label>
      </div>

      {/* Table */}
      <div className="card flex-1 overflow-hidden flex flex-col min-h-0">
        <div ref={tableRef} className="overflow-y-auto flex-1">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-surface-card border-b border-surface-border">
              <tr className="text-xs text-gray-500 uppercase">
                <th className="text-left py-2 px-3">Time</th>
                <th className="text-left py-2 px-3">Module</th>
                <th className="text-left py-2 px-3">Type</th>
                <th className="text-left py-2 px-3">Identity</th>
                <th className="text-left py-2 px-3">Confidence</th>
                <th className="text-left py-2 px-3">Method</th>
                <th className="text-left py-2 px-3">Cam</th>
              </tr>
            </thead>
            <tbody>
              {events.length === 0 && (
                <tr>
                  <td colSpan={7} className="text-center py-12 text-gray-500">
                    {loading ? 'Loading…' : 'No events found'}
                  </td>
                </tr>
              )}
              {events.map((ev, i) => (
                <tr key={ev.id ?? i} className="table-row">
                  <td className="py-2 px-3 font-mono text-xs text-gray-400 whitespace-nowrap">
                    {ev.timestamp ? format(new Date(ev.timestamp), 'MM-dd HH:mm:ss') : '—'}
                  </td>
                  <td className="py-2 px-3">
                    <span className={clsx('badge', ev.module === 'face' ? 'badge-blue' : 'badge-gray')}>
                      {ev.module}
                    </span>
                  </td>
                  <td className="py-2 px-3">
                    <span className={clsx('badge', badgeClass(ev.event_type))}>
                      {ev.event_type}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-white truncate max-w-[140px]">
                    {ev.identity || <span className="text-gray-600">—</span>}
                  </td>
                  <td className="py-2 px-3">
                    {confidenceBar(ev.confidence)}
                  </td>
                  <td className="py-2 px-3 text-gray-400 text-xs">
                    {ev.method || '—'}
                  </td>
                  <td className="py-2 px-3 text-gray-400 text-xs text-center">
                    {ev.camera_id ?? 0}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
