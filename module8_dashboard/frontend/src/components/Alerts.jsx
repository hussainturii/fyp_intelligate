import { useEffect, useState } from 'react'
import { Bell, CheckCircle, RefreshCw, AlertTriangle, AlertOctagon, Info } from 'lucide-react'
import { getAlerts, resolveAlert } from '../api/client'
import { format } from 'date-fns'
import clsx from 'clsx'

const SEVERITY_ICON = {
  CRITICAL: { Icon: AlertOctagon,  cls: 'text-red-400',    bg: 'bg-red-900/30',    border: 'border-red-800'    },
  HIGH:     { Icon: AlertTriangle, cls: 'text-orange-400', bg: 'bg-orange-900/30', border: 'border-orange-800' },
  MEDIUM:   { Icon: AlertTriangle, cls: 'text-yellow-400', bg: 'bg-yellow-900/30', border: 'border-yellow-800' },
  LOW:      { Icon: Info,          cls: 'text-blue-400',   bg: 'bg-blue-900/20',   border: 'border-blue-800'   },
}

function AlertCard({ alert, onResolve }) {
  const sev  = SEVERITY_ICON[alert.severity] ?? SEVERITY_ICON.LOW
  const Icon = sev.Icon

  return (
    <div className={clsx(
      'card flex gap-4 border',
      sev.bg, sev.border,
      alert.resolved && 'opacity-50'
    )}>
      <div className={clsx('mt-0.5 shrink-0', sev.cls)}>
        <Icon size={20} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <div>
            <p className="font-semibold text-white text-sm">{alert.title}</p>
            <p className="text-xs text-gray-400 mt-0.5">{alert.description}</p>
          </div>
          <div className="shrink-0 flex flex-col items-end gap-1">
            <span className={clsx('badge text-xs', {
              'badge-red':    alert.severity === 'CRITICAL',
              'badge-yellow': alert.severity === 'HIGH' || alert.severity === 'MEDIUM',
              'badge-blue':   alert.severity === 'LOW',
            })}>
              {alert.severity}
            </span>
            <span className="text-xs text-gray-500 font-mono">{alert.category}</span>
          </div>
        </div>
        <div className="flex items-center justify-between mt-3">
          <p className="text-xs text-gray-500">
            {alert.timestamp ? format(new Date(alert.timestamp), 'MMM dd, HH:mm:ss') : ''}
          </p>
          {!alert.resolved && (
            <button
              onClick={() => onResolve(alert.id)}
              className="flex items-center gap-1 text-xs text-green-400 hover:text-green-300 transition-colors"
            >
              <CheckCircle size={13} /> Mark resolved
            </button>
          )}
          {alert.resolved && (
            <p className="text-xs text-gray-600">
              Resolved {alert.resolved_at ? format(new Date(alert.resolved_at), 'MMM dd HH:mm') : ''}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

export default function Alerts({ ws }) {
  const [alerts,       setAlerts]       = useState([])
  const [showResolved, setShowResolved] = useState(false)
  const [loading,      setLoading]      = useState(false)

  const load = (resolved = showResolved) => {
    setLoading(true)
    getAlerts(resolved)
      .then(setAlerts)
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [showResolved])

  // Prepend live alerts from WebSocket
  useEffect(() => {
    if (!ws?.alerts?.length) return
    const newest = ws.alerts[0]
    if (!newest || showResolved) return
    setAlerts(prev => {
      if (prev[0]?.id === newest.id) return prev
      return [newest, ...prev]
    })
  }, [ws?.alerts])

  const handleResolve = async (id) => {
    await resolveAlert(id).catch(() => {})
    load()
  }

  const open   = alerts.filter(a => !a.resolved)
  const closed = alerts.filter(a =>  a.resolved)

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Security Alerts</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            {open.length} open · {closed.length} resolved
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={showResolved}
              onChange={e => setShowResolved(e.target.checked)}
              className="accent-brand-500"
            />
            Show resolved
          </label>
          <button onClick={() => load()} className="btn-ghost">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Alert types legend */}
      <div className="card flex flex-wrap gap-4">
        {Object.entries(SEVERITY_ICON).map(([sev, { Icon, cls }]) => (
          <div key={sev} className={clsx('flex items-center gap-1.5 text-xs', cls)}>
            <Icon size={13} /> {sev}
          </div>
        ))}
        <span className="text-xs text-gray-600 ml-auto">
          Categories: repeat_visitor · spoof_attempt · forced_entry · system
        </span>
      </div>

      {/* Alerts list */}
      {loading && <p className="text-gray-500 text-sm">Loading…</p>}

      {!loading && alerts.length === 0 && (
        <div className="card text-center py-16">
          <Bell size={40} className="mx-auto text-gray-600 mb-3" />
          <p className="text-gray-400 font-medium">No alerts</p>
          <p className="text-xs text-gray-600 mt-1">
            {showResolved ? 'No resolved alerts.' : 'System is clear.'}
          </p>
        </div>
      )}

      <div className="space-y-3">
        {alerts.map(alert => (
          <AlertCard key={alert.id} alert={alert} onResolve={handleResolve} />
        ))}
      </div>
    </div>
  )
}
