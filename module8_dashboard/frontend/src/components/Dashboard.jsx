import { useEffect, useState } from 'react'
import { Users, Car, AlertTriangle, CheckCircle, XCircle, Activity } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { getStats, getHealth, getEvents } from '../api/client'
import { format } from 'date-fns'
import clsx from 'clsx'

function StatCard({ icon: Icon, label, value, color, sub }) {
  return (
    <div className="stat-card">
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-400">{label}</span>
        <div className={clsx('w-9 h-9 rounded-lg flex items-center justify-center', color)}>
          <Icon size={18} className="text-white" />
        </div>
      </div>
      <p className="stat-number">{value ?? '—'}</p>
      {sub && <p className="text-xs text-gray-500">{sub}</p>}
    </div>
  )
}

function SystemGauge({ label, value, color }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">{label}</span>
        <span className={color}>{value}%</span>
      </div>
      <div className="h-1.5 bg-surface rounded-full overflow-hidden">
        <div
          className={clsx('h-full rounded-full transition-all duration-500', color.replace('text-', 'bg-'))}
          style={{ width: `${Math.min(value, 100)}%` }}
        />
      </div>
    </div>
  )
}

function eventBadge(type) {
  const map = {
    authorized: 'badge-green',
    known:      'badge-green',
    stranger:   'badge-yellow',
    unknown:    'badge-yellow',
    spoof:      'badge-red',
  }
  return map[type] || 'badge-gray'
}

export default function Dashboard({ ws }) {
  const [stats,  setStats]  = useState(null)
  const [health, setHealth] = useState(null)
  const [recent, setRecent] = useState([])

  // Fetch on mount
  useEffect(() => {
    getStats().then(setStats).catch(() => {})
    getHealth().then(setHealth).catch(() => {})
    getEvents({ limit: 10 }).then(rows => setRecent(rows)).catch(() => {})
  }, [])

  // Update health from WebSocket system ticks
  useEffect(() => {
    if (ws.health) setHealth(h => ({ ...h, ...ws.health }))
  }, [ws.health])

  // Prepend new live events
  useEffect(() => {
    if (ws.events.length > 0) {
      setRecent(prev => [ws.events[0], ...prev.slice(0, 9)])
    }
  }, [ws.events])

  const healthHistory = health?.history?.map(h => ({
    t:   h.timestamp?.slice(11, 16),
    cpu: h.cpu_pct,
    mem: h.mem_pct,
  })) ?? []

  return (
    <div className="p-6 space-y-6">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Overview</h1>
          <p className="text-sm text-gray-500 mt-0.5">Last 24 hours</p>
        </div>
        <div className="flex items-center gap-2">
          <div className={clsx(
            'relative w-2 h-2 rounded-full live-dot',
            ws.connected ? 'text-green-400' : 'text-gray-600'
          )} />
          <span className="text-xs text-gray-400">
            {ws.connected ? 'Live' : 'Offline'}
          </span>
        </div>
      </div>

      {/* Stat cards row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={CheckCircle}
          label="Authorized Persons"
          value={stats?.faces_authorized_24h}
          color="bg-green-700"
          sub="Face recognized"
        />
        <StatCard
          icon={AlertTriangle}
          label="Unknown Faces"
          value={stats?.faces_stranger_24h}
          color="bg-yellow-700"
          sub="Not in database"
        />
        <StatCard
          icon={Car}
          label="Known Vehicles"
          value={stats?.vehicles_known_24h}
          color="bg-blue-700"
          sub="Plate / Re-ID matched"
        />
        <StatCard
          icon={XCircle}
          label="Open Alerts"
          value={stats?.open_alerts}
          color="bg-red-700"
          sub="Needs attention"
        />
      </div>

      {/* Middle row: health + recent events */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* System health card */}
        <div className="card space-y-4 col-span-1">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-white">System Health</h2>
            <Activity size={16} className="text-gray-500" />
          </div>
          <SystemGauge
            label="CPU"
            value={health?.cpu ?? 0}
            color="text-blue-400"
          />
          <SystemGauge
            label="Memory"
            value={health?.memory ?? 0}
            color="text-purple-400"
          />
          <SystemGauge
            label="GPU"
            value={health?.gpu ?? 0}
            color="text-green-400"
          />
          <div className="flex justify-between text-xs pt-1 border-t border-surface-border">
            <span className="text-gray-500">Camera FPS</span>
            <span className="text-white font-mono">{health?.fps ?? '—'}</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-gray-500">Camera</span>
            <span className={health?.camera_online ? 'text-green-400' : 'text-red-400'}>
              {health?.camera_online ? 'Online' : 'Offline'}
            </span>
          </div>

          {/* Sparkline */}
          {healthHistory.length > 2 && (
            <ResponsiveContainer width="100%" height={60}>
              <LineChart data={healthHistory}>
                <Line type="monotone" dataKey="cpu" stroke="#60a5fa" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="mem" stroke="#a78bfa" dot={false} strokeWidth={1.5} />
                <Tooltip
                  contentStyle={{ background: '#1a1d27', border: '1px solid #2d3148', borderRadius: 8 }}
                  labelStyle={{ color: '#9ca3af' }}
                  itemStyle={{ color: '#e5e7eb' }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Recent events */}
        <div className="card col-span-2 flex flex-col">
          <h2 className="font-semibold text-white mb-4">Recent Activity</h2>
          <div className="flex-1 overflow-y-auto space-y-2">
            {recent.length === 0 && (
              <p className="text-sm text-gray-500 text-center py-8">No events yet</p>
            )}
            {recent.map((ev, i) => (
              <div key={ev.id ?? i} className="flex items-center gap-3 py-2 border-b border-surface-border last:border-0">
                <div className={clsx('w-2 h-2 rounded-full shrink-0', {
                  'bg-green-400': ev.event_type === 'authorized' || ev.event_type === 'known',
                  'bg-yellow-400': ev.event_type === 'stranger'  || ev.event_type === 'unknown',
                  'bg-red-400':    ev.event_type === 'spoof',
                })} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white truncate">
                    {ev.identity || 'Unknown'}{' '}
                    <span className="text-gray-500 font-normal">
                      via {ev.method || ev.type?.replace('_event', '') || '—'}
                    </span>
                  </p>
                  <p className="text-xs text-gray-500">
                    {ev.timestamp ? format(new Date(ev.timestamp), 'HH:mm:ss') : ''}
                  </p>
                </div>
                <span className={clsx('badge shrink-0', eventBadge(ev.event_type))}>
                  {ev.event_type}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom: spoof attempts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="card col-span-1">
          <h2 className="font-semibold text-white mb-3">24h Summary</h2>
          <dl className="space-y-2 text-sm">
            {[
              ['Total events',         stats?.total_events_24h,      'text-white'],
              ['Spoof attempts',        stats?.faces_spoof_24h,       'text-red-400'],
              ['Unknown vehicles',      stats?.vehicles_unknown_24h,  'text-yellow-400'],
            ].map(([label, val, cls]) => (
              <div key={label} className="flex justify-between border-b border-surface-border pb-2 last:border-0">
                <span className="text-gray-400">{label}</span>
                <span className={clsx('font-mono font-semibold', cls)}>{val ?? '—'}</span>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </div>
  )
}
