import { NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard, Camera, List, Users, Car, Bell, Shield, Wifi, WifiOff
} from 'lucide-react'
import clsx from 'clsx'

const NAV = [
  { to: '/',         icon: LayoutDashboard, label: 'Overview'       },
  { to: '/cameras',  icon: Camera,          label: 'Live Cameras'   },
  { to: '/events',   icon: List,            label: 'Event Log'      },
  { to: '/faces',    icon: Users,           label: 'Face Database'  },
  { to: '/vehicles', icon: Car,             label: 'Vehicle DB'     },
  { to: '/alerts',   icon: Bell,            label: 'Alerts'         },
]

export default function Layout({ children, ws }) {
  const { connected, alerts } = ws
  const openAlerts = alerts.filter(a => !a.resolved).length

  return (
    <div className="flex h-screen overflow-hidden bg-surface">
      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <aside className="w-60 shrink-0 flex flex-col bg-surface-card border-r border-surface-border">
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-surface-border">
          <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center">
            <Shield size={18} className="text-white" />
          </div>
          <div>
            <p className="font-bold text-white leading-none">IntelliGate</p>
            <p className="text-xs text-gray-500 mt-0.5">Security Dashboard</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all',
                  isActive
                    ? 'bg-brand-600/20 text-brand-400 font-medium'
                    : 'text-gray-400 hover:bg-white/5 hover:text-white'
                )
              }
            >
              {({ isActive }) => (
                <>
                  <Icon size={17} className={isActive ? 'text-brand-400' : ''} />
                  {label}
                  {label === 'Alerts' && openAlerts > 0 && (
                    <span className="ml-auto text-xs bg-red-600 text-white rounded-full w-5 h-5 flex items-center justify-center font-bold">
                      {openAlerts > 9 ? '9+' : openAlerts}
                    </span>
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Connection status */}
        <div className="px-5 py-4 border-t border-surface-border">
          <div className={clsx(
            'flex items-center gap-2 text-xs',
            connected ? 'text-green-400' : 'text-red-400'
          )}>
            {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
            {connected ? 'Live — Connected' : 'Reconnecting…'}
          </div>
          <p className="text-xs text-gray-600 mt-1">COMSATS University</p>
        </div>
      </aside>

      {/* ── Main content ────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto">
        {children}
      </main>
    </div>
  )
}
