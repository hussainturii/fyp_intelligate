import { useEffect, useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './components/Dashboard'
import CameraView from './components/CameraView'
import EventLog from './components/EventLog'
import FaceDatabase from './components/FaceDatabase'
import VehicleDatabase from './components/VehicleDatabase'
import Alerts from './components/Alerts'
import Operations from './components/Operations'
import TestCenter from './components/TestCenter'
import { useWebSocket } from './api/useWebSocket'
import { request } from './api/client'

function SignedIn({ user, logout }) {
  const ws = useWebSocket()
  return <Layout ws={ws}>
    <div className="flex flex-wrap justify-between items-center gap-2 px-6 py-3 border-b border-surface-border text-sm">
      <span className="text-amber-300">CPU software prototype · live/replay inputs · virtual gate</span>
      <span>{user.username} · {user.role} <button onClick={logout} className="btn-ghost ml-2">Sign out</button></span>
    </div>
    <Routes>
      <Route path="/" element={<Dashboard ws={ws} />} />
      <Route path="/cameras" element={<CameraView ws={ws} user={user} />} />
      <Route path="/operations" element={<Operations user={user} />} />
      <Route path="/testing" element={<TestCenter user={user} />} />
      <Route path="/events" element={<EventLog ws={ws} />} />
      <Route path="/faces" element={<FaceDatabase user={user} />} />
      <Route path="/vehicles" element={<VehicleDatabase user={user} />} />
      <Route path="/alerts" element={<Alerts ws={ws} />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </Layout>
}

export default function App() {
  const [status, setStatus] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const load = () => request('/auth/status').then(setStatus).catch(e => setError(e.message))
  useEffect(() => { load() }, [])
  async function submit(event) {
    event.preventDefault(); setBusy(true); setError('')
    const fields = Object.fromEntries(new FormData(event.target))
    try {
      if (status.setup_required) await request('/auth/setup', { method: 'POST', body: JSON.stringify(fields) })
      const result = await request('/auth/login', { method: 'POST', body: JSON.stringify(fields) })
      setStatus({ setup_required: false, user: result.user })
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  async function logout() {
    try { await request('/auth/logout', { method: 'POST' }); setStatus({ setup_required: false, user: null }) }
    catch (e) { setError(e.message) }
  }
  if (status?.user) return <SignedIn user={status.user} logout={logout} />
  return <div className="min-h-screen flex items-center justify-center p-6">
    <form onSubmit={submit} className="card space-y-4 w-full max-w-md">
      <h1 className="text-2xl font-bold">IntelliGate</h1>
      <p className="text-amber-300 text-sm">Software prototype · CPU · no physical gate</p>
      <p>{status?.setup_required ? 'Create the first local administrator account.' : 'Sign in to the security dashboard.'}</p>
      <input className="input w-full" name="username" placeholder="Username" autoComplete="username" required maxLength={64} />
      <input className="input w-full" name="password" type="password" placeholder={status?.setup_required ? 'Password (at least 10 characters)' : 'Password'} autoComplete={status?.setup_required ? 'new-password' : 'current-password'} required minLength={status?.setup_required ? 10 : 1} maxLength={128} />
      {error && <p role="alert" className="text-red-400 break-words">{error}</p>}
      <button className="btn-primary w-full" disabled={busy || !status}>{busy ? 'Please wait…' : status?.setup_required ? 'Create account and sign in' : 'Sign in'}</button>
      {!status && <button type="button" onClick={load} className="btn-ghost">Retry connection</button>}
    </form>
  </div>
}
