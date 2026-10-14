import { useEffect, useState } from 'react'
import { request } from '../api/client'

export default function Operations({ user }) {
  const [gate, setGate] = useState(null)
  const [reviews, setReviews] = useState([])
  const [modules, setModules] = useState(null)
  const [visitors, setVisitors] = useState([])
  const [report, setReport] = useState(null)
  const [query, setQuery] = useState('events today')
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [config, setConfig] = useState(null)
  const canControl = user.role !== 'viewer'
  async function load() {
    try {
      const [g, r, m, v] = await Promise.all([request('/gate'), request('/reviews'), request('/modules'), request('/visitors')])
      setGate(g); setReviews(r); setModules(m); setVisitors(v)
      setConfig(previous => previous || m.scene_config)
    } catch (e) { setError(e.message) }
  }
  useEffect(() => { load(); const timer = setInterval(load, 2000); return () => clearInterval(timer) }, [])
  async function action(path, body) {
    setBusy(true); setError('')
    try { await request(path, { method: 'POST', body: JSON.stringify(body) }); await load() }
    catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  return <div className="p-6 space-y-6">
    <h1 className="text-2xl font-bold">Software operations</h1>
    {error && <div className="card text-red-400" role="alert">{error}</div>}
    <div className="grid lg:grid-cols-2 gap-4">
      <section className="card space-y-4">
        <h2 className="font-semibold">Virtual gate <span className="badge badge-yellow">Simulation</span></h2>
        <p className="text-3xl font-mono">{gate?.state || 'Loading…'}</p>
        <p className="text-sm text-gray-400">{gate?.reason}</p>
        <p>Obstruction: {gate?.obstruction ? 'present' : 'clear'} · Lockdown: {gate?.lockdown ? 'active' : 'off'}</p>
        <div className="flex flex-wrap gap-2">{['OPEN','CLOSE','STOP','LOCKDOWN','RESET'].map(command => <button key={command} disabled={busy || !canControl} className={command === 'STOP' ? 'btn-danger' : 'btn-ghost'} onClick={() => action('/gate/command', { command, command_id: crypto.randomUUID(), expires: Date.now()/1000+5 })}>{command}</button>)}</div>
        <button className="btn-ghost" disabled={busy || !canControl} onClick={() => action('/gate/obstruction', { present: !gate?.obstruction })}>{gate?.obstruction ? 'Clear simulated obstruction' : 'Simulate obstruction'}</button>
      </section>
      <section className="card space-y-3">
        <h2 className="font-semibold">Access review</h2>
        <p className="text-sm text-gray-400">Approvals expire after two minutes. Face liveness and appearance matches require review.</p>
        {reviews.filter(r => r.status === 'PENDING').length === 0 && <p>No pending reviews.</p>}
        {reviews.filter(r => r.status === 'PENDING').map(r => <div key={r.id} className="border-t pt-2 flex flex-wrap justify-between gap-2"><span>{r.identity || 'Unknown'} · event #{r.event_id}</span><div className="flex gap-2"><button disabled={busy || !canControl} className="btn-primary" onClick={() => action(`/reviews/${r.id}`, { approve: true })}>Approve</button><button disabled={busy || !canControl} className="btn-danger" onClick={() => action(`/reviews/${r.id}`, { approve: false })}>Deny</button></div></div>)}
      </section>
    </div>
    <section className="card space-y-3">
      <h2 className="font-semibold">Scene rules and queue</h2>
      <p className="text-sm text-gray-400">Blue ROI counts tracked detections; red ROI represents the virtual gate path. Loitering uses track dwell, not intent prediction.</p>
      <p>Current ROI count: {modules?.scene?.count ?? 'No scene processed'} · {modules?.scene?.modality_warning}</p>
      {config && <form onSubmit={e => { e.preventDefault(); action('/scene/config', config) }} className="flex flex-wrap gap-3 items-end">
        {['queue_roi','gate_roi'].map(key => <label key={key} className="text-sm">{key.replace('_',' ')} (0–1)<div className="flex gap-1">{config[key].map((n, i) => <input key={i} className="input w-20" type="number" min="0" max="1" step="0.01" value={n} onChange={e => setConfig({ ...config, [key]: config[key].map((v,j) => i === j ? Number(e.target.value) : v) })} />)}</div></label>)}
        <label className="text-sm">Queue threshold<input className="input block w-24" type="number" min="1" max="100" value={config.queue_threshold} onChange={e => setConfig({ ...config, queue_threshold: Number(e.target.value) })} /></label>
        <label className="text-sm">Dwell seconds<input className="input block w-24" type="number" min="1" max="3600" value={config.loiter_seconds} onChange={e => setConfig({ ...config, loiter_seconds: Number(e.target.value) })} /></label>
        <button disabled={busy || !canControl} className="btn-primary">Save rules</button>
      </form>}
    </section>
    <section className="card space-y-3">
      <h2 className="font-semibold">Stranger visits</h2>
      <p className="text-sm text-gray-400">One visit per continuous appearance, with a two-second absence grace. Three distinct appearances within 24 hours trigger an alert.</p>
      {visitors.length === 0 ? <p>No stranger visits yet.</p> : visitors.map(v => <p key={v.identity} className="text-sm font-mono break-all">{v.identity.slice(0,12)} · {v.visits} visits · {new Date(v.last_seen*1000).toLocaleString()}</p>)}
    </section>
    <section className="card space-y-3">
      <h2 className="font-semibold">Evidence report <span className="badge badge-gray">Templates · not a VLM</span></h2>
      <button className="btn-primary" onClick={() => request('/reports').then(setReport).catch(e => setError(e.message))}>Generate last 24 hours</button>
      {report && <><p>{report.summary}</p><p className="text-sm text-gray-400">Evidence preview: {report.evidence.length} of {report.total} events. Generated {new Date(report.generated_at).toLocaleString()}.</p><button className="btn-ghost" onClick={() => { const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'})); const a=document.createElement('a'); a.href=url; a.download='intelligate-report.json'; a.click(); URL.revokeObjectURL(url) }}>Download evidence JSON</button></>}
      <form onSubmit={async e => { e.preventDefault(); try { setResults(await request(`/search?q=${encodeURIComponent(query)}`)); setError('') } catch(e) { setError(e.message) } }} className="flex gap-2"><input className="input flex-1" value={query} onChange={e => setQuery(e.target.value)} placeholder="face today, vehicle, spoof, obstruction…" /><button className="btn-ghost">Search events</button></form>
      {results && <div className="max-h-64 overflow-auto space-y-2">{results.events.length === 0 ? 'No matching events.' : results.events.map(e => <p key={e.id} className="text-sm">#{e.id} · {e.timestamp} UTC · {e.module} · {e.event_type} · {e.identity}</p>)}</div>}
    </section>
    <section className="card space-y-2"><h2 className="font-semibold">Module implementation status</h2>{modules?.modules.map(m => <p key={m.id} className="text-sm"><span className="text-brand-400">M{m.id} {m.name}</span> — {m.status}</p>)}</section>
    {user.role === 'admin' && <form className="card flex flex-wrap gap-3" onSubmit={e => { e.preventDefault(); action('/auth/users', Object.fromEntries(new FormData(e.target))) }}><h2 className="w-full font-semibold">Create dashboard account</h2><input className="input" name="username" placeholder="Username" required /><input className="input" name="password" type="password" placeholder="Password (10+ characters)" minLength={10} required /><select className="input" name="role"><option value="viewer">Viewer</option><option value="security_staff">Security staff</option><option value="admin">Administrator</option></select><button disabled={busy} className="btn-primary">Create account</button></form>}
  </div>
}
