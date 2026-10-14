import { useEffect, useState } from 'react'
import { listFaces, deleteFace, request } from '../api/client'

export default function FaceDatabase({ user }) {
  const [data, setData] = useState({ persons: {}, count: 0 })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  async function registerCamera(e) {
    e.preventDefault();setBusy(true);setError('')
    try { const fields=Object.fromEntries(new FormData(e.target));fields.camera_id=Number(fields.camera_id);await request('/faces/register-camera',{method:'POST',body:JSON.stringify(fields)});e.target.reset();await load() }
    catch(e) {setError(e.message)}finally{setBusy(false)}
  }
  async function load() { try { setData(await listFaces()) } catch(e) { setError(e.message) } }
  useEffect(() => { load() }, [])
  async function register(e) {
    e.preventDefault(); setBusy(true); setError('')
    try { await request('/faces/register', { method:'POST', body:new FormData(e.target) }); e.target.reset(); await load() }
    catch(e) { setError(e.message) } finally { setBusy(false) }
  }
  return <div className="p-6 space-y-6">
    <h1 className="text-2xl font-bold">Face enrolment</h1>
    <p className="text-gray-400">SFace CPU embeddings are encrypted at rest. These records use a separate gallery from the legacy ViT model. Recognition requires manual access review.</p>
    {error && <div className="card text-red-400" role="alert">{error}</div>}
    {user.role === 'admin' && <form onSubmit={register} className="card flex flex-wrap items-end gap-3">
      <label>Name<input name="name" className="input block" required maxLength={100} /></label>
      <label>Member role<select name="role" className="input block"><option>resident</option><option>employee</option><option>security_staff</option></select></label>
      <label>One clear face image<input name="image" type="file" accept="image/*" className="input block" required /></label>
      <button disabled={busy} className="btn-primary">{busy ? 'Enrolling…' : 'Enrol face'}</button>
    </form>}
    {user.role==='admin' && <form onSubmit={registerCamera} className="card flex flex-wrap gap-3 items-end"><h2 className="w-full font-semibold">Enrol from the current camera frame</h2><p className="w-full text-sm text-gray-400">Open a camera first. Only the person being enrolled should be visible. Use a clear frontal view; enrol each person separately.</p><label>Name<input name="name" className="input block" required maxLength={100}/></label><label>Role<select name="role" className="input block"><option>resident</option><option>employee</option><option>security_staff</option></select></label><label>Source index<input name="camera_id" type="number" min="0" max="7" defaultValue="0" className="input block w-24"/></label><button className="btn-primary" disabled={busy}>{busy?'Enrolling…':'Enrol current frame'}</button></form>}
    <button onClick={load} className="btn-ghost">Refresh · {data.count} registered</button>
    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">{Object.entries(data.persons).map(([name,person]) => <div key={name} className="card space-y-2"><h2 className="font-semibold">{name}</h2><p>{person.role} · {person.model}</p><p className="text-sm text-gray-400">Enrolled {new Date(person.registered_at).toLocaleString()}</p>{user.role === 'admin' && <button className="btn-danger" onClick={async () => { if (!window.confirm(`Remove ${name}?`)) return; try { await deleteFace(name); await load() } catch(e) { setError(e.message) } }}>Remove</button>}</div>)}</div>
  </div>
}
