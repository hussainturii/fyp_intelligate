import { useEffect, useState } from 'react'
import { Car, Trash2, RefreshCw } from 'lucide-react'
import { listVehicles, deleteVehicle, request } from '../api/client'
import clsx from 'clsx'

export default function VehicleDatabase({ user }) {
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  async function register(event) {
    event.preventDefault(); setBusy(true); setError('')
    try { await request('/vehicles/register', { method: 'POST', body: new FormData(event.target) }); event.target.reset(); load() }
    catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [search,  setSearch]  = useState('')
  const [confirm, setConfirm] = useState(null)

  const load = () => {
    setLoading(true)
    listVehicles()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const handleDelete = async (plate) => {
    try { await deleteVehicle(plate) } catch(e) { setError(e.message); return }
    setConfirm(null)
    load()
  }

  const vehicles = data?.vehicles ?? {}
  const plates   = Object.keys(vehicles).filter(p =>
    p.toLowerCase().includes(search.toLowerCase()) ||
    (vehicles[p]?.owner ?? '').toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Vehicle Database</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            {data?.count ?? '…'} registered vehicle{data?.count !== 1 ? 's' : ''}
          </p>
        </div>
        <button onClick={load} className="btn-ghost">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {error && <p role="alert" className="text-red-400">{error}</p>}
      {user.role === 'admin' && <form onSubmit={register} className="card flex flex-wrap gap-3 items-end"><label>Plate<input className="input block" name="plate" required maxLength={20} /></label><label>Owner<input className="input block" name="owner" required maxLength={100} /></label><label>Cropped vehicle image (optional)<input className="input block" name="image" type="file" accept="image/*" /></label><button className="btn-primary" disabled={busy}>{busy ? 'Registering…' : 'Register vehicle'}</button></form>}
      {/* Appearance matching is a review-only baseline. */}
      <div className="card bg-brand-600/10 border-brand-600/30">
        <p className="text-sm text-brand-400 font-medium mb-1">CPU vehicle registry</p>
        <p className="text-xs text-gray-400 mb-2">
          Register through the form above. Plate matching uses OCR; appearance candidates use a colour-histogram baseline and require review.
        </p>
      </div>

      {/* Search */}
      <input
        placeholder="Search by plate or owner…"
        value={search}
        onChange={e => setSearch(e.target.value)}
        className="input w-full max-w-sm"
      />

      {loading && <p className="text-gray-500 text-sm">Loading…</p>}

      {!loading && plates.length === 0 && (
        <div className="card text-center py-12">
          <Car size={40} className="mx-auto text-gray-600 mb-3" />
          <p className="text-gray-500">
            {search ? 'No results.' : 'No vehicles registered yet.'}
          </p>
        </div>
      )}

      {/* Vehicle table */}
      {plates.length > 0 && (
        <div className="card overflow-hidden p-0">
          <table className="w-full text-sm">
            <thead className="border-b border-surface-border">
              <tr className="text-xs text-gray-500 uppercase">
                <th className="text-left py-3 px-4">Plate</th>
                <th className="text-left py-3 px-4">Owner</th>
                <th className="text-left py-3 px-4">Category</th>
                <th className="text-left py-3 px-4">Registered</th>
                <th className="text-left py-3 px-4">Last Seen</th>
                <th className="text-left py-3 px-4">Visits</th>
                <th className="py-3 px-4"></th>
              </tr>
            </thead>
            <tbody>
              {plates.map(plate => {
                const v = vehicles[plate] ?? {}
                return (
                  <tr key={plate} className="table-row">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-7 h-7 rounded-md bg-blue-900/40 flex items-center justify-center">
                          <Car size={14} className="text-blue-400" />
                        </div>
                        <span className="font-mono font-semibold text-white">{plate}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-gray-300">{v.owner ?? '—'}</td>
                    <td className="py-3 px-4">
                      <span className="badge badge-blue">{v.category ?? 'authorized'}</span>
                    </td>
                    <td className="py-3 px-4 text-xs text-gray-500">
                      {v.registered_at ? new Date(v.registered_at).toLocaleDateString() : '—'}
                    </td>
                    <td className="py-3 px-4 text-xs text-gray-500">
                      {v.last_seen ? new Date(v.last_seen).toLocaleDateString() : '—'}
                    </td>
                    <td className="py-3 px-4 text-xs font-mono text-center text-gray-300">
                      {v.visit_count ?? '—'}
                    </td>
                    <td className="py-3 px-4">
                      {confirm === plate ? (
                        <div className="flex gap-1">
                          <button
                            onClick={() => handleDelete(plate)}
                            className="text-xs text-red-400 hover:text-red-300 font-medium"
                          >
                            Confirm
                          </button>
                          <span className="text-gray-600">/</span>
                          <button
                            onClick={() => setConfirm(null)}
                            className="text-xs text-gray-400 hover:text-white"
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button disabled={user.role !== 'admin'}
                          onClick={() => setConfirm(plate)}
                          className="text-gray-600 hover:text-red-400 transition-colors"
                          title="Delete"
                        >
                          <Trash2 size={14} />
                        </button>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
