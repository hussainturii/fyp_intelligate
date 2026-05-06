import { useEffect, useState } from 'react'
import { Car, Trash2, RefreshCw } from 'lucide-react'
import { listVehicles, deleteVehicle } from '../api/client'
import clsx from 'clsx'

export default function VehicleDatabase() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [search,  setSearch]  = useState('')
  const [confirm, setConfirm] = useState(null)

  const load = () => {
    setLoading(true)
    listVehicles()
      .then(setData)
      .catch(() => setData({ count: 0, vehicles: {} }))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const handleDelete = async (plate) => {
    await deleteVehicle(plate).catch(() => {})
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

      {/* Register tip */}
      <div className="card bg-brand-600/10 border-brand-600/30">
        <p className="text-sm text-brand-400 font-medium mb-1">Register a new vehicle</p>
        <p className="text-xs text-gray-400 mb-2">
          Run from your terminal inside the{' '}
          <code className="font-mono bg-surface px-1 rounded">intelligate/</code> directory:
        </p>
        <pre className="text-xs font-mono bg-surface rounded-lg p-3 text-green-400 overflow-x-auto">
{`# With plate number only:
python -m module2_vehicle.main --mode register --plate "ABC-1234" --owner "Hussain"

# With image file:
python -m module2_vehicle.main --mode register --plate "ABC-1234" --owner "Hussain" --image car.jpg`}
        </pre>
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
                        <button
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
