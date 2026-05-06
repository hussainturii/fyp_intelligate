import { useEffect, useState } from 'react'
import { Users, Trash2, RefreshCw, UserCheck, UserX } from 'lucide-react'
import { listFaces, deleteFace } from '../api/client'
import clsx from 'clsx'

export default function FaceDatabase() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [search,  setSearch]  = useState('')
  const [confirm, setConfirm] = useState(null)   // name to confirm delete

  const load = () => {
    setLoading(true)
    listFaces()
      .then(setData)
      .catch(() => setData({ count: 0, persons: {} }))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const handleDelete = async (name) => {
    await deleteFace(name).catch(() => {})
    setConfirm(null)
    load()
  }

  const persons   = data?.persons ?? {}
  const names     = Object.keys(persons).filter(n =>
    n.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Face Database</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            {data?.count ?? '…'} registered person{data?.count !== 1 ? 's' : ''}
          </p>
        </div>
        <button onClick={load} className="btn-ghost">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Register tip */}
      <div className="card bg-brand-600/10 border-brand-600/30">
        <p className="text-sm text-brand-400 font-medium mb-1">Register a new person</p>
        <p className="text-xs text-gray-400">
          Run from your terminal inside the{' '}
          <code className="font-mono bg-surface px-1 rounded">intelligate/</code> directory:
        </p>
        <pre className="mt-2 text-xs font-mono bg-surface rounded-lg p-3 text-green-400 overflow-x-auto">
          python -m module1_face_recognition.main --mode register --name "Hussain"
        </pre>
      </div>

      {/* Search */}
      <div className="flex gap-3">
        <input
          placeholder="Search by name…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="input flex-1"
        />
      </div>

      {/* Person cards */}
      {loading && <p className="text-gray-500 text-sm">Loading…</p>}

      {!loading && names.length === 0 && (
        <div className="card text-center py-12">
          <Users size={40} className="mx-auto text-gray-600 mb-3" />
          <p className="text-gray-500">
            {search ? 'No results for your search.' : 'No persons registered yet.'}
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {names.map(name => {
          const meta = persons[name] ?? {}
          return (
            <div key={name} className="card flex flex-col gap-3">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-brand-600/20 flex items-center justify-center text-brand-400 font-bold text-lg">
                  {name[0]?.toUpperCase()}
                </div>
                <div className="min-w-0">
                  <p className="font-semibold text-white truncate">{name}</p>
                  <p className="text-xs text-gray-500">
                    {meta.registered_at
                      ? new Date(meta.registered_at).toLocaleDateString()
                      : 'Registered'}
                  </p>
                </div>
              </div>

              <div className="text-xs text-gray-500 space-y-1 border-t border-surface-border pt-2">
                <div className="flex justify-between">
                  <span>Embeddings</span>
                  <span className="text-white font-mono">{meta.n_embeddings ?? '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Access count</span>
                  <span className="text-white font-mono">{meta.access_count ?? '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Last seen</span>
                  <span className="text-white">{meta.last_seen ? new Date(meta.last_seen).toLocaleDateString() : '—'}</span>
                </div>
              </div>

              {confirm === name ? (
                <div className="flex gap-2 mt-auto">
                  <button
                    onClick={() => handleDelete(name)}
                    className="btn-danger flex-1 text-xs py-1.5"
                  >
                    Confirm Delete
                  </button>
                  <button
                    onClick={() => setConfirm(null)}
                    className="btn-ghost flex-1 text-xs py-1.5"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setConfirm(name)}
                  className="btn-ghost text-red-400 hover:text-red-300 text-xs mt-auto flex items-center gap-1"
                >
                  <Trash2 size={13} /> Remove
                </button>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
