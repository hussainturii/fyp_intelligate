import { useState, useEffect, useRef } from 'react'
import { Camera, Play, Square, RefreshCw, Maximize2, Loader } from 'lucide-react'
import { startCamera, stopCamera } from '../api/client'

// Stream goes DIRECTLY to FastAPI — bypasses Vite proxy which buffers MJPEG
const BACKEND = 'http://localhost:8000'

function CameraCard({ camId, onRemove }) {
  const [status,   setStatus]   = useState('connecting') // connecting | live | offline
  const [fullscreen, setFullscreen] = useState(false)
  const [retryKey, setRetryKey] = useState(0)
  const timerRef = useRef(null)

  const src = `${BACKEND}/api/stream/${camId}?t=${retryKey}`

  // Give the stream 6 seconds to show first frame before declaring offline
  useEffect(() => {
    setStatus('connecting')
    timerRef.current = setTimeout(() => {
      setStatus(s => s === 'connecting' ? 'offline' : s)
    }, 6000)
    return () => clearTimeout(timerRef.current)
  }, [retryKey])

  const handleLoad = () => {
    clearTimeout(timerRef.current)
    setStatus('live')
  }

  const handleError = () => {
    clearTimeout(timerRef.current)
    setStatus('offline')
  }

  const retry = () => {
    setRetryKey(k => k + 1)
  }

  return (
    <>
      <div className="card flex flex-col gap-3">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Camera size={16} className="text-brand-400" />
            <span className="font-medium text-white text-sm">Camera {camId}</span>
          </div>
          <div className="flex items-center gap-1">
            {status === 'live' && (
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-green-900/40 text-green-400 border border-green-800">
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                Live
              </span>
            )}
            {status === 'connecting' && (
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-900/40 text-yellow-400 border border-yellow-800">
                Connecting…
              </span>
            )}
            {status === 'offline' && (
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-red-900/40 text-red-400 border border-red-800">
                Offline
              </span>
            )}
            {status === 'live' && (
              <button
                onClick={() => setFullscreen(true)}
                className="btn-ghost p-1.5 rounded-lg ml-1"
                title="Fullscreen"
              >
                <Maximize2 size={14} />
              </button>
            )}
            <button onClick={retry} className="btn-ghost p-1.5 rounded-lg" title="Retry">
              <RefreshCw size={13} />
            </button>
          </div>
        </div>

        {/* Stream */}
        <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
          {/* Always render the img — browser will stream MJPEG automatically */}
          <img
            key={retryKey}
            src={src}
            alt={`Camera ${camId}`}
            className="w-full h-full object-cover"
            onLoad={handleLoad}
            onError={handleError}
            style={{ display: status === 'live' ? 'block' : 'none' }}
          />

          {/* Overlay for non-live states */}
          {status !== 'live' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-gray-500">
              {status === 'connecting' ? (
                <>
                  <Loader size={28} className="animate-spin text-brand-400" />
                  <p className="text-xs text-gray-400">Connecting to camera…</p>
                  <p className="text-xs text-gray-600 font-mono">{BACKEND}/api/stream/{camId}</p>
                </>
              ) : (
                <>
                  <Camera size={28} className="text-gray-600" />
                  <p className="text-xs text-gray-400">Stream unavailable</p>
                  <button
                    onClick={retry}
                    className="text-xs text-brand-400 hover:text-brand-300 border border-brand-600/40 px-3 py-1 rounded-lg"
                  >
                    Retry
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        {/* Footer info */}
        <div className="text-xs text-gray-600 font-mono">
          {BACKEND}/api/stream/{camId}
        </div>
      </div>

      {/* Fullscreen modal */}
      {fullscreen && (
        <div
          className="fixed inset-0 bg-black/95 z-50 flex items-center justify-center p-4"
          onClick={() => setFullscreen(false)}
        >
          <div className="relative max-w-5xl w-full" onClick={e => e.stopPropagation()}>
            <button
              className="absolute -top-10 right-0 text-gray-400 hover:text-white text-sm"
              onClick={() => setFullscreen(false)}
            >
              ✕ Close
            </button>
            <img
              src={src}
              alt={`Camera ${camId} fullscreen`}
              className="w-full rounded-xl"
            />
          </div>
        </div>
      )}
    </>
  )
}

// ─────────────────────────────────────────────────────────────────────────────

export default function CameraView() {
  const [camIds, setCamIds] = useState([0])   // start with cam 0
  const [addId,  setAddId]  = useState('')

  const handleAdd = async () => {
    const id = parseInt(addId, 10)
    if (isNaN(id) || camIds.includes(id)) return
    await startCamera(id).catch(() => {})
    setCamIds(prev => [...prev, id])
    setAddId('')
  }

  const handleRemove = (id) => {
    stopCamera(id).catch(() => {})
    setCamIds(prev => prev.filter(c => c !== id))
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Live Cameras</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            MJPEG streams — direct to backend on port 8000
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            type="number"
            placeholder="Camera index"
            value={addId}
            onChange={e => setAddId(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleAdd()}
            className="input w-32"
            min="0"
          />
          <button onClick={handleAdd} className="btn-primary">
            <Play size={14} /> Add
          </button>
        </div>
      </div>

      {/* Camera grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {camIds.map(id => (
          <CameraCard
            key={id}
            camId={id}
            onRemove={handleRemove}
          />
        ))}
      </div>

      {/* Help card */}
      <div className="card text-xs text-gray-500 space-y-1">
        <p className="text-white font-medium text-sm">Troubleshooting</p>
        <p>• Make sure the backend is running:
          <code className="ml-1 font-mono bg-surface px-1 rounded text-gray-300">
            uvicorn module8_dashboard.backend.main:app --port 8000
          </code>
        </p>
        <p>• Test the stream directly in your browser: <code className="font-mono bg-surface px-1 rounded text-gray-300">http://localhost:8000/api/stream/0</code></p>
        <p>• If the stream page shows a JPEG image, the backend is working. If it's blank, check that your camera index (0) is correct.</p>
      </div>
    </div>
  )
}
