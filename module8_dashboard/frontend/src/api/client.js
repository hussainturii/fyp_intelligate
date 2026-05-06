/**
 * client.js — IntelliGate Dashboard API Client
 * All fetch calls to the FastAPI backend go through here.
 */

const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const err = await res.text()
    throw new Error(err || `HTTP ${res.status}`)
  }
  return res.json()
}

// ── Stats ──────────────────────────────────────────────────────────────────
export const getStats = () => request('/stats')

// ── Events ────────────────────────────────────────────────────────────────
export const getEvents = (params = {}) => {
  const qs = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v != null)
  ).toString()
  return request(`/events${qs ? '?' + qs : ''}`)
}

export const logEvent = (body) =>
  request('/events', { method: 'POST', body: JSON.stringify(body) })

// ── Alerts ────────────────────────────────────────────────────────────────
export const getAlerts = (resolved = false) =>
  request(`/alerts?resolved=${resolved}`)

export const createAlert = (body) =>
  request('/alerts', { method: 'POST', body: JSON.stringify(body) })

export const resolveAlert = (id) =>
  request(`/alerts/${id}/resolve`, { method: 'PUT' })

// ── Faces ─────────────────────────────────────────────────────────────────
export const listFaces    = () => request('/faces')
export const deleteFace   = (name) => request(`/faces/${encodeURIComponent(name)}`, { method: 'DELETE' })

// ── Vehicles ──────────────────────────────────────────────────────────────
export const listVehicles  = () => request('/vehicles')
export const deleteVehicle = (plate) => request(`/vehicles/${encodeURIComponent(plate)}`, { method: 'DELETE' })

// ── Cameras ───────────────────────────────────────────────────────────────
export const listCameras  = () => request('/cameras')
export const startCamera  = (id) => request(`/cameras/${id}/start`, { method: 'POST' })
export const stopCamera   = (id) => request(`/cameras/${id}/stop`,  { method: 'POST' })
// MJPEG stream must bypass the Vite proxy — connect directly to FastAPI.
// The Vite proxy buffers responses, which breaks multipart/x-mixed-replace streaming.
const STREAM_BASE = 'http://localhost:8000/api'
export const streamUrl = (id) => `${STREAM_BASE}/stream/${id}`

// ── Health ────────────────────────────────────────────────────────────────
export const getHealth = () => request('/health')
