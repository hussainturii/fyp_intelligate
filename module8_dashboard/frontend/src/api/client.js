/**
 * client.js — IntelliGate Dashboard API Client
 * All fetch calls to the FastAPI backend go through here.
 */

const BASE = '/api'

export async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    credentials: 'same-origin',
    headers: { ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...options.headers },
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

export async function uploadVehicleVideo(id, file, mode = 'vehicle') {
  const formData = new FormData()
  formData.append('video', file)
  formData.append('mode', mode)

  const res = await fetch(`${BASE}/cameras/${id}/upload-video`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const err = await res.text()
    throw new Error(err || `HTTP ${res.status}`)
  }

  return res.json()
}

// Same-origin streaming keeps HttpOnly session authentication available.
const STREAM_BASE = '/api'
export const streamUrl = (id) => `${STREAM_BASE}/stream/${id}`

// ── Health ────────────────────────────────────────────────────────────────
export const getHealth = () => request('/health')
