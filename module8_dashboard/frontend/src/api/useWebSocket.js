/**
 * useWebSocket.js — React hook for the IntelliGate live event WebSocket.
 *
 * Automatically connects, reconnects on drop, and exposes the latest
 * events array + system health to consuming components.
 *
 * Usage:
 *   const { events, health, connected } = useWebSocket()
 */

import { useEffect, useRef, useState, useCallback } from 'react'

const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`
const MAX_EVENTS = 200   // keep last 200 events in memory

export function useWebSocket() {
  const [connected, setConnected]   = useState(false)
  const [events,    setEvents]      = useState([])
  const [alerts,    setAlerts]      = useState([])
  const [health,    setHealth]      = useState(null)
  const wsRef     = useRef(null)
  const retryRef  = useRef(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      clearInterval(retryRef.current)
      // Start keep-alive ping every 20s
      retryRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping')
      }, 20_000)
    }

    ws.onclose = () => {
      setConnected(false)
      clearInterval(retryRef.current)
      // Reconnect after 3s
      setTimeout(connect, 3_000)
    }

    ws.onerror = () => {
      ws.close()
    }

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        handleMessage(msg)
      } catch (_) {}
    }
  }, [])

  function handleMessage(msg) {
    const { type, timestamp, data } = msg

    if (type === 'face_event' || type === 'vehicle_event') {
      setEvents(prev => [
        { ...data, type, timestamp, id: Date.now() + Math.random() },
        ...prev.slice(0, MAX_EVENTS - 1),
      ])
    } else if (type === 'alert') {
      setAlerts(prev => [
        { ...data, timestamp, id: Date.now() + Math.random() },
        ...prev.slice(0, 50),
      ])
    } else if (type === 'system') {
      setHealth(data)
    } else if (type === 'init') {
      // Initial stats snapshot — ignore for now (pages fetch their own)
    }
  }

  useEffect(() => {
    connect()
    return () => {
      clearInterval(retryRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { connected, events, alerts, health }
}
