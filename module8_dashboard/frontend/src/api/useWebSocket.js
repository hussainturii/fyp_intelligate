import { useEffect, useState } from 'react'

export function useWebSocket() {
  const [connected,setConnected]=useState(false)
  const [events,setEvents]=useState([])
  const [alerts,setAlerts]=useState([])
  const [health,setHealth]=useState(null)
  useEffect(() => {
    let disposed=false, socket, retry, ping
    function connect() {
      if(disposed) return
      socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`)
      socket.onopen=() => { setConnected(true); ping=setInterval(() => { if(socket.readyState===WebSocket.OPEN) socket.send('ping') },20000) }
      socket.onclose=() => { clearInterval(ping); if(!disposed) { setConnected(false); retry=setTimeout(connect,3000) } }
      socket.onerror=() => socket.close()
      socket.onmessage=e => {
        try {
          const {type,data,timestamp}=JSON.parse(e.data)
          if(['face_event','vehicle_event','event'].includes(type)) setEvents(prev=>[{...data,type,timestamp,id:data.event_id},...prev].slice(0,200))
          if(type==='alert') setAlerts(prev=>[{...data,timestamp,id:data.alert_id},...prev].slice(0,50))
          if(type==='system') setHealth(data)
        } catch (_) { /* Ignore malformed messages without breaking reconnection. */ }
      }
    }
    connect()
    return () => { disposed=true; clearTimeout(retry); clearInterval(ping); socket?.close() }
  },[])
  return {connected,events,alerts,health}
}
