import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './components/Dashboard'
import CameraView from './components/CameraView'
import EventLog from './components/EventLog'
import FaceDatabase from './components/FaceDatabase'
import VehicleDatabase from './components/VehicleDatabase'
import Alerts from './components/Alerts'
import { useWebSocket } from './api/useWebSocket'

export default function App() {
  // Single WebSocket connection shared across all pages via context
  const ws = useWebSocket()

  return (
    <Layout ws={ws}>
      <Routes>
        <Route path="/"           element={<Dashboard ws={ws} />} />
        <Route path="/cameras"    element={<CameraView ws={ws} />} />
        <Route path="/events"     element={<EventLog ws={ws} />} />
        <Route path="/faces"      element={<FaceDatabase />} />
        <Route path="/vehicles"   element={<VehicleDatabase />} />
        <Route path="/alerts"     element={<Alerts ws={ws} />} />
        <Route path="*"           element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}
