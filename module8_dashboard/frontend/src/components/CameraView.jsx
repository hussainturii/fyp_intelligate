import { useEffect, useState } from 'react'
import { listCameras, uploadVehicleVideo, stopCamera, streamUrl, request } from '../api/client'

const MODES = { raw: 'Raw playback', face: 'Face candidates (manual liveness review)', vehicle: 'Vehicle detection / OCR', scene: 'Queue / obstruction / dwell rules', low_light: 'RGB low-light enhancement', thermal: 'Genuine infrared input (unvalidated detector)' }

export default function CameraView({ user }) {
  const [sources, setSources] = useState({})
  const [mode, setMode] = useState('face')
  const [cameraId, setCameraId] = useState(0)
  const [file, setFile] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [version, setVersion] = useState(0)
  const [metrics, setMetrics] = useState({})
  const [samples, setSamples] = useState([])
  const [sample, setSample] = useState('local_vehicle')
  const [remoteUrl, setRemoteUrl] = useState('')
  const [remoteToken, setRemoteToken] = useState('')
  async function load() {
    try { setSources(await listCameras()); const status = await request('/modules'); setMetrics(status.pipelines) }
    catch (e) { setError(e.message) }
  }
  useEffect(() => { load(); request('/demo/sources').then(setSamples).catch(e => setError(e.message)); const timer=setInterval(load,2000); return () => clearInterval(timer) }, [])
  async function upload(e) {
    e.preventDefault(); setBusy(true); setError('')
    try { await uploadVehicleVideo(0,file,mode); setVersion(v => v+1); await load() }
    catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  async function connectRemote(e) {
    e.preventDefault(); setBusy(true); setError('')
    try {
      await request(`/cameras/${cameraId}/network`, {
        method: 'POST',
        body: JSON.stringify({ url: remoteUrl, token: remoteToken, mode })
      })
      setVersion(v => v+1); await load()
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  return <div className="p-6 space-y-6">
    <h1 className="text-2xl font-bold">Live camera and video replay</h1>
    <p className="text-gray-400">One source at a time on CPU. Capture runs independently of inference. Known faces are green; unknown faces are red. Recognition is cached while each person remains tracked.</p>
    <div className="card flex flex-wrap gap-3 items-end"><label>Webcam index<input type="number" min="0" max="7" className="input block w-24" value={cameraId} onChange={e=>setCameraId(Number(e.target.value))} /></label><label>Processing mode<select className="input block" value={mode} onChange={e=>setMode(e.target.value)}>{Object.entries(MODES).map(([id,label])=><option key={id} value={id}>{label}</option>)}</select></label><button className="btn-primary" disabled={busy || user?.role==='viewer'} onClick={async()=>{setBusy(true);setError('');try{await request(`/cameras/${cameraId}/start?mode=${mode}`,{method:'POST'});setVersion(v=>v+1);await load()}catch(e){setError(e.message)}finally{setBusy(false)}}}>{busy?'Opening…':'Open local camera'}</button><p className="text-xs text-gray-400 w-full">Index 0 is usually the built-in webcam. The browser displays the backend camera stream; it does not request browser camera permission. Close other camera apps if opening fails. Gate control remains simulated.</p></div>
    <form onSubmit={connectRemote} className="card flex flex-wrap gap-3 items-end">
      <label className="flex-1 min-w-[240px]">Remote / Laptop Stream URL (LAN)<input className="input block w-full" placeholder="http://192.168.1.15:8080/stream" required value={remoteUrl} onChange={e => setRemoteUrl(e.target.value)} /></label>
      <label className="w-48">Auth Token<input className="input block w-full" placeholder="Token from laptop script" value={remoteToken} onChange={e => setRemoteToken(e.target.value)} /></label>
      <label>Processing mode<select className="input block" value={mode} onChange={e => setMode(e.target.value)}>{Object.entries(MODES).map(([id,label]) => <option key={id} value={id}>{label}</option>)}</select></label>
      <button className="btn-primary" disabled={busy || !remoteUrl || user?.role==='viewer'}>{busy ? 'Connecting…' : 'Connect remote camera'}</button>
      <p className="text-xs text-gray-400 w-full">Streams live webcam from your laptop via tools/laptop_cam_streamer.py over LAN. Main PC executes all AI detection, recognition, PAD and tracking.</p>
    </form>
    <form onSubmit={upload} className="card flex flex-wrap gap-3 items-end">
      <label>Video (up to 250 MB)<input type="file" className="input block" accept=".mp4,.avi,.mov,.mkv,.webm" required onChange={e => setFile(e.target.files[0])} /></label>
      <label>Processing mode<select className="input block" value={mode} onChange={e => setMode(e.target.value)}>{Object.entries(MODES).map(([id,label]) => <option key={id} value={id}>{label}</option>)}</select></label>
      <button disabled={busy || !file || user?.role==='viewer'} className="btn-primary">{busy ? 'Loading pipeline…' : 'Upload and replay'}</button>
    </form>
    {samples.length > 0 && <div className="card flex flex-wrap gap-3 items-center"><select className="input" value={sample} onChange={e => setSample(e.target.value)}>{samples.map(s => <option key={s.id} value={s.id}>{s.label}</option>)}</select><button disabled={busy} className="btn-ghost" onClick={async () => { setBusy(true); setError(''); try { await request('/demo/replay', { method:'POST', body:JSON.stringify({source:sample,mode}) }); setVersion(v=>v+1); await load() } catch(e) { setError(e.message) } finally { setBusy(false) } }}>Replay sample in selected mode</button><p className="text-xs text-gray-400 w-full">LLVIP samples are official illustration material, not an annotated evaluation split. Static face samples cannot validate liveness.</p></div>}
    {mode === 'thermal' && <p className="text-amber-300">Use genuine infrared/thermal recordings. RGB false-colour images do not validate thermal performance. This detector was trained on RGB.</p>}
    {error && <div className="card text-red-400" role="alert">{error}</div>}
    {Object.entries(sources).map(([id, source]) => <section key={id} className="card space-y-3">
      <div className="flex justify-between"><h2>Source {id} · {source.running ? source.remote ? 'Remote camera (LAN)' : source.replayed ? 'Replaying' : 'Live camera' : 'Stopped / finished'}</h2><button className="btn-danger" disabled={user?.role==='viewer'} onClick={async () => { try { await stopCamera(id); await load() } catch(e) { setError(e.message) } }}>Stop source</button></div>
      <img key={`${id}-${version}`} src={streamUrl(id)} alt="Live or recorded video with recognition overlays" className="w-full max-w-3xl rounded-lg bg-black" />
      <p className="text-sm text-gray-400">Position {source.source_time_seconds}s · capture/preview {source.fps} FPS · detector {source.inference_fps} FPS (cap {source.inference_sampling_fps}) · last inference {metrics[id]?.latency_ms ?? '—'} ms</p>
      <p className="text-sm text-gray-400">Visible tracks {metrics[id]?.visible_tracks ?? 0} · identity/OCR runs {metrics[id]?.recognition_runs ?? 0} · cache hits {metrics[id]?.recognition_cache_hits ?? 0}</p>
      <p className={source.processing_status==='error' || source.processing_status==='stalled' ? 'text-red-400':'text-gray-300'}>Active mode: {source.processing_mode} · recognition worker: {source.processing_status} · processed frames: {source.processed_frames ?? 0}</p>
      {source.processing_mode==='raw' && <p className="text-amber-300">Raw preview does not recognize faces. Select Face mode and reopen the camera.</p>}
      {source.processing_mode==='face' && source.processing_status==='running' && <p className="text-sm text-gray-300">Faces detected: {metrics[id]?.faces_detected ?? 0} · known: {metrics[id]?.recognized_faces ?? 0} · unknown: {metrics[id]?.unknown_faces ?? 0} · too small: {metrics[id]?.pending_faces ?? 0}{metrics[id]?.pad_attack_suspected > 0 && <span className="text-red-400 font-semibold ml-2">· Spoof suspected: {metrics[id].pad_attack_suspected}</span>}{metrics[id]?.pad_inconclusive > 0 && <span className="text-yellow-400 ml-2">· Inconclusive: {metrics[id].pad_inconclusive}</span>}</p>}
      {source.processing_mode==='face' && source.processing_status==='running' && metrics[id]?.faces_detected===0 && <p className="text-amber-300">The detector is running but sees no face. Try a clearer frontal view, better lighting or move closer.</p>}
      {source.processing_status==='error' && !source.last_error && <p className="text-red-400">Recognition worker stopped unexpectedly. Stop this source and reopen it.</p>}
      {(source.last_error || metrics[id]?.last_error) && <p className="text-red-400">Pipeline error: {source.last_error || metrics[id]?.last_error}</p>}
    </section>)}
    {Object.keys(sources).length === 0 && <p className="card text-gray-400">Open your webcam or upload a video to start.</p>}
  </div>
}
