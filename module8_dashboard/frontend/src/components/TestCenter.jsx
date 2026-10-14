import { useEffect,useState } from 'react'
import { request } from '../api/client'

const MODES={face:'Face recognition',vehicle:'Vehicles / OCR',scene:'Queue / obstruction / dwell',low_light:'Low-light enhancement',thermal:'Infrared detection (RGB-trained baseline)'}
const color=status=>['PASS','completed'].includes(status)?'text-green-400':['FAIL','error'].includes(status)?'text-red-400':'text-amber-300'

export default function TestCenter({user}) {
  const [history,setHistory]=useState([])
  const [run,setRun]=useState(null)
  const [id,setId]=useState(null)
  const [error,setError]=useState('')
  const [mode,setMode]=useState('face')
  const [busy,setBusy]=useState(false)
  async function refresh() {try{setHistory(await request('/testing/runs'))}catch(e){setError(e.message)}}
  useEffect(()=>{refresh()},[])
  useEffect(()=>{
    if(!id)return
    let disposed=false
    let timer
    const load=async()=>{try{const value=await request(`/testing/runs/${id}`);if(!disposed){setRun(value);if(['completed','error'].includes(value.status)){clearInterval(timer);refresh()}}}catch(e){if(!disposed)setError(e.message)}}
    load();timer=setInterval(load,1200)
    return()=>{disposed=true;clearInterval(timer)}
  },[id])
  async function start(path,body) {
    setBusy(true);setError('')
    try {const result=await request(path,{method:'POST',body});setRun(null);setId(result.id);await refresh()}
    catch(e){setError(e.message)}finally{setBusy(false)}
  }
  const running=busy || (id&&!run) || ['queued','running'].includes(run?.status)
  const allowed=user.role!=='viewer'
  return <div className="p-6 space-y-5">
    <h1 className="text-2xl font-bold">Test Center</h1>
    <p className="text-gray-400">Run component checks or upload your own image/video. Tests use an isolated copy of the gallery and temporary device/events state. Test observations do not flood the live event list or operate its virtual gate.</p>
    <p className="text-amber-300 text-sm">Passing a narrow check does not mean the whole module is complete. Unimplemented research features and missing accuracy data are listed explicitly. Stop live sources before timing benchmarks to avoid CPU contention.</p>
    {error&&<p className="card text-red-400" role="alert">{error}</p>}
    <section className="card space-y-3"><h2 className="font-semibold">1. Check project components</h2><p className="text-sm text-gray-400">Model smoke test, vehicle registry, visits, counting logic, safety interlocks, reports, frontend assets, virtual gate and authentication/persistence.</p><button className="btn-primary" disabled={running||!allowed} onClick={()=>start('/testing/checks')}>Run component checks</button></section>
    <form className="card flex flex-wrap gap-3 items-end" onSubmit={event=>{event.preventDefault();const fields=Object.fromEntries(new FormData(event.target));fields.camera_id=Number(fields.camera_id);if(!fields.expected_identity)delete fields.expected_identity;start('/testing/camera',JSON.stringify(fields))}}><h2 className="w-full font-semibold">Diagnose the current camera frame</h2><p className="w-full text-sm text-gray-400">Open your camera first. This runs face detection and gallery matching directly on one current frame and shows names/similarity scores, independently of the live recognition worker.</p><label>Source index<input className="input block w-24" name="camera_id" type="number" min="0" max="7" defaultValue="0"/></label><label>Expected name (optional)<input className="input block" name="expected_identity" placeholder="Your registered name" maxLength={100}/></label><button className="btn-primary" disabled={running||!allowed}>Test current camera frame</button></form>
    <form className="card flex flex-wrap gap-3 items-end" onSubmit={event=>{event.preventDefault();const data=new FormData(event.target);for(const key of ['expected_identity','expected_count'])if(!data.get(key))data.delete(key);start('/testing/runs',data)}}>
      <h2 className="w-full font-semibold">2. Test with your own data</h2>
      <label>Module<select className="input block" name="mode" value={mode} onChange={e=>setMode(e.target.value)}>{Object.entries(MODES).map(([key,label])=><option key={key} value={key}>{label}</option>)}</select></label>
      <label>Image or video<input className="input block" name="file" type="file" accept=".jpg,.jpeg,.png,.webp,.bmp,.mp4,.avi,.mov,.mkv,.webm" required/></label>
      <label>Maximum sampled frames<input className="input block w-28" name="max_frames" type="number" defaultValue="30" min="1" max="300"/></label>
      {mode==='face'&&<label>Expected name (optional)<input className="input block" name="expected_identity" placeholder="Registered name, or Unknown" maxLength={100}/></label>}
      {mode!=='low_light'&&<label>Expected count per frame (optional)<input className="input block w-40" name="expected_count" type="number" min="0" max="100"/></label>}
      <button className="btn-primary" disabled={running||!allowed}>Run data test</button>
      <p className="w-full text-sm text-gray-400">Use a separate photo/video from enrolment. Video samples run at roughly 5 frames per source second from the start. Expected counts apply to every sampled frame; leave blank for varying scenes. Use at least 180 samples of a long enough scene clip to test the default 30-second dwell rule. Without expected results the verdict is OBSERVED, not PASS.</p>
    </form>
    {run&&<section className="card space-y-4">
      <h2 className={`font-semibold ${color(run.status)}`}>{run.kind} · {run.status} {run.verdict&&`· ${run.verdict}`}</h2>
      {run.error&&<p className="text-red-400" role="alert">{run.error}</p>}
      {run.scope&&<p className="text-sm text-gray-400">{run.scope}</p>}
      {run.checks&&<div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr><th className="text-left p-2">Module / check</th><th className="text-left p-2">Result</th><th className="text-left p-2">Evidence / limit</th></tr></thead><tbody>{run.checks.map((check,index)=><tr key={index} className="border-t"><td className="p-2">{check.module} · {check.name}</td><td className={`p-2 ${color(check.status)}`}>{check.status}</td><td className="p-2 text-gray-400">{check.detail}</td></tr>)}</tbody></table></div>}
      {run.frames_processed&&<><p>Processed {run.frames_processed} frames · gallery entries {run.gallery_count??'n/a'} · events {run.event_count} · median {run.timing?.median_ms} ms · p95 {run.timing?.p95_ms} ms</p><p>Recognition runs {run.metrics?.recognition_runs} · cache hits {run.metrics?.recognition_cache_hits}</p></>}
      {run.preview_url&&<img className="max-w-full rounded-lg" src={run.preview_url} alt="Annotated last frame of the supplied test data"/>}
      {run.assertions?.map((check,index)=><p key={index} className={color(check.status)}>{check.status}: {check.name} · expected {String(check.expected)} · observed {JSON.stringify(check.observed)}</p>)}
      {run.sampled_frames&&<div className="max-h-64 overflow-auto space-y-1 text-sm font-mono">{run.sampled_frames.map((frame,index)=><p key={index}>{frame.source_seconds}s · faces {frame.faces_detected} · objects in ROI {frame.count} · obstruction {String(frame.obstruction)} {frame.faces.map(face=>`${face.label} (${face.similarity})`).join(', ')}</p>)}</div>}
      {run.limitations?.map(item=><p className="text-amber-300 text-sm" key={item}>{item}</p>)}
      <button className="btn-ghost" onClick={()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(run,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=`intelligate-test-${run.id}.json`;link.click();URL.revokeObjectURL(url)}}>Download full JSON evidence</button>
    </section>}
    <section className="card space-y-2"><h2 className="font-semibold">Recent test runs</h2>{history.map(item=><button className="btn-ghost block text-left" key={item.id} onClick={()=>{setId(item.id);setRun(null)}}>{item.created_at} · {item.kind} {item.mode} · {item.status} {item.verdict}</button>)}</section>
  </div>
}
