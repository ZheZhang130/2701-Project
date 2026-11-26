import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

async function api(path, options) {
  const response = await fetch('/api' + path, options);
  if (!response.ok) { const body = await response.json(); throw new Error(body.detail || 'Request failed'); }
  return response.json();
}
const post = (path) => api(path, {method:'POST'});
const time = value => new Date(value).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
function App() {
  const [data,setData] = useState({drivers:[],notifications:[],rule:{window:10,threshold:5}});
  const [selected,setSelected] = useState(null), [driver,setDriver] = useState(null);
  const [health,setHealth] = useState(false), [error,setError] = useState(''), [name,setName] = useState('');
  const [tab,setTab] = useState('all'), [bell,setBell] = useState(false), [busy,setBusy] = useState(false);
  const [source,setSource] = useState('images'), [videoUrl,setVideoUrl] = useState(''), [running,setRunning] = useState(false);
  const [interval,setIntervalSeconds] = useState(1), [rule,setRule] = useState({window:10,threshold:5});
  const video = useRef(), stream = useRef(), active = useRef(false), generation = useRef(0), selectedRef = useRef(null);
  const refresh = async () => {const d=await api('/dashboard'); setData(d); return d;};
  const loadDriver = async id => {const d=await api('/drivers/'+id); if(selectedRef.current===id) setDriver(d);};
  const attempt = async fn => {try {setError(''); await fn();} catch(e) {setError(e.message);}};
  useEffect(() => {
    let socket, retry, disposed = false;
    function connect() {
      socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/demo/heartbeat`);
      socket.onmessage = () => {}; // Server keeps the lease alive without throttled browser timers.
      socket.onclose = () => { if (!disposed) retry = setTimeout(connect, 2000); };
    }
    connect();
    return () => { disposed = true; clearTimeout(retry); socket?.close(); };
  }, []);
  useEffect(()=>{attempt(async()=>{const d=await refresh();setRule(d.rule);}); const t=setInterval(()=>{refresh().catch(()=>{});api('/health').then(h=>setHealth(h.ml_connected)).catch(()=>setHealth(false));if(selectedRef.current)loadDriver(selectedRef.current).catch(()=>{});},2000);return()=>{clearInterval(t);active.current=false;stream.current?.getTracks().forEach(t=>t.stop());};},[]);
  function stop() {active.current=false;generation.current++;setRunning(false);video.current?.pause();}
  function select(id) {stop();selectedRef.current=id;setSelected(id);setDriver(null);attempt(()=>loadDriver(id));}
  async function send(blob,id) {await api('/drivers/'+id+'/frames',{method:'POST',headers:{'Content-Type':blob.type || 'image/jpeg'},body:blob});await refresh();await loadDriver(id);}
  async function upload(files) {if(!selected)return;const id=selected;setBusy(true);await attempt(async()=>{for(const file of files)await send(file,id);});setBusy(false);}
  async function start() {
    if(!selected || busy)return;
    const id=selected, token=++generation.current;
    setBusy(true);
    await attempt(async()=>{
      if(source==='camera') {stream.current?.getTracks().forEach(t=>t.stop());stream.current=await navigator.mediaDevices.getUserMedia({video:true});video.current.srcObject=stream.current;}
      else {video.current.srcObject=null;if(!videoUrl)throw new Error('Choose a video first.');}
      await video.current.play();active.current=true;setRunning(true);
      while(active.current && token===generation.current) {
        if(video.current.ended)break;
        if(video.current.videoWidth) {
          const canvas=document.createElement('canvas');const ratio=Math.min(1,960/video.current.videoWidth);canvas.width=video.current.videoWidth*ratio;canvas.height=video.current.videoHeight*ratio;
          canvas.getContext('2d').drawImage(video.current,0,0,canvas.width,canvas.height);
          const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.85));await send(blob,id);
        }
        await new Promise(resolve=>setTimeout(resolve,interval*1000));
      }
    });
    stop();stream.current?.getTracks().forEach(t=>t.stop());setBusy(false);
  }
  const tagged=data.drivers.filter(d=>d.tagged), unread=data.notifications.filter(n=>!n.read).length;
  return <div className="app">
    <aside><a className="brand" href="/">◈ <span>SafeDrive<span className="brand-dot">.</span></span></a><div className="workspace">FLEET WORKSPACE</div><button className="nav active" onClick={()=>setTab('all')}>▦ &nbsp; Overview</button><button className="nav" onClick={()=>{setTab('tagged');}}>⚑ &nbsp; Tagged drivers <b>{tagged.length}</b></button><div className="aside-bottom"><span className="demo-dot"/> LOCAL DEMO <p>Driver intelligence<br/>Human review, informed by data.</p></div></aside>
    <main><header><div><div className="eyebrow">OPERATIONS / LIVE MONITORING</div><h1>Fleet overview</h1><p className="sub">A clearer view of every journey.</p></div><div className="header-right"><span className={'connection '+(health?'online':'')}>● &nbsp; ML {health?'connected':'offline'}</span><button className="bell" onClick={()=>setBell(!bell)}>♧ Notifications {unread>0&&<b>{unread}</b>}</button></div></header>
    {error&&<div className="error" role="alert">{error}<button onClick={()=>setError('')}>×</button></div>}
    {bell&&<section className="panel notification"><div className="panel-heading"><h3>Notifications</h3><button onClick={()=>attempt(async()=>{await post('/notifications/read');await refresh();})}>Mark all read</button></div>{data.notifications.length===0?<p>No alerts yet. New driver tags appear here.</p>:data.notifications.map(n=><button key={n.id} className={'notice '+(!n.read?'unread':'')} onClick={()=>{select(n.driver_id);setBell(false);}}><strong>{n.driver_name} flagged for review</strong><span>{n.message} · {time(n.timestamp)}</span></button>)}</section>}
    <div className="stats"><div><span>MONITORED DRIVERS</span><strong>{String(data.drivers.length).padStart(2,'0')}</strong><small>Registered in this workspace</small></div><div><span>TAGGED FOR REVIEW</span><strong className="orange">{String(tagged.length).padStart(2,'0')} <i>↗</i></strong><small>Accumulated dangerous behavior</small></div><div><span>FRAME HISTORY</span><strong>{data.drivers.reduce((a,d)=>a+d.frame_count,0).toLocaleString()}</strong><small>Live predictions + labeled demo samples</small></div><div><span>ACTIVE REVIEW RULE</span><strong>{data.rule.threshold}<em> / {data.rule.window}</em></strong><small>Dangerous frames in rolling window</small></div></div>
    <div className="columns"><section className="panel roster"><div className="panel-heading"><h2>Driver directory</h2><button onClick={()=>attempt(async()=>{await post('/demo/import');await refresh();})}>Import demo drivers</button></div><div className="tabs"><button className={tab==='all'?'chosen':''} onClick={()=>setTab('all')}>All drivers</button><button className={tab==='tagged'?'chosen':''} onClick={()=>setTab('tagged')}>Tagged <span>{tagged.length}</span></button></div><div className="driver-list">{(tab==='tagged'?tagged:data.drivers).map(d=><button className={'driver '+(selected===d.id?'selected':'')} key={d.id} onClick={()=>select(d.id)}><div className="avatar">{d.demo&&d.latest?<img src={d.latest.image_url} alt="Demo source"/>:d.name.slice(0,2).toUpperCase()}</div><div className="driver-info"><strong>{d.name}</strong><small>{d.demo?'DEMO · ':''}{d.frame_count} frames · {d.latest?d.latest.label:'Awaiting stream'}</small></div><span className={'status '+(d.tagged?'tagged':'')}>{d.tagged?'Tagged':'Monitoring'}</span></button>)}{!(tab==='tagged'?tagged:data.drivers).length&&<div className="empty">{tab==='tagged'?'No drivers tagged for review.':'Add your first driver to start monitoring.'}</div>}</div><form className="add-driver" onSubmit={e=>{e.preventDefault();attempt(async()=>{const d=await api('/drivers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});setName('');await refresh();select(d.id);});}}><input placeholder="Driver name" value={name} onChange={e=>setName(e.target.value)} required/><button className="primary">+ Add driver</button></form></section>
    <section className="panel detail">{!driver?<div className="empty big"><div className="empty-icon">◎</div><h2>Select a driver</h2><p>Explore their stream, predictions, and review history.</p></div>:<><div className="panel-heading"><div><div className="eyebrow">DRIVER WORKSPACE</div><h2>{driver.name}</h2></div><span className={'status '+(driver.tagged?'tagged':'')}>{driver.tagged?'⚑ Tagged for review':'● Monitoring'}</span></div><div className="evidence">{driver.demo&&<div className="demo-note"><strong>Synthetic demo · distinct frame sequence</strong><p>Six different poses of the same fictional driver. AI-generated images with manual demo annotations, not ML predictions.</p><a href={driver.source.source_url} target="_blank" rel="noreferrer">Images: {driver.source.author}</a> · <a href={driver.source.license_url} target="_blank" rel="noreferrer">{driver.source.license}</a></div>}<span>Recent evidence <strong>{driver.recent.filter(Boolean).length} / {data.rule.threshold} to tag</strong></span><div className="squares">{Array.from({length:data.rule.window},(_,i)=><i key={i} className={i<driver.recent.length?(driver.recent[i]?'danger':'safe'):''}/>)}</div><small>Classes c1–c9 count as dangerous. c0 is safe driving.</small>{driver.tagged&&<button onClick={()=>attempt(async()=>{await post('/drivers/'+selected+'/clear');await refresh();await loadDriver(selected);})}>Clear tag & restart evidence window</button>}</div>
    <div className="capture"><div className="capture-heading"><h3>Stream input</h3><select disabled={busy} value={source} onChange={e=>{stop();stream.current?.getTracks().forEach(t=>t.stop());setSource(e.target.value);}}><option value="images">Image sequence</option><option value="video">Video file</option><option value="camera">Webcam</option></select></div>{source==='images'?<label className={'upload '+(busy?'disabled':'')}>↑ <strong>{busy?'Classifying your images…':'Choose JPEG / PNG images'}</strong><span>Multiple images are sent sequentially in selection order</span><input type="file" accept="image/jpeg,image/png" multiple disabled={busy} onChange={e=>{upload(Array.from(e.target.files));e.target.value='';}}/></label>:<><video ref={video} src={source==='video'?videoUrl:undefined} muted playsInline controls={source==='video'} /><div className="stream-controls">{source==='video'&&<input disabled={busy} type="file" accept="video/*" onChange={e=>{if(videoUrl)URL.revokeObjectURL(videoUrl);if(e.target.files[0])setVideoUrl(URL.createObjectURL(e.target.files[0]));}}/>}<label>Interval <input type="number" min="0.2" step="0.2" disabled={busy} value={interval} onChange={e=>setIntervalSeconds(Math.max(.2,Number(e.target.value)))}/> s</label><button className="primary" disabled={busy&&!running} onClick={running?stop:start}>{running?'Stop stream':busy?'Finishing…':'Start stream'}</button></div><small>One frame per interval after inference finishes. Video playback continues between captures.</small></>}</div>
    <div className="history"><div className="panel-heading"><h3>Stream history <span className="pill">{driver.frames.length}</span></h3><small>Newest first · click image to expand</small></div>{!driver.frames.length?<div className="empty">Your classified frames will appear here.</div>:<div className="frame-grid">{[...driver.frames].reverse().map(f=><article className="frame" key={f.id}><a href={f.image_url} target="_blank" rel="noreferrer"><img src={f.image_url} loading="lazy"/></a><div><span className={'class-label '+(f.dangerous?'danger-text':'')}>{f.class_id===null?'':`c${f.class_id} · `}{f.label}</span><small>{time(f.timestamp)} <b>{f.demo?'DEMO':`${Math.round(f.confidence*100)}%`}</b></small></div></article>)}</div>}</div></>}</section></div>
    <section className="panel rule"><div><h3>Review rule</h3><p>Simple frame counting · Tags persist until manually cleared · Saved locally</p></div><form onSubmit={e=>{e.preventDefault();attempt(async()=>{await api('/rule',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(rule)});await refresh();if(selected)await loadDriver(selected);});}}><label>Dangerous frames <input type="number" min="1" max={rule.window} value={rule.threshold} onChange={e=>setRule({...rule,threshold:Number(e.target.value)})}/></label><label>Window size <input type="number" min={rule.threshold} max="100" value={rule.window} onChange={e=>setRule({...rule,window:Number(e.target.value)})}/></label><button>Apply rule</button></form></section><footer>SAFEDRIVE / DEMO WORKSPACE <span>Predictions support human review. Demo thresholds are not safety validated.</span></footer>
    </main></div>;
}
createRoot(document.getElementById('root')).render(<App/>);
