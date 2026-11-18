from pathlib import Path
import json,base64,csv
root=Path(__file__).parent;e=root/'evidence'
constant=lambda n,obj:'const '+n+'='+json.dumps(obj,separators=(',',':'))+';\n'
code=(root/'inference.mjs').read_text().replace('export function','function')+'\n'
code+=constant('HTML',(root/'index.html').read_text())
for name,file in [('MODEL','model.json'),('REPORT','report.json'),('MANIFEST','manifest.json'),('SAMPLES','samples.json'),('RELEASE','release.json')]:code+=constant(name,json.loads((e/file).read_text()))
sample_ids={str(x['id']) for x in json.loads((e/'samples.json').read_text())}
replay={id:[] for id in sample_ids}
with (e/'features.csv').open(newline='') as f:
 for row in csv.DictReader(f):
  if row['recording'] in replay:
   replay[row['recording']].append([float(row[k]) for k in ('rms','kurtosis','crest_factor')])
code+=constant('REPLAY',replay)
code+=constant('SLIDES',json.loads((root/'presentation.json').read_text()))
files={'/assets/motor.jpg':(root/'assets/motor.jpg','image/jpeg','motor.jpg'),'/downloads/evidence.sqlite':(e/'evidence.sqlite','application/vnd.sqlite3','evidence.sqlite'),'/downloads/predictions.csv':(e/'predictions.csv','text/csv','predictions.csv')}
code+=constant('FILES',{k:dict(data=base64.b64encode(p.read_bytes()).decode(),type=t,name=n) for k,(p,t,n) in files.items()})
code+='''
const json=(x,status=200)=>new Response(JSON.stringify(x),{status,headers:{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}});
async function readLimited(request){if(!request.body)throw Error('JSON body required.');const reader=request.body.getReader();let size=0,chunks=[];while(true){const {value,done}=await reader.read();if(done)break;size+=value.length;if(size>100000){await reader.cancel();throw Error('Request exceeds 100 KB.')}chunks.push(value)}const all=new Uint8Array(size);let offset=0;for(const c of chunks){all.set(c,offset);offset+=c.length}return new TextDecoder().decode(all)}
export default {async fetch(request){const path=new URL(request.url).pathname;
 if(path==='/api/predict'){
  if(request.method!=='POST')return json({error:'Use POST with a JSON body.'},405);
  if(!(request.headers.get('content-type')||'').includes('application/json'))return json({error:'Content-Type must be application/json.'},415);
  let body;try{body=JSON.parse(await readLimited(request))}catch(e){return json({error:e.message.startsWith('Request exceeds')?e.message:'Valid JSON body required.'},400)}
  if(!body||typeof body!=='object'||Array.isArray(body))return json({error:'JSON object required.'},400);
  if(body.sampling_hz!==48000)return json({error:'This release accepts declared 48000 Hz CWRU-scale windows only.'},400);
  const name=body.model||MODEL.selected;if(!Object.hasOwn(MODEL.thresholds,name))return json({error:'Unknown model.'},400);
  try{const start=performance.now();const result=infer(body.samples,MODEL,name);result.processing_ms=performance.now()-start;
   const hash=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(JSON.stringify(body.samples)));result.input_sha256=Array.from(new Uint8Array(hash),x=>x.toString(16).padStart(2,'0')).join('');result.release=RELEASE.release;result.model_sha256=RELEASE.model_sha256;return json(result);
  }catch(e){return json({error:e.message},400)}
 }
 if(request.method!=='GET'&&request.method!=='HEAD')return json({error:'Method not allowed.'},405);
 if(path==='/api/health')return json({status:'ok',release:RELEASE.release,selected_model:MODEL.selected,model_sha256:RELEASE.model_sha256,mode:'real_benchmark_inference',live_sensors:false,persistence:'none'});
 if(path==='/api/replay'){
  const q=new URL(request.url).searchParams,id=q.get('recording'),index=Number(q.get('window')),name=q.get('model')||MODEL.selected;
  if(!Object.hasOwn(REPLAY,id)||!Number.isInteger(index)||index<0||index>=REPLAY[id].length||!Object.hasOwn(MODEL.thresholds,name))return json({error:'Unknown recording, window or model.'},400);
  const features=REPLAY[id][index],score=scoreFeatures(features,MODEL,name),threshold=MODEL.thresholds[name],ratio=score/threshold,health=Math.max(0,Math.min(1,1-0.25*ratio**3));
  return json({recording:id,window:index,window_count:REPLAY[id].length,model:name,features:Object.fromEntries(MODEL.features.map((k,i)=>[k,features[i]])),score,threshold,health,band:health>=0.9?'healthy':health>=0.75?'watch':'inspect',source:'CWRU recorded feature sequence; no connected sensor'});
 }
 if(path==='/api/report')return json(REPORT);if(path==='/api/manifest')return json(MANIFEST);if(path==='/api/model')return json(MODEL);if(path==='/api/release')return json(RELEASE);if(path==='/api/slides')return json(SLIDES);
 if(path==='/api/samples')return json(SAMPLES.map(({id,name,label,load_hp,window_count})=>({id,name,label,load_hp,window_count})));
 if(path.startsWith('/api/samples/')){const s=SAMPLES.find(s=>s.id===path.split('/').pop());return s?json(s):json({error:'Unknown recording.'},404)}
 if(FILES[path]){const f=FILES[path],bytes=Uint8Array.from(atob(f.data),c=>c.charCodeAt(0));return new Response(bytes,{headers:{'Content-Type':f.type,...(path.startsWith('/downloads/')?{'Content-Disposition':'attachment; filename="'+f.name+'"'}:{}),'X-Content-Type-Options':'nosniff'}})}
 if(path==='/'||path==='/index.html')return new Response(HTML,{headers:{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}});
 return json({error:'Not found.'},404);
}};
'''
(root/'dist/server').mkdir(parents=True,exist_ok=True)
(root/'public').mkdir(exist_ok=True)
(root/'worker.mjs').write_text(code);(root/'dist/server/index.js').write_text(code)
print('Built Worker:',len(code),'bytes')
