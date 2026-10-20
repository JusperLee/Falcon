const result = document.querySelector('#result');
const button = document.querySelector('#run');
const canvas = document.querySelector('#preview');
const ctx = canvas.getContext('2d');
button.onclick = async () => {
  button.disabled = true;
  const worker = new Worker('/static/video-worker.js', { type: 'module' });
  const messages = [], waits = [];
  worker.onmessage = ({data}) => {
    messages.push(data);
    for (const wait of [...waits]) if (wait.match(data)) { waits.splice(waits.indexOf(wait), 1); clearTimeout(wait.timeout); wait.resolve(data); }
  };
  const next = (match, timeoutMs = 20000) => new Promise((resolve,reject) => {
    const w = {match,resolve,timeout:setTimeout(() => reject(new Error('Timed out waiting for worker: '+JSON.stringify(messages.filter(x=>x.type !== 'frame')))),timeoutMs)};
    waits.push(w);
  });
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  try {
    result.textContent = '加载本地检测模型…';
    const loaded = next(m => m.type === 'tracking' && (m.detectorReady || m.status === 'error'));
    worker.postMessage({type:'start',tracking:'auto',trackingEpoch:0,roi:{cx:.5,cy:.6,size:.2}});
    const initial = await loaded;
    assert(initial.status !== 'error', initial.message);
    const portrait = new Image(); portrait.src = 'portrait.jpg'; await portrait.decode();
    const fit = Math.min(640/portrait.width,480/portrait.height);
    async function frame(blank = false, dx = 0) {
      ctx.fillStyle = '#888'; ctx.fillRect(0,0,640,480);
      if (!blank) ctx.drawImage(portrait,(640-portrait.width*fit)/2+dx,0,portrait.width*fit,portrait.height*fit);
      const bitmap = await createImageBitmap(canvas);
      const before = messages.length;
      const ack = next(m=>m.type==='ack');
      worker.postMessage({type:'frame',bitmap,wallTime:performance.timeOrigin+performance.now()},[bitmap]);
      await ack;
      return messages.slice(before);
    }
    const found = [];
    for(let i=0;i<40;i++) { await new Promise(r=>setTimeout(r,45)); found.push(...await frame(false,Math.min(i,7)*2)); }
    const tracks = found.filter(m=>m.type==='tracking'&&m.status==='tracking');
    assert(tracks.length>=5, 'Expected stable real face landmark detections');
    assert(tracks.every(m=>m.lips.length>=10&&m.roi.size>0&&m.roi.size<1),'Invalid lip contour or ROI');
    assert(found.filter(m=>m.type==='frame').length>=5,'Real face must produce model input crops');
    const absent = [];
    for(let i=0;i<3;i++) { await new Promise(r=>setTimeout(r,45)); absent.push(...await frame(true)); }
    // A missing face never pauses output: crops continue at the last ROI.
    assert(absent.every(m=>m.type!=='tracking'||m.status==='tracking'),'A brief miss must not change the public status');
    assert(absent.filter(m=>m.type==='frame').length===3 && absent.filter(m=>m.type==='frame').every(m=>m.coasted),'Brief miss keeps emitting coasted crops');
    await new Promise(r=>setTimeout(r,600));
    const searching = await frame(true);
    assert(searching.some(m=>m.type==='tracking'&&m.status==='tracking'&&m.searching),'Long loss reports searching, not lost');
    assert(searching.some(m=>m.type==='frame'),'Long loss still outputs crops at the last ROI');
    let relocked = false;
    for (let i=0;i<12&&!relocked;i++) { await new Promise(r=>setTimeout(r,45)); relocked = (await frame(false,14)).some(m=>m.type==='tracking'&&m.status==='tracking'&&!m.searching&&!m.coasted&&m.lips.length>=10); }
    assert(relocked,'Same face must re-lock automatically after a long loss');
    worker.postMessage({type:'tracking',mode:'manual',trackingEpoch:1,roi:{cx:.5,cy:.6,size:.2}});
    const manual = await frame(true);
    assert(manual.some(m=>m.type==='frame'),'Explicit manual mode must retain crop functionality');
    const detections = tracks.map(m=>m.detectionMs).filter(Number.isFinite);
    const steady = detections.slice(5).sort((a,b)=>a-b);
    const report = {test:'PASS',library:'MediaPipe 1.0.1',fixture:'Official MediaPipe portrait.jpg',faceFrames:tracks.length,lipPoints:tracks[0].lips.length,firstROI:tracks[0].roi,lastROI:tracks.at(-1).roi,meanDetectionMs:detections.reduce((a,b)=>a+b,0)/detections.length,steadyMedianMs:steady[Math.floor(steady.length/2)],steadyP95Ms:steady[Math.floor(steady.length*.95)],noPauseWithoutFace:true,autoRelock:true,manualFallback:true};
    result.textContent = JSON.stringify(report,null,2);
  } catch(error) { result.textContent = 'FAIL: '+error.message; }
  finally { worker.postMessage({type:'stop'}); setTimeout(()=>worker.terminate(),500); button.disabled=false; }
};
