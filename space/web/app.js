// Falcon Studio on a ZeroGPU Space: a fork of studio/web/app.js (same capture, mouth
// tracking and recording export). ZeroGPU attaches a GPU per call, so instead of the
// live 0.5 s socket loop the page records one clip and sends it once recording stops;
// the server separates it on the live Studio's window schedule (space/clip.py).
import { setText, setAttr, initializeI18n } from './i18n.js';
import { createAudioInspector } from './audio-inspector.js';

const $ = (id) => document.getElementById(id);
const RATE = 16000;
const FPS = 25;
const STALE = .16 * RATE; // Studio refuses video further than 160 ms from a slot
const state = {
  status: {}, stream: null, context: null, source: null, worklet: null,
  frameTimer: null, samples: 0, audioOrigin: null, generation: 0,
  running: false, opening: false, separating: false, controller: null,
  maxSeconds: 60, clipStart: 0, clipAudio: null, clipSamples: 0, clipFrames: [], remaining: null,
  recording: [], recordingSamples: 0, recordingUrl: null, recordingFinalizedSamples: 0,
  mixture: [], mixtureUrl: null, recordingTrack: 'target',
  inputWave: [], outputWave: [],
  roi: { cx: .5, cy: .60, size: .20 }, lastError: '', backendReachable: false,
  videoWorker: null, videoMode: 'none',
  trackingMode: 'auto', trackingStatus: 'idle', trackingEpoch: 0,
  lips: [], trackingMessage: '', detectionMs: null, requiresReacquire: false, searching: false,
  bitmapBusy: false, bitmapTimeout: null,
};
const cropCanvas = document.createElement('canvas');
cropCanvas.width = cropCanvas.height = 96;
const cropContext = cropCanvas.getContext('2d', { willReadFrequently: true });
const clamp = (n, low, high) => Math.min(high, Math.max(low, n));
const duration = (seconds) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
const audioInspector = createAudioInspector({ player: $('recordingPlayer'), onError: (message) => notice(message, true) });
// Only the microphone streams live here; the separated clip appears in the recording view.
document.querySelector('#outputWave + .audio-live-spectrum')?.setAttribute('hidden', '');
const liveNote = document.querySelector('.audio-live-note');
if (liveNote) setText(liveNote, '实时频谱显示麦克风输入；分离结果在录音结束、GPU 分离完成后显示在下方。');

// Aggregate-only troubleshooting: no PCM, images, cookies or device identifiers.
const captureHealth = { inputAt: 0, frameAt: 0, inputRms: 0, startedAt: 0, warning: '' };
window.falconDiagnostics = () => ({
  running: state.running, separating: state.separating, opening: state.opening, audioState: state.context?.state || 'closed',
  videoMode: state.videoMode, tracking: state.trackingStatus, searching: state.searching,
  samples: state.samples, clipSeconds: state.clipSamples / RATE, clipFrames: state.clipFrames.length,
  recordingSeconds: state.recordingSamples / RATE, inputRms: captureHealth.inputRms,
  inputAgeMs: Math.round(performance.now() - captureHealth.inputAt),
  frameAgeMs: Math.round(performance.now() - captureHealth.frameAt), warning: captureHealth.warning,
});
function checkCaptureHealth() {
  if (!state.running || performance.now() - captureHealth.startedAt < 3000) return;
  const now = performance.now();
  let warning = '';
  if (now - captureHealth.inputAt > 3000) warning = '麦克风采集已暂停，请检查设备或浏览器是否挂起。';
  else if (now - captureHealth.frameAt > 3000) warning = '尚无新的有效嘴部画面，请检查摄像头、取景和自动检测状态。';
  else if (captureHealth.inputRms < .00001) warning = '麦克风输入接近静音，请确认选中了正在使用的麦克风。';
  if (warning) setText($('performanceHint'), warning);
  else if (captureHealth.warning) setText($('performanceHint'), '录音中 · 停止后整段上传分离');
  if (warning && warning !== captureHealth.warning) console.warn('Falcon capture diagnostics', window.falconDiagnostics(), warning);
  captureHealth.warning = warning;
}
setInterval(checkCaptureHealth, 1000);
function bounded(promise, milliseconds, message) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), milliseconds);
    Promise.resolve(promise).then(resolve, reject).finally(() => clearTimeout(timer));
  });
}
function notice(message, error = false, params = {}) {
  setText($('noticeText'), message, params);
  $('notice').classList.toggle('error', error);
  $('notice').hidden = false;
}
function errorMessage(error) {
  const names = {
    NotAllowedError: '摄像头或麦克风访问被拒绝。请在浏览器地址栏和系统隐私设置中允许访问，然后重试。',
    NotFoundError: '没有找到可用的摄像头或麦克风。请连接设备后重试。',
    NotReadableError: '设备暂时无法使用。请关闭其他正在使用摄像头或麦克风的应用后重试。',
    OverconstrainedError: '所选设备已不可用，请重新选择设备。',
  };
  return names[error?.name] || error?.message || String(error);
}
function setSession(title, subtitle, params = {}) {
  setText($('sessionTitle'), title);
  if (subtitle) setText($('sessionSubtitle'), subtitle, params);
}
function updateButtons() {
  const primary = $('primaryButton');
  const ready = !!state.status.ready && state.backendReachable;
  primary.disabled = state.opening || state.separating || (!state.running && !!state.stream && !ready);
  setText(primary.querySelector('span'), state.opening ? '正在开启设备…' : state.separating ? '正在分离…' : state.running ? '完成并分离' : state.stream ? '开始录制' : '开启设备');
  primary.querySelector('svg').innerHTML = state.running ? '<path d="M6 6h12v12H6z"/>' : state.stream ? '<circle cx="12" cy="12" r="6"/>' : '<path d="M12 4v16M4 12h16"/>';
  $('stopButton').hidden = !state.stream && !state.opening;
  $('cameraSelect').disabled = !state.stream || state.opening || state.running;
  $('micSelect').disabled = !state.stream || state.opening || state.running;
  $('modelSelect').disabled = state.running || state.separating;
  const recording = state.running || state.separating || !state.recordingSamples;
  $('sessionLight').classList.toggle('running', state.running && trackingAllowsInference());
  setText($('captureLabel'), state.running ? trackingAllowsInference() ? '正在录制' : '等待目标' : '设备预览');
  $('captureLabel').parentElement.classList.toggle('running', state.running && trackingAllowsInference());
  $('playRecording').disabled = $('downloadButton').disabled = recording;
  $('trackTarget').disabled = $('trackMixture').disabled = recording;
  $('reacquireButton').disabled = !state.stream || state.opening || state.trackingMode !== 'auto';
}
function applyStatus(status) {
  state.status = status;
  state.backendReachable = true;
  if (Number.isFinite(status.max_seconds)) state.maxSeconds = clamp(status.max_seconds, 1, 300);
  if (Array.isArray(status.variants)) {
    const selected = $('modelSelect').value || 'small';
    $('modelSelect').replaceChildren(...status.variants.map((variant) => {
      const option = document.createElement('option');
      option.value = variant.id;
      setText(option, variant.id === 'small' ? '{model} · 论文模型' : '{model} · 高容量', { model: variant.name || variant.id });
      option.disabled = !variant.available;
      return option;
    }));
    if ([...$('modelSelect').options].some((option) => option.value === selected)) $('modelSelect').value = selected;
  }
  $('enginePill').classList.toggle('ready', !!status.ready);
  $('enginePill').classList.remove('loading');
  setText($('enginePill').querySelector('span'), status.ready ? '模型已就绪' : '等待模型');
  setText($('modelState'), status.ready ? '就绪 · 可开始' : '尚未加载');
  setText($('modelDevice'), status.device === 'zerogpu' ? 'ZeroGPU' : (status.device || '自动选择').toUpperCase());
  setText($('modelFootnote'), status.device === 'zerogpu' ? '录音结束后整段送入 ZeroGPU 分离 · 不保存音视频' : '录音结束后整段在本服务分离 · 不保存音视频');
  updateButtons();
}
async function fetchStatus() {
  try {
    const response = await fetch('/api/status', { signal: AbortSignal.timeout(8000) });
    if (!response.ok) throw new Error(String(response.status));
    applyStatus(await response.json());
  } catch {
    state.backendReachable = false;
    setText($('enginePill').querySelector('span'), '服务未连接');
    $('enginePill').classList.remove('ready', 'loading');
    setText($('modelState'), '服务未连接');
    setText($('modelFootnote'), 'Space 可能正在启动或休眠，请稍后刷新页面。');
    updateButtons();
  }
}

async function enumerateDevices() {
  const generation = state.generation;
  const devices = await bounded(navigator.mediaDevices.enumerateDevices(), 8000, '读取设备列表超时，可继续使用当前设备。');
  if (generation !== state.generation) return;
  const tracks = state.stream?.getTracks() || [];
  for (const [id, kind, trackKind, label] of [['cameraSelect', 'videoinput', 'video', '摄像头'], ['micSelect', 'audioinput', 'audio', '麦克风']]) {
    const select = $(id);
    const selected = tracks.find((track) => track.kind === trackKind)?.getSettings().deviceId || select.value;
    select.replaceChildren();
    devices.filter((device) => device.kind === kind).forEach((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      if (device.label) option.textContent = device.label;
      else setText(option, label === '摄像头' ? '摄像头 {number}' : '麦克风 {number}', { number: index + 1 });
      select.append(option);
    });
    if ([...select.options].some((option) => option.value === selected)) select.value = selected;
  }
  updateButtons();
}
async function openCapture(deviceIds = {}) {
  if (state.opening || state.stream) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('设备访问需要 HTTPS，请从 Hugging Face Space 页面打开。');
  const generation = ++state.generation;
  state.opening = true;
  updateButtons();
  setSession('等待设备授权', '请允许访问摄像头和麦克风');
  let stream = null;
  let context = null;
  try {
    // Resume in the user gesture, before the permission prompt resolves (Safari requirement).
    context = new AudioContext({ latencyHint: 'interactive' });
    // Invoke resume in the click gesture, but do not block the permission request on it.
    const resumed = context.resume();
    resumed.catch(() => {});
    const requested = navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: FPS }, ...(deviceIds.camera ? { deviceId: { exact: deviceIds.camera } } : {}) },
      audio: { channelCount: { ideal: 1 }, echoCancellation: false, noiseSuppression: false, autoGainControl: false, ...(deviceIds.microphone ? { deviceId: { exact: deviceIds.microphone } } : {}) },
    });
    // A permission/device request can finish after cancellation or its deadline.
    requested.then((lateStream) => {
      if (generation !== state.generation) lateStream.getTracks().forEach((track) => track.stop());
    }, () => {});
    stream = await bounded(requested, 45000, '设备授权或启动超过 45 秒，请检查浏览器权限及设备是否被占用后重试。');
    if (generation !== state.generation) { stream.getTracks().forEach((track) => track.stop()); void context.close().catch(() => {}); return; }
    setSession('正在启动音频设备');
    await bounded(resumed, 8000, '音频设备启动超时，请停止其他占用麦克风的程序后重试。');
    if (generation !== state.generation) { stream.getTracks().forEach((track) => track.stop()); void context.close().catch(() => {}); return; }
    state.stream = stream;
    state.context = context;
    Object.assign(captureHealth, { inputAt: performance.now(), frameAt: performance.now(), inputRms: 0, warning: '' });
    state.audioOrigin = null;
    state.samples = 0;
    state.inputWave = [];
    audioInspector.resetLive();
    $('cameraVideo').srcObject = stream;
    setSession('正在启动摄像头预览');
    await bounded($('cameraVideo').play(), 12000, '摄像头预览启动超时，请检查摄像头是否被其他程序占用后重试。');
    if (generation !== state.generation) return;
    setSession('正在加载音频采集模块');
    await bounded(context.audioWorklet.addModule('/static/capture-worklet.js'), 15000, '音频采集模块加载超时，请检查网络后重试。');
    if (generation !== state.generation) return;
    state.source = context.createMediaStreamSource(stream);
    state.worklet = new AudioWorkletNode(context, 'falcon-capture', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    state.worklet.port.onmessage = (event) => { if (generation === state.generation) onPCM(event.data); };
    state.source.connect(state.worklet);
    state.worklet.connect(context.destination);
    stream.getTracks().forEach((track) => track.addEventListener('ended', () => {
      if (generation === state.generation) { stopCapture(); notice('设备连接已结束，请重新开启摄像头和麦克风。', true); }
    }));
    await startVideoCapture(generation);
    if (generation !== state.generation) return;
    $('cameraPlaceholder').hidden = true;
    $('cameraOverlay').hidden = false;
    $('cameraState').classList.add('active');
    setText($('cameraState').querySelector('span'), '设备已连接');
    setText($('inputCaption'), '实时麦克风输入');
    refreshTrackingUI();
    // Enumeration is optional; a slow device driver must not keep Start disabled.
    void enumerateDevices().catch((error) => { if (generation === state.generation) notice(errorMessage(error), true); });
    positionROI();
    if (state.videoMode === 'fallback') captureFrame();
  } catch (error) {
    stream?.getTracks().forEach((track) => track.stop());
    if (context && context.state !== 'closed') void context.close().catch(() => {});
    if (generation === state.generation) { stopCapture(); notice(errorMessage(error), true); }
  } finally {
    if (generation === state.generation) { state.opening = false; updateButtons(); }
  }
}
function onPCM({ samples, startTime }) {
  if (!state.context || !samples?.length) return;
  if (state.audioOrigin === null) state.audioOrigin = startTime;
  // A suspended context or lost packet invalidates the old audio clock; crops already
  // placed on the clip keep their sample positions, later ones use the new origin.
  const expectedTime = state.audioOrigin + state.samples / RATE;
  if (Math.abs(startTime - expectedTime) > .08) state.audioOrigin = startTime - state.samples / RATE;
  let energy = 0;
  for (const value of samples) {
    const sample = Number.isFinite(value) ? value : 0;
    if (state.running && state.clipSamples < state.clipAudio.length) state.clipAudio[state.clipSamples++] = sample;
    state.samples++;
    energy += sample * sample;
  }
  const rms = Math.sqrt(energy / samples.length);
  captureHealth.inputAt = performance.now(); captureHealth.inputRms = rms;
  $('inputLevel').textContent = rms > .00001 ? `${Math.max(-99, 20 * Math.log10(rms)).toFixed(0)} dB` : '−∞ dB';
  state.inputWave.push(rms);
  if (state.inputWave.length > 140) state.inputWave.splice(0, state.inputWave.length - 140);
  audioInspector.pushInput(samples);
  const elapsed = state.running ? state.clipSamples / RATE : state.samples / RATE;
  $('inputDuration').textContent = duration(elapsed);
  $('captureTime').textContent = duration(elapsed);
  if (!state.running) return;
  const remaining = Math.max(0, Math.ceil(state.maxSeconds - elapsed));
  if (remaining !== state.remaining) { state.remaining = remaining; refreshTrackingUI(); }
  if (state.clipSamples >= state.clipAudio.length) {
    notice('已达 {seconds} 秒上限，开始分离。', false, { seconds: state.maxSeconds });
    void finishRecording();
  }
}
function videoGeometry() {
  const video = $('cameraVideo');
  const rect = $('cameraStage').getBoundingClientRect();
  const width = video.videoWidth || 1280;
  const height = video.videoHeight || 720;
  const scale = Math.max(rect.width / width, rect.height / height);
  return { width, height, rect, scale, ox: (rect.width - width * scale) / 2, oy: (rect.height - height * scale) / 2 };
}
function positionROI() {
  const { width, height, rect, scale, ox, oy } = videoGeometry();
  const size = Math.min(width, height) * state.roi.size;
  const marginX = size / width / 2;
  const marginY = size / height / 2;
  // Clamp to the displayed portion of an object-fit:cover video, as well as the source.
  const minX = Math.max(marginX, (-ox + size * scale / 2) / (width * scale));
  const maxX = Math.min(1 - marginX, (rect.width - ox - size * scale / 2) / (width * scale));
  const minY = Math.max(marginY, (-oy + size * scale / 2 + 26) / (height * scale));
  const maxY = Math.min(1 - marginY, (rect.height - oy - size * scale / 2 - 37) / (height * scale));
  if (state.trackingMode === 'manual') {
    state.roi.cx = clamp(state.roi.cx, minX, Math.max(minX, maxX));
    state.roi.cy = clamp(state.roi.cy, minY, Math.max(minY, maxY));
  }
  const roi = $('mouthRoi');
  roi.style.left = `${ox + (state.roi.cx * width - size / 2) * scale}px`;
  roi.style.top = `${oy + (state.roi.cy * height - size / 2) * scale}px`;
  roi.style.width = roi.style.height = `${size * scale}px`;
  roi.setAttribute('aria-valuenow', Math.round(state.roi.size * 100));
  setAttr(roi, 'aria-valuetext', '横向 {x}%，纵向 {y}%，大小 {size}%', { x: Math.round(state.roi.cx * 100), y: Math.round(state.roi.cy * 100), size: Math.round(state.roi.size * 100) });
  if (state.trackingMode === 'manual') state.videoWorker?.postMessage({ type: 'roi', roi: state.roi });
  const overlay = $('lipOverlay');
  overlay.setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
  $('lipContour').setAttribute('points', state.lips.map((point) => `${ox + point.x * width * scale},${oy + point.y * height * scale}`).join(' '));
}
function trackingAllowsInference() {
  return state.trackingMode === 'manual' || state.trackingStatus === 'tracking';
}
function refreshTrackingUI() {
  const automatic = state.trackingMode === 'auto';
  const tracked = automatic && state.trackingStatus === 'tracking';
  const states = { idle: '待开启设备', loading: '正在加载嘴部检测…', tracking: '嘴部跟踪中', lost: '未找到清晰嘴部', error: '自动检测不可用', manual: '手动取景模式' };
  $('autoTrackingButton').setAttribute('aria-pressed', String(automatic));
  $('manualTrackingButton').setAttribute('aria-pressed', String(!automatic));
  $('trackingStatus').dataset.status = state.trackingStatus;
  const trackingLabel = state.requiresReacquire ? '目标变化，请重新捕捉' : state.searching ? '未检测到人脸 · 沿用上次取景' : states[state.trackingStatus] || '等待检测';
  setText($('trackingStatus'), trackingLabel);
  if (state.trackingMessage) setAttr($('trackingStatus'), 'title', state.trackingMessage);
  else setAttr($('trackingStatus'), 'title', Number.isFinite(state.detectionMs) ? '检测耗时 {milliseconds} ms' : '', { milliseconds: Number.isFinite(state.detectionMs) ? state.detectionMs.toFixed(1) : '' });
  $('mouthRoi').classList.toggle('automatic', automatic);
  $('mouthRoi').setAttribute('aria-valuemin', automatic ? '1' : '9');
  $('mouthRoi').setAttribute('aria-valuemax', automatic ? '100' : '48');
  $('mouthRoi').hidden = automatic && !tracked;
  setText($('mouthRoi').querySelector('.roi-label'), automatic ? 'AUTO / 嘴部' : 'MANUAL / 嘴部');
  $('lipOverlay').hidden = !tracked || !state.lips.length;
  $('centerRoi').hidden = automatic;
  setText($('roiHint'), automatic ? tracked ? '嘴部轮廓实时跟踪 · 拖动框可切换手动' : state.requiresReacquire ? '请确认目标说话者后，点击「重新捕捉」' : '请正对摄像头，保持嘴部清晰、无遮挡' : '拖动取景框对准嘴部 · 右下角调整大小');
  if (state.stream && !state.opening && !state.separating) {
    const remaining = { seconds: state.remaining ?? state.maxSeconds };
    if (!trackingAllowsInference()) {
      setSession(state.trackingStatus === 'loading' ? '正在准备自动嘴部检测' : '等待清晰的目标嘴部', state.trackingStatus === 'error' ? '检测未就绪 · 可点击「手动框选」继续' : state.requiresReacquire ? '目标发生变化 · 请点击「重新捕捉」' : state.running ? '录音中 · 没有嘴部画面的开头和结尾会被裁掉' : '锁定嘴部后再开始录制');
    } else if (state.running) {
      setSession('正在录制目标说话者', '还可录制 {seconds} 秒 · 停止后整段上传分离', remaining);
    } else if (state.recordingSamples) {
      setSession('分离完成', '可在下方试听、导出 WAV，或再录一段');
    } else {
      setSession(automatic ? '已锁定嘴部，可开始录制' : '将橙色取景框对准嘴部', '最长 {seconds} 秒 · 录完后整段送入 GPU 分离', { seconds: state.maxSeconds });
    }
  }
  updateButtons();
}
function setTrackingStatus(message) {
  if (message.trackingEpoch !== undefined && message.trackingEpoch !== state.trackingEpoch) return;
  const next = message.status;
  if (!['loading', 'tracking', 'lost', 'error', 'manual', 'idle'].includes(next)) return;
  if (state.trackingMode === 'manual' && next !== 'manual' && next !== 'idle') return;
  if (state.trackingMode === 'auto' && next === 'manual') return;
  state.trackingStatus = next;
  state.requiresReacquire = !!message.requiresReacquire;
  state.searching = next === 'tracking' && !!message.searching;
  state.trackingMessage = message.message || '';
  state.detectionMs = message.detectionMs ?? null;
  if (next === 'tracking' && message.roi && ['cx', 'cy', 'size'].every((key) => Number.isFinite(message.roi[key]))) {
    state.roi = { ...message.roi };
    state.lips = (message.lips || []).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  } else if (next !== 'tracking') state.lips = [];
  positionROI();
  refreshTrackingUI();
}
function setTrackingMode(mode, reacquire = false) {
  if (!['auto', 'manual'].includes(mode)) return;
  if (mode === state.trackingMode && !reacquire) return;
  state.trackingMode = mode;
  ++state.trackingEpoch;
  state.trackingStatus = state.stream ? mode === 'auto' ? 'loading' : 'manual' : 'idle';
  state.requiresReacquire = false;
  state.trackingMessage = '';
  state.lips = [];
  if (state.videoWorker) state.videoWorker.postMessage({ type: 'tracking', mode, roi: state.roi, trackingEpoch: state.trackingEpoch });
  else if (mode === 'auto' && state.stream) state.trackingStatus = 'error';
  positionROI();
  refreshTrackingUI();
}
async function startVideoCapture(generation) {
  ++state.trackingEpoch;
  state.trackingStatus = state.trackingMode === 'auto' ? 'loading' : 'manual';
  state.requiresReacquire = false;
  state.lips = [];
  state.bitmapBusy = false;
  refreshTrackingUI();
  const canUseWorker = 'Worker' in window && 'OffscreenCanvas' in window;
  const attempts = canUseWorker ? ('MediaStreamTrackProcessor' in window ? ['worker', 'fallback'] : ['fallback']) : [];
  for (const mode of attempts) {
    let worker;
    try {
      worker = new Worker('/static/video-worker.js', { type: 'module' });
      state.videoWorker = worker;
      const message = { type: 'start', roi: state.roi, tracking: state.trackingMode, trackingEpoch: state.trackingEpoch };
      const transfer = [];
      if (mode === 'worker') {
        const processor = new MediaStreamTrackProcessor({ track: state.stream.getVideoTracks()[0], maxBufferSize: 1 });
        message.readable = processor.readable;
        transfer.push(processor.readable);
      }
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('视频采集初始化超时')), 5000);
        worker.onerror = (event) => { clearTimeout(timeout); reject(new Error(event.message || '视频采集不可用')); };
        worker.onmessage = (event) => {
          if (event.data.type === 'ready') { clearTimeout(timeout); resolve(); }
          else if (event.data.type === 'tracking' && generation === state.generation) setTrackingStatus(event.data);
          else if (event.data.type === 'error') { clearTimeout(timeout); reject(new Error(event.data.message)); }
        };
        try { worker.postMessage(message, transfer); } catch (error) { clearTimeout(timeout); reject(error); }
      });
      if (generation !== state.generation) { worker.postMessage({ type: 'stop' }); worker.terminate(); return; }
      state.videoMode = mode;
      worker.onmessage = (event) => {
        const message = event.data;
        if (message.type === 'stopped') {
          worker.terminate();
          if (generation === state.generation && state.stream) {
            stopCapture();
            notice('摄像头帧流已结束，设备已关闭。请重新开启设备。', true);
          }
          return;
        }
        if (generation !== state.generation || !state.context) return;
        if (message.type === 'ack' || message.type === 'frame-done') {
          state.bitmapBusy = false;
          clearTimeout(state.bitmapTimeout);
          state.bitmapTimeout = null;
          return;
        }
        if (message.trackingEpoch !== undefined && message.trackingEpoch !== state.trackingEpoch) return;
        if (message.type === 'tracking') setTrackingStatus(message);
        else if (message.type === 'frame') {
          if (!trackingAllowsInference()) return;
          const wallNow = performance.timeOrigin + performance.now();
          const time = state.context.currentTime - Math.max(0, wallNow - message.wallTime) / 1000;
          const bytes = new Uint8Array(message.jpeg);
          let binary = '';
          for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
          addFrame(time, btoa(binary));
        } else if (message.type === 'error') {
          stopCapture();
          notice('视频采集中断，设备已关闭：{message}', true, { message: message.message });
        }
      };
      worker.onerror = () => {
        if (generation === state.generation) { stopCapture(); notice('视频采集中断，已停止以保持音视频同步。请重新开启设备。', true); }
      };
      if (mode === 'fallback') {
        state.frameTimer = setInterval(captureFrame, 1000 / FPS);
        notice('当前浏览器使用前台视频采集。自动检测仍可使用，请保持 Falcon 页面可见。');
      }
      return;
    } catch {
      worker?.terminate();
      if (generation !== state.generation) return;
      state.videoWorker = null;
    }
  }
  state.videoMode = 'fallback';
  if (state.trackingMode === 'auto') setTrackingStatus({ status: 'error', message: '当前浏览器无法运行本地嘴部检测。请使用 Chrome，或明确选择手动框选。' });
  state.frameTimer = setInterval(captureFrame, 1000 / FPS);
  notice('当前浏览器无法运行自动嘴部检测。请使用 Chrome，或点击「手动框选」后继续。', true);
}
// Each crop keeps its capture time on the clip's sample axis; the server picks the crop
// nearest to every video slot by that time, like Studio's frameWindow.
function addFrame(time, data) {
  if (!trackingAllowsInference()) return;
  captureHealth.frameAt = performance.now();
  if (!state.running || state.audioOrigin === null) return;
  const position = (time - state.audioOrigin) * RATE - state.clipStart;
  if (position < -RATE / FPS || position > state.clipAudio.length + RATE || state.clipFrames.length >= state.maxSeconds * FPS * 2) return;
  state.clipFrames.push({ position, data });
}
async function captureFrame() {
  const video = $('cameraVideo');
  if (!state.context || video.readyState < 2 || !video.videoWidth || state.bitmapBusy) return;
  if (state.videoWorker) {
    const generation = state.generation;
    const wallTime = performance.timeOrigin + performance.now();
    const trackingEpoch = state.trackingEpoch;
    state.bitmapBusy = true;
    let bitmap;
    try {
      bitmap = await createImageBitmap(video);
      if (generation !== state.generation || !state.videoWorker) { bitmap.close(); return; }
      if (trackingEpoch !== state.trackingEpoch) { bitmap.close(); state.bitmapBusy = false; return; }
      state.videoWorker.postMessage({ type: 'frame', bitmap, wallTime, trackingEpoch }, [bitmap]);
      state.bitmapTimeout = setTimeout(() => {
        if (generation === state.generation && state.bitmapBusy) {
          stopCapture();
          notice('视频处理未响应，已关闭设备。请重新开启或使用 Chrome。', true);
        }
      }, 10000);
    } catch (error) {
      bitmap?.close();
      if (generation === state.generation) { state.bitmapBusy = false; stopCapture(); notice('无法读取摄像头帧：{message}', true, { message: errorMessage(error) }); }
    }
    return;
  }
  // An explicit manual mode remains available when worker/ML APIs are unsupported.
  if (state.trackingMode !== 'manual') return;
  const time = state.context.currentTime;
  const size = Math.min(video.videoWidth, video.videoHeight) * state.roi.size;
  cropContext.drawImage(video, state.roi.cx * video.videoWidth - size / 2, state.roi.cy * video.videoHeight - size / 2, size, size, 0, 0, 96, 96);
  const frame = cropContext.getImageData(0, 0, 96, 96);
  for (let i = 0; i < frame.data.length; i += 4) {
    const gray = Math.round(frame.data[i] * .299 + frame.data[i + 1] * .587 + frame.data[i + 2] * .114);
    frame.data[i] = frame.data[i + 1] = frame.data[i + 2] = gray;
  }
  cropContext.putImageData(frame, 0, 0);
  addFrame(time, cropCanvas.toDataURL('image/jpeg', .88).split(',')[1]);
}
function encodeFloat32(samples) {
  const bytes = new Uint8Array(samples.length * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < samples.length; i++) view.setFloat32(i * 4, samples[i], true);
  let result = '';
  for (let i = 0; i < bytes.length; i += 8192) result += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(result);
}
function decodeFloat32(encoded) {
  const binary = atob(encoded);
  if (!binary.length || binary.length % 4) throw new Error('模型返回了无效的音频数据。');
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const samples = new Float32Array(bytes.length / 4);
  for (let i = 0; i < samples.length; i++) {
    const value = view.getFloat32(i * 4, true);
    if (!Number.isFinite(value)) throw new Error('模型输出包含无效数值，已停止分离。');
    samples[i] = value;
  }
  return samples;
}
function startRecording() {
  if (!state.stream || state.running || state.separating || !state.status.ready) return;
  state.clipAudio = new Float32Array(state.maxSeconds * RATE);
  state.clipSamples = 0;
  state.clipFrames = [];
  state.clipStart = state.samples;
  state.remaining = null;
  state.running = true;
  captureHealth.startedAt = performance.now();
  captureHealth.warning = '';
  audioInspector.setRunning(true);
  $('notice').hidden = true;
  setText($('performanceHint'), '录音中 · 停止后整段上传分离');
  setText($('outputCaption'), '录音结束后显示分离结果');
  refreshTrackingUI();
}
// The clip starts half a frame before the first crop and ends 160 ms after the last one
// (Studio's stale-video limit): audio without any mouth crop at its ends is not sent.
function buildClip() {
  const frames = state.clipFrames.slice().sort((a, b) => a.position - b.position);
  if (!frames.length) throw new Error('录音中没有有效的嘴部画面。请锁定嘴部后重新录制。');
  const half = RATE / FPS / 2;
  const start = Math.max(0, Math.floor(frames[0].position - half));
  const end = Math.min(state.clipSamples, Math.ceil(frames.at(-1).position + STALE));
  if (end - start < RATE) throw new Error('有效录音不足 1 秒，请锁定嘴部后录制更长一些。');
  const audio = state.clipAudio.slice(start, end);
  if (frames.length < (end - start) / RATE * FPS / 4) throw new Error('嘴部画面过少，请保持嘴部在画面内后重新录制。');
  return { audio, frames: frames.map((frame) => frame.data), times: frames.map((frame) => Math.max(0, frame.position - start)) };
}
async function finishRecording() {
  if (!state.running) return;
  state.running = false;
  audioInspector.setRunning(false);
  let clip;
  try {
    clip = buildClip();
  } catch (error) {
    notice(errorMessage(error), true);
    setText($('performanceHint'), '未发送录音');
    refreshTrackingUI();
    return;
  } finally {
    state.clipAudio = null;
    state.clipFrames = [];
  }
  await separateClip(clip);
}
// Inside the huggingface.co Space page, the parent hands each request a signed token so
// ZeroGPU counts the visitor's own quota (the handshake of @gradio/client 2.x).
const ZEROGPU_ORIGIN = location.hostname.endsWith('.hf.space') ? 'https://huggingface.co' : null;
let zeroGpuHeaders = false;
window.addEventListener('message', (event) => {
  if (event.origin === ZEROGPU_ORIGIN && event.data === 'supports-zerogpu-headers') zeroGpuHeaders = true;
});
if (ZEROGPU_ORIGIN && window.parent !== window) window.parent.postMessage('supports-zerogpu-headers', ZEROGPU_ORIGIN);
function requestZeroGpuHeaders() {
  if (!zeroGpuHeaders || window.parent === window) return Promise.resolve({});
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    const timer = setTimeout(() => { channel.port1.close(); resolve({}); }, 3000);
    channel.port1.onmessage = ({ data }) => {
      clearTimeout(timer);
      channel.port1.close();
      resolve(Object.fromEntries(Object.entries(data || {}).filter(([, value]) => typeof value === 'string')));
    };
    window.parent.postMessage('zerogpu-headers', ZEROGPU_ORIGIN, [channel.port2]);
  });
}
// Gradio's /call protocol: POST returns an event id, GET streams the result as SSE.
async function callSeparate(payload, signal) {
  const headers = { 'Content-Type': 'application/json', 'x-gradio-user': 'app', ...await requestZeroGpuHeaders() };
  const posted = await fetch('/gradio_api/call/separate', { method: 'POST', headers, body: JSON.stringify({ data: [payload] }), signal });
  const body = await posted.json().catch(() => ({}));
  if (!posted.ok || !body.event_id) throw new Error(typeof body.detail === 'string' ? body.detail : '分离请求被拒绝（{status}）。'.replace('{status}', String(posted.status)));
  const response = await fetch(`/gradio_api/call/separate/${encodeURIComponent(body.event_id)}`, { signal });
  if (!response.ok || !response.body) throw new Error('分离请求被拒绝（{status}）。'.replace('{status}', String(response.status)));
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let split;
    while ((split = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const event = /^event: *(.*)$/m.exec(block)?.[1];
      const data = /^data: *(.*)$/m.exec(block)?.[1];
      if (event === 'complete') { reader.cancel().catch(() => {}); return JSON.parse(data)[0]; }
      if (event === 'error') {
        reader.cancel().catch(() => {});
        let detail = null;
        try { detail = JSON.parse(data); } catch {}
        const message = typeof detail === 'string' ? detail : detail?.error || '分离失败，请稍后重试。';
        // ZeroGPU's quota and queue messages give no next step inside this page.
        throw new Error(/^ZeroGPU/.test(detail?.title || '') ? '{message} 登录 Hugging Face 后可获得更多 GPU 配额和更高的排队优先级。'.replace('{message}', message.trim()) : message);
      }
    }
  }
  throw new Error('分离连接意外结束，请重试。');
}
async function separateClip(clip) {
  const seconds = clip.audio.length / RATE;
  const variant = $('modelSelect').value || 'small';
  state.separating = true;
  state.controller = new AbortController();
  const timer = setTimeout(() => state.controller?.abort(), 300000);
  setSession('正在分离 {seconds} 秒录音', '上传后在 ZeroGPU 上排队 · 通常只需几秒', { seconds: seconds.toFixed(1) });
  setText($('outputCaption'), '正在上传并分离…');
  setText($('performanceHint'), '正在上传并分离…');
  updateButtons();
  try {
    const result = await callSeparate({ variant, audio: encodeFloat32(clip.audio), frames: clip.frames, times: clip.times }, state.controller.signal);
    applyResult(result, clip, variant);
  } catch (error) {
    const aborted = error?.name === 'AbortError';
    if (!aborted || state.controller) notice(aborted ? '分离超过 5 分钟未完成，已取消。请稍后重试。' : '分离失败：{message}', true, { message: errorMessage(error) });
    setText($('outputCaption'), '分离未完成');
    setText($('performanceHint'), '分离未完成 · 可重新录制');
  } finally {
    clearTimeout(timer);
    state.controller = null;
    state.separating = false;
    if (state.stream) refreshTrackingUI();
    else setSession(state.recordingSamples ? '设备已关闭，声音已留存' : '设备已关闭', '开启设备即可开始新的会话');
    updateButtons();
  }
}
function envelope(samples, bars = 140) {
  const values = [];
  const step = Math.max(1, Math.floor(samples.length / bars));
  for (let i = 0; i + step <= samples.length; i += step) {
    let energy = 0;
    for (let j = 0; j < step; j++) energy += samples[i + j] ** 2;
    values.push(Math.sqrt(energy / step));
  }
  return values;
}
function applyResult(result, clip, variant) {
  if (!result || result.sample_rate !== RATE) throw new Error('服务返回了不支持的分离结果。');
  const target = decodeFloat32(result.audio);
  if (target.length !== clip.audio.length) throw new Error('服务返回的音频长度与录音不一致。');
  resetRecording();
  state.recording = [target];
  state.mixture = [clip.audio];
  state.recordingSamples = target.length;
  state.outputWave = envelope(target);
  const milliseconds = Number(result.gpu_ms), rtf = Number(result.rtf);
  $('latencyValue').innerHTML = Number.isFinite(milliseconds) ? `${Math.round(milliseconds)}<small>ms</small>` : '—<small>ms</small>';
  $('rtfValue').innerHTML = Number.isFinite(rtf) ? `${rtf.toFixed(3)}<small>×</small>` : '—<small>×</small>';
  setText($('performanceHint'), '{windows} 个 2 秒窗口 · 与实时工作台相同的 0.5 秒步长', { windows: Number(result.windows) || 0 });
  setText($('outputCaption'), '分离完成 · {model}', { model: variant === 'big' ? 'Falcon Big' : 'Falcon Small' });
  $('outputDuration').textContent = duration(target.length / RATE);
  $('recordingCount').textContent = '1';
  finalizeRecording();
}
// The separation in flight belongs to the clip already recorded, so closing the devices
// does not cancel it. Closing during a recording finishes and separates that recording.
function stopCapture() {
  if (state.running) void finishRecording();
  ++state.generation;
  state.opening = false;
  clearInterval(state.frameTimer);
  state.frameTimer = null;
  clearTimeout(state.bitmapTimeout);
  state.bitmapTimeout = null;
  state.bitmapBusy = false;
  if (state.videoWorker) {
    const worker = state.videoWorker;
    worker.postMessage({ type: 'stop' });
    setTimeout(() => worker.terminate(), 300);
    state.videoWorker = null;
  }
  state.videoMode = 'none';
  state.trackingStatus = 'idle';
  state.trackingMessage = '';
  state.detectionMs = null;
  state.lips = [];
  state.requiresReacquire = false;
  ++state.trackingEpoch;
  if (state.worklet) { state.worklet.port.onmessage = null; state.worklet.disconnect(); state.worklet = null; }
  state.source?.disconnect();
  state.source = null;
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  if (state.context && state.context.state !== 'closed') void state.context.close();
  state.context = null;
  $('cameraVideo').srcObject = null;
  $('cameraPlaceholder').hidden = false;
  $('cameraOverlay').hidden = true;
  $('cameraState').classList.remove('active');
  setText($('cameraState').querySelector('span'), '设备已关闭');
  setText($('inputCaption'), '采集已停止');
  $('inputLevel').textContent = '−∞ dB';
  if (!state.separating) {
    setText($('performanceHint'), '采集已停止 · 开启设备后继续');
    setSession(state.recordingSamples ? '设备已关闭，声音已留存' : '设备已关闭', '开启设备即可开始新的会话');
  }
  refreshTrackingUI();
}
function resetRecording() {
  $('recordingPlayer').pause();
  $('recordingPlayer').removeAttribute('src');
  if (state.recordingUrl) URL.revokeObjectURL(state.recordingUrl);
  if (state.mixtureUrl) URL.revokeObjectURL(state.mixtureUrl);
  state.recordingUrl = state.mixtureUrl = null;
  state.recordingFinalizedSamples = 0;
  audioInspector.clearRecording();
  state.recording = [];
  state.mixture = [];
  state.recordingSamples = 0;
  $('recordingCount').textContent = '0';
  $('outputDuration').textContent = '00:00';
  setText($('playRecording').querySelector('span'), '试听');
}
// Play, spectrum, seek and export follow the selected track; switching keeps the position.
function showRecordingTrack() {
  const mixture = state.recordingTrack === 'mixture';
  const url = mixture ? state.mixtureUrl : state.recordingUrl;
  const player = $('recordingPlayer');
  $('trackTarget').setAttribute('aria-pressed', String(!mixture));
  $('trackMixture').setAttribute('aria-pressed', String(mixture));
  if (!url) return;
  if (player.src !== url) {
    const position = player.currentTime || 0, playing = !player.paused;
    player.src = url;
    player.addEventListener('loadedmetadata', () => {
      player.currentTime = Math.min(position, player.duration || position);
      if (playing) player.play().catch(() => {});
    }, { once: true });
  }
  audioInspector.setRecording(mixture ? state.mixture : state.recording, state.recordingSamples);
  setText($('recordingTitle'), mixture ? '原始混合声音 · {duration}' : '目标说话者 · {duration}', { duration: duration(state.recordingSamples / RATE) });
}
function wavBlob(chunks, totalSamples) {
  const data = new ArrayBuffer(44 + totalSamples * 2);
  const view = new DataView(data);
  const writeString = (offset, string) => { for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i)); };
  writeString(0, 'RIFF'); view.setUint32(4, 36 + totalSamples * 2, true);
  writeString(8, 'WAVE'); writeString(12, 'fmt '); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, RATE, true);
  view.setUint32(28, RATE * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  writeString(36, 'data'); view.setUint32(40, totalSamples * 2, true);
  let offset = 44;
  for (const chunk of chunks) for (const value of chunk) {
    const sample = clamp(value, -1, 1);
    view.setInt16(offset, sample < 0 ? sample * 32768 : sample * 32767, true);
    offset += 2;
  }
  return new Blob([data], { type: 'audio/wav' });
}
function finalizeRecording() {
  if (!state.recordingSamples) return;
  if (state.recordingUrl && state.recordingFinalizedSamples === state.recordingSamples) return;
  if (state.recordingUrl) URL.revokeObjectURL(state.recordingUrl);
  if (state.mixtureUrl) URL.revokeObjectURL(state.mixtureUrl);
  state.recordingUrl = URL.createObjectURL(wavBlob(state.recording, state.recordingSamples));
  state.mixtureUrl = URL.createObjectURL(wavBlob(state.mixture, state.recordingSamples));
  state.recordingFinalizedSamples = state.recordingSamples;
  showRecordingTrack();
  setText($('recordingSubtitle'), '16 kHz · 单声道 · 本次片段已就绪');
}
function drawWave(canvas, values, color) {
  const ratio = Math.min(devicePixelRatio || 1, 2);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) { canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#74895b16'; ctx.lineWidth = 1;
  for (let y = height / 4; y < height; y += height / 4) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
  ctx.strokeStyle = '#80956529'; ctx.setLineDash([2, 5]); ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); ctx.stroke(); ctx.setLineDash([]);
  const bars = Math.floor(width / 4);
  const count = Math.min(bars, values.length);
  ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.lineCap = 'round';
  for (let n = 0; n < count; n++) {
    const value = values[values.length - count + n];
    const amplitude = Math.min(height * .46, Math.max(1, Math.sqrt(value) * height * 1.4));
    const x = width - (count - n) * 4;
    ctx.beginPath(); ctx.moveTo(x, height / 2 - amplitude); ctx.lineTo(x, height / 2 + amplitude); ctx.stroke();
  }
}
let lastDraw = 0;
function animate(now) {
  if (now - lastDraw > 60) {
    drawWave($('inputWave'), state.inputWave, '#93a77b');
    drawWave($('outputWave'), state.outputWave, '#c7e691');
    lastDraw = now;
  }
  requestAnimationFrame(animate);
}

$('primaryButton').addEventListener('click', () => {
  if (!state.stream) void openCapture().catch((error) => notice(errorMessage(error), true));
  else if (state.running) void finishRecording();
  else startRecording();
});
$('stopButton').addEventListener('click', stopCapture);
for (const id of ['cameraSelect', 'micSelect']) $(id).addEventListener('change', async () => {
  const ids = { camera: $('cameraSelect').value, microphone: $('micSelect').value };
  stopCapture();
  notice('已切换输入设备。确认嘴部取景后，点击「开始录制」。');
  await openCapture(ids);
});
$('autoTrackingButton').addEventListener('click', () => setTrackingMode('auto'));
$('manualTrackingButton').addEventListener('click', () => setTrackingMode('manual'));
$('reacquireButton').addEventListener('click', () => setTrackingMode('auto', true));
$('centerRoi').addEventListener('click', () => { state.roi = { cx: .5, cy: .6, size: .2 }; setTrackingMode('manual', true); });
let drag = null;
$('mouthRoi').addEventListener('pointerdown', (event) => {
  if (event.button !== 0) return;
  setTrackingMode('manual');
  const geometry = videoGeometry();
  drag = { x: event.clientX, y: event.clientY, roi: { ...state.roi }, geometry, resize: event.target === $('roiResize'), pointerId: event.pointerId };
  $('mouthRoi').setPointerCapture(event.pointerId);
  event.preventDefault();
});
$('mouthRoi').addEventListener('pointermove', (event) => {
  if (!drag || event.pointerId !== drag.pointerId) return;
  const dx = (event.clientX - drag.x) / drag.geometry.scale;
  const dy = (event.clientY - drag.y) / drag.geometry.scale;
  if (drag.resize) state.roi.size = clamp(drag.roi.size + Math.max(dx, dy) / Math.min(drag.geometry.width, drag.geometry.height) * 2, .09, .48);
  else { state.roi.cx = drag.roi.cx + dx / drag.geometry.width; state.roi.cy = drag.roi.cy + dy / drag.geometry.height; }
  positionROI();
});
function finishDrag() { if (drag) setTrackingMode('manual', true); drag = null; }
$('mouthRoi').addEventListener('pointerup', finishDrag);
$('mouthRoi').addEventListener('pointercancel', finishDrag);
$('mouthRoi').addEventListener('keydown', (event) => {
  const step = event.shiftKey ? .025 : .005;
  const keys = { ArrowLeft: () => state.roi.cx -= step, ArrowRight: () => state.roi.cx += step, ArrowUp: () => state.roi.cy -= step, ArrowDown: () => state.roi.cy += step, '+': () => state.roi.size = clamp(state.roi.size + .01, .09, .48), '=': () => state.roi.size = clamp(state.roi.size + .01, .09, .48), '-': () => state.roi.size = clamp(state.roi.size - .01, .09, .48) };
  if (keys[event.key]) { event.preventDefault(); setTrackingMode('manual'); keys[event.key](); setTrackingMode('manual', true); }
});
new ResizeObserver(positionROI).observe($('cameraStage'));
$('cameraVideo').addEventListener('loadedmetadata', positionROI);
$('navStudio').addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
$('navRecordings').addEventListener('click', () => $('recordings').scrollIntoView({ behavior: 'smooth', block: 'start' }));
$('settingsButton').addEventListener('click', () => {
  $('deviceCard').scrollIntoView({ behavior: 'smooth', block: 'center' });
  $('deviceCard').classList.remove('attention');
  requestAnimationFrame(() => $('deviceCard').classList.add('attention'));
});
$('guideButton').addEventListener('click', () => $('guideDialog').showModal());
$('closeGuide').addEventListener('click', () => $('guideDialog').close());
$('guideDone').addEventListener('click', () => $('guideDialog').close());
$('guideDialog').addEventListener('click', (event) => { if (event.target === $('guideDialog')) { const rect = $('guideDialog').getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $('guideDialog').close(); } });
$('dismissNotice').addEventListener('click', () => $('notice').hidden = true);
$('playRecording').addEventListener('click', async () => {
  const player = $('recordingPlayer');
  if (!state.recordingUrl || state.running) return;
  try {
    if (player.paused) { await player.play(); setText($('playRecording').querySelector('span'), '暂停'); }
    else { player.pause(); setText($('playRecording').querySelector('span'), '试听'); }
  } catch (error) { notice('无法播放录音：{message}', true, { message: errorMessage(error) }); }
});
$('recordingPlayer').addEventListener('ended', () => setText($('playRecording').querySelector('span'), '试听'));
$('recordingPlayer').addEventListener('play', () => setText($('playRecording').querySelector('span'), '暂停'));
$('recordingPlayer').addEventListener('pause', () => setText($('playRecording').querySelector('span'), '试听'));
$('downloadButton').addEventListener('click', () => {
  if (!state.recordingUrl || state.running) return;
  const mixture = state.recordingTrack === 'mixture';
  const anchor = document.createElement('a');
  anchor.href = mixture ? state.mixtureUrl : state.recordingUrl;
  anchor.download = `falcon-${mixture ? 'mixture' : 'separated'}-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.wav`;
  document.body.append(anchor); anchor.click(); anchor.remove();
});
$('trackTarget').addEventListener('click', () => { state.recordingTrack = 'target'; showRecordingTrack(); });
$('trackMixture').addEventListener('click', () => { state.recordingTrack = 'mixture'; showRecordingTrack(); });
document.addEventListener('visibilitychange', () => {
  if (document.hidden && state.stream && state.videoMode !== 'worker') {
    const recording = state.running;
    stopCapture();
    notice(recording ? '页面进入后台，已停止采集；录到的部分会继续分离。回到此页面后可重新开启设备。' : '页面进入后台，已停止采集。回到此页面后可重新开启设备。');
  }
});
window.addEventListener('pagehide', (event) => {
  state.running = false; // leaving the page: do not upload a half-finished recording
  state.controller?.abort();
  state.controller = null;
  stopCapture();
  if (!event.persisted) { audioInspector.destroy(); if (state.recordingUrl) URL.revokeObjectURL(state.recordingUrl); }
});
navigator.mediaDevices?.addEventListener?.('devicechange', () => { if (state.stream) void enumerateDevices().catch(() => {}); });
initializeI18n();
void fetchStatus();
setInterval(fetchStatus, 15000);
requestAnimationFrame(animate);
