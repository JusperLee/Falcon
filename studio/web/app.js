import { setText, setAttr, initializeI18n } from './i18n.js';
import { createAudioInspector } from './audio-inspector.js';
import { createPlaybackUpsampler } from './playback-resampler.js';

const $ = (id) => document.getElementById(id);
const RATE = 16000;
const WINDOW = 2 * RATE;
const HOP = RATE / 2;
const GUARD = 1280; // 80 ms of right context the server keeps past each output chunk
const LEFT_CONTEXT = WINDOW - HOP - GUARD; // 1.42 s of silence before each fresh target
const FPS = 25;
const MAX_RECORDING_SECONDS = 300;
const state = {
  status: {}, stream: null, context: null, source: null, worklet: null, socket: null,
  frameTimer: null, frames: [], ring: new Float32Array(RATE * 8), samples: 0,
  audioOrigin: null, generation: 0, sessionEpoch: 0, running: false, starting: false,
  opening: false, loading: false, inFlight: null, lastSentEnd: 0, sequence: 0,
  needsReset: true, recording: [], recordingSamples: 0, recordingUrl: null, recordingFinalizedSamples: 0,
  mixture: [], mixtureUrl: null, recordingTrack: 'target', recordedEnd: 0, draining: null,
  sessionStart: 0, skipped: 0, playAt: 0, monitorSources: new Set(), frameId: 0,
  inputWave: [], outputWave: [], lastOutput: 0, lastResultEnd: null,
  roi: { cx: .5, cy: .60, size: .20 }, lastError: '', backendReachable: false,
  monitorGain: null, meetingGain: null, meetingDestination: null, meetingActive: false,
  meetingOutputs: [], meetingEpoch: 0, requestedVariant: null,
  videoWorker: null, videoMode: 'none',
  recordingCapped: false,
  trackingMode: 'auto', trackingStatus: 'idle', trackingEpoch: 0, inferenceEpoch: 0,
  lips: [], trackingMessage: '', detectionMs: null, requiresReacquire: false, searching: false,
  bitmapBusy: false, bitmapTimeout: null,
  playbackUpsampler: null,
};
const cropCanvas = document.createElement('canvas');
cropCanvas.width = cropCanvas.height = 96;
const cropContext = cropCanvas.getContext('2d', { willReadFrequently: true });
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const clamp = (n, low, high) => Math.min(high, Math.max(low, n));
const duration = (seconds) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
const audioInspector = createAudioInspector({ player: $('recordingPlayer'), onError: (message) => notice(message, true) });

// Aggregate-only troubleshooting: no PCM, images, cookies or device identifiers.
const captureHealth = { inputAt: 0, frameAt: 0, resultAt: 0, inputRms: 0, outputRms: 0, startedAt: 0, warning: '' };
window.falconDiagnostics = () => ({
  running: state.running, opening: state.opening, audioState: state.context?.state || 'closed',
  videoMode: state.videoMode, tracking: state.trackingStatus, searching: state.searching,
  samples: state.samples, frames: state.frames.length, sentWindows: state.sequence,
  recordingSeconds: state.recordingSamples / RATE,
  pendingMs: state.inFlight ? Math.round(performance.now() - state.inFlight.time) : null,
  inputRms: captureHealth.inputRms, outputRms: captureHealth.outputRms,
  inputAgeMs: Math.round(performance.now() - captureHealth.inputAt),
  frameAgeMs: Math.round(performance.now() - captureHealth.frameAt),
  resultAgeMs: Math.round(performance.now() - captureHealth.resultAt), warning: captureHealth.warning,
});
function checkCaptureHealth() {
  if (!state.running || state.draining || performance.now() - captureHealth.startedAt < 5000) return;
  const now = performance.now();
  let warning = '';
  if (now - captureHealth.inputAt > 3000) warning = '麦克风采集已暂停，请检查设备或浏览器是否挂起。';
  else if (now - captureHealth.frameAt > 3000) warning = '尚无新的有效嘴部画面，请检查摄像头、取景和自动检测状态。';
  else if (state.inFlight && now - state.inFlight.time > 5000) warning = '正在等待模型响应，请检查网络连接。';
  else if (!state.inFlight && now - captureHealth.resultAt > 5000) warning = '音视频尚未对齐，暂未发送新的推理请求。';
  else if (captureHealth.inputRms < .00001) warning = '麦克风输入接近静音，请确认选中了正在使用的麦克风。';
  else if (captureHealth.outputRms < .000001 && now - captureHealth.resultAt < 2000) warning = '模型返回接近静音，请确认嘴部取景与当前说话者一致。';
  if (warning) setText($('performanceHint'), warning);
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
    NotAllowedError: '摄像头或麦克风访问被拒绝。请在浏览器地址栏和 macOS「隐私与安全性」中允许访问，然后重试。',
    NotFoundError: '没有找到可用的摄像头或麦克风。请连接设备后重试。',
    NotReadableError: '设备暂时无法使用。请关闭其他正在使用摄像头或麦克风的应用后重试。',
    OverconstrainedError: '所选设备已不可用，请重新选择设备。',
  };
  return names[error?.name] || error?.message || String(error);
}
function setSession(title, subtitle) {
  setText($('sessionTitle'), title);
  if (subtitle) setText($('sessionSubtitle'), subtitle);
}
function updateButtons() {
  const primary = $('primaryButton');
  primary.disabled = state.opening || state.starting || state.running;
  setText(primary.querySelector('span'), state.opening ? '正在开启设备…' : state.starting ? '准备模型中…' : state.running ? trackingAllowsInference() ? '正在分离' : '等待目标' : state.stream ? '开始分离' : '开启设备');
  primary.querySelector('svg').innerHTML = state.stream ? '<path d="m8 4 12 8-12 8V4Z"/>' : '<path d="M12 4v16M4 12h16"/>';
  $('stopButton').hidden = !state.stream && !state.opening;
  $('cameraSelect').disabled = !state.stream || state.opening;
  $('micSelect').disabled = !state.stream || state.opening;
  $('loadButton').disabled = state.status.ready || state.status.loading || state.loading;
  setText($('loadButton').querySelector('span'), state.status.ready ? '模型已就绪' : state.status.loading || state.loading ? '正在加载模型…' : '加载本地模型');
  $('modelSelect').disabled = state.running || state.starting || state.loading || !!state.status.loading;
  $('meetingSelect').disabled = !state.stream || state.meetingActive || !state.meetingOutputs.length;
  $('meetingToggle').disabled = !state.stream || !state.meetingOutputs.length || !('setSinkId' in HTMLMediaElement.prototype);
  $('sessionLight').classList.toggle('running', state.running && trackingAllowsInference());
  setText($('captureLabel'), state.running ? trackingAllowsInference() ? '正在分离' : '等待目标' : '设备预览');
  $('captureLabel').parentElement.classList.toggle('running', state.running && trackingAllowsInference());
  $('playRecording').disabled = state.running || !state.recordingSamples;
  $('downloadButton').disabled = state.running || !state.recordingSamples;
  $('trackTarget').disabled = $('trackMixture').disabled = state.running || !state.recordingSamples;
  $('reacquireButton').disabled = !state.stream || state.opening || state.trackingMode !== 'auto';
}
function applyStatus(status) {
  state.status = status;
  state.backendReachable = true;
  if (Array.isArray(status.variants)) {
    const selected = state.requestedVariant || status.variant || $('modelSelect').value || 'small';
    $('modelSelect').replaceChildren(...status.variants.map((variant) => {
      const option = document.createElement('option');
      option.value = variant.id;
      const label = !variant.available ? '{model} · 不可用' : variant.id === 'small' ? '{model} · 实时优先' : '{model} · 高容量';
      // Weights not in models/ yet are fetched from Hugging Face on first load.
      setText(option, variant.source === 'huggingface' ? `${label} · 首次使用将下载` : label, { model: variant.name || variant.id });
      option.disabled = !variant.available;
      return option;
    }));
    $('modelSelect').value = selected;
  }
  const loading = !!status.loading || state.loading;
  const title = status.ready ? '模型已就绪' : loading ? '模型加载中' : status.error ? '加载失败' : '等待模型';
  $('enginePill').classList.toggle('ready', !!status.ready);
  $('enginePill').classList.toggle('loading', loading && !status.ready);
  setText($('enginePill').querySelector('span'), title);
  setText($('modelState'), status.ready ? '就绪 · 可开始' : loading ? '加载中…' : status.error ? '加载失败' : '尚未加载');
  setText($('modelDevice'), (status.device || '自动选择').toUpperCase());
  if (status.ready) setText($('modelFootnote'), '本机推理 · 摄像头与麦克风不上传云端');
  else if (status.error) setText($('modelFootnote'), status.error);
  else setText($('modelFootnote'), loading ? '正在读取权重，首次启动可能需要稍等。' : '首次加载需要片刻，完成后即可开始。');
  if (status.error && status.error !== state.lastError) {
    state.lastError = status.error;
    notice('模型加载失败：{message}', true, { message: status.error });
  }
  updateButtons();
}
async function fetchStatus() {
  try {
    const response = await fetch('/api/status', { signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error(`服务响应 ${response.status}`);
    applyStatus(await response.json());
  } catch {
    state.backendReachable = false;
    setText($('enginePill').querySelector('span'), '服务未连接');
    $('enginePill').classList.remove('ready', 'loading');
    setText($('modelState'), '服务未连接');
    setText($('modelFootnote'), '请保持本地 Falcon 服务运行，然后重试。');
  }
}
async function loadModel(variant = $('modelSelect').value || 'small') {
  if ((state.status.ready && (!state.status.variant || state.status.variant === variant)) || state.status.loading || state.loading) return;
  state.loading = true;
  state.lastError = '';
  updateButtons();
  try {
    const response = await fetch('/api/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ variant }), signal: AbortSignal.timeout(180000) });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || body.error || `模型加载请求失败 (${response.status})`);
    applyStatus(body);
  } catch (error) {
    notice(errorMessage(error), true);
    throw error;
  } finally {
    state.loading = false;
    state.requestedVariant = null;
    updateButtons();
    void fetchStatus();
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
  const previous = $('meetingSelect').value;
  state.meetingOutputs = devices.filter((device) => device.kind === 'audiooutput' && device.deviceId !== 'default' && device.deviceId !== 'communications' && /blackhole|loopback|vb[- ]?audio|virtual cable|cable input/i.test(device.label));
  const outputSelect = $('meetingSelect');
  outputSelect.replaceChildren();
  const placeholder = document.createElement('option');
  placeholder.value = '';
  setText(placeholder, state.meetingOutputs.length ? '选择虚拟声卡' : '未找到虚拟声卡');
  outputSelect.append(placeholder);
  for (const device of state.meetingOutputs) {
    const option = document.createElement('option');
    option.value = device.deviceId;
    option.textContent = device.label;
    outputSelect.append(option);
  }
  if (state.meetingOutputs.some((device) => device.deviceId === previous)) outputSelect.value = previous;
  else if (state.meetingActive) {
    disableMeeting();
    notice('会议虚拟声卡已断开，会议输出已关闭。重新连接并选择设备后再启用。', true);
  }
  if (!('setSinkId' in HTMLMediaElement.prototype)) setText($('meetingHint'), '当前浏览器不支持指定输出声卡，请使用 Chrome 打开本地页面。');
  else if (!state.meetingOutputs.length) setText($('meetingHint'), '请先安装 BlackHole 2ch 并重启电脑，或连接已有的虚拟声卡。');
  else if (!state.meetingActive) setText($('meetingHint'), '选择虚拟声卡，并在会议软件中将它设为麦克风。');
  updateButtons();
}
async function openCapture(deviceIds = {}) {
  if (state.opening || state.stream) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('设备访问需要安全连接。请使用启动脚本打开的 http://127.0.0.1 地址。');
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
    state.monitorGain = context.createGain();
    state.monitorGain.gain.value = $('monitorToggle').checked ? 1 : 0;
    state.monitorGain.connect(context.destination);
    state.meetingGain = context.createGain();
    state.meetingGain.gain.value = 0;
    Object.assign(captureHealth, { inputAt: performance.now(), frameAt: performance.now(), resultAt: performance.now(), inputRms: 0, outputRms: 0, warning: '' });
    state.audioOrigin = null;
    state.samples = 0;
    state.frames = [];
    state.inputWave = [];
    audioInspector.resetLive();
    state.ring.fill(0);
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
  if (!state.context || !samples?.length || state.draining) return;
  if (state.audioOrigin === null) state.audioOrigin = startTime;
  // A suspended context or lost packet invalidates old audio/video alignment.
  const expectedTime = state.audioOrigin + state.samples / RATE;
  if (Math.abs(startTime - expectedTime) > .08) {
    state.audioOrigin = startTime - state.samples / RATE;
    resetTrackingWindow();
  }
  let energy = 0;
  for (const sample of samples) {
    state.ring[state.samples % state.ring.length] = Number.isFinite(sample) ? sample : 0;
    state.samples++;
    energy += sample * sample;
  }
  const rms = Math.sqrt(energy / samples.length);
  captureHealth.inputAt = performance.now(); captureHealth.inputRms = rms;
  $('inputLevel').textContent = rms > .00001 ? `${Math.max(-99, 20 * Math.log10(rms)).toFixed(0)} dB` : '−∞ dB';
  state.inputWave.push(rms);
  if (state.inputWave.length > 140) state.inputWave.splice(0, state.inputWave.length - 140);
  audioInspector.pushInput(samples);
  const elapsed = state.running ? (state.samples - state.sessionStart) / RATE : state.samples / RATE;
  $('inputDuration').textContent = duration(elapsed);
  $('captureTime').textContent = duration(elapsed);
  maybeInfer();
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
function resetTrackingWindow() {
  ++state.inferenceEpoch;
  state.frames = [];
  state.sessionEpoch = state.samples;
  state.lastSentEnd = 0;
  state.lastResultEnd = null;
  state.needsReset = true;
  state.outputWave = [];
  audioInspector.resetOutput();
  stopMonitoring();
  // Keep the one outstanding server request until it resolves, but invalidate its
  // result by epoch. Clearing inFlight here would let slow GPU work accumulate.
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
  if (state.stream && !state.starting) {
    if (!trackingAllowsInference()) {
      setSession(state.trackingStatus === 'loading' ? '正在准备自动嘴部检测' : '等待清晰的目标嘴部', state.trackingStatus === 'error' ? '检测未就绪 · 可点击「手动框选」继续' : state.requiresReacquire ? '目标发生变化 · 请点击「重新捕捉」' : '未锁定时暂停分离，避免提取错误的声音');
      if (state.running) {
        setText($('outputCaption'), '等待有效嘴部 · 输出已暂停');
        setText($('performanceHint'), '目标恢复后约 0.6 秒有效输入即可恢复');
      }
    } else {
      setSession(state.running ? '正在聚焦你的目标声音' : automatic ? '已锁定嘴部，可开始分离' : '将橙色取景框对准嘴部', automatic ? '自动跟随嘴部 · 失去目标时暂停输出' : '手动取景 · 请保持嘴部在框内');
      if (state.running && !state.frames.length) {
        setText($('outputCaption'), '正在积累新的有效上下文…');
        setText($('performanceHint'), '约 0.6 秒有效音视频后恢复输出');
      }
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
  if (next !== state.trackingStatus) {
    state.trackingStatus = next;
    resetTrackingWindow();
  }
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
  resetTrackingWindow();
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
  resetTrackingWindow();
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
function addFrame(time, data) {
  if (!trackingAllowsInference() || state.draining) return;
  captureHealth.frameAt = performance.now();
  state.frames.push({ time, data });
  while (state.frames.length > FPS * 6 || (state.frames[0] && state.frames[0].time < time - 6)) state.frames.shift();
  maybeInfer();
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
function frameWindow(startTime, endTime) {
  const epochTime = state.audioOrigin + state.sessionEpoch / RATE;
  const padded = startTime < epochTime;
  // Synthetic history uses this target's first valid frame, never a preview or old target.
  let index = padded ? state.frames.findIndex((frame) => frame.time >= epochTime) : 0;
  if (index < 0 || state.frames.length - index < 2
      || state.frames[index].time > Math.max(startTime, epochTime) + .12
      || state.frames.at(-1).time < endTime - .12) return null;
  const first = index;
  const frames = [];
  for (let n = 0; n < FPS * 2; n++) {
    const targetTime = startTime + (n + .5) / FPS;
    if (padded && targetTime < epochTime) { frames.push(state.frames[first].data); continue; }
    while (index + 1 < state.frames.length && Math.abs(state.frames[index + 1].time - targetTime) < Math.abs(state.frames[index].time - targetTime)) index++;
    // Refuse stale video, e.g. background-tab timer throttling, instead of desynchronizing silently.
    if (Math.abs(state.frames[index].time - targetTime) > .16) return null;
    frames.push(state.frames[index].data);
  }
  return frames;
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
function maybeInfer() {
  if (!state.running || state.inFlight || state.socket?.readyState !== WebSocket.OPEN || state.audioOrigin === null) return;
  if (!trackingAllowsInference()) return;
  if (state.samples - state.sessionEpoch < HOP + GUARD) return;
  let end = state.sessionEpoch + GUARD + Math.floor((state.samples - state.sessionEpoch - GUARD) / HOP) * HOP;
  // Preserve the first half-second even when the video/inference scheduler starts late.
  if (!state.lastSentEnd) end = state.sessionEpoch + HOP + GUARD;
  // While stopping, separate every remaining window in order instead of skipping to the newest.
  if (state.draining && state.lastSentEnd) end = Math.min(end, state.lastSentEnd + HOP);
  if (end < state.lastSentEnd + HOP) return;
  const startTime = state.audioOrigin + (end - WINDOW) / RATE;
  const frames = frameWindow(startTime, state.audioOrigin + end / RATE);
  if (!frames) return;
  const audio = new Float32Array(WINDOW);
  for (let i = 0; i < WINDOW; i++) {
    const sample = end - WINDOW + i;
    if (sample >= state.sessionEpoch) audio[i] = state.ring[sample % state.ring.length];
  }
  const missed = state.lastSentEnd ? Math.max(0, Math.round((end - state.lastSentEnd) / HOP) - 1) : 0;
  const reset = state.needsReset || missed > 0;
  if (missed) {
    state.skipped += missed;
    setText($('performanceHint'), '已跳过 {count} 个旧窗口，优先处理最新声音', { count: state.skipped });
  }
  const id = ++state.sequence;
  state.inFlight = { id, end, audio, timestampOffset: LEFT_CONTEXT, inferenceEpoch: state.inferenceEpoch, time: performance.now(), timeout: setTimeout(() => {
    if (state.inFlight?.id === id) { stopCapture(); notice('本次推理超过 60 秒，已停止采集。请检查模型服务和计算设备后重试。', true); }
  }, 60000) };
  try {
    state.socket.send(JSON.stringify({ type: 'infer', id, audio: encodeFloat32(audio), frames, window_seconds: 2, hop_seconds: .5, end_sample: end + LEFT_CONTEXT, reset }));
    state.lastSentEnd = end;
    state.needsReset = false;
  } catch (error) {
    stopCapture();
    notice('音视频发送失败：{message}', true, { message: errorMessage(error) });
  }
}
async function startSession() {
  if (!state.stream || state.running || state.starting) return;
  const generation = state.generation;
  state.starting = true;
  updateButtons();
  try {
    await state.context.resume();
    if (!state.status.ready) {
      setSession('正在准备 Falcon 模型', '设备预览可用 · 权重加载完成后开始');
      await loadModel();
      const started = Date.now();
      while (!state.status.ready && generation === state.generation && Date.now() - started < 180000) {
        if (state.status.error && !state.status.loading) throw new Error(state.status.error);
        await delay(1500);
        await fetchStatus();
      }
      if (generation !== state.generation) return;
      if (!state.status.ready) throw new Error('模型仍未就绪，请稍后重试或检查服务终端。');
    }
    if (generation !== state.generation) return;
    const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    state.socket = socket;
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('连接分离服务超时。')), 10000);
      socket.onopen = () => { clearTimeout(timeout); resolve(); };
      socket.onerror = () => { clearTimeout(timeout); reject(new Error('无法连接分离服务，请确认本地服务仍在运行。')); };
      socket.onclose = () => { clearTimeout(timeout); reject(new Error('分离服务在连接时关闭。')); };
    });
    if (generation !== state.generation) { socket.close(); return; }
    socket.onmessage = (event) => { if (generation === state.generation && state.running) handleResult(event.data); };
    socket.onerror = () => { if (state.running) { stopCapture(); notice('分离服务连接出错，录音已保留。请检查服务后重试。', true); } };
    socket.onclose = () => { if (generation === state.generation && state.running) { stopCapture(); notice('分离服务连接已关闭，录音已保留。', true); } };
    resetRecording();
    state.running = true;
    captureHealth.startedAt = captureHealth.resultAt = performance.now();
    captureHealth.warning = '';
    audioInspector.setRunning(true);
    ++state.inferenceEpoch;
    state.sessionEpoch = state.samples;
    state.sessionStart = state.samples;
    state.lastSentEnd = 0;
    state.lastResultEnd = null;
    state.needsReset = true;
    state.skipped = 0;
    state.playAt = 0;
    state.outputWave = [];
    $('notice').hidden = true;
    $('latencyValue').innerHTML = '—<small>ms</small>';
    $('rtfValue').innerHTML = '—<small>×</small>';
    setText($('performanceHint'), '正在积累首段音视频，开头会完整保留');
    setText($('outputCaption'), '正在积累首段上下文…');
    setText($('recordingTitle'), '正在记录目标说话者');
    setText($('recordingSubtitle'), '录音最多保留 5 分钟 · 实时分离可持续运行');
    setSession('正在聚焦你的目标声音', '首段约需 0.6 秒输入 · 每 0.5 秒更新');
  } catch (error) {
    if (generation === state.generation) { state.socket?.close(); state.socket = null; notice(errorMessage(error), true); setSession('暂时无法开始分离', '设备仍可预览 · 请检查模型或服务状态'); }
  } finally {
    if (generation === state.generation) { state.starting = false; refreshTrackingUI(); }
  }
}
function handleResult(raw) {
  try {
    const result = JSON.parse(raw);
    if (result.type !== 'result' && result.type !== 'error') return;
    if (!state.inFlight || result.id !== state.inFlight.id) return;
    clearTimeout(state.inFlight.timeout);
    const received = state.inFlight;
    state.inFlight = null;
    if (received.inferenceEpoch !== state.inferenceEpoch || !trackingAllowsInference()) {
      maybeInfer();
      continueDrain();
      return;
    }
    if (result.type === 'error') throw new Error(result.message || '模型推理失败。');
    if (result.sample_rate !== RATE) throw new Error(`输出采样率不受支持：${result.sample_rate}`);
    const samples = decodeFloat32(result.audio);
    if (samples.length !== HOP) throw new Error('服务返回的音频长度不符合 0.5 秒步长，为避免重复播放已停止。');
    // Protocol timestamps include the synthetic left history; all UI/recording time stays real.
    const offset = received.timestampOffset || 0;
    if (!Number.isInteger(result.start_sample) || !Number.isInteger(result.end_sample)
        || result.end_sample !== received.end + offset - GUARD || result.start_sample !== result.end_sample - HOP) {
      throw new Error('模型输出时间戳与本次音视频窗口不匹配，已停止以避免播放错位。');
    }
    result.start_sample -= offset;
    result.end_sample -= offset;
    if (Number.isFinite(result.end_sample) && state.lastResultEnd !== null && result.end_sample <= state.lastResultEnd) return;
    if (Number.isFinite(result.end_sample)) state.lastResultEnd = result.end_sample;
    captureHealth.resultAt = performance.now();
    captureHealth.outputRms = Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
    audioInspector.pushOutput(samples);
    const remaining = MAX_RECORDING_SECONDS * RATE - state.recordingSamples;
    const kept = samples.subarray(0, Math.max(0, remaining));
    if (kept.length) {
      const mixture = inputSpan(result.start_sample, kept.length, received);
      state.recording.push(kept);
      state.mixture.push(mixture);
      state.recordingSamples += kept.length;
      state.recordedEnd = result.start_sample + kept.length;
    }
    if (state.recordingSamples >= MAX_RECORDING_SECONDS * RATE && !state.recordingCapped) {
      state.recordingCapped = true;
      notice('录音已达 5 分钟上限，实时分离和会议输出继续。停止后可导出已保存的片段。');
      setText($('recordingTitle'), '录音已保留 5 分钟');
      setText($('recordingSubtitle'), '实时分离和会议输出继续 · 新片段不再写入录音');
    }
    state.lastOutput = performance.now();
    for (let i = 0; i < samples.length; i += 160) {
      let energy = 0;
      const length = Math.min(160, samples.length - i);
      for (let j = 0; j < length; j++) energy += samples[i + j] ** 2;
      state.outputWave.push(Math.sqrt(energy / length));
    }
    if (state.outputWave.length > 140) state.outputWave.splice(0, state.outputWave.length - 140);
    const milliseconds = Number.isFinite(result.inference_ms) ? result.inference_ms : performance.now() - received.time;
    const rtf = Number.isFinite(result.rtf) ? result.rtf : milliseconds / 500;
    $('latencyValue').innerHTML = `${Math.round(milliseconds)}<small>ms</small>`;
    $('rtfValue').innerHTML = `${rtf.toFixed(2)}<small>×</small>`;
    setText($('performanceHint'), state.skipped ? '已跳过 {count} 个旧窗口，录音仅保留已完成片段' : rtf > 1 ? '推理慢于采集速度，将自动跳过积压窗口' : '跟得上实时采集 · 输出保护区 {delay} ms', { count: state.skipped, delay: result.output_delay_ms ?? 80 });
    setText($('outputCaption'), state.draining ? '正在处理最后一段声音…' : $('monitorToggle').checked ? '分离中 · 耳机监听已开启' : '分离中 · 监听已关闭');
    $('outputDuration').textContent = duration(state.recordingSamples / RATE);
    $('recordingCount').textContent = '1';
    if ($('monitorToggle').checked || state.meetingActive) playSamples(samples, result.contiguous);
    maybeInfer();
    continueDrain();
  } catch (error) {
    stopCapture();
    notice(errorMessage(error), true);
  }
}
function playSamples(samples, contiguous) {
  const context = state.context;
  if (!context || context.state !== 'running') return;
  if (state.playAt > context.currentTime + 1.5) {
    stopMonitoring(); // Drop any exceptional backlog instead of increasing live latency forever.
  }
  // Chrome resamples an AudioBuffer whose sampleRate differs from the
  // AudioContext's with plain 2-point linear interpolation and no low-pass
  // filtering (RenderFromBuffer/ProcessInterpolatedPath in
  // audio_buffer_source_handler.cc), which lets the separated voice's
  // spectral images above 8 kHz through almost unattenuated and audibly
  // changes its timbre. Upsample explicitly to context.sampleRate first, with
  // our own low-pass, so the AudioBuffer is created at the context's native
  // rate and Chrome takes the unfiltered fast path (playback rate exactly 1)
  // instead. A discontinuous result (model reset, or a fresh playAt schedule
  // after stopMonitoring/backlog-drop) must not blend FIR history or
  // fractional phase from unrelated prior audio into the new stream.
  let toPlay = samples;
  if (context.sampleRate !== RATE) {
    if (!state.playbackUpsampler || state.playbackUpsampler.targetRate !== context.sampleRate) {
      state.playbackUpsampler = createPlaybackUpsampler(context.sampleRate, RATE);
    } else if (contiguous === false) {
      state.playbackUpsampler.reset();
    }
    toPlay = state.playbackUpsampler.push(samples);
    if (!toPlay.length) return;
  }
  // A chunk that does not continue the previous one (model reset or a playback gap)
  // starts with a 5 ms fade-in so the hard edge does not click.
  if (contiguous === false || state.playAt < context.currentTime + .03) {
    toPlay = Float32Array.from(toPlay);
    const fade = Math.min(toPlay.length, Math.round((context.sampleRate || RATE) * .005));
    for (let i = 0; i < fade; i++) toPlay[i] *= i / fade;
  }
  const buffer = context.createBuffer(1, toPlay.length, context.sampleRate);
  buffer.copyToChannel(toPlay, 0);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(state.monitorGain);
  source.connect(state.meetingGain);
  const when = Math.max(context.currentTime + .03, state.playAt);
  source.start(when);
  state.playAt = when + buffer.duration;
  state.monitorSources.add(source);
  source.onended = () => { source.disconnect(); state.monitorSources.delete(source); };
}
function stopMonitoring() {
  for (const source of state.monitorSources) { try { source.stop(); source.disconnect(); } catch {} }
  state.monitorSources.clear();
  state.playAt = 0;
  state.playbackUpsampler?.reset();
}
async function enableMeeting() {
  const context = state.context;
  const deviceId = $('meetingSelect').value;
  const target = state.meetingOutputs.find((device) => device.deviceId === deviceId);
  const generation = state.generation;
  const epoch = ++state.meetingEpoch;
  const player = $('meetingPlayer');
  try {
    if (!context || !state.stream) throw new Error('请先开启摄像头和物理麦克风。');
    if (!('setSinkId' in HTMLMediaElement.prototype)) throw new Error('请使用 Chrome 开启会议输出；当前浏览器无法指定虚拟声卡。');
    if (!target || !deviceId) throw new Error('请明确选择 BlackHole、Loopback 或 VB-Audio 虚拟输出设备。');
    const microphone = state.stream.getAudioTracks()[0];
    if (/blackhole|loopback|vb[- ]?audio|virtual cable|cable (input|output)/i.test(microphone?.label || '')) throw new Error('Falcon 输入需要选择物理麦克风。虚拟麦克风作为输入会产生反馈回路。');
    $('meetingToggle').disabled = true;
    setText($('meetingHint'), '正在连接虚拟声卡…');
    await context.resume();
    // Route to the explicit sink before connecting/playing any processed audio.
    await player.setSinkId(deviceId);
    if (generation !== state.generation || epoch !== state.meetingEpoch) return;
    if (player.sinkId !== deviceId) throw new Error('浏览器未确认所选虚拟声卡，会议输出已取消。');
    state.meetingDestination = context.createMediaStreamDestination();
    state.meetingGain.connect(state.meetingDestination);
    player.srcObject = state.meetingDestination.stream;
    player.volume = 1;
    player.muted = false;
    await player.play();
    if (generation !== state.generation || epoch !== state.meetingEpoch) return;
    state.meetingGain.gain.value = 1;
    state.meetingActive = true;
    $('meetingToggle').checked = true;
    setText($('meetingHint'), state.videoMode === 'worker' ? '已连接 {device} · 后台采集已启用，可切换至会议窗口。' : '已连接 {device} · 请保持 Falcon 窗口可见。', { device: target.label });
  } catch (error) {
    if (generation === state.generation && epoch === state.meetingEpoch) {
      disableMeeting();
      notice(errorMessage(error), true);
      setText($('meetingHint'), '会议输出未开启，请检查所选虚拟声卡。');
    }
  } finally {
    updateButtons();
  }
}
function disableMeeting() {
  ++state.meetingEpoch;
  state.meetingActive = false;
  if (state.meetingGain) {
    state.meetingGain.gain.value = 0;
    state.meetingGain.disconnect();
  }
  if (state.meetingDestination) {
    state.meetingDestination.stream.getTracks().forEach((track) => track.stop());
    state.meetingDestination.disconnect();
    state.meetingDestination = null;
  }
  const player = $('meetingPlayer');
  player.pause();
  player.srcObject = null;
  $('meetingToggle').checked = false;
  setText($('meetingHint'), '会议输出已关闭；启用后将分离声音发送至所选虚拟声卡。');
  updateButtons();
}
// Stopping must not swallow the last words: the newest ~0.6-1.1 s is not separated yet
// (the window in flight, the samples since the last 0.5 s hop and the 80 ms right context).
// Pad it with silence and the last mouth frame, separate the remaining windows in order,
// trim the recording to the real end, then close. A second click stops at once.
function requestStop() {
  if (state.draining || !state.running || state.samples <= state.sessionEpoch || !state.frames.length || !trackingAllowsInference()) {
    stopCapture();
    return;
  }
  const end = state.samples;
  const target = state.sessionEpoch + GUARD + Math.ceil((end - state.sessionEpoch) / HOP) * HOP;
  for (; state.samples < target; state.samples++) state.ring[state.samples % state.ring.length] = 0;
  const last = state.frames.at(-1);
  for (let time = last.time + 1 / FPS; time < state.audioOrigin + target / RATE + 1 / FPS; time += 1 / FPS) {
    state.frames.push({ time, data: last.data });
  }
  const generation = state.generation;
  state.draining = { end, target, finished: false, timer: setTimeout(() => { if (generation === state.generation) finishDrain(); }, 8000) };
  setText($('outputCaption'), '正在处理最后一段声音…');
  maybeInfer();
  continueDrain();
}
function continueDrain() {
  const drain = state.draining;
  if (drain && !drain.finished && !state.inFlight && (state.recordedEnd >= drain.end || state.lastSentEnd >= drain.target)) finishDrain();
}
function finishDrain() {
  const drain = state.draining;
  if (!drain || drain.finished) return;
  drain.finished = true;
  clearTimeout(drain.timer);
  if (state.inFlight) { clearTimeout(state.inFlight.timeout); state.inFlight = null; }
  // Drop the padded silence past the real end from both aligned recordings.
  let excess = Math.max(0, state.recordedEnd - drain.end);
  while (excess > 0 && state.recording.length) {
    const last = state.recording.length - 1, cut = Math.min(excess, state.recording[last].length);
    state.recording[last] = state.recording[last].subarray(0, state.recording[last].length - cut);
    state.mixture[last] = state.mixture[last].subarray(0, state.mixture[last].length - cut);
    if (!state.recording[last].length) { state.recording.pop(); state.mixture.pop(); }
    state.recordingSamples -= cut; state.recordedEnd -= cut; excess -= cut;
  }
  // Let audio already scheduled for headphones or the meeting finish playing.
  const generation = state.generation;
  const wait = state.context ? Math.min(2, Math.max(0, state.playAt - state.context.currentTime)) : 0;
  drain.timer = setTimeout(() => { if (generation === state.generation) stopCapture(); }, wait * 1000);
}
function stopCapture() {
  clearTimeout(state.draining?.timer);
  state.draining = null;
  ++state.generation;
  state.running = false;
  audioInspector.setRunning(false);
  state.starting = false;
  state.opening = false;
  if (state.inFlight) clearTimeout(state.inFlight.timeout);
  state.inFlight = null;
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
  ++state.inferenceEpoch;
  state.frames = [];
  if (state.socket) {
    state.socket.onmessage = state.socket.onerror = state.socket.onclose = null;
    state.socket.close();
    state.socket = null;
  }
  stopMonitoring();
  disableMeeting();
  state.monitorGain?.disconnect();
  state.meetingGain?.disconnect();
  state.monitorGain = state.meetingGain = null;
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
  setText($('performanceHint'), '采集已停止 · 开启设备后继续');
  $('inputLevel').textContent = '−∞ dB';
  setText($('outputCaption'), state.recordingSamples ? '已保存本次分离片段' : '等待真实模型输出');
  setSession(state.recordingSamples ? '设备已关闭，声音已留存' : '设备已关闭', '开启设备即可开始新的会话');
  finalizeRecording();
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
  state.recordedEnd = 0;
  state.recordingSamples = 0;
  state.recordingCapped = false;
  $('recordingCount').textContent = '0';
  $('outputDuration').textContent = '00:00';
  setText($('playRecording').querySelector('span'), '试听');
}
// The microphone input the model saw for one output chunk, so the mixture lines up with
// the separated voice sample for sample (same gaps where windows were skipped).
function inputSpan(start, length, request) {
  const offset = start - (request.end - WINDOW);
  if (!(request.audio instanceof Float32Array) || offset < 0 || offset + length > request.audio.length) {
    throw new Error('原始音频窗口已失效，请重新开始分离。');
  }
  // The ring continues receiving microphone samples during inference. Use the
  // immutable request snapshot even if a slow response outlives the 8 s ring.
  return request.audio.slice(offset, offset + length);
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
  if (!state.recordingSamples) {
    setText($('recordingTitle'), '尚未生成分离片段');
    setText($('recordingSubtitle'), '开始分离，等待模型返回后即可保存');
    return;
  }
  // Stopping a preview must not rebuild an unchanged recording or lose its seek position.
  if (state.recordingUrl && state.recordingFinalizedSamples === state.recordingSamples) return;
  if (state.recordingUrl) URL.revokeObjectURL(state.recordingUrl);
  if (state.mixtureUrl) URL.revokeObjectURL(state.mixtureUrl);
  state.recordingUrl = URL.createObjectURL(wavBlob(state.recording, state.recordingSamples));
  state.mixtureUrl = URL.createObjectURL(wavBlob(state.mixture, state.recordingSamples));
  state.recordingFinalizedSamples = state.recordingSamples;
  showRecordingTrack();
  setText($('recordingSubtitle'), state.skipped ? '已跳过 {count} 个旧窗口 · WAV 按已完成片段顺序拼接' : '16 kHz · 单声道 · 本次片段已就绪', { count: state.skipped });
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
  else void startSession();
});
$('stopButton').addEventListener('click', requestStop);
$('loadButton').addEventListener('click', () => void loadModel().catch(() => {}));
$('monitorToggle').addEventListener('change', async () => {
  if (state.monitorGain) state.monitorGain.gain.value = $('monitorToggle').checked ? 1 : 0;
  if (!$('monitorToggle').checked && !state.meetingActive) stopMonitoring();
  else if (state.context?.state === 'suspended') await state.context.resume();
  if (state.running && state.recordingSamples && trackingAllowsInference()) setText($('outputCaption'), $('monitorToggle').checked ? '分离中 · 耳机监听已开启' : '分离中 · 监听已关闭');
});
$('modelSelect').addEventListener('change', async () => {
  if (state.running || state.starting) return;
  state.requestedVariant = $('modelSelect').value;
  await loadModel(state.requestedVariant).catch(() => {});
});
$('meetingToggle').addEventListener('change', () => {
  if ($('meetingToggle').checked) void enableMeeting();
  else { disableMeeting(); if (!$('monitorToggle').checked) stopMonitoring(); }
});
$('meetingPlayer').addEventListener('error', () => {
  if (state.meetingActive) { disableMeeting(); notice('虚拟声卡输出失败，已关闭会议输出。请重新选择虚拟设备。', true); }
});
$('meetingGuideButton').addEventListener('click', () => $('guideDialog').showModal());
for (const id of ['cameraSelect', 'micSelect']) $(id).addEventListener('change', async () => {
  const ids = { camera: $('cameraSelect').value, microphone: $('micSelect').value };
  stopCapture();
  notice('已切换输入设备并停止上一次分离。确认嘴部取景后，点击「开始分离」。');
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
  if (document.hidden && state.running && state.videoMode !== 'worker') {
    stopCapture();
    notice('页面进入后台，已停止采集并保留片段。回到此页面后重新开启设备，以保持音视频同步。');
  }
});
window.addEventListener('pagehide', (event) => {
  stopCapture();
  if (!event.persisted) { audioInspector.destroy(); if (state.recordingUrl) URL.revokeObjectURL(state.recordingUrl); }
});
navigator.mediaDevices?.addEventListener?.('devicechange', () => { if (state.stream) void enumerateDevices().catch(() => {}); });
initializeI18n();
void fetchStatus();
setInterval(fetchStatus, 4000);
requestAnimationFrame(animate);
