/* Control-flow tests using the real app.js with minimal DOM/audio stubs.
 * No camera, detector, model, browser rendering, or conference call is simulated
 * as a quality test here. Run: node tests/check_tracking_ui.cjs
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');
// Localization has separate real-module tests. Keep these lifecycle tests
// dependency-free while executing the application body and language callback.
const executableSource = source.replace(/^import\s+[^;]+?from\s+['"]\.\/(?:i18n|audio-inspector|playback-resampler)\.js['"];?\s*$/gm, '');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));

function harness() {
  const elements = new Map();
  const languageCallbacks = [];
  const inspectorCalls = [];
  function element(id = '') {
    const attributes = new Map();
    const children = new Map();
    const node = {
      id, style: {}, dataset: {}, value: '', checked: false, hidden: false,
      disabled: false, textContent: '', innerHTML: '', options: [],
      videoWidth: 1280, videoHeight: 720, readyState: 4, listeners: new Map(),
      classList: { add() {}, remove() {}, toggle() {} },
      setAttribute(key, value) { attributes.set(key, value); },
      getAttribute(key) { return attributes.get(key); },
      removeAttribute(key) { attributes.delete(key); },
      getBoundingClientRect() { return { width: 640, height: 360, left: 0, top: 0, right: 640, bottom: 360 }; },
      querySelector(selector) { if (!children.has(selector)) children.set(selector, element()); return children.get(selector); },
      addEventListener(type, handler) { this.listeners.set(type, handler); },
      replaceChildren(...items) { this.options = items; }, append(item) { this.options.push(item); },
      getContext() { return {}; }, pause() {}, play() { return Promise.resolve(); },
      setPointerCapture() {}, scrollIntoView() {}, showModal() {}, close() {}, remove() {}, click() {},
    };
    node.parentElement = node;
    return node;
  }
  for (const id of ids) elements.set(id, element(id));
  const sandbox = {
    console, Float32Array, Uint8Array, ArrayBuffer, DataView, Blob, URL, AbortSignal,
    btoa: (data) => Buffer.from(data, 'binary').toString('base64'),
    atob: (data) => Buffer.from(data, 'base64').toString('binary'),
    document: {
      hidden: false, body: element(), addEventListener() {},
      getElementById(id) { assert.ok(elements.has(id), `Missing DOM id: ${id}`); return elements.get(id); },
      createElement: element,
    },
    window: { addEventListener() {}, scrollTo() {} },
    navigator: { mediaDevices: { addEventListener() {} } },
    HTMLMediaElement: class { setSinkId() {} },
    ResizeObserver: class { observe() {} },
    WebSocket: { OPEN: 1 },
    performance: { timeOrigin: 0, now: () => 1000 },
    fetch: () => new Promise(() => {}), // No real network and no startup status mutation.
    setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
    requestAnimationFrame: () => 1,
    t: (key, params = {}) => key.replace(/\{(\w+)\}/g, (match, name) => Object.hasOwn(params, name) ? String(params[name]) : match),
    translateMessage: (message) => message,
    initializeI18n() {},
    onLanguageChange: (callback) => languageCallbacks.push(callback),
    createAudioInspector: () => Object.fromEntries(['pushInput','pushOutput','resetLive','resetOutput','setRunning','clearRecording','setRecording','destroy'].map((method) => [method, (...args) => inspectorCalls.push({method,args})])),
    // Playback resampling has real-signal tests in tests/check_playback_resampler.mjs.
    // Here, a mock context with no sampleRate (undefined !== RATE) would otherwise
    // force these DOM/control-flow tests to depend on the real resampler; pass
    // samples straight through so this file keeps testing only control flow.
    createPlaybackUpsampler: () => ({ targetRate: undefined, reset() {}, push: (samples) => samples }),
  };
  sandbox.setText = (target, key, params) => {
    const node = typeof target === 'string' ? sandbox.document.getElementById(target) : target;
    node.textContent = sandbox.t(key, params);
  };
  sandbox.setAttr = (target, attribute, key, params) => {
    const node = typeof target === 'string' ? sandbox.document.getElementById(target) : target;
    node.setAttribute(attribute, sandbox.t(key, params));
  };
  vm.createContext(sandbox);
  vm.runInContext(`${executableSource}\nglobalThis.api = { state, setTrackingStatus, setTrackingMode, resetTrackingWindow, trackingAllowsInference, positionROI, maybeInfer, handleResult, stopCapture, captureFrame, finalizeRecording, resetRecording, openCapture, checkCaptureHealth, onPCM };`, sandbox, { filename: 'app.js' });
  return { ...sandbox.api, elements, sandbox, languageCallbacks, inspectorCalls };
}

let passed = 0;
function test(name, run) {
  run();
  passed++;
  console.log(`PASS ${name}`);
}

test('all literal DOM ids in app.js exist in index.html', () => {
  for (const match of source.matchAll(/\$\('([^']+)'\)/g)) assert.ok(ids.has(match[1]), match[1]);
  harness();
});

test('loss clears queued sound, holds single flight, and drops the old result', () => {
  const h = harness();
  const s = h.state;
  let stopped = 0;
  Object.assign(s, { running: true, trackingStatus: 'tracking', samples: 90000, frames: [{ data: 'old' }], playAt: 5 });
  s.inFlight = { id: 9, inferenceEpoch: s.inferenceEpoch, timeout: 1 };
  s.monitorSources.add({ stop() { stopped++; }, disconnect() {} });
  h.setTrackingStatus({ status: 'lost', trackingEpoch: s.trackingEpoch });
  assert.equal(h.trackingAllowsInference(), false);
  assert.equal(s.frames.length, 0);
  assert.equal(s.sessionEpoch, 90000);
  assert.equal(s.playAt, 0);
  assert.equal(stopped, 1);
  assert.equal(s.inFlight.id, 9, 'do not queue another request while the old one computes');
  h.handleResult(JSON.stringify({ type: 'result', id: 9 }));
  assert.equal(s.inFlight, null);
  assert.equal(s.recordingSamples, 0);
  assert.ok(h.inspectorCalls.some((call) => call.method === 'resetOutput'));
  assert.ok(!h.inspectorCalls.some((call) => call.method === 'pushOutput'), 'old target audio must not reach the spectrum');
  assert.equal(s.running, true, 'discard stale result without killing the session');
});

test('a missing face keeps the window, the queued audio and inference running', () => {
  const h = harness();
  const s = h.state;
  let stopped = 0;
  Object.assign(s, { running: true, trackingStatus: 'tracking', samples: 90000, frames: [{ data: 'kept' }], playAt: 5 });
  s.monitorSources.add({ stop() { stopped++; }, disconnect() {} });
  const epoch = s.inferenceEpoch;
  h.setTrackingStatus({ status: 'tracking', searching: true, roi: { cx: .5, cy: .6, size: .2 }, trackingEpoch: s.trackingEpoch });
  assert.equal(h.trackingAllowsInference(), true);
  assert.equal(s.inferenceEpoch, epoch);
  assert.equal(s.frames.length, 1);
  assert.equal(s.playAt, 5);
  assert.equal(stopped, 0);
  assert.equal(h.elements.get('trackingStatus').textContent, '未检测到人脸 · 沿用上次取景');
  h.setTrackingStatus({ status: 'tracking', roi: { cx: .5, cy: .6, size: .2 }, trackingEpoch: s.trackingEpoch });
  assert.equal(s.searching, false);
  assert.equal(s.inferenceEpoch, epoch, 'the face returning does not reset either');
});

test('manual switch is explicit and late detector events cannot change it', () => {
  const h = harness();
  const s = h.state;
  const messages = [];
  Object.assign(s, { stream: {}, videoWorker: { postMessage: (message) => messages.push(message) }, trackingStatus: 'lost' });
  const oldEpoch = s.trackingEpoch;
  h.setTrackingMode('manual');
  assert.equal(s.trackingMode, 'manual');
  assert.equal(s.trackingStatus, 'manual');
  assert.equal(h.trackingAllowsInference(), true);
  assert.ok(messages.some((message) => message.type === 'tracking' && message.mode === 'manual'));
  h.setTrackingStatus({ status: 'tracking', trackingEpoch: oldEpoch });
  assert.equal(s.trackingStatus, 'manual');
  assert.equal(h.elements.get('manualTrackingButton').getAttribute('aria-pressed'), 'true');
});

test('returning to tracking still rejects results from a previous target epoch', () => {
  const h = harness();
  const s = h.state;
  Object.assign(s, { running: true, stream: {}, trackingStatus: 'tracking', videoWorker: { postMessage() {} } });
  s.inFlight = { id: 3, inferenceEpoch: s.inferenceEpoch, timeout: 1 };
  h.setTrackingMode('manual');
  h.setTrackingMode('auto');
  h.setTrackingStatus({ status: 'tracking', trackingEpoch: s.trackingEpoch });
  assert.equal(h.trackingAllowsInference(), true);
  h.handleResult(JSON.stringify({ type: 'result', id: 3 }));
  assert.equal(s.recordingSamples, 0);
  assert.equal(s.running, true);
});

test('automatic ROI rendering preserves source coordinates and does not write back', () => {
  const h = harness();
  const messages = [];
  h.state.videoWorker = { postMessage: (message) => messages.push(message) };
  h.state.roi = { cx: .01, cy: .99, size: .2 };
  h.positionROI();
  assert.equal(h.state.roi.cx, .01);
  assert.equal(h.state.roi.cy, .99);
  assert.equal(messages.length, 0);
  assert.equal(h.elements.get('mouthRoi').style.width, '72px');
});

test('a reacquired target preserves its onset with fresh synchronized frames', () => {
  const h = harness();
  const s = h.state;
  const sent = [];
  Object.assign(s, { running: true, samples: 32000, audioOrigin: 0, trackingStatus: 'lost', socket: { readyState: 1, send: (data) => sent.push(JSON.parse(data)) } });
  h.setTrackingStatus({ status: 'tracking', trackingEpoch: s.trackingEpoch });
  s.samples = 41279;
  s.frames = Array.from({ length: 50 }, (_, index) => ({ time: 2 + (index + .5) / 25, data: `frame-${index}` }));
  h.maybeInfer();
  assert.equal(sent.length, 0);
  s.samples = 41280;
  h.maybeInfer();
  assert.equal(sent.length, 1);
  assert.equal(sent[0].frames.length, 50);
  assert.equal(sent[0].end_sample, 64000);
  assert.equal(s.inFlight.end, 41280);
  assert.equal(sent[0].reset, true);
  assert.equal(s.inFlight.inferenceEpoch, s.inferenceEpoch);
  const snapshot = s.inFlight.audio;
  assert.equal(Buffer.from(snapshot.buffer, snapshot.byteOffset, snapshot.byteLength).toString('base64'), sent[0].audio);
  const original = snapshot[0];
  s.ring.fill(.9);
  assert.equal(snapshot[0], original, 'sent input snapshot does not share memory with the rolling ring');
});

test('the five-minute recording cap does not stop live meeting playback', () => {
  const h = harness();
  const s = h.state;
  let played = 0;
  Object.assign(s, {
    running: true, trackingStatus: 'tracking', recordingSamples: 300 * 16000,
    meetingActive: true, monitorGain: {}, meetingGain: {},
    context: { currentTime: 3, state: 'running',
      createBuffer: () => ({ duration: .5, copyToChannel() {} }),
      createBufferSource: () => ({ connect() {}, start() { played++; }, disconnect() {} }),
    },
  });
  s.inFlight = { id: 4, end: 48000, inferenceEpoch: s.inferenceEpoch, timeout: 1, time: 900 };
  h.handleResult(JSON.stringify({ type: 'result', id: 4, sample_rate: 16000,
    audio: Buffer.alloc(8000 * 4).toString('base64'), start_sample: 38720, end_sample: 46720,
    inference_ms: 100, rtf: .2, output_delay_ms: 80 }));
  assert.equal(s.recordingSamples, 300 * 16000);
  assert.equal(s.recordingCapped, true);
  assert.equal(s.running, true);
  assert.equal(played, 1);
  assert.equal(h.inspectorCalls.filter((call) => call.method === 'pushOutput').length, 1, 'spectrum continues after the recording cap');
});

test('the mixture is recorded aligned with the separated voice and can be played and exported', () => {
  const h = harness();
  const s = h.state;
  Object.assign(s, { running: true, trackingStatus: 'tracking', samples: 48000 });
  for (let i = 0; i < s.ring.length; i++) s.ring[i] = i / s.ring.length;
  s.inFlight = { id: 7, end: 48000, audio: s.ring.slice(16000, 48000), inferenceEpoch: s.inferenceEpoch, timeout: 1, time: 900 };
  h.handleResult(JSON.stringify({ type: 'result', id: 7, sample_rate: 16000,
    audio: Buffer.alloc(8000 * 4).toString('base64'), start_sample: 38720, end_sample: 46720,
    inference_ms: 100, rtf: .2, output_delay_ms: 80 }));
  assert.equal(s.recording.length, 1);
  assert.equal(s.mixture.length, 1);
  assert.equal(s.mixture[0].length, 8000, 'same length as the separated chunk');
  assert.equal(s.mixture[0][0], s.ring[38720 % s.ring.length], 'starts at the input sample of the output chunk');
  assert.equal(s.mixture[0][7999], s.ring[46719 % s.ring.length]);
  s.running = false;
  h.finalizeRecording();
  assert.ok(s.recordingUrl && s.mixtureUrl && s.recordingUrl !== s.mixtureUrl);
  const player = h.elements.get('recordingPlayer');
  assert.equal(player.src, s.recordingUrl, 'the separated voice plays by default');
  h.elements.get('trackMixture').listeners.get('click')();
  assert.equal(player.src, s.mixtureUrl);
  assert.equal(h.elements.get('trackMixture').getAttribute('aria-pressed'), 'true');
  assert.equal(h.elements.get('recordingTitle').textContent, '原始混合声音 · 00:00');
  assert.equal(h.inspectorCalls.filter((call) => call.method === 'setRecording').at(-1).args[0], s.mixture, 'spectrum shows the mixture');
  const anchors = [];
  h.sandbox.document.createElement = (tag) => { const a = { tag, click() { anchors.push(this); }, remove() {} }; return a; };
  h.elements.get('downloadButton').listeners.get('click')();
  assert.equal(anchors[0].href, s.mixtureUrl);
  assert.match(anchors[0].download, /^falcon-mixture-.*\.wav$/);
  h.elements.get('trackTarget').listeners.get('click')();
  assert.equal(player.src, s.recordingUrl);
  h.resetRecording();
  assert.equal(s.mixture.length, 0);
  assert.equal(s.mixtureUrl, null);
});

test('a delayed result records the sent input snapshot after the microphone ring was overwritten', () => {
  const h = harness();
  const s = h.state;
  const audio = Float32Array.from({ length: 32000 }, (_, i) => i / 32000);
  Object.assign(s, { running: true, trackingStatus: 'tracking', samples: 320000 });
  s.ring.fill(-.75);
  s.inFlight = { id: 11, end: 48000, audio, inferenceEpoch: s.inferenceEpoch, timeout: 1, time: 900 };
  h.handleResult(JSON.stringify({ type: 'result', id: 11, sample_rate: 16000,
    audio: Buffer.alloc(8000 * 4).toString('base64'), start_sample: 38720, end_sample: 46720,
    inference_ms: 100, rtf: .2, output_delay_ms: 80 }));
  assert.equal(s.recordingSamples, 8000);
  assert.deepEqual(Array.from(s.mixture[0]), Array.from(audio.slice(22720, 30720)));
});

function stoppingSession() {
  const h = harness();
  const s = h.state;
  const sent = [], timers = [];
  h.sandbox.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return timers.length; };
  Object.assign(s, { running: true, trackingStatus: 'tracking', samples: 9280, audioOrigin: 0, sessionEpoch: 0,
    context: { currentTime: 2.4, state: 'running', close: async () => {} },
    socket: { readyState: 1, send: (data) => sent.push(JSON.parse(data)), close() {} } });
  s.ring.fill(.25);
  s.frames = Array.from({ length: 60 }, (_, n) => ({ time: (n + .5) / 25, data: `f${n}` }));
  const reply = (message) => h.handleResult(JSON.stringify({ type: 'result', id: s.inFlight.id, sample_rate: 16000,
    audio: Buffer.alloc(8000 * 4).toString('base64'), start_sample: message.end_sample - 1280 - 8000,
    end_sample: message.end_sample - 1280, inference_ms: 100, rtf: .2, output_delay_ms: 80 }));
  h.maybeInfer();
  reply(sent[0]);
  s.samples = 38400; // 0.4 s more speech: not a full hop yet, so not separated
  return { h, s, sent, timers, reply };
}

test('stopping separates the last words, then trims the padding and closes', () => {
  const { h, s, sent, timers, reply } = stoppingSession();
  assert.equal(s.recordedEnd, 8000, 'the first output starts at zero and ends at half a second');
  h.elements.get('stopButton').listeners.get('click')();
  assert.equal(s.running, true, 'the session stays open while the tail is separated');
  assert.equal(sent.length, 2);
  assert.equal(sent[1].end_sample, 40000, 'the next window continues the real sample timeline');
  assert.equal(sent[1].reset, false, 'the tail continues the previous output');
  assert.equal(sent[1].frames.length, 50);
  while (s.inFlight) reply(sent.at(-1));
  assert.equal(s.recordedEnd, 38400, 'the recording now reaches the moment Stop was clicked');
  assert.equal(s.recordingSamples, 38400);
  assert.ok(s.mixture.at(-1).every((value) => value === .25), 'padding silence is trimmed from the mixture');
  assert.equal(s.recording.at(-1).length, s.mixture.at(-1).length);
  timers.filter((timer) => timer.ms < 8000).at(-1).fn();
  assert.equal(s.running, false);
  assert.ok(s.recordingUrl && s.mixtureUrl);
});

test('a second click stops at once', () => {
  const { h, s } = stoppingSession();
  h.elements.get('stopButton').listeners.get('click')();
  assert.equal(s.running, true);
  h.elements.get('stopButton').listeners.get('click')();
  assert.equal(s.running, false);
  assert.equal(s.draining, null);
});

function onsetSession(epoch = 0) {
  const h = harness(), s = h.state, sent = [];
  Object.assign(s, { running: true, trackingStatus: 'tracking', samples: epoch, sessionEpoch: epoch,
    sessionStart: epoch, audioOrigin: 0, context: { currentTime: epoch / 16000, state: 'running', close: async () => {} },
    socket: { readyState: 1, send: (data) => sent.push(JSON.parse(data)), close() {} } });
  s.ring.fill(-.9); // preview/previous-target audio must not become synthetic history
  const reply = () => {
    const message = sent.at(-1), request = s.inFlight;
    const output = Buffer.from(request.audio.buffer, request.audio.byteOffset + 22720 * 4, 8000 * 4).toString('base64');
    h.handleResult(JSON.stringify({ type: 'result', id: message.id, sample_rate: 16000,
      audio: output, start_sample: message.end_sample - 9280, end_sample: message.end_sample - 1280,
      inference_ms: 100, rtf: .2, output_delay_ms: 80 }));
  };
  const collect = (count) => {
    for (let n = 0; n < count; n++) { s.ring[s.samples % s.ring.length] = (s.samples - epoch + 1) / 64000; s.samples++; }
    s.frames = [{ time: (epoch - 160) / 16000, data: 'old-target' },
      ...Array.from({ length: Math.ceil((s.samples - epoch) / 640) }, (_, n) => ({ time: epoch / 16000 + (n + .5) / 25, data: `new-${n}` }))];
  };
  return { h, s, sent, reply, collect };
}

test('the first words start at sample zero, padding stays out, and following chunks join exactly', () => {
  const { h, s, sent, reply, collect } = onsetSession();
  collect(9279);
  h.maybeInfer();
  assert.equal(sent.length, 0, 'wait for a real half-second plus the right context');
  collect(1);
  h.maybeInfer();
  assert.equal(sent[0].end_sample, 32000, 'valid protocol window even without a device preview');
  assert.ok(s.inFlight.audio.slice(0, 22720).every((value) => value === 0));
  assert.ok(sent[0].frames.slice(0, 35).every((frame) => frame === 'new-0'));
  assert.ok(!sent[0].frames.includes('old-target'));
  reply();
  assert.equal(s.recordedEnd, 8000);
  assert.equal(s.recordingSamples, 8000);
  assert.equal(s.mixture[0][0], Math.fround(1 / 64000), 'the very first real microphone sample is saved');
  for (let n = 0; n < 4; n++) { collect(8000); h.maybeInfer(); reply(); }
  assert.equal(s.recordedEnd, 40000);
  assert.equal(s.recordingSamples, 40000);
  assert.equal(sent.length, 5);
  assert.equal(s.skipped, 0);
  const actual = Array.from(s.mixture).flatMap((chunk) => Array.from(chunk));
  assert.deepEqual(actual, Array.from({ length: 40000 }, (_, i) => Math.fround((i + 1) / 64000)));
  assert.deepEqual(Array.from(s.recording).flatMap((chunk) => Array.from(chunk)), actual, 'identity model output and mixture stay aligned');
});

test('stopping before the first result saves a short clip without leading or trailing padding', () => {
  const { h, s, sent, reply, collect } = onsetSession(40000);
  collect(3200); // 0.2 seconds, no normal window has been sent
  h.elements.get('stopButton').listeners.get('click')();
  assert.equal(sent.length, 1);
  assert.equal(s.running, true);
  reply();
  assert.equal(s.recordedEnd, 43200);
  assert.equal(s.recordingSamples, 3200);
  assert.equal(s.recording[0].length, 3200);
  assert.deepEqual(Array.from(s.mixture[0]), Array.from({ length: 3200 }, (_, i) => Math.fround((i + 1) / 64000)));
  assert.ok(s.draining.finished);
});

test('stopping with the first window in flight retains both the opening and the tail', () => {
  const { h, s, sent, reply, collect } = onsetSession();
  collect(9280); h.maybeInfer();
  collect(2400); // 0.73 s since Start, before the first response
  h.elements.get('stopButton').listeners.get('click')();
  assert.equal(sent.length, 1, 'keep the single outstanding request');
  while (s.inFlight) reply();
  assert.equal(sent.length, 2);
  assert.equal(s.recordedEnd, 11680);
  assert.equal(s.recordingSamples, 11680);
  assert.deepEqual(Array.from(s.mixture).flatMap((chunk) => Array.from(chunk)),
    Array.from({ length: 11680 }, (_, i) => Math.fround((i + 1) / 64000)));
  assert.ok(s.draining.finished);
});

test('a delayed first response still records the opening from its immutable input', () => {
  const { h, s, sent, reply, collect } = onsetSession();
  collect(9280); h.maybeInfer();
  const expected = Array.from(s.inFlight.audio.slice(22720, 30720));
  s.ring.fill(-.75); s.samples += 160000;
  reply();
  assert.deepEqual(Array.from(s.mixture[0]), expected);
  assert.equal(s.recordingSamples, 8000);
});

test('preview audio reaches the spectrum and unchanged recordings keep their seek position', () => {
  const h = harness();
  h.state.context = { currentTime: 0 };
  h.onPCM({ samples: new Float32Array(320), startTime: 0 });
  assert.equal(h.inspectorCalls.filter((call) => call.method === 'pushInput').length, 1);
  Object.assign(h.state, { recordingSamples: 16000, recordingFinalizedSamples: 16000, recordingUrl: 'blob:existing' });
  h.elements.get('recordingPlayer').currentTime = .75;
  h.finalizeRecording();
  assert.equal(h.elements.get('recordingPlayer').currentTime, .75);
  assert.ok(!h.inspectorCalls.some((call) => call.method === 'setRecording'));
  h.resetRecording();
  assert.ok(h.inspectorCalls.some((call) => call.method === 'clearRecording'));
  assert.equal(h.state.recordingFinalizedSamples, 0);
});

test('a frozen microphone is reported separately from an inference stall', () => {
  const h = harness();
  h.sandbox.console = { ...console, warn() {} };
  h.sandbox.performance.now = () => 10000;
  Object.assign(h.state, { running: true, context: { currentTime: 0 }, trackingMode: 'manual' });
  h.checkCaptureHealth();
  assert.match(h.elements.get('performanceHint').textContent, /麦克风采集已暂停/);
  const diagnostic = h.sandbox.window.falconDiagnostics();
  assert.equal(diagnostic.recordingSeconds, 0);
  assert.ok(!('audio' in diagnostic) && !('framesData' in diagnostic), 'diagnostics contain aggregate state only');
});

function startupHarness({ resume, play, enumerate, permission } = {}) {
  const h = harness(), timers = [], track = { kind: 'audio', stopped: 0, stop() { this.stopped++; }, addEventListener() {}, getSettings: () => ({}) };
  const stream = { getTracks: () => [track] };
  const node = () => ({ connect() {}, disconnect() {}, gain: { value: 0 } });
  let requested = false;
  class Context {
    constructor() { this.state = 'running'; this.currentTime = 0; this.destination = {}; this.audioWorklet = { addModule: async () => {} }; }
    resume() { return resume ? resume() : Promise.resolve(); }
    close() { this.state = 'closed'; return Promise.resolve(); }
    createGain() { return node(); }
    createMediaStreamSource() { return node(); }
  }
  h.sandbox.AudioContext = Context;
  h.sandbox.AudioWorkletNode = class { constructor() { this.port = {}; } connect() {} disconnect() {} };
  h.sandbox.window.isSecureContext = true;
  h.sandbox.navigator.mediaDevices.getUserMedia = () => { requested = true; return permission ? permission(stream) : Promise.resolve(stream); };
  h.sandbox.navigator.mediaDevices.enumerateDevices = () => enumerate ? enumerate() : Promise.resolve([]);
  h.sandbox.setTimeout = (fn, ms) => { const timer = { fn, ms, active: true }; timers.push(timer); return timer; };
  h.sandbox.clearTimeout = (timer) => { if (timer && typeof timer === 'object') timer.active = false; };
  if (play) h.elements.get('cameraVideo').play = play;
  return { h, timers, track, stream, requested: () => requested,
    fire: (ms) => { const timer = timers.find((timer) => timer.active && timer.ms === ms); assert.ok(timer, `Missing ${ms} ms deadline`); timer.active = false; timer.fn(); } };
}
const settle = async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); };

(async () => {
  {
    const { h, fire } = startupHarness({ enumerate: () => new Promise(() => {}) });
    await h.openCapture();
    assert.equal(h.state.opening, false, 'a stuck device list cannot disable Start');
    assert.ok(h.state.stream);
    fire(8000); await settle();
    assert.match(h.elements.get('noticeText').textContent, /设备列表超时/);
    h.stopCapture(); passed++; console.log('PASS device enumeration never blocks ready capture');
  }
  {
    const { h, fire, track } = startupHarness({ play: () => new Promise(() => {}) });
    const pending = h.openCapture(); await settle();
    assert.equal(h.state.opening, true);
    fire(12000); await pending;
    assert.equal(h.state.opening, false); assert.equal(h.state.stream, null);
    assert.ok(track.stopped); assert.match(h.elements.get('noticeText').textContent, /摄像头预览启动超时/);
    h.elements.get('cameraVideo').play = () => Promise.resolve();
    await h.openCapture(); assert.ok(h.state.stream, 'retry succeeds after timeout');
    h.stopCapture(); passed++; console.log('PASS preview deadline releases devices and permits retry');
  }
  {
    const { h, fire, requested, track } = startupHarness({ resume: () => new Promise(() => {}) });
    const pending = h.openCapture(); await settle();
    assert.equal(requested(), true, 'permission request must not wait on AudioContext.resume');
    fire(8000); await pending;
    assert.equal(h.state.opening, false); assert.ok(track.stopped);
    assert.match(h.elements.get('noticeText').textContent, /音频设备启动超时/);
    passed++; console.log('PASS suspended audio cannot hide permission request or hang startup');
  }
  {
    let grant;
    const { h, fire, stream, track } = startupHarness({ permission: () => new Promise((resolve) => { grant = resolve; }) });
    const pending = h.openCapture(); await settle(); fire(45000); await pending;
    grant(stream); await settle();
    assert.ok(track.stopped, 'a late permission grant must release its device');
    assert.equal(h.state.stream, null); assert.equal(h.state.opening, false);
    passed++; console.log('PASS late permissions after timeout do not leak devices');
  }
  const h = harness();
  const messages = [];
  let resolveBitmap, closed = 0;
  h.sandbox.createImageBitmap = () => new Promise((resolve) => { resolveBitmap = resolve; });
  Object.assign(h.state, { stream: {}, context: { currentTime: 0 }, videoWorker: { postMessage: (message) => messages.push(message) } });
  const pending = h.captureFrame();
  assert.equal(h.state.bitmapBusy, true);
  h.setTrackingMode('manual');
  resolveBitmap({ close() { closed++; } });
  await pending;
  assert.equal(closed, 1, 'old-mode bitmap must be closed');
  assert.equal(messages.filter((message) => message.type === 'frame').length, 0, 'old-mode bitmap must never reach the worker');
  assert.equal(h.state.bitmapBusy, false);
  assert.equal(h.state.bitmapTimeout, null);
  passed++;
  console.log('PASS a bitmap finishing after a mode change is closed without submission');
  console.log(`${passed} tracking UI control-flow checks passed. Real detector/camera/meeting validation is separate.`);
})().catch((error) => { console.error(error); process.exitCode = 1; });
