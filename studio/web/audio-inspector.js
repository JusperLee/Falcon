import { setText, setAttr, onLanguageChange } from './i18n.js';

const RATE = 16000;
const LIVE_WIDTH = 200;
const HEIGHT = 128;
const MAX_LIVE_SAMPLES = RATE;
const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));
const timeLabel = (seconds, tenths = false) => {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  return `${Math.floor(safe / 60).toString().padStart(2, '0')}:${Math.floor(safe % 60).toString().padStart(2, '0')}${tenths ? `.${Math.floor(safe * 10) % 10}` : ''}`;
};
const COLORS = (() => {
  const stops = [[0, 13, 20, 28], [45, 18, 40, 51], [105, 24, 78, 87], [160, 56, 132, 114], [215, 172, 202, 101], [255, 246, 227, 146]];
  const colors = new Uint8ClampedArray(256 * 4);
  for (let value = 0; value < 256; value++) {
    let index = 1;
    while (stops[index][0] < value) index++;
    const a = stops[index - 1], b = stops[index], fraction = (value - a[0]) / (b[0] - a[0]);
    for (let channel = 0; channel < 3; channel++) colors[value * 4 + channel] = a[channel + 1] + fraction * (b[channel + 1] - a[channel + 1]);
    colors[value * 4 + 3] = 255;
  }
  return colors;
})();

function make(tag, className, key = null) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (key !== null) setText(element, key);
  return element;
}
function paintSpectrum(canvas, pixels, width, height = HEIGHT) {
  canvas.width = width; canvas.height = height;
  const image = new ImageData(width, height);
  for (let i = 0; i < pixels.length; i++) image.data.set(COLORS.subarray(pixels[i] * 4, pixels[i] * 4 + 4), i * 4);
  canvas.getContext('2d').putImageData(image, 0, 0);
}

export function createAudioInspector({ player, onError = () => {} }) {
  if (!(player instanceof HTMLMediaElement)) throw new TypeError('A recording audio player is required');
  if (!document.getElementById('audioInspectorStyles')) {
    const sheet = document.createElement('link'); sheet.id = 'audioInspectorStyles';
    sheet.rel = 'stylesheet'; sheet.href = '/static/audio-inspector.css'; document.head.append(sheet);
  }
  let destroyed = false, running = false, failed = false, worker = null;
  let recordingEpoch = 0, hasRecording = false, recordingReady = false, duration = 0;
  let zoom = 1, viewStart = 0, latestRequest = 0, viewInFlight = null, wantedView = null;
  let currentView = null, dirtyView = true, draggingSeek = false;
  let liveInFlight = null, lastChannel = 'output', lastPaint = 0, animation = 0;
  let lastRecording = null, lastRecordingSamples = 0;
  const listeners = [];
  const live = {};
  const listen = (target, event, handler, options) => { target.addEventListener(event, handler, options); listeners.push(() => target.removeEventListener(event, handler, options)); };

  for (const channel of ['input', 'output']) {
    const anchor = document.getElementById(channel === 'input' ? 'inputWave' : 'outputWave');
    const section = make('div', 'audio-live-spectrum');
    const heading = make('div', 'audio-spectrum-heading');
    heading.append(make('span', '', '实时频谱'), make('span', '', '最近 8 秒 · 0–8 kHz'));
    const canvas = make('canvas'); canvas.width = LIVE_WIDTH; canvas.height = HEIGHT;
    setAttr(canvas, 'aria-label', '实时频谱');
    const livePlot = make('div', 'audio-live-plot');
    const frequencyAxis = make('div', 'audio-live-frequencies');
    for (const value of ['8 kHz', '4 kHz', '0 Hz']) { const label = make('span'); label.textContent = value; frequencyAxis.append(label); }
    livePlot.append(canvas, frequencyAxis); section.append(heading, livePlot);
    anchor?.insertAdjacentElement('afterend', section);
    live[channel] = { epoch: 0, pending: new Float32Array(0), pixels: new Uint8Array(LIVE_WIDTH * HEIGHT), dirty: true, canvas, section };
  }
  const liveNote = make('p', 'audio-live-note', '输入和分离输出各自滚动显示，输出包含处理延迟。');
  document.querySelector('.audio-grid')?.insertAdjacentElement('afterend', liveNote);
  const colorLegend = make('div', 'audio-spectrum-legend');
  const legendLow = make('span'), legendHigh = make('span');
  legendLow.textContent = '−90 dBFS'; legendHigh.textContent = '0 dBFS';
  colorLegend.append(legendLow, make('i'), legendHigh);
  liveNote.insertAdjacentElement('afterend', colorLegend);
  const root = make('section', 'audio-inspector');
  root.id = 'audioInspector';
  const header = make('div', 'audio-inspector-header');
  const title = make('div', 'audio-inspector-heading');
  title.append(make('h3', '', '录音细览'), make('span', '', '波形 + 频谱概览'));
  const controls = make('div', 'audio-zoom-controls');
  const zoomLabel = make('label', '', '放大');
  zoomLabel.htmlFor = 'audioInspectorZoom';
  const zoomSelect = make('select'); zoomSelect.id = 'audioInspectorZoom';
  setAttr(zoomSelect, 'aria-label', '放大');
  for (const value of [1, 2, 4, 8, 16]) { const option = make('option'); option.value = String(value); option.textContent = `${value}×`; zoomSelect.append(option); }
  const fitButton = make('button', 'audio-fit-button', '全览');
  fitButton.type = 'button'; setAttr(fitButton, 'title', '恢复整段录音视图');
  controls.append(zoomLabel, zoomSelect, fitButton); header.append(title, controls);
  const seekRow = make('div', 'audio-seek-row');
  const currentTime = make('span', 'audio-current-time'); currentTime.textContent = '00:00.0';
  const totalTime = make('span', 'audio-total-time'); totalTime.textContent = '00:00.0';
  const seek = make('input', 'audio-range'); seek.id = 'audioInspectorSeek'; seek.type = 'range'; seek.min = '0'; seek.max = '0'; seek.step = '0.01'; seek.value = '0';
  setAttr(seek, 'aria-label', '播放位置');
  seekRow.append(currentTime, seek, totalTime);
  const plot = make('div', 'audio-recording-plot');
  plot.id = 'audioInspectorPlot'; plot.setAttribute('role', 'slider'); plot.setAttribute('aria-valuemin', '0');
  setAttr(plot, 'aria-label', '录音波形和频谱，点击或使用方向键定位');
  const waveform = make('canvas', 'audio-recording-wave'); waveform.height = 77;
  const spectrum = make('canvas', 'audio-recording-spectrum'); spectrum.height = HEIGHT;
  const cursor = make('div', 'audio-play-cursor'); cursor.hidden = true;
  const empty = make('div', 'audio-plot-empty', '尚无录音 · 停止分离后可查看和定位');
  const frequencies = make('div', 'audio-frequency-labels');
  for (const text of ['8 kHz', '4 kHz', '0 Hz']) { const label = make('span'); label.textContent = text; frequencies.append(label); }
  plot.append(waveform, spectrum, make('span', 'audio-plot-label wave-label', '波形'), make('span', 'audio-plot-label spectrum-label', '频谱概览'), frequencies, cursor, empty);
  const axis = make('div', 'audio-time-axis');
  const axisLabels = [make('span'), make('span'), make('span')]; axis.append(...axisLabels);
  const viewRow = make('div', 'audio-view-row');
  const panLabel = make('label', '', '视区位置'); panLabel.htmlFor = 'audioInspectorPan';
  const pan = make('input', 'audio-range'); pan.id = 'audioInspectorPan'; pan.type = 'range'; pan.min = '0'; pan.max = '0'; pan.step = '0.01'; pan.value = '0';
  setAttr(pan, 'aria-label', '视区位置');
  const viewRange = make('span', 'audio-view-range');
  viewRow.append(panLabel, pan, viewRange);
  const footer = make('div', 'audio-inspector-footer');
  footer.append(make('p', '', '点击波形或频谱定位播放；放大查看局部。'), make('p', '', '录音按已完成的分离片段拼接，时间轴可能省略跳过的窗口。'));
  const errorNote = make('p', 'audio-spectrum-error'); errorNote.hidden = true;
  root.append(header, seekRow, plot, axis, viewRow, footer, errorNote);
  document.querySelector('.recording-bar')?.insertAdjacentElement('afterend', root);

  const span = () => duration ? Math.min(duration, Math.max(.1, duration / zoom)) : 0;
  const canSeek = () => !destroyed && !running && hasRecording && duration > 0 && player.readyState >= 1 && Number.isFinite(player.duration);
  function updateLabels() {
    const end = Math.min(duration, viewStart + span());
    axisLabels[0].textContent = timeLabel(viewStart, zoom > 1);
    axisLabels[1].textContent = timeLabel((viewStart + end) / 2, zoom > 1);
    axisLabels[2].textContent = timeLabel(end, zoom > 1);
    viewRange.textContent = `${timeLabel(viewStart, true)} — ${timeLabel(end, true)}`;
    setAttr(pan, 'aria-valuetext', '可见范围：{start} 至 {end}', { start: timeLabel(viewStart, true), end: timeLabel(end, true) });
    totalTime.textContent = timeLabel(duration, true);
  }
  function updateControls() {
    const enabled = canSeek();
    seek.disabled = !enabled; zoomSelect.disabled = !hasRecording || running; fitButton.disabled = !hasRecording || running;
    pan.disabled = !hasRecording || running || zoom === 1;
    plot.tabIndex = enabled ? 0 : -1; plot.setAttribute('aria-disabled', String(!enabled));
    seek.max = String(duration); plot.setAttribute('aria-valuemax', String(duration));
    pan.max = String(Math.max(0, duration - span())); pan.value = String(viewStart);
    empty.hidden = hasRecording && !running && !!currentView;
    if (running) setText(empty, '正在分离 · 停止后可定位播放');
    else if (!hasRecording) setText(empty, '尚无录音 · 停止分离后可查看和定位');
    else setText(empty, failed ? '频谱概览不可用；仍可使用进度条定位。' : '正在生成频谱概览…');
    updateLabels(); updatePosition();
  }
  function updatePosition() {
    if (destroyed) return;
    const time = hasRecording ? clamp(player.currentTime || 0, 0, duration) : 0;
    currentTime.textContent = timeLabel(time, true);
    if (!draggingSeek) seek.value = String(time);
    seek.style.setProperty('--audio-progress', `${duration ? time / duration * 100 : 0}%`);
    setAttr(seek, 'aria-valuetext', '当前位置 {current}，总时长 {duration}', { current: timeLabel(time, true), duration: timeLabel(duration, true) });
    plot.setAttribute('aria-valuenow', String(time));
    const fraction = span() ? (time - viewStart) / span() : -1;
    cursor.hidden = !hasRecording || running || fraction < 0 || fraction > 1 || !currentView;
    cursor.style.left = `${clamp(fraction, 0, 1) * 100}%`;
  }
  function seekTo(seconds) {
    if (!canSeek()) return;
    try { player.currentTime = clamp(seconds, 0, Math.min(duration, player.duration)); updatePosition(); }
    catch { onError('无法定位录音，请等待音频加载后重试。'); }
  }
  function reportFailure() {
    if (failed || destroyed) return;
    failed = true; worker?.terminate(); worker = null; liveInFlight = null; viewInFlight = null; wantedView = null;
    live.input.pending = live.output.pending = new Float32Array(0);
    errorNote.hidden = false; setText(errorNote, '音频可视化暂不可用，分离与播放仍可继续。');
    updateControls();
    try { onError('音频可视化暂不可用，分离与播放仍可继续。'); } catch {}
  }
  function dispatchLive() {
    if (!worker || destroyed || failed || liveInFlight) return;
    const order = lastChannel === 'input' ? ['output', 'input'] : ['input', 'output'];
    const channel = order.find((name) => live[name].pending.length);
    if (!channel) return;
    const stream = live[channel];
    const samples = stream.pending; stream.pending = new Float32Array(0);
    liveInFlight = { channel, epoch: stream.epoch }; lastChannel = channel;
    try { worker.postMessage({ type: 'live', channel, epoch: stream.epoch, samples }, [samples.buffer]); }
    catch { reportFailure(); }
  }
  function push(channel, samples) {
    if (destroyed || failed || !worker || !samples?.length) return;
    const stream = live[channel];
    // Always copy: the capture/recording path retains ownership of its samples.
    const keepNew = Math.min(samples.length, MAX_LIVE_SAMPLES);
    const keepOld = Math.min(stream.pending.length, MAX_LIVE_SAMPLES - keepNew);
    const combined = new Float32Array(keepOld + keepNew);
    if (keepOld) combined.set(stream.pending.subarray(stream.pending.length - keepOld));
    combined.set(samples.subarray(samples.length - keepNew), keepOld);
    stream.pending = combined; dispatchLive();
  }
  function dispatchView() {
    if (!worker || failed || !recordingReady || !wantedView || viewInFlight) return;
    const request = wantedView; wantedView = null;
    viewInFlight = { epoch: request.epoch, requestId: request.requestId };
    try { worker.postMessage(request); } catch { reportFailure(); }
  }
  function requestView() {
    if (!hasRecording || destroyed) return;
    const columns = clamp(Math.round(plot.clientWidth || 800), 64, 1600);
    wantedView = { type: 'view', epoch: recordingEpoch, requestId: ++latestRequest, start: viewStart, end: Math.min(duration, viewStart + span()), columns };
    dispatchView(); updateLabels();
  }
  function setView(start, factor = zoom) {
    zoom = factor; viewStart = clamp(start, 0, Math.max(0, duration - span()));
    currentView = null; dirtyView = true;
    zoomSelect.value = String(zoom); updateControls(); requestView();
  }
  function paintRecording() {
    const width = Math.max(1, Math.round(waveform.clientWidth * Math.min(devicePixelRatio || 1, 2)));
    const height = Math.max(1, Math.round((waveform.clientHeight || 77) * Math.min(devicePixelRatio || 1, 2)));
    waveform.width = width; waveform.height = height;
    const context = waveform.getContext('2d');
    context.clearRect(0, 0, width, height);
    context.strokeStyle = '#69845066'; context.lineWidth = 1;
    context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
    if (!currentView) { spectrum.width = 200; spectrum.height = HEIGHT; spectrum.getContext('2d').clearRect(0, 0, 200, HEIGHT); return; }
    context.strokeStyle = '#c3e791'; context.lineWidth = Math.max(1, width / currentView.width * .72);
    context.beginPath();
    for (let i = 0; i < currentView.width; i++) {
      const x = (i + .5) / currentView.width * width;
      context.moveTo(x, height / 2 - clamp(currentView.waveMax[i], -1, 1) * height * .43);
      context.lineTo(x, height / 2 - clamp(currentView.waveMin[i], -1, 1) * height * .43);
    }
    context.stroke(); paintSpectrum(spectrum, currentView.pixels, currentView.width, currentView.height);
  }
  function acceptLive(message) {
    if (liveInFlight?.channel === message.channel && liveInFlight.epoch === message.epoch) liveInFlight = null;
    const stream = live[message.channel];
    if (stream && message.epoch === stream.epoch && message.width > 0) {
      const count = Math.min(LIVE_WIDTH, message.width), offset = message.width - count;
      for (let row = 0; row < HEIGHT; row++) {
        const start = row * LIVE_WIDTH;
        stream.pixels.copyWithin(start, start + count, start + LIVE_WIDTH);
        stream.pixels.set(message.columns.subarray(row * message.width + offset, row * message.width + message.width), start + LIVE_WIDTH - count);
      }
      stream.dirty = true;
    }
    dispatchLive();
  }
  try {
    worker = new Worker('/static/spectrum-worker.js', { type: 'module' });
    worker.onerror = reportFailure;
    worker.onmessage = ({ data: message }) => {
      if (destroyed) return;
      if (message.type === 'error') { reportFailure(); return; }
      if (message.type === 'live') { acceptLive(message); return; }
      if (message.type === 'recording-ready' && message.epoch === recordingEpoch && hasRecording) {
        recordingReady = true; requestView(); return;
      }
      if (message.type === 'view') {
        if (viewInFlight?.epoch === message.epoch && viewInFlight.requestId === message.requestId) viewInFlight = null;
        if (message.epoch === recordingEpoch && message.requestId === latestRequest && hasRecording) {
          currentView = message; dirtyView = true; updateControls();
        }
        dispatchView();
      }
    };
  } catch { reportFailure(); }

  listen(seek, 'pointerdown', () => { draggingSeek = true; });
  listen(seek, 'input', () => seekTo(Number(seek.value)));
  listen(window, 'pointerup', () => { if (draggingSeek) { draggingSeek = false; updatePosition(); } });
  listen(seek, 'blur', () => { draggingSeek = false; });
  listen(zoomSelect, 'change', () => {
    const newZoom = Number(zoomSelect.value);
    const position = player.currentTime || 0;
    const center = position >= viewStart && position <= viewStart + span() ? position : viewStart + span() / 2;
    setView(center - duration / newZoom / 2, newZoom);
  });
  listen(fitButton, 'click', () => setView(0, 1));
  listen(pan, 'input', () => setView(Number(pan.value)));
  let plotDragging = false;
  const pointerSeek = (event) => { const rect = plot.getBoundingClientRect(); seekTo(viewStart + clamp((event.clientX - rect.left) / rect.width, 0, 1) * span()); };
  listen(plot, 'pointerdown', (event) => { if (!canSeek() || event.button !== 0) return; plotDragging = true; plot.setPointerCapture(event.pointerId); pointerSeek(event); event.preventDefault(); });
  listen(plot, 'pointermove', (event) => { if (plotDragging) pointerSeek(event); });
  listen(plot, 'pointerup', () => { plotDragging = false; });
  listen(plot, 'pointercancel', () => { plotDragging = false; });
  listen(plot, 'keydown', (event) => {
    if (!canSeek()) return;
    const step = event.shiftKey ? 5 : 1;
    const targets = { ArrowLeft: player.currentTime - step, ArrowDown: player.currentTime - step, ArrowRight: player.currentTime + step, ArrowUp: player.currentTime + step, Home: 0, End: duration };
    if (Object.hasOwn(targets, event.key)) { event.preventDefault(); seekTo(targets[event.key]); }
  });
  for (const event of ['timeupdate', 'seeking', 'seeked', 'play', 'pause', 'ended']) listen(player, event, updatePosition);
  for (const event of ['loadedmetadata', 'durationchange', 'emptied']) listen(player, event, updateControls);
  const observer = new ResizeObserver(() => { dirtyView = true; if (hasRecording) requestView(); }); observer.observe(plot);
  const unsubscribe = onLanguageChange(() => { updateLabels(); updatePosition(); });
  function draw(now) {
    if (destroyed) return;
    if (!document.hidden && now - lastPaint >= 40) {
      for (const channel of ['input', 'output']) { const stream = live[channel]; if (stream.dirty) { paintSpectrum(stream.canvas, stream.pixels, LIVE_WIDTH); stream.dirty = false; } }
      if (dirtyView) { paintRecording(); dirtyView = false; }
      updatePosition(); lastPaint = now;
    }
    animation = requestAnimationFrame(draw);
  }
  animation = requestAnimationFrame(draw);
  updateControls();

  function resetChannel(channel) {
    const stream = live[channel]; stream.epoch++; stream.pending = new Float32Array(0); stream.pixels.fill(0); stream.dirty = true;
    if (liveInFlight?.channel === channel) liveInFlight = null;
    if (worker) { try { worker.postMessage({ type: 'reset-live', channel, epoch: stream.epoch }); } catch { reportFailure(); } }
  }
  function clearRecording() {
    ++recordingEpoch; hasRecording = false; recordingReady = false; duration = 0; zoom = 1; viewStart = 0;
    latestRequest++; wantedView = null; viewInFlight = null; currentView = null; dirtyView = true;
    lastRecording = null; lastRecordingSamples = 0;
    player.pause(); player.removeAttribute('src'); player.load();
    if (worker) worker.postMessage({ type: 'clear-recording', epoch: recordingEpoch });
    zoomSelect.value = '1'; seek.value = '0'; updateControls();
  }
  function setRecording(chunks, totalSamples) {
    if (destroyed || !totalSamples || !chunks?.length) return;
    if (chunks === lastRecording && totalSamples === lastRecordingSamples && hasRecording) { updateControls(); return; }
    const count = Math.min(totalSamples, RATE * 300);
    const samples = new Float32Array(count);
    let offset = 0;
    for (const chunk of chunks) {
      const length = Math.min(chunk.length, count - offset);
      for (let i = 0; i < length; i++) samples[offset + i] = Number.isFinite(chunk[i]) ? clamp(chunk[i], -1, 1) : 0;
      offset += length; if (offset >= count) break;
    }
    ++recordingEpoch; latestRequest++; hasRecording = true; recordingReady = false; duration = offset / RATE;
    zoom = 1; viewStart = 0; wantedView = null; viewInFlight = null; currentView = null; dirtyView = true;
    lastRecording = chunks; lastRecordingSamples = totalSamples; zoomSelect.value = '1';
    const payloadSamples = offset === samples.length ? samples : samples.slice(0, offset);
    if (worker) { try { worker.postMessage({ type: 'recording', epoch: recordingEpoch, samples: payloadSamples }, [payloadSamples.buffer]); } catch { reportFailure(); } }
    updateControls();
  }
  return {
    pushInput: (samples) => push('input', samples),
    pushOutput: (samples) => push('output', samples),
    resetLive() { resetChannel('input'); resetChannel('output'); dispatchLive(); },
    resetOutput() { resetChannel('output'); dispatchLive(); },
    setRunning(value) { running = !!value; if (running) player.pause(); updateControls(); },
    clearRecording,
    setRecording,
    destroy() {
      if (destroyed) return;
      destroyed = true; recordingEpoch++; worker?.terminate(); worker = null;
      cancelAnimationFrame(animation); observer.disconnect(); unsubscribe(); listeners.forEach((remove) => remove());
      root.remove(); liveNote.remove(); colorLegend.remove(); live.input.section.remove(); live.output.section.remove();
      live.input.pending = live.output.pending = new Float32Array(0); currentView = null; lastRecording = null;
    },
  };
}
