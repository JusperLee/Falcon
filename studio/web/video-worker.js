/* Local landmarks + camera cropping, independent of page-render timers.
 * Every VideoFrame/ImageBitmap is closed, including skipped input. Automatic
 * mode coasts on the last ROI only within the tracker's grace period. */
import { clampROI, newMouthTrack, updateMouthTrack } from './mouth-tracker.js';

let reader = null, active = false, processingBitmap = false, session = 0;
let mode = 'auto', trackingEpoch = 0;
let manualROI = { cx: .5, cy: .6, size: .2 };
let trackState = newMouthTrack(), nextTimestamp = null, lastDetectTimestamp = -1;
let detector = null, detectorPromise = null, detectorError = null;
let lastTrackingMessage = '', stoppedSent = false;
const canvas = new OffscreenCanvas(96, 96);
const context = canvas.getContext('2d', { willReadFrequently: true });

function tracking(status, details = {}, epoch = trackingEpoch) {
  if (!active || epoch !== trackingEpoch) return;
  const message = { type: 'tracking', status, trackingEpoch: epoch, detectorReady: !!detector, ...details };
  if (status !== 'tracking' && JSON.stringify(message) === lastTrackingMessage) return;
  lastTrackingMessage = JSON.stringify(message);
  self.postMessage(message);
}
async function ensureDetector(epoch) {
  if (detector) { if (mode === 'auto') tracking('lost', { message: '等待目标面部。' }, epoch); return; }
  tracking('loading', { message: '正在加载本地嘴部跟踪模型…' }, epoch);
  const startedSession = session;
  if (!detectorPromise) {
    detectorError = null;
    detectorPromise = (async () => {
      const { FaceLandmarker, FilesetResolver } = await import('./vendor/mediapipe/vision_bundle.mjs');
      const files = await FilesetResolver.forVisionTasks('/static/vendor/mediapipe/wasm', true);
      const instance = await FaceLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: '/static/vendor/mediapipe/face_landmarker.task', delegate: 'CPU' },
        // Two faces so a returning target can be told apart from a second person.
        canvas: new OffscreenCanvas(1, 1), runningMode: 'VIDEO', numFaces: 2,
        minFaceDetectionConfidence: .6, minFacePresenceConfidence: .6, minTrackingConfidence: .6,
        outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false,
      });
      if (!active || session !== startedSession) { instance.close(); return; }
      detector = instance;
      lastDetectTimestamp = -1;
    })().catch((error) => {
      if (session === startedSession) detectorError = error?.message || String(error);
    }).finally(() => { if (session === startedSession) detectorPromise = null; });
  }
  await detectorPromise;
  if (!active || session !== startedSession || mode !== 'auto' || epoch !== trackingEpoch) return;
  if (detector) tracking('lost', { message: '等待目标面部。' }, epoch);
  else tracking('error', { message: `自动跟踪无法加载，请重试或切换手动模式。${detectorError || ''}` }, epoch);
}
function closeDetector() { const old = detector; detector = null; try { old?.close(); } catch {} }
function stopped() { if (!stoppedSent) { stoppedSent = true; self.postMessage({ type: 'stopped' }); } }

async function processFrame(frame, wallTime, epoch) {
  try {
    if (!active || epoch !== trackingEpoch) return;
    const timestamp = Number.isFinite(frame.timestamp) ? frame.timestamp : wallTime * 1000;
    if (nextTimestamp === null || timestamp < nextTimestamp - 1000000) nextTimestamp = timestamp;
    if (timestamp + 1 < nextTimestamp) return;
    nextTimestamp = Math.max(nextTimestamp + 40000, timestamp);
    const width = frame.displayWidth || frame.width, height = frame.displayHeight || frame.height;
    if (!(width > 0 && height > 0)) throw new Error('摄像头帧尺寸无效。');
    const frameMode = mode;
    let roi, lips, detectionMs = null, coasted = false;
    if (frameMode === 'auto') {
      if (!detector) return;
      const started = performance.now();
      const monotonicMs = Math.max(lastDetectTimestamp + .001, timestamp / 1000);
      lastDetectTimestamp = monotonicMs;
      let detected;
      try { detected = detector.detectForVideo(frame, monotonicMs); }
      catch (error) {
        detectorError = error?.message || String(error);
        tracking('error', { message: `自动跟踪失败，请重新捕捉或切换手动模式。${detectorError}` }, epoch);
        closeDetector(); return;
      }
      detectionMs = performance.now() - started;
      const tracked = updateMouthTrack(trackState, detected.faceLandmarks ?? [], width, height, wallTime);
      if (!active || epoch !== trackingEpoch) return;
      trackState = tracked.state;
      const { state: _, ...details } = tracked;
      tracking(tracked.status, { ...details, detectionMs }, epoch);
      if (tracked.status !== 'tracking') return;
      roi = tracked.roi; lips = tracked.lips; coasted = !!tracked.coasted;
    } else {
      roi = clampROI(manualROI, width, height);
      tracking('manual', { roi }, epoch);
    }
    const size = Math.min(width, height) * roi.size;
    context.drawImage(frame, roi.cx * width - size / 2, roi.cy * height - size / 2, size, size, 0, 0, 96, 96);
    const image = context.getImageData(0, 0, 96, 96);
    for (let i = 0; i < image.data.length; i += 4) {
      const gray = Math.round(image.data[i] * .299 + image.data[i + 1] * .587 + image.data[i + 2] * .114);
      image.data[i] = image.data[i + 1] = image.data[i + 2] = gray;
    }
    context.putImageData(image, 0, 0);
    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: .88 });
    const jpeg = await blob.arrayBuffer();
    if (active && epoch === trackingEpoch && mode === frameMode) {
      self.postMessage({ type: 'frame', jpeg, wallTime, trackingEpoch: epoch,
        trackingStatus: frameMode === 'auto' ? 'tracking' : 'manual', roi, lips, detectionMs, coasted }, [jpeg]);
    }
  } finally { frame.close(); }
}
async function pump(readable, startedSession) {
  try {
    reader = readable.getReader();
    while (active && startedSession === session) {
      const { value: frame, done } = await reader.read();
      if (done) break;
      await processFrame(frame, performance.timeOrigin + performance.now(), trackingEpoch);
    }
  } catch (error) {
    if (active && startedSession === session) self.postMessage({ type: 'error', message: error?.message || String(error) });
  } finally {
    active = false;
    try { await reader?.cancel(); } catch {}
    try { reader?.releaseLock(); } catch {}
    reader = null; closeDetector(); stopped();
  }
}
self.onmessage = async (event) => {
  const message = event.data;
  if (message.type === 'roi') { manualROI = message.roi; return; }
  if (message.type === 'tracking') {
    if (!active || !['auto', 'manual'].includes(message.mode)) return;
    trackingEpoch = Number.isSafeInteger(message.trackingEpoch) ? message.trackingEpoch : trackingEpoch + 1;
    mode = message.mode;
    if (message.roi) manualROI = message.roi;
    trackState = newMouthTrack(); lastTrackingMessage = ''; nextTimestamp = null;
    if (mode === 'auto') void ensureDetector(trackingEpoch);
    else tracking('manual', { roi: manualROI });
    return;
  }
  if (message.type === 'stop') {
    active = false; closeDetector();
    if (reader) { try { await reader.cancel(); } catch {} }
    else stopped();
    return;
  }
  if (message.type === 'frame') {
    const epoch = Number.isSafeInteger(message.trackingEpoch) ? message.trackingEpoch : trackingEpoch;
    let ownsProcessing = false;
    try {
      if (!message.bitmap) return;
      if (!active || epoch !== trackingEpoch || reader || processingBitmap) { message.bitmap.close(); return; }
      processingBitmap = ownsProcessing = true;
      await processFrame(message.bitmap, Number.isFinite(message.wallTime) ? message.wallTime : performance.timeOrigin + performance.now(), epoch);
    } catch (error) {
      if (active) self.postMessage({ type: 'error', message: error?.message || String(error) });
    } finally {
      if (ownsProcessing) processingBitmap = false;
      self.postMessage({ type: 'ack', trackingEpoch: epoch });
    }
    return;
  }
  if (message.type !== 'start' || active) return;
  ++session; active = true; stoppedSent = false;
  mode = message.tracking === 'manual' ? 'manual' : 'auto';
  trackingEpoch = Number.isSafeInteger(message.trackingEpoch) ? message.trackingEpoch : 0;
  manualROI = message.roi || manualROI;
  trackState = newMouthTrack(); lastTrackingMessage = ''; nextTimestamp = null;
  self.postMessage({ type: 'ready' });
  if (mode === 'auto') void ensureDetector(trackingEpoch);
  else tracking('manual', { roi: manualROI });
  if (message.readable) await pump(message.readable, session);
};
