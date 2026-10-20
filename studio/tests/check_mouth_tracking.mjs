/* Deterministic geometry, target continuity, crop ownership and epoch tests.
 * Real MediaPipe/browser inference is covered by tests/browser/test.js. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { newMouthTrack, updateMouthTrack, mouthGeometry, clampROI, OUTER_LIP_INDICES } from '../web/mouth-tracker.js';

const root = fileURLToPath(new URL('../web/', import.meta.url));
function landmarks(dx = 0, opening = .025) {
  const points = Array.from({ length: 478 }, () => ({ x: .5 + dx, y: .5 }));
  OUTER_LIP_INDICES.forEach((index, n) => {
    const angle = Math.PI - n * Math.PI / 10;
    points[index] = { x: .5 + dx + .08 * Math.cos(angle), y: .62 + opening * Math.sin(angle) };
  });
  points[234] = { x: .3 + dx, y: .48 };
  points[454] = { x: .7 + dx, y: .48 };
  points[1] = { x: .5 + dx, y: .52 };
  return points;
}
const initial = updateMouthTrack(newMouthTrack(), landmarks(), 640, 480, 1000);
assert.equal(initial.status, 'tracking');
assert.equal(initial.lips.length, 20);
assert(Math.abs(initial.roi.size * 480 - 2.13 * .16 * 640) < 1e-8);
const translated = updateMouthTrack(initial.state, landmarks(.03), 640, 480, 1040);
assert(translated.roi.cx > initial.roi.cx && translated.roi.cx < .53, 'Translation must be smoothed');
const opened = updateMouthTrack(initial.state, landmarks(0, .07), 640, 480, 1040);
assert.equal(opened.roi.size, initial.roi.size, 'Lip opening must not pump crop scale');
assert.notDeepEqual(opened.lips, initial.lips, 'Articulation landmarks must remain unsmoothed');
// A miss within the grace period coasts on the last ROI without changing status.
const missing = updateMouthTrack(initial.state, null, 640, 480, 1080);
assert.equal(missing.status, 'tracking');
assert.equal(missing.coasted, true);
assert.deepEqual(missing.roi, initial.roi);
assert.equal(updateMouthTrack(missing.state, landmarks(.01), 640, 480, 1200).coasted, false);
const graceEnd = updateMouthTrack(initial.state, null, 640, 480, 1501);
// Beyond grace, output keeps going on the last crop while the tracker searches.
assert.equal(graceEnd.status, 'tracking');
assert.equal(graceEnd.searching, true);
assert.deepEqual(graceEnd.roi, initial.roi, 'A missing face keeps the last crop');
assert.equal(updateMouthTrack(newMouthTrack(), null, 640, 480, 1000).status, 'lost', 'No target seen yet: nothing to hold');
// After a long loss the same target re-locks automatically after 4 consistent frames, without a click.
let reTrack = updateMouthTrack(initial.state, null, 640, 480, 3100);
assert.equal(reTrack.searching, true);
const relockStatuses = [];
for (let n = 0; n < 4; n++) {
  reTrack = updateMouthTrack(reTrack.state, landmarks(.01), 640, 480, 3140 + n * 40);
  relockStatuses.push(!!reTrack.searching);
}
assert.deepEqual(relockStatuses, [true, true, true, false]);
assert.equal(reTrack.relocked, 'consistent');
// An intermittent consistent candidate restarts the confirmation count.
let flicker = updateMouthTrack(initial.state, null, 640, 480, 3100);
for (const [t, lm] of [[3140, landmarks()], [3180, landmarks()], [3220, null], [3260, landmarks()], [3300, landmarks()]]) {
  flicker = updateMouthTrack(flicker.state, lm, 640, 480, t);
}
assert.equal(flicker.searching, true);
const narrow = landmarks(); narrow[234].x = .35; narrow[454].x = .65;
const anchor = updateMouthTrack(newMouthTrack(), narrow, 640, 480, 1000);
const moved = narrow.map((point) => ({ ...point, x: point.x + .24 }));
// An obvious jump is never followed during grace; afterwards it is trusted only after 2.5 s of stability.
const jump = updateMouthTrack(anchor.state, moved, 640, 480, 1040);
assert.equal(jump.coasted, true);
assert.deepEqual(jump.roi, anchor.roi, 'A jumped face must not move the crop');
let jumped = updateMouthTrack(jump.state, moved, 640, 480, 1600);
assert.equal(jumped.searching, true);
assert.deepEqual(jumped.roi, anchor.roi, 'A jumped face never moves the crop before confirmation');
for (let t = 1640; t < 4100; t += 40) jumped = updateMouthTrack(jumped.state, moved, 640, 480, t);
assert.equal(jumped.searching, true, 'Inconsistent face is not accepted before 2.5 s');
jumped = updateMouthTrack(jumped.state, moved, 640, 480, 4140);
assert.equal(jumped.searching, undefined);
assert.equal(jumped.relocked, 'inconsistent');
// With the original target and a second person both visible, the consistent face wins.
const other = landmarks(-.25);
let pick = updateMouthTrack(initial.state, [other, landmarks(.01)], 640, 480, 1040);
assert.equal(pick.status, 'tracking');
assert(Math.abs(pick.state.face.x - .51) < 1e-9, 'Consistent face is selected among several');
// Two faces that are both inconsistent with the target: ask the user, never guess.
let ambiguous = updateMouthTrack(anchor.state, null, 640, 480, 2000);
ambiguous = updateMouthTrack(ambiguous.state, [moved, other], 640, 480, 2040);
assert.equal(ambiguous.searching, true);
assert.match(ambiguous.message, /多个人脸/);
assert.equal(updateMouthTrack(newMouthTrack(), landmarks(.25), 640, 480, 1080).status, 'tracking');
const invalid = landmarks(); invalid[61].x = NaN;
assert.equal(mouthGeometry(invalid, 640, 480), null);
const edge = clampROI({ cx: 0, cy: 1, size: .4 }, 640, 480);
assert(edge.cx * 640 >= .4 * 480 / 2 && edge.cy * 480 <= 480 - .4 * 480 / 2);

const messages = [], crops = [];
let gate = null, throwEncode = false;
const ctx = {
  drawImage: (...args) => crops.push(args),
  getImageData: () => ({ data: new Uint8ClampedArray([200, 100, 50, 255]) }),
  putImageData() {},
};
const fakeDetector = { detectForVideo: () => ({ faceLandmarks: [landmarks()] }), close() {} };
const realm = vm.createContext({ self: { postMessage: (message) => messages.push(message) },
  performance: { timeOrigin: 100000, now: () => 1 }, mockDetector: fakeDetector,
  OffscreenCanvas: class { getContext() { return ctx; } async convertToBlob() {
    if (gate) await gate;
    if (throwEncode) throw new Error('test encoder failure');
    return { arrayBuffer: async () => new Uint8Array([255, 216]).buffer };
  } },
});
const geometrySource = fs.readFileSync(`${root}mouth-tracker.js`, 'utf8').replace(/^export /gm, '');
const workerSource = fs.readFileSync(`${root}video-worker.js`, 'utf8').replace(/^import .*mouth-tracker\.js';\n/m, '');
vm.runInContext(geometrySource, realm);
vm.runInContext(workerSource, realm);
const send = (message) => realm.self.onmessage({ data: message });
const bitmap = () => ({ width: 640, height: 480, closed: 0, close() { this.closed++; } });
await send({ type: 'start', tracking: 'manual', trackingEpoch: 1 });
assert.equal(messages[0].type, 'ready', 'Ready must precede detector initialization');
const first = bitmap();
await send({ type: 'frame', bitmap: first, wallTime: 100000, trackingEpoch: 1 });
assert.equal(first.closed, 1);
assert.equal(messages.at(-1).type, 'ack');
assert(messages.some((m) => m.type === 'frame' && m.trackingEpoch === 1));

// A bitmap captured before a mode switch must not be relabeled as the new epoch.
await send({ type: 'tracking', mode: 'manual', trackingEpoch: 2 });
const old = bitmap(), oldCount = messages.filter((m) => m.type === 'frame').length;
await send({ type: 'frame', bitmap: old, wallTime: 100040, trackingEpoch: 1 });
assert.equal(old.closed, 1);
assert.equal(messages.at(-1).trackingEpoch, 1);
assert.equal(messages.filter((m) => m.type === 'frame').length, oldCount);

// A mode switch while JPEG encoding awaits completion must discard that JPEG.
let release;
gate = new Promise((resolve) => { release = resolve; });
const pending = bitmap();
const work = send({ type: 'frame', bitmap: pending, wallTime: 100080, trackingEpoch: 2 });
await Promise.resolve();
await send({ type: 'tracking', mode: 'manual', trackingEpoch: 3 });
release(); await work; gate = null;
assert.equal(pending.closed, 1);
assert.equal(messages.filter((m) => m.type === 'frame').length, oldCount);

// A missing face in automatic mode keeps emitting crops at the last ROI.
vm.runInContext('detector = mockDetector', realm);
await send({ type: 'tracking', mode: 'auto', trackingEpoch: 4 });
const tracked = bitmap();
await send({ type: 'frame', bitmap: tracked, wallTime: 100120, trackingEpoch: 4 });
assert.equal(tracked.closed, 1);
assert(messages.some((m) => m.type === 'tracking' && m.status === 'tracking' && m.trackingEpoch === 4));
fakeDetector.detectForVideo = () => ({ faceLandmarks: [] });
const beforeLoss = messages.filter((m) => m.type === 'frame').length;
const coastBitmap = bitmap();
await send({ type: 'frame', bitmap: coastBitmap, wallTime: 100160, trackingEpoch: 4 });
const coastFrame = messages.filter((m) => m.type === 'frame').at(-1);
assert.equal(coastFrame.coasted, true, 'A brief miss keeps emitting crops at the last ROI');
const afterCoast = messages.filter((m) => m.type === 'frame').length;
const blank = bitmap();
await send({ type: 'frame', bitmap: blank, wallTime: 100800, trackingEpoch: 4 });
assert.equal(blank.closed, 1);
assert.equal(beforeLoss + 1, afterCoast);
assert.equal(messages.filter((m) => m.type === 'frame').length, afterCoast + 1, 'Output continues after the grace period');
assert(messages.some((m) => m.type === 'tracking' && m.searching && m.trackingEpoch === 4));
assert.equal(messages.at(-1).type, 'ack');

// Exceptions and stop both release fallback ownership and acknowledge input.
await send({ type: 'tracking', mode: 'manual', trackingEpoch: 5 });
throwEncode = true;
const failed = bitmap();
await send({ type: 'frame', bitmap: failed, wallTime: 100200, trackingEpoch: 5 });
assert.equal(failed.closed, 1);
assert.equal(messages.at(-1).type, 'ack');
assert(messages.some((m) => m.type === 'error'));
await send({ type: 'stop' });
const stoppedBitmap = bitmap();
await send({ type: 'frame', bitmap: stoppedBitmap, wallTime: 100240, trackingEpoch: 5 });
assert.equal(stoppedBitmap.closed, 1);
assert.equal(messages.at(-1).type, 'ack');
console.log('PASS: mouth geometry, stable scale/articulation, continuous output without a face, automatic re-lock, jump confirmation, multi-face choice, fallback ownership, epoch races, encoding failure and stop.');
