// Run with: node web/video-worker.test.cjs
// Checks the actual worker's frame selection, ROI, grayscale conversion and ownership.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(`${__dirname}/video-worker.js`, 'utf8').replace(/^import .*mouth-tracker\.js';\n/m, '');
const geometry = fs.readFileSync(`${__dirname}/mouth-tracker.js`, 'utf8').replace(/^export /gm, '');

async function run({ failEncode = false, count = 60 } = {}) {
  const messages = [], draws = [], pixels = [], frames = [];
  let index = 0, cancelled = false, released = false, now = 0;
  for (let i = 0; i < count; i++) frames.push({ timestamp: i * 1e6 / 30, displayWidth: 640, displayHeight: 480, closed: 0, close() { this.closed++; } });
  const reader = {
    async read() { if (index >= frames.length) return { done: true }; const value = frames[index++]; now = value.timestamp / 1000; return { value, done: false }; },
    async cancel() { cancelled = true; }, releaseLock() { released = true; },
  };
  const self = { postMessage: (message) => messages.push(message) };
  const context = {
    drawImage: (...args) => draws.push(args),
    getImageData: () => ({ data: new Uint8ClampedArray([200, 100, 50, 255]) }),
    putImageData: (image) => pixels.push([...image.data]),
  };
  const realm = vm.createContext({ self, performance: { timeOrigin: 100000, now: () => now }, OffscreenCanvas: class {
    getContext() { return context; }
    async convertToBlob() { if (failEncode) throw new Error('encode failure'); return { arrayBuffer: async () => new Uint8Array([255, 216]).buffer }; }
  } });
  vm.runInContext(geometry, realm);
  vm.runInContext(source, realm);
  await self.onmessage({ data: { type: 'start', tracking: 'manual', trackingEpoch: 0, readable: { getReader: () => reader }, roi: { cx: .5, cy: .6, size: .2 } } });
  return { messages, draws, pixels, frames, cancelled, released };
}
(async () => {
  const normal = await run();
  const outputs = normal.messages.filter((message) => message.type === 'frame');
  assert.equal(outputs.length, 50, '30 fps input sampled to 25 fps without systematic half-rate decimation');
  assert.ok(normal.frames.every((frame) => frame.closed === 1), 'every delivered/skipped frame closed exactly once');
  assert.deepEqual(normal.draws[0].slice(1), [272, 240, 96, 96, 0, 0, 96, 96], 'normalized mouth ROI maps to source pixels');
  assert.deepEqual(normal.pixels[0], [124, 124, 124, 255], 'luma grayscale is correct and alpha preserved');
  assert.ok(outputs.every((output) => output.wallTime >= 100000 && output.jpeg.byteLength === 2), 'wall clock and transferable JPEG present');
  assert.ok(normal.cancelled && normal.released, 'reader cancelled and ownership released');
  const failed = await run({ failEncode: true, count: 1 });
  assert.equal(failed.frames[0].closed, 1, 'encoding error still closes VideoFrame');
  assert.ok(failed.messages.some((message) => message.type === 'error'), 'encoding error surfaces');
  assert.ok(failed.cancelled && failed.released, 'encoding error releases reader');
  console.log('PASS: 25 fps sampling, grayscale ROI, wall-clock messages, VideoFrame closure, error cleanup.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
