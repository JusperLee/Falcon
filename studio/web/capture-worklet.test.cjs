// Run with: node web/capture-worklet.test.cjs
// Real DSP checks: output clock/count, passband preservation, stopband alias rejection,
// and bounded recovery after a render-clock discontinuity. No microphone is used.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const code = fs.readFileSync(`${__dirname}/capture-worklet.js`, 'utf8');

function makeProcessor(rate) {
  const packets = [];
  let Processor;
  const context = vm.createContext({
    sampleRate: rate, currentFrame: 0,
    AudioWorkletProcessor: class { constructor() { this.port = { postMessage: (packet) => packets.push(packet) }; } },
    registerProcessor: (_name, Class) => { Processor = Class; },
  });
  vm.runInContext(code, context);
  return { processor: new Processor(), context, packets };
}
function tone(rate, frequency, seconds = 1) {
  const fixture = makeProcessor(rate);
  for (let frame = 0; frame < rate * seconds; frame += 128) {
    fixture.context.currentFrame = frame;
    const input = Float32Array.from({ length: 128 }, (_, i) => .8 * Math.sin(2 * Math.PI * frequency * (frame + i) / rate));
    fixture.processor.process([[input]], [[new Float32Array(128)]]);
  }
  const samples = fixture.packets.flatMap((packet) => [...packet.samples]);
  const settled = samples.slice(640);
  const rms = Math.sqrt(settled.reduce((sum, value) => sum + value ** 2, 0) / settled.length);
  return { ...fixture, samples, rms };
}
for (const rate of [16000, 44100, 48000]) {
  const result = tone(rate, 1000);
  assert.ok(Math.abs(result.samples.length - 16000) < 320, `${rate}: output count ${result.samples.length}`);
  assert.ok(Math.abs(result.rms - .8 / Math.sqrt(2)) < .015, `${rate}: passband RMS ${result.rms}`);
  assert.ok(Math.abs(result.packets[0].startTime + 31 / rate) < 1e-9, 'group-delay timestamp compensation');
  for (let i = 1; i < result.packets.length; i++) {
    assert.ok(Math.abs(result.packets[i].startTime - result.packets[i - 1].startTime - .02) < 1e-8, 'continuous packet clock');
  }
}
for (const rate of [44100, 48000]) {
  const result = tone(rate, 12000);
  assert.ok(result.rms < .006, `${rate}: 12 kHz alias rejected, actual RMS ${result.rms}`);
}
const gap = tone(48000, 1000, .1);
const before = gap.packets.length;
gap.context.currentFrame = 48000 * 60;
gap.processor.process([[new Float32Array(128)]], [[new Float32Array(128)]]);
assert.equal(gap.packets.length, before, 'a clock gap must not synthesize a minute of stale samples');
assert.ok(gap.processor.packetLength < 64, 'clock-gap packet buffer starts afresh');
console.log('PASS: 16/44.1/48 kHz clocks, 1 kHz passband, 12 kHz alias rejection, FIR timing, clock-gap recovery.');
