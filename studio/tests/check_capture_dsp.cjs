/* Real worklet DSP in a deterministic render-quantum harness; no microphone. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../web/capture-worklet.js'), 'utf8');
function harness(rate) {
  const packets = [];
  let Processor;
  const context = {
    sampleRate: rate, currentFrame: 0, Float32Array, Math,
    AudioWorkletProcessor: class { constructor() { this.port = { postMessage: (packet) => packets.push(packet) }; } },
    registerProcessor: (_, constructor) => { Processor = constructor; },
  };
  vm.runInNewContext(source, context);
  const processor = new Processor();
  return { context, processor, packets };
}
function tone(rate, frequency) {
  const run = harness(rate);
  for (let start = 0; start < rate * 2; start += 128) {
    const count = Math.min(128, rate * 2 - start);
    const channel = Float32Array.from({ length: count }, (_, i) => Math.sin(2 * Math.PI * frequency * (start + i) / rate));
    const muted = new Float32Array(count).fill(1);
    run.context.currentFrame = start;
    run.processor.process([[channel]], [[muted]]);
    assert(muted.every((x) => x === 0), 'Capture output must not monitor raw audio');
  }
  const samples = run.packets.flatMap((packet) => Array.from(packet.samples));
  assert.equal(samples.length, 32000, 'Two seconds must contain exactly 32000 samples');
  run.packets.forEach((packet, i) => {
    const expectedTime = -31 / rate + i * 0.02;
    assert(Math.abs(packet.startTime - expectedTime) < 1e-7, 'Compensated timestamps drift');
  });
  const stable = samples.slice(1600);
  const rms = Math.sqrt(stable.reduce((sum, x) => sum + x * x, 0) / stable.length);
  let error = 0;
  for (let i = 1600; i < samples.length; i++) {
    const expected = Math.sin(2 * Math.PI * frequency * (i / 16000 - 31 / rate));
    error += (samples[i] - expected) ** 2;
  }
  return { rms, referenceErrorRms: Math.sqrt(error / stable.length) };
}
const results = [];
for (const rate of [44100, 48000]) {
  const pass = tone(rate, 1000);
  const stop = tone(rate, 12000);
  const attenuation = 20 * Math.log10(stop.rms / pass.rms);
  assert(Math.abs(pass.rms - Math.SQRT1_2) < 0.01, 'Speech passband gain changed');
  assert(pass.referenceErrorRms < 0.003, 'FIR timestamps disagree with actual signal phase');
  assert(attenuation < -40, 'Out-of-band audio aliases into the 16kHz output');
  results.push({ input_rate: rate, output_rate: 16000, passband_rms: pass.rms,
    timestamp_reference_error_rms: pass.referenceErrorRms, alias_attenuation_db: attenuation });
}
const gap = harness(48000);
gap.processor.process([[new Float32Array(128)]], [[new Float32Array(128)]]);
for (let start = 48000; start < 49024; start += 128) {
  gap.context.currentFrame = start;
  gap.processor.process([[new Float32Array(128)]], [[new Float32Array(128)]]);
}
assert.equal(gap.packets.length, 1, 'A clock gap must not generate stale audio packets');
assert(Math.abs(gap.packets[0].startTime - (48000 - 31) / 48000) < 1e-7);
console.log(JSON.stringify({ test: 'PASS', rates: results, clock_gap_reset: true }, null, 2));
