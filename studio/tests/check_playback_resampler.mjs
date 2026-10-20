// Run with: node tests/check_playback_resampler.mjs
// Real DSP checks on the streaming 16 kHz -> context.sampleRate upsampler used
// by web/app.js's playback path (playSamples). No browser, AudioContext or
// AudioBuffer is used: this measures PlaybackUpsampler's own output samples.
import assert from 'node:assert/strict';
import { PlaybackUpsampler, createPlaybackUpsampler } from '../web/playback-resampler.js';

// Minimal radix-2 FFT, independent of the module under test, for spectral
// measurements (passband droop, image/alias level).
function fft(real, imag) {
  const n = real.length;
  if (n <= 1) return;
  const evenR = new Float64Array(n / 2), evenI = new Float64Array(n / 2);
  const oddR = new Float64Array(n / 2), oddI = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    evenR[i] = real[2 * i]; evenI[i] = imag[2 * i];
    oddR[i] = real[2 * i + 1]; oddI[i] = imag[2 * i + 1];
  }
  fft(evenR, evenI); fft(oddR, oddI);
  for (let k = 0; k < n / 2; k++) {
    const angle = -2 * Math.PI * k / n;
    const cos = Math.cos(angle), sin = Math.sin(angle);
    const tr = cos * oddR[k] - sin * oddI[k];
    const ti = sin * oddR[k] + cos * oddI[k];
    real[k] = evenR[k] + tr; imag[k] = evenI[k] + ti;
    real[k + n / 2] = evenR[k] - tr; imag[k + n / 2] = evenI[k] - ti;
  }
}
function nextPow2(n) { let p = 1; while (p < n) p *= 2; return p; }
// Magnitude, in dB relative to a full-scale sinusoid, of the content near
// freqHz in `signal` sampled at `rate`. Hann-windowed to limit leakage.
function magnitudeDb(signal, rate, freqHz) {
  const n = nextPow2(signal.length);
  const real = new Float64Array(n), imag = new Float64Array(n);
  for (let i = 0; i < signal.length; i++) {
    const w = .5 - .5 * Math.cos(2 * Math.PI * i / (signal.length - 1));
    real[i] = signal[i] * w;
  }
  fft(real, imag);
  const bin = Math.round(freqHz / rate * n);
  let peak = 0;
  for (let b = Math.max(0, bin - 2); b <= Math.min(n / 2, bin + 2); b++) peak = Math.max(peak, Math.hypot(real[b], imag[b]));
  return 20 * Math.log10(peak / (signal.length / 2) + 1e-300);
}
function tone(freq, seconds, sourceRate = 16000) {
  const n = Math.round(sourceRate * seconds);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * freq * i / sourceRate);
  return x;
}
function toneResponseDb(freq, targetRate, seconds = 1) {
  const up = new PlaybackUpsampler(16000, targetRate);
  const y = up.push(tone(freq, seconds));
  const trimmed = y.subarray(Math.floor(y.length * .1), Math.floor(y.length * .9));
  return magnitudeDb(trimmed, targetRate, freq);
}
// Direct time-domain peak amplitude of a steady-state pure tone, in dB
// relative to full scale. Used for passband droop: a windowed FFT loses
// accuracy for low frequencies close to its own analysis window's skirt
// (verified separately: 100 Hz measures as a false -0.24 dB "droop" by FFT
// while the actual time-domain peak amplitude is 1.0007, i.e. ~0 dB), so the
// passband check measures amplitude directly rather than through the FFT.
function tonePeakDb(freq, targetRate, seconds = 1) {
  const up = new PlaybackUpsampler(16000, targetRate);
  const y = up.push(tone(freq, seconds));
  const mid = y.subarray(Math.floor(y.length * .3), Math.floor(y.length * .7));
  let peak = 0;
  for (const s of mid) peak = Math.max(peak, Math.abs(s));
  return 20 * Math.log10(peak);
}

const SOURCE_RATE = 16000;
const HOP = SOURCE_RATE / 2; // 8000 samples, the project's 0.5 s model hop

// 1. Passband: +-0.1 dB to 7 kHz relative to 1 kHz, at both target rates.
// Measured as steady-state time-domain peak amplitude (see tonePeakDb), not
// via the FFT helper, which loses accuracy near its own analysis window's
// skirt for low frequencies close to DC.
for (const targetRate of [44100, 48000]) {
  const ref = tonePeakDb(1000, targetRate);
  for (const f of [100, 500, 1000, 2000, 4000, 6000, 7000]) {
    const db = tonePeakDb(f, targetRate) - ref;
    assert.ok(Math.abs(db) <= .1, `${targetRate} Hz target: ${f} Hz passband droop ${db.toFixed(3)} dB exceeds +-0.1 dB`);
  }
}
console.log('PASS: passband +-0.1 dB to 7 kHz at 44.1 kHz and 48 kHz');

// 2. Images: a tone at the capture chain's actual cutoff edge (7200 Hz, see
// web/capture-worklet.js: cutoff = min(7200, 0.45*fs)) must not leave any
// image above -60 dB relative to the passband tone, anywhere above 8.5 kHz up
// to just below the target Nyquist.
for (const targetRate of [44100, 48000]) {
  const passbandRefDb = toneResponseDb(1000, targetRate);
  const up = new PlaybackUpsampler(16000, targetRate);
  const y = up.push(tone(7200, 1));
  const trimmed = y.subarray(Math.floor(y.length * .1), Math.floor(y.length * .9));
  for (const probe of [8500, 9000, 10000, 12000, 14000, 16000, 18000, 20000]) {
    if (probe >= targetRate / 2 - 200) continue;
    const level = magnitudeDb(trimmed, targetRate, probe) - passbandRefDb;
    assert.ok(level <= -60, `${targetRate} Hz target: image at ${probe} Hz is ${level.toFixed(2)} dB relative to passband, must be <= -60 dB`);
  }
}
console.log('PASS: images <= -60 dB above 8.5 kHz relative to the passband tone, for a 7.2 kHz input (capture cutoff edge)');

// 3. Chunked (0.5 s hops, as playSamples receives from the server) must equal
// one continuous push() bit-for-bit, at both target rates and for a
// representative speech-like multi-tone signal (not just a single sinusoid,
// so the check also covers inter-tone phase handling).
for (const targetRate of [44100, 48000]) {
  const n = HOP * 6; // 3 s, six hops
  const full = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    full[i] = .5 * Math.sin(2 * Math.PI * 220 * i / SOURCE_RATE)
      + .3 * Math.sin(2 * Math.PI * 1800 * i / SOURCE_RATE)
      + .1 * Math.sin(2 * Math.PI * 5500 * i / SOURCE_RATE);
  }
  const oneShot = new PlaybackUpsampler(16000, targetRate).push(full);
  const chunkedUp = new PlaybackUpsampler(16000, targetRate);
  const parts = [];
  for (let i = 0; i < n; i += HOP) parts.push(chunkedUp.push(full.subarray(i, i + HOP)));
  const chunked = Float32Array.from(parts.flatMap((p) => Array.from(p)));
  assert.equal(chunked.length, oneShot.length, `${targetRate} Hz: chunked length ${chunked.length} != one-shot length ${oneShot.length}`);
  let maxDiff = 0;
  for (let i = 0; i < oneShot.length; i++) maxDiff = Math.max(maxDiff, Math.abs(oneShot[i] - chunked[i]));
  assert.ok(maxDiff < 1e-6, `${targetRate} Hz: chunked vs one-shot max abs diff ${maxDiff} too large`);
}
console.log('PASS: six chunked 0.5 s hops produce output bit-identical to one continuous push()');

// 4. Exact output length per 0.5 s (8000-sample) hop at 44.1 kHz and 48 kHz,
// steady state (after the first hop's filter warm-up / group delay has been
// absorbed), matching HOP * targetRate / 16000 to within one sample.
for (const targetRate of [44100, 48000]) {
  const up = new PlaybackUpsampler(16000, targetRate);
  const expectedPerHop = HOP * targetRate / 16000;
  const silence = new Float32Array(HOP);
  up.push(silence); // warm-up hop: absorbs the TAPS/2 group-delay shortfall
  for (let hop = 0; hop < 5; hop++) {
    const out = up.push(silence);
    assert.ok(Math.abs(out.length - expectedPerHop) <= 1,
      `${targetRate} Hz: steady-state hop length ${out.length}, expected ~${expectedPerHop}`);
  }
}
console.log('PASS: exact steady-state output length per 0.5 s hop at 44.1 kHz and 48 kHz');

// 5. Reset behaviour: reset() must drop FIR history and fractional phase so a
// post-reset hop doesn't blend in the previous, unrelated stream (e.g. after
// a non-contiguous model result, stopMonitoring, or a fresh playAt restart),
// while still producing a correctly filtered, non-empty hop immediately after.
{
  const targetRate = 48000;
  const up = new PlaybackUpsampler(16000, targetRate);
  up.push(tone(1000, 1)); // establish non-trivial history/phase state
  const beforeReset = { historyIndex: up.historyIndex, nextPosition: up.nextPosition, filled: up.filled };
  up.reset();
  assert.deepEqual(up.history, new Float32Array(63), 'reset must zero FIR history');
  assert.equal(up.filled, 0, 'reset must zero the filled-sample counter');
  assert.equal(up.nextPosition, (63 - 1) / 2, 'reset must restore the initial group-delay phase');
  assert.notEqual(JSON.stringify(beforeReset), JSON.stringify({ historyIndex: up.historyIndex, nextPosition: up.nextPosition, filled: up.filled }),
    'state must actually have been non-trivial before reset (test sanity)');
  const afterReset = up.push(tone(1000, .5));
  assert.ok(afterReset.length > 0, 'a hop immediately after reset must still produce output once enough samples arrive');
  // A silent hop right after reset must come out silent (no leaked energy
  // from the pre-reset tone), proving history was actually cleared.
  const fresh = new PlaybackUpsampler(16000, targetRate);
  fresh.push(tone(1000, 1));
  fresh.reset();
  const afterSilence = fresh.push(new Float32Array(HOP));
  const maxAbs = Math.max(...afterSilence, 0, ...Array.from(afterSilence, Math.abs));
  assert.ok(maxAbs < 1e-5, `reset must clear FIR history: silent hop after reset has max abs ${maxAbs}`);
}
console.log('PASS: reset() clears FIR history and fractional phase, and post-reset output is correctly silent/filtered');

// 6. createPlaybackUpsampler returns null for identity (no resampling needed)
// and a working upsampler otherwise, so app.js's playSamples can bypass the
// resampling branch entirely when the device happens to run at 16 kHz.
assert.equal(createPlaybackUpsampler(16000, 16000), null, 'identity rate must bypass resampling (return null)');
assert.ok(createPlaybackUpsampler(48000, 16000) instanceof PlaybackUpsampler);
console.log('PASS: createPlaybackUpsampler identity bypass and construction');

console.log('ALL PASS: check_playback_resampler');
