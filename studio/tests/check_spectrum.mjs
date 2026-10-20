import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// Import the browser ESM without requiring a package.json change.
const source = await readFile(new URL('../web/spectrum-dsp.js', import.meta.url), 'utf8');
const { SpectrumDSP, LiveSpectrum, SpectrumProcessor, recordingView,
  FFT_SIZE, HOP_SIZE, FREQUENCY_ROWS, SAMPLE_RATE, MAX_RECORDING_SAMPLES } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const tone = (frequency, amplitude = 1, length = FFT_SIZE) => Float32Array.from({ length }, (_, i) => amplitude * Math.sin(2 * Math.PI * frequency * i / SAMPLE_RATE));
const expectedRow = frequency => FREQUENCY_ROWS - 1 - Math.min(FREQUENCY_ROWS - 1, Math.floor(frequency / (SAMPLE_RATE / 2) * FREQUENCY_ROWS));
const decodedDb = value => value * 90 / 255 - 90;
const checks = [];

const dsp = new SpectrumDSP();
for (const frequency of [1000, 3000, 7000]) {
  for (const amplitude of [1, .5, .1]) {
    const column = dsp.column(tone(frequency, amplitude));
    const peak = Math.max(...column), row = column.indexOf(peak);
    assert.equal(row, expectedRow(frequency), `${frequency} Hz appears in the wrong linear-frequency row`);
    assert(Math.abs(decodedDb(peak) - 20 * Math.log10(amplitude)) <= .2, 'Hann coherent-gain dBFS calibration changed');
  }
}
assert.equal(dsp.column(tone(1000))[expectedRow(1000)], 255);
assert.equal(dsp.column(new Float32Array(FFT_SIZE).fill(1))[127], 255, 'DC must not get doubled');
const nyquist = Float32Array.from({ length: FFT_SIZE }, (_, i) => i % 2 ? -1 : 1);
assert.equal(dsp.column(nyquist)[0], 255, 'Nyquist must not get doubled');
checks.push('frequency ordering and full-scale / -6 / -20 dBFS tone calibration');

assert(dsp.column(new Float32Array(FFT_SIZE)).every(value => value === 0));
const invalid = new Float32Array(FFT_SIZE); invalid[1] = NaN; invalid[50] = Infinity; invalid[99] = -Infinity;
assert(dsp.column(invalid).every(value => value === 0));
checks.push('silence and non-finite samples remain silent');

const audio = tone(1300, .4, 32000);
for (let i = 8000; i < 16000; i++) audio[i] = .7 * Math.sin(2 * Math.PI * 6200 * i / SAMPLE_RATE);
const entire = new LiveSpectrum().push(audio);
assert.equal(entire.width, 1 + Math.floor((audio.length - FFT_SIZE) / HOP_SIZE));
const chunks = [], split = new LiveSpectrum();
let position = 0, chunkIndex = 0;
const lengths = [1, 320, 127, 1024, 23, 4096, 8000, 640];
while (position < audio.length) {
  const end = Math.min(audio.length, position + lengths[chunkIndex++ % lengths.length]);
  chunks.push(split.push(audio.subarray(position, end)));
  position = end;
}
const joined = new Uint8Array(entire.columns.length);
let columnPosition = 0;
for (const chunk of chunks) {
  for (let row = 0; row < FREQUENCY_ROWS; row++) joined.set(chunk.columns.subarray(row * chunk.width, (row + 1) * chunk.width), row * entire.width + columnPosition);
  columnPosition += chunk.width;
}
assert.deepEqual(joined, entire.columns, 'Live spectra must be independent of capture packet boundaries');
split.reset();
assert.equal(split.push(new Float32Array(1023)).width, 0);
assert.equal(split.push(new Float32Array(1)).width, 1);
assert(split.push(new Float32Array(HOP_SIZE)).columns.every(value => value === 0));
checks.push('fixed 640-sample hop, exact chunk-boundary continuity and cleared history');

const spikes = new Float32Array(16000);
spikes[19] = .75; spikes[99] = -.8; spikes[7999] = -.6; spikes[15999] = .9;
const overview = recordingView(spikes, 0, 1, 160, dsp);
assert.equal(overview.pixels.length, 160 * FREQUENCY_ROWS);
assert.equal(overview.waveMax[0], .75);
assert(Math.abs(overview.waveMin[0] + .8) < 1e-6);
assert(Math.abs(overview.waveMin[79] + .6) < 1e-6);
assert(Math.abs(overview.waveMax[159] - .9) < 1e-6, 'Last sample peak was lost');
const zoom = recordingView(spikes, .99, 1, 320, dsp);
assert(Math.abs(Math.max(...zoom.waveMax) - .9) < 1e-6, 'Subsample zoom must preserve the final peak');
const bounded = recordingView(spikes, -4, 99, 100000, dsp);
assert.equal(bounded.start, 0); assert.equal(bounded.end, 1); assert.equal(bounded.width, 1600);
const empty = recordingView(new Float32Array(0), 2, 4, 300, dsp);
assert.equal(empty.start, 0); assert.equal(empty.end, 0);
assert(empty.pixels.every(value => value === 0) && empty.waveMin.every(value => value === 0));
const steady = recordingView(tone(3000, .5, 16000), .1, .9, 100, dsp);
for (let col = 0; col < steady.width; col++) assert(Math.abs(decodedDb(steady.pixels[expectedRow(3000) * steady.width + col]) + 6.0206) < .2);
checks.push('recording spectrum, bounded view, waveform impulses and empty audio');

// Five minutes at 1600 columns spans 3000 samples per column: a single central
// 1024-sample window used to miss both of these early-column transients entirely.
const longAudio = new Float32Array(MAX_RECORDING_SAMPLES);
const transientColumn = 147, impulseColumn = 1111, overviewWidth = 1600;
const columnSamples = MAX_RECORDING_SAMPLES / overviewWidth;
longAudio.set(tone(3000, 1, 128), transientColumn * columnSamples + 64);
longAudio[impulseColumn * columnSamples + 80] = .9;
const formerCentreStart = Math.round((transientColumn + .5) * columnSamples - FFT_SIZE / 2);
assert(dsp.column(longAudio, formerCentreStart).every(value => value === 0), 'Fixture must be outside the former central window');
class CountingDSP extends SpectrumDSP {
  count = 0;
  column(...args) { this.count++; return super.column(...args); }
}
const overviewDSP = new CountingDSP();
const overviewStarted = performance.now();
const longView = recordingView(longAudio, 0, 300, overviewWidth, overviewDSP);
const overviewMilliseconds = Math.round((performance.now() - overviewStarted) * 10) / 10;
assert(decodedDb(longView.pixels[expectedRow(3000) * overviewWidth + transientColumn]) > -25, 'Short tone far from a column centre vanished');
assert(decodedDb(longView.pixels[expectedRow(3000) * overviewWidth + impulseColumn]) > -65, 'Off-centre impulse vanished');
assert.equal(overviewDSP.count, overviewWidth * Math.ceil(columnSamples / (FFT_SIZE / 2)));
assert(overviewDSP.count <= Math.ceil(MAX_RECORDING_SAMPLES / (FFT_SIZE / 2)) + overviewWidth, 'Overview FFT work must remain bounded');
assert(Math.abs(longView.waveMax[impulseColumn] - .9) < 1e-6);
checks.push('five-minute overview preserves off-centre short tones and impulses with bounded overlapping FFT coverage');

const messages = [];
const processor = new SpectrumProcessor((message, transfer = []) => {
  for (const buffer of transfer) assert(buffer instanceof ArrayBuffer);
  messages.push(message);
});
processor.handle({ type: 'live', channel: 'input', epoch: 1, samples: tone(1000, 1, 320) });
assert.equal(messages.at(-1).width, 0, 'Incomplete live frames must acknowledge receipt');
processor.handle({ type: 'live', channel: 'input', epoch: 1, samples: tone(1000, 1, 320) });
processor.handle({ type: 'live', channel: 'output', epoch: 1, samples: tone(7000, 1, 1024) });
assert.equal(messages.at(-1).columns.indexOf(255), expectedRow(7000));
processor.handle({ type: 'reset-live', channel: 'output', epoch: 2 });
processor.handle({ type: 'live', channel: 'input', epoch: 1, samples: new Float32Array(384) });
assert.equal(messages.at(-1).width, 1, 'Output reset must preserve input history');
processor.handle({ type: 'live', channel: 'output', epoch: 2, samples: new Float32Array(1024) });
assert(messages.at(-1).columns.every(value => value === 0));
const beforeOld = messages.length;
processor.handle({ type: 'live', channel: 'output', epoch: 1, samples: tone(1000) });
assert.equal(messages.length, beforeOld, 'Old epochs must not contaminate a reset channel');
processor.handle({ type: 'live', channel: 'input', epoch: 3, samples: new Float32Array(1024) });
assert(messages.at(-1).columns.every(value => value === 0), 'A new epoch must clear channel history automatically');
processor.handle({ type: 'reset-live', epoch: 4 });
assert.equal(processor.live.input.total, 0); assert.equal(processor.live.output.total, 0);
processor.handle({ type: 'recording', epoch: 10, samples: Float32Array.of(NaN, .2, Infinity, -.4) });
assert.deepEqual(messages.at(-1), { type: 'recording-ready', epoch: 10, duration: 4 / SAMPLE_RATE });
assert.deepEqual([...processor.recording], [0, Math.fround(.2), 0, Math.fround(-.4)]);
processor.handle({ type: 'view', epoch: 10, requestId: 17, start: 0, end: 1, columns: 2 });
assert.equal(messages.at(-1).requestId, 17); assert.equal(messages.at(-1).type, 'view');
processor.handle({ type: 'clear-recording', epoch: 11 });
const beforeStale = messages.length;
processor.handle({ type: 'recording', epoch: 10, samples: tone(1000) });
processor.handle({ type: 'view', epoch: 10, requestId: 18, start: 0, end: 1, columns: 2 });
assert.equal(messages.length, beforeStale); assert.equal(processor.recording.length, 0);
assert.throws(() => processor.handle({ type: 'recording', epoch: 12, samples: new Float32Array(MAX_RECORDING_SAMPLES + 1) }), /4800000/);
checks.push('worker acknowledgements, channel isolation, epochs, recording release and size bounds');

console.log(JSON.stringify({ test: 'PASS', checks, fft: FFT_SIZE, hop: HOP_SIZE, frequency_rows: FREQUENCY_ROWS, sample_rate: SAMPLE_RATE, matrix_layout: 'rows-major, highest frequency at row 0', maximum_overview: { samples: MAX_RECORDING_SAMPLES, columns: overviewWidth, fft_windows: overviewDSP.count, milliseconds: overviewMilliseconds } }, null, 2));
