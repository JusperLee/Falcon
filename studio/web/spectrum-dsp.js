/* Local 16 kHz audio analysis. Matrices are rows-major, highest frequency first.
 * A full-scale, bin-centred sine is approximately 0 dBFS after Hann gain
 * correction. Each display row takes the peak of its linear-frequency bins.
 * No audio normalization is applied: input/output levels remain comparable. */
export const SAMPLE_RATE = 16000;
export const FFT_SIZE = 1024;
export const HOP_SIZE = 640;
export const FREQUENCY_ROWS = 128;
export const DB_FLOOR = -90;
export const MAX_VIEW_COLUMNS = 1600;
export const MAX_RECORDING_SAMPLES = 4800000;

const finiteSample = value => Number.isFinite(value) ? value : 0;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

export class SpectrumDSP {
  constructor() {
    this.real = new Float64Array(FFT_SIZE);
    this.imag = new Float64Array(FFT_SIZE);
    this.window = new Float64Array(FFT_SIZE);
    this.reverse = new Uint16Array(FFT_SIZE);
    this.cos = new Float64Array(FFT_SIZE / 2);
    this.sin = new Float64Array(FFT_SIZE / 2);
    this.peaks = new Float64Array(FREQUENCY_ROWS);
    this.result = new Uint8Array(FREQUENCY_ROWS);
    this.windowSum = 0;
    for (let i = 0; i < FFT_SIZE; i++) {
      this.window[i] = .5 - .5 * Math.cos(2 * Math.PI * i / (FFT_SIZE - 1));
      this.windowSum += this.window[i];
      let reversed = 0, n = i;
      for (let bit = 0; bit < 10; bit++) { reversed = (reversed << 1) | (n & 1); n >>= 1; }
      this.reverse[i] = reversed;
    }
    for (let i = 0; i < FFT_SIZE / 2; i++) {
      this.cos[i] = Math.cos(-2 * Math.PI * i / FFT_SIZE);
      this.sin[i] = Math.sin(-2 * Math.PI * i / FFT_SIZE);
    }
  }

  // The returned 128-byte column is scratch storage, valid until the next call.
  column(samples, offset = 0) {
    for (let i = 0; i < FFT_SIZE; i++) {
      const index = offset + i;
      this.real[this.reverse[i]] = (index >= 0 && index < samples.length ? finiteSample(samples[index]) : 0) * this.window[i];
    }
    return this.transform();
  }

  circularColumn(ring, oldest) {
    for (let i = 0; i < FFT_SIZE; i++) this.real[this.reverse[i]] = ring[(oldest + i) % FFT_SIZE] * this.window[i];
    return this.transform();
  }

  transform() {
    this.imag.fill(0);
    for (let size = 2; size <= FFT_SIZE; size <<= 1) {
      const half = size >> 1, step = FFT_SIZE / size;
      for (let base = 0; base < FFT_SIZE; base += size) {
        for (let j = 0; j < half; j++) {
          const a = base + j, b = a + half, twiddle = j * step;
          const br = this.real[b] * this.cos[twiddle] - this.imag[b] * this.sin[twiddle];
          const bi = this.real[b] * this.sin[twiddle] + this.imag[b] * this.cos[twiddle];
          this.real[b] = this.real[a] - br;
          this.imag[b] = this.imag[a] - bi;
          this.real[a] += br;
          this.imag[a] += bi;
        }
      }
    }
    this.peaks.fill(0);
    for (let bin = 0; bin <= FFT_SIZE / 2; bin++) {
      const magnitude = Math.hypot(this.real[bin], this.imag[bin]) * (bin === 0 || bin === FFT_SIZE / 2 ? 1 : 2) / this.windowSum;
      const row = FREQUENCY_ROWS - 1 - Math.min(FREQUENCY_ROWS - 1, Math.floor(bin * FREQUENCY_ROWS / (FFT_SIZE / 2)));
      this.peaks[row] = Math.max(this.peaks[row], magnitude);
    }
    for (let row = 0; row < FREQUENCY_ROWS; row++) {
      const db = this.peaks[row] > 0 ? 20 * Math.log10(this.peaks[row]) : DB_FLOOR;
      this.result[row] = Math.round(clamp((db - DB_FLOOR) / -DB_FLOOR, 0, 1) * 255);
    }
    return this.result;
  }
}

export class LiveSpectrum {
  constructor(dsp = new SpectrumDSP()) {
    this.dsp = dsp;
    this.ring = new Float32Array(FFT_SIZE);
    this.reset();
  }
  reset() {
    this.ring.fill(0);
    this.position = 0;
    this.total = 0;
    this.nextFrame = FFT_SIZE;
  }
  push(samples) {
    const width = Math.max(0, Math.floor((this.total + samples.length - this.nextFrame) / HOP_SIZE) + 1);
    const columns = new Uint8Array(width * FREQUENCY_ROWS);
    let columnIndex = 0;
    for (let i = 0; i < samples.length; i++) {
      this.ring[this.position] = finiteSample(samples[i]);
      this.position = (this.position + 1) % FFT_SIZE;
      ++this.total;
      if (this.total === this.nextFrame) {
        const values = this.dsp.circularColumn(this.ring, this.position);
        for (let row = 0; row < FREQUENCY_ROWS; row++) columns[row * width + columnIndex] = values[row];
        ++columnIndex;
        this.nextFrame += HOP_SIZE;
      }
    }
    return { columns, width, height: FREQUENCY_ROWS };
  }
}

export function recordingView(samples, start, end, requestedColumns, dsp = new SpectrumDSP()) {
  const duration = samples.length / SAMPLE_RATE;
  start = clamp(Number.isFinite(start) ? start : 0, 0, duration);
  end = clamp(Number.isFinite(end) ? end : duration, start, duration);
  const width = clamp(Number.isFinite(requestedColumns) ? Math.floor(requestedColumns) : 1, 1, MAX_VIEW_COLUMNS);
  const pixels = new Uint8Array(FREQUENCY_ROWS * width);
  const waveMin = new Float32Array(width), waveMax = new Float32Array(width);
  if (!samples.length || end <= start) return { start, end, width, height: FREQUENCY_ROWS, pixels, waveMin, waveMax };
  const beginSample = start * SAMPLE_RATE, samplesPerColumn = (end - start) * SAMPLE_RATE / width;
  // A long overview column can span far more than one FFT window. Partition
  // its interval at <= 50% window hops and aggregate peaks, so short sounds
  // outside the central window cannot fall into unanalysed gaps. Narrow views
  // retain a single centred window. Total FFTs <= viewSamples / 512 + width.
  const windowsPerColumn = Math.max(1, Math.ceil(samplesPerColumn / (FFT_SIZE / 2)));
  const windowStep = samplesPerColumn / windowsPerColumn;
  for (let column = 0; column < width; column++) {
    const columnStart = beginSample + column * samplesPerColumn;
    for (let window = 0; window < windowsPerColumn; window++) {
      const centre = columnStart + (window + .5) * windowStep;
      const spectrum = dsp.column(samples, Math.round(centre - FFT_SIZE / 2));
      for (let row = 0; row < FREQUENCY_ROWS; row++) {
        const index = row * width + column;
        pixels[index] = Math.max(pixels[index], spectrum[row]);
      }
    }
    // floor boundaries partition the source without dropping narrow impulses.
    // When zoomed past individual samples, display the nearest available sample.
    const from = Math.min(samples.length - 1, Math.floor(beginSample + column * samplesPerColumn));
    const to = Math.min(samples.length, Math.max(from + 1, Math.floor(beginSample + (column + 1) * samplesPerColumn)));
    let minimum = Infinity, maximum = -Infinity;
    for (let i = from; i < to; i++) {
      const value = finiteSample(samples[i]);
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    waveMin[column] = minimum;
    waveMax[column] = maximum;
  }
  return { start, end, width, height: FREQUENCY_ROWS, pixels, waveMin, waveMax };
}

// Stateless transport around bounded, reusable DSP state; also exercised without
// a browser in tests. Old epochs are ignored, including late recording uploads.
export class SpectrumProcessor {
  constructor(send) {
    this.send = send;
    this.dsp = new SpectrumDSP();
    this.live = { input: new LiveSpectrum(this.dsp), output: new LiveSpectrum(this.dsp) };
    this.liveEpoch = { input: -1, output: -1 };
    this.recordingEpoch = -1;
    this.recording = new Float32Array(0);
  }
  handle(message) {
    const epoch = message?.epoch;
    if (!Number.isSafeInteger(epoch) || epoch < 0) return;
    if (message.type === 'reset-live') {
      if (message.channel !== undefined && !['input', 'output'].includes(message.channel)) return;
      const channels = message.channel ? [message.channel] : ['input', 'output'];
      for (const channel of channels) if (epoch >= this.liveEpoch[channel]) {
        this.liveEpoch[channel] = epoch;
        this.live[channel].reset();
      }
      return;
    }
    if (message.type === 'live') {
      const channel = message.channel;
      if (channel !== 'input' && channel !== 'output' || epoch < this.liveEpoch[channel]) return;
      if (!(message.samples instanceof Float32Array) || message.samples.length > MAX_RECORDING_SAMPLES) throw new Error('Invalid live audio buffer');
      if (epoch !== this.liveEpoch[channel]) { this.liveEpoch[channel] = epoch; this.live[channel].reset(); }
      const result = this.live[channel].push(message.samples);
      this.send({ type: 'live', channel, epoch, ...result }, [result.columns.buffer]);
      return;
    }
    if (message.type === 'clear-recording') {
      if (epoch < this.recordingEpoch) return;
      this.recordingEpoch = epoch;
      this.recording = new Float32Array(0);
      return;
    }
    if (message.type === 'recording') {
      if (epoch < this.recordingEpoch) return;
      if (!(message.samples instanceof Float32Array) || message.samples.length > MAX_RECORDING_SAMPLES) throw new Error('Recording must contain at most 4800000 float32 samples');
      this.recordingEpoch = epoch;
      this.recording = message.samples;
      for (let i = 0; i < this.recording.length; i++) if (!Number.isFinite(this.recording[i])) this.recording[i] = 0;
      this.send({ type: 'recording-ready', epoch, duration: this.recording.length / SAMPLE_RATE });
      return;
    }
    if (message.type === 'view' && epoch === this.recordingEpoch) {
      const result = recordingView(this.recording, message.start, message.end, message.columns, this.dsp);
      this.send({ type: 'view', epoch, requestId: message.requestId, ...result }, [result.pixels.buffer, result.waveMin.buffer, result.waveMax.buffer]);
    }
  }
}
