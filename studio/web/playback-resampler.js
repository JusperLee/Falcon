/* Stateful streaming polyphase windowed-sinc upsampler, 16 kHz -> an arbitrary
 * AudioContext sample rate (44100/48000/96000 Hz, including non-integer ratios
 * such as 44100/16000 = 2.75625). Mirrors capture-worklet.js's downsampler: a
 * Hamming-windowed sinc low-pass is pre-tabulated into PHASES fractional
 * sub-sample positions, and FIR history plus fractional phase are carried
 * across consecutive chunks so a 2 s model window split into 0.5 s playback
 * chunks resamples identically to one continuous buffer. Chrome's own
 * AudioBufferSourceNode resampler (see audio_buffer_source_handler.cc
 * RenderFromBuffer/ProcessInterpolatedPath) does plain 2-point linear
 * interpolation with no low-pass filtering when buffer.sampleRate differs
 * from context.sampleRate; that leaves the full 16 kHz-sampled spectrum,
 * including its mirror image above 8 kHz, almost unattenuated after
 * upsampling. This module performs the upsampling with an explicit low-pass
 * (cutoff just above the capture chain's ~7.2 kHz passband edge) before Chrome
 * ever sees the buffer, so AudioBuffer is created at context.sampleRate and
 * RenderFromBuffer's playback rate stays 1 (ProcessFastPath, no resampling,
 * no further filtering). */
const TAPS = 63;
const PHASES = 2048;
const CUTOFF_HZ = 7400;

function buildPhases(cutoffRatio) {
  const groupDelay = (TAPS - 1) / 2;
  const phases = new Array(PHASES);
  for (let phase = 0; phase < PHASES; phase++) {
    const coefficients = new Float32Array(TAPS);
    let sum = 0;
    for (let k = 0; k < TAPS; k++) {
      const x = k - groupDelay - phase / PHASES;
      const sinc = Math.abs(x) < 1e-8 ? 2 * cutoffRatio : Math.sin(2 * Math.PI * cutoffRatio * x) / (Math.PI * x);
      coefficients[k] = sinc * (.54 - .46 * Math.cos(2 * Math.PI * k / (TAPS - 1)));
      sum += coefficients[k];
    }
    for (let k = 0; k < TAPS; k++) coefficients[k] /= sum;
    phases[phase] = coefficients;
  }
  return phases;
}

const phaseCache = new Map();
function phasesFor(sourceRate) {
  let table = phaseCache.get(sourceRate);
  if (!table) {
    table = buildPhases(CUTOFF_HZ / sourceRate);
    phaseCache.set(sourceRate, table);
  }
  return table;
}

/* A streaming upsampler bound to one (sourceRate -> targetRate) pair. Call
 * push(chunk) with consecutive Float32Array chunks of `sourceRate`-rate audio;
 * it returns a Float32Array at `targetRate`. Call reset() whenever the input
 * stream is discontinuous (a non-contiguous model result, a restart after
 * stopMonitoring, or a fresh playAt schedule) so stale FIR history and
 * fractional phase from the previous, unrelated audio are discarded instead
 * of leaking a short edit click or pitch artifact into the new stream. */
export class PlaybackUpsampler {
  constructor(sourceRate, targetRate) {
    this.sourceRate = sourceRate;
    this.targetRate = targetRate;
    this.ratio = targetRate / sourceRate; // output samples per input sample
    this.step = sourceRate / targetRate; // input samples advanced per output sample
    this.phases = phasesFor(sourceRate);
    this.groupDelay = (TAPS - 1) / 2;
    this.reset();
  }
  reset() {
    this.history = new Float32Array(TAPS);
    this.historyIndex = 0;
    this.filled = 0; // input samples written since the last reset, capped at TAPS
    // Fractional read position, in input-sample units, relative to the next
    // unwritten history sample. Starts at groupDelay so the first output
    // sample is centred once TAPS/2 input samples have arrived (matches the
    // causal-FIR group-delay convention used by the capture worklet).
    this.nextPosition = this.groupDelay;
  }
  /* Number of target-rate samples push() would emit for `inputLength` more
   * source-rate samples, given the current fractional phase. Exposed for
   * tests that need exact output-length accounting without calling push(). */
  pendingOutputCount(inputLength) {
    const end = this.filled + inputLength;
    let position = this.nextPosition;
    let count = 0;
    while (position <= end - 1 + 1e-9) { count++; position += this.step; }
    return count;
  }
  push(chunk) {
    const outputs = [];
    for (let i = 0; i < chunk.length; i++) {
      this.history[this.historyIndex] = chunk[i];
      this.historyIndex = (this.historyIndex + 1) % TAPS;
      this.filled++;
      const position = this.filled - 1; // index of the sample just written, in input-sample units since reset
      while (this.nextPosition <= position + 1e-9) {
        // Quantize nextPosition to the PHASES sub-sample grid with a single
        // rounding operation, then split into integer base + phase. Flooring
        // nextPosition and separately rounding its fractional part can
        // disagree at exact phase-grid points (e.g. nextPosition landing at
        // 31.999999999999996 instead of 32 after repeated float addition,
        // for the exact ratio 16000/48000 = 1/3): floor gives 31 while the
        // fractional part rounds to phase 0, silently shifting every tap by
        // one history slot. Rounding once avoids that disagreement entirely.
        const quantized = Math.round(this.nextPosition * PHASES);
        const intPos = Math.floor(quantized / PHASES);
        const phase = ((quantized % PHASES) + PHASES) % PHASES;
        const coefficients = this.phases[phase];
        let sample = 0;
        // Tap k samples the input that is (intPos - k) behind `position`,
        // i.e. (position - (intPos - k)) = k - (intPos - position) slots
        // back in history from the one just written.
        const back0 = position - intPos;
        for (let k = 0; k < TAPS; k++) {
          const back = back0 + k;
          if (back < 0 || back >= Math.min(this.filled, TAPS)) continue;
          const index = (this.historyIndex - 1 - back + TAPS * 4) % TAPS;
          sample += this.history[index] * coefficients[TAPS - 1 - k];
        }
        outputs.push(sample);
        this.nextPosition += this.step;
      }
    }
    return Float32Array.from(outputs);
  }
}

export function createPlaybackUpsampler(targetRate, sourceRate = 16000) {
  if (targetRate === sourceRate) return null; // identity: caller should bypass resampling entirely
  return new PlaybackUpsampler(sourceRate, targetRate);
}
