import { SpectrumProcessor } from './spectrum-dsp.js';

const processor = new SpectrumProcessor((message, transfer = []) => self.postMessage(message, transfer));
self.onmessage = ({ data }) => {
  try { processor.handle(data); }
  catch (error) {
    self.postMessage({ type: 'error', epoch: data?.epoch, requestId: data?.requestId, message: error?.message || String(error) });
  }
};
