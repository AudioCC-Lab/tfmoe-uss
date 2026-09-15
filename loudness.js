/** Constant-gain RMS matching for separated sources; never changes model tensors. */
export const NORMALIZATION = Object.freeze({targetRms:0.1,peakLimit:0.95,maxGain:100,silenceRms:1e-6});
export function normalizeSource(samples) {
  let energy=0,peak=0;
  for(const value of samples){if(!Number.isFinite(value))throw new Error('The output contains invalid samples.');energy+=value*value;peak=Math.max(peak,Math.abs(value));}
  const inputRms=samples.length?Math.sqrt(energy/samples.length):0;
  // Leave numerical silence quiet rather than magnifying residual noise.
  const silent=inputRms<=NORMALIZATION.silenceRms;
  const gain=silent?1:Math.min(NORMALIZATION.targetRms/inputRms,NORMALIZATION.peakLimit/peak,NORMALIZATION.maxGain);
  const normalized=Float32Array.from(samples,value=>value*gain);
  return {samples:normalized,normalization:{gain,gainDb:20*Math.log10(gain),inputRms,outputRms:inputRms*gain,inputPeak:peak,outputPeak:peak*gain,silent,peakLimited:!silent&&gain===NORMALIZATION.peakLimit/peak,gainLimited:!silent&&gain===NORMALIZATION.maxGain}};
}
