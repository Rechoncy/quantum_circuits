/**
 * Shared physical drive & pulse envelope calculations (mirrors backend `project_engine.py` & `qutip_backend.py`).
 * Relation:
 *   theta = 2 * pi * rabi_mhz * 1e-3 * kappa * duration_ns
 */

export type PulseShapeType = 'cosine' | 'gaussian' | 'rectangular' | 'trapezoid';
export type DriveModeType = 'fixed_duration' | 'fixed_rabi';

const VIRTUAL_1Q_GATES = new Set([
  'RZ',
  'Z',
  'S',
  'SDG',
  'T',
  'TDG',
  'P',
  'PHASE',
  'I',
  'ID',
  'IDLE',
  'DELAY',
  'BARRIER',
]);

export function envelopeAreaFactor(
  pulseShape: string | undefined,
  durationNs: number,
  riseNs = 0,
  fallNs = 0
): number {
  const shape = (pulseShape || 'cosine').toLowerCase();
  if (shape === 'rectangular' || shape === 'trapezoid') {
    if (durationNs <= 0) return 1.0;
    let r = Math.max(0, riseNs);
    let f = Math.max(0, fallNs);
    if (r + f > durationNs) {
      const s = durationNs / (r + f);
      r *= s;
      f *= s;
    }
    return Math.max(1e-6, (durationNs - 0.5 * (r + f)) / durationNs);
  }
  if (shape === 'gaussian') {
    // Truncated zero-edge Gaussian with sigma = T / 4.5: analytical area / peak ≈ 0.5205
    return 0.5204998;
  }
  return 0.5; // Hann raised-cosine
}

export function rabiPulseDurationNs(
  thetaRad: number,
  rabiMhz: number,
  pulseShape: string | undefined,
  riseNs = 0,
  fallNs = 0
): number {
  const th = Math.abs(thetaRad);
  if (th < 1e-9 || rabiMhz <= 0) return 0;
  const w = 2 * Math.PI * rabiMhz * 1e-3; // rad / ns at peak
  const shape = (pulseShape || 'cosine').toLowerCase();
  if (shape === 'rectangular' || shape === 'trapezoid') {
    return th / w + 0.5 * (Math.max(0, riseNs) + Math.max(0, fallNs));
  }
  return th / (w * envelopeAreaFactor(shape, 1.0));
}

export function impliedRabiMhz(
  thetaRad: number,
  durationNs: number,
  pulseShape: string | undefined,
  riseNs = 0,
  fallNs = 0
): number {
  if (durationNs <= 0) return 0;
  const kappa = envelopeAreaFactor(pulseShape, durationNs, riseNs, fallNs);
  return Math.abs(thetaRad) / (2 * Math.PI * 1e-3 * kappa * durationNs);
}

/**
 * Peak-normalized envelope value in [0, 1] at time `tNs` within `[0, durationNs]`.
 */
export function samplePeakNormalizedEnvelope(
  tNs: number,
  durationNs: number,
  pulseShape: string | undefined,
  riseNs = 0,
  fallNs = 0
): number {
  if (durationNs <= 0 || tNs < 0 || tNs > durationNs) return 0;
  const shape = (pulseShape || 'cosine').toLowerCase();
  if (shape === 'rectangular' || shape === 'trapezoid') {
    let r = Math.max(0, riseNs);
    let f = Math.max(0, fallNs);
    if (r + f > durationNs) {
      const s = durationNs / (r + f);
      r *= s;
      f *= s;
    }
    if (r === 0 && f === 0) {
      return 1.0;
    }
    if (r > 0 && tNs < r) {
      return shape === 'trapezoid' ? tNs / r : 0.5 * (1 - Math.cos((Math.PI * tNs) / r));
    }
    if (f > 0 && tNs > durationNs - f) {
      const rem = durationNs - tNs;
      return shape === 'trapezoid' ? rem / f : 0.5 * (1 - Math.cos((Math.PI * rem) / f));
    }
    return 1.0;
  }
  const u = tNs / durationNs;
  if (shape === 'gaussian') {
    const sigma = 1 / 4.5;
    const edge = Math.exp(-0.5 * Math.pow(0.5 / sigma, 2));
    return Math.max(0, (Math.exp(-0.5 * Math.pow((u - 0.5) / sigma, 2)) - edge) / (1 - edge));
  }
  return 0.5 * (1 - Math.cos(2 * Math.PI * u));
}

export function singleQubitRotationAngle(
  gateName: string,
  params: (number | string)[],
  variables: Record<string, number>
): number {
  const u = gateName.toUpperCase();
  if (VIRTUAL_1Q_GATES.has(u)) return 0;
  if (u === 'SX' || u === 'SXDG' || u === 'H') return 0.5 * Math.PI;
  if (u === 'X' || u === 'Y') return Math.PI;
  const resolve0 = (): number => {
    const raw = params[0];
    if (typeof raw === 'number') return raw;
    if (typeof raw === 'string') return variables[raw] ?? 0.5 * Math.PI;
    return 0.5 * Math.PI;
  };
  if (u === 'RX' || u === 'RY' || u === 'RXY' || u === 'U3') {
    return Math.abs(resolve0());
  }
  return 0.5 * Math.PI;
}

export function isVirtual1QGate(gateName: string): boolean {
  return VIRTUAL_1Q_GATES.has(gateName.toUpperCase());
}

/**
 * Compute DRAG quadrature waveform Ω_Q(t)/2π (MHz) = -β * (dΩ_I/dt) / (2π * α_GHz).
 */
export function sampleDragQuadratureMhz(
  tNs: number,
  durationNs: number,
  peakRabiMhz: number,
  anharmonicityMhz: number,
  dragBeta: number,
  pulseShape: string | undefined,
  riseNs = 0,
  fallNs = 0
): number {
  if (
    durationNs <= 0.1 ||
    tNs < 0 ||
    tNs > durationNs ||
    Math.abs(anharmonicityMhz) < 1e-2 ||
    Math.abs(dragBeta) < 1e-6
  ) {
    return 0;
  }
  const eps = Math.max(0.02, durationNs * 1e-3);
  const t0 = Math.max(0, tNs - eps);
  const t1 = Math.min(durationNs, tNs + eps);
  const y0 = peakRabiMhz * samplePeakNormalizedEnvelope(t0, durationNs, pulseShape, riseNs, fallNs);
  const y1 = peakRabiMhz * samplePeakNormalizedEnvelope(t1, durationNs, pulseShape, riseNs, fallNs);
  const dOmegaDt = (y1 - y0) / Math.max(1e-6, t1 - t0); // MHz / ns
  const wAlpha = 2 * Math.PI * anharmonicityMhz * 1e-3; // rad / ns
  return (-dragBeta * dOmegaDt) / wAlpha;
}

/**
 * First-order spectral Fourier estimate of |1> -> |2> leakage probability P_leak in [0, 0.5]
 * for a driven transmon pulse of rotation angle `thetaRad` and duration `durationNs`.
 */
export function estimateTransmonLeakage(
  thetaRad: number,
  durationNs: number,
  anharmonicityMhz: number,
  dragBeta: number,
  pulseShape: string | undefined,
  riseNs = 0,
  fallNs = 0
): number {
  if (durationNs <= 0.1 || Math.abs(anharmonicityMhz) < 1.0) return 0;
  const nSteps = 64;
  const dt = durationNs / nSteps;
  const peakMhz = impliedRabiMhz(thetaRad, durationNs, pulseShape, riseNs, fallNs);
  const wAlpha = 2 * Math.PI * anharmonicityMhz * 1e-3;
  let re = 0;
  let im = 0;
  for (let i = 0; i < nSteps; i++) {
    const t = (i + 0.5) * dt;
    const omI = 2 * Math.PI * 1e-3 * peakMhz * samplePeakNormalizedEnvelope(t, durationNs, pulseShape, riseNs, fallNs);
    const omQ = 2 * Math.PI * 1e-3 * sampleDragQuadratureMhz(t, durationNs, peakMhz, anharmonicityMhz, dragBeta, pulseShape, riseNs, fallNs);
    // |1> -> |2> coupling carries factor sqrt(2) * (1/2) = 1 / sqrt(2):
    const ph = wAlpha * t;
    const c = Math.cos(ph);
    const s = Math.sin(ph);
    // (omI - i omQ) * (cos(w_alpha t) + i sin(w_alpha t))
    re += (omI * c + omQ * s) * dt;
    im += (omI * s - omQ * c) * dt;
  }
  const ampSq = 0.5 * (re * re + im * im);
  return Math.min(0.5, Math.max(0, ampSq));
}
