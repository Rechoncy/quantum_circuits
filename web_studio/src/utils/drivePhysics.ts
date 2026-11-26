/**
 * Shared physical drive & pulse envelope calculations (mirrors backend `project_engine.py` & `qutip_backend.py`).
 * Relation:
 *   theta = 2 * pi * rabi_mhz * 1e-3 * kappa * duration_ns
 */

export type PulseShapeType = 'cosine' | 'gaussian' | 'rectangular' | 'trapezoid';
export type DriveModeType = 'fixed_duration' | 'fixed_rabi';

const VIRTUAL_1Q_GATES = new Set(['RZ', 'Z', 'S', 'SDG', 'T', 'TDG', 'P', 'PHASE', 'I', 'ID']);

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
