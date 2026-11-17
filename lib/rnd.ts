// Breeden–Litzenberger risk-neutral density extracted directly from MARKET
// option prices: q(K) = e^{rT} · ∂²C/∂K².
//
// The raw call-price curve C(K) is interpolated with a natural (C²) cubic
// spline — the discrete second differences of noisy mids are unusable —
// and the second derivative is evaluated analytically on a fine uniform
// grid. Construction follows the classic pipeline:
//
//   market call mids (parity-synthesized below the forward)
//     → natural cubic spline C(K)
//     → q(K) = e^{rT} · C″(K)  on a uniform grid
//     → clip negative tail mass (reported, never hidden)
//     → trapezoid renormalization so the density integrates to 1
//     → CDF by cumulative quadrature
//
// Nothing here is fitted to a smile — this is the density the raw quotes
// imply. The numbers are only as good as the mids: expect width noise in
// the wings and read `clippedFraction` / `rawArea` as the integrity check.

export interface SlicePointLike {
  strike: number;
  mid: number;
  relSpread: number;
  type: 'call' | 'put';
}

export interface CallCurvePoint {
  strike: number;
  /** mid used as C(K) — direct call above F, put-call parity below F */
  call: number;
  /** true when C(K) was synthesized from the put mid via put–call parity */
  parity: boolean;
}

export interface CallCurve {
  points: CallCurvePoint[];
  /** rows dropped for non-positive mid or relSpread above the filter */
  dropped: number;
  nDirect: number;
  nParity: number;
  strikeMin: number;
  strikeMax: number;
}

const MAX_REL_SPREAD = 0.5;
const MIN_POINTS = 8;

/**
 * Assemble the call-price curve from raw slice points.
 * Above the forward: direct call mid (falling back to parity if absent).
 * Below the forward: put–call parity, C(K) = P(K) + (F − K)·e^{−rT}
 * (ITM calls are thin and wide; the put side is the liquid one).
 * Returns null when the usable curve is too thin to differentiate.
 */
export function buildCallCurve(
  points: SlicePointLike[],
  opts: { forward: number; t: number; rate: number },
): CallCurve | null {
  const { forward, t, rate } = opts;
  const discT = Math.exp(-rate * t);
  let dropped = 0;
  const calls = new Map<number, SlicePointLike>();
  const puts = new Map<number, SlicePointLike>();
  for (const p of points) {
    const ok =
      isFinite(p.strike) && p.strike > 0 &&
      isFinite(p.mid) && p.mid > 0 &&
      isFinite(p.relSpread) && p.relSpread <= MAX_REL_SPREAD;
    if (!ok) { dropped++; continue; }
    const m = p.type === 'call' ? calls : puts;
    const cur = m.get(p.strike);
    if (!cur || p.relSpread < cur.relSpread) m.set(p.strike, p);
  }

  const strikes = Array.from(new Set([...calls.keys(), ...puts.keys()])).sort((a, b) => a - b);
  const pts: CallCurvePoint[] = [];
  for (const K of strikes) {
    const c = calls.get(K);
    const p = puts.get(K);
    let call: number | null = null;
    let parity = false;
    if (K >= forward) {
      if (c) call = c.mid;
      else if (p) { call = p.mid + discT * (forward - K); parity = true; }
    } else {
      if (p) { call = p.mid + discT * (forward - K); parity = true; }
      else if (c) call = c.mid;
    }
    if (call === null || !isFinite(call) || call < 0) continue;
    pts.push({ strike: K, call, parity });
  }

  const below = pts.reduce((s, p) => s + (p.strike < forward ? 1 : 0), 0);
  const above = pts.length - below;
  if (pts.length < MIN_POINTS || below < 2 || above < 2) return null;
  return {
    points: pts,
    dropped,
    nDirect: pts.filter((p) => !p.parity).length,
    nParity: pts.filter((p) => p.parity).length,
    strikeMin: pts[0].strike,
    strikeMax: pts[pts.length - 1].strike,
  };
}

/**
 * Natural cubic spline (C²) through (xs, ys) — the scipy CubicSpline
 * counterpart. The second derivative is a piecewise-linear function, which
 * is what Breeden–Litzenberger consumes.
 */
export function naturalCubicSpline(xs: number[], ys: number[]): {
  value: (x: number) => number;
  second: (x: number) => number;
} {
  const n = xs.length;
  if (n < 3) throw new Error('naturalCubicSpline needs at least 3 points');
  const h: number[] = [];
  for (let i = 0; i < n - 1; i++) h.push(xs[i + 1] - xs[i]);

  // Tridiagonal system for the second derivatives M_i (natural BC: M_0 = M_{n-1} = 0)
  const a = new Array<number>(n).fill(0);
  const b = new Array<number>(n).fill(1);
  const c = new Array<number>(n).fill(0);
  const d = new Array<number>(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    a[i] = h[i - 1];
    b[i] = 2 * (h[i - 1] + h[i]);
    c[i] = h[i];
    d[i] =
      6 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
  }
  // Thomas algorithm
  for (let i = 1; i < n; i++) {
    const w = a[i] / b[i - 1];
    b[i] -= w * c[i - 1];
    d[i] -= w * d[i - 1];
  }
  const M = new Array<number>(n).fill(0);
  M[n - 1] = d[n - 1] / b[n - 1];
  for (let i = n - 2; i >= 0; i--) M[i] = (d[i] - c[i] * M[i + 1]) / b[i];

  const segment = (x: number): number => {
    // xs is ascending; clamp to range
    if (x <= xs[0]) return 0;
    if (x >= xs[n - 1]) return n - 2;
    let lo = 0, hi = n - 2;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (xs[mid] <= x) lo = mid; else hi = mid - 1;
    }
    return lo;
  };

  const second = (x: number): number => {
    const i = segment(x);
    const hh = h[i];
    return (M[i] * (xs[i + 1] - x) + M[i + 1] * (x - xs[i])) / hh;
  };

  const value = (x: number): number => {
    const i = segment(x);
    const hh = h[i];
    const aK = (xs[i + 1] - x) / hh;
    const bK = (x - xs[i]) / hh;
    return (
      aK * ys[i] +
      bK * ys[i + 1] +
      ((aK * aK * aK - aK) * M[i] + (bK * bK * bK - bK) * M[i + 1]) * ((hh * hh) / 6)
    );
  };

  return { value, second };
}

export interface MarketRnd {
  /** ascending fine-grid strikes */
  strikes: number[];
  /** clipped, trapezoid-normalized density (integrates to ≈ 1) */
  density: number[];
  /** cumulative distribution, 0 → 1 */
  cdf: number[];
  /** raw minimum of q(K) before clipping — negative when the smile is not perfectly convex */
  minDensity: number;
  /** integral of the clipped density BEFORE renormalization — 1.00 is a clean smile */
  rawArea: number;
  /** |negative tail mass| / total |q| — the honest noise meter of the second derivative */
  clippedFraction: number;
  nUsed: number;
  nDirect: number;
  nParity: number;
  dropped: number;
  strikeMin: number;
  strikeMax: number;
  t: number;
  rate: number;
  forward: number;
  spot: number;
}

/**
 * q(K) = e^{rT} · C″(K) on a uniform grid over the observed strike range,
 * clipped at 0 (diagnostics kept) and renormalized by trapezoid quadrature.
 */
export function impliedDensityFromCalls(
  curve: CallCurve,
  opts: { spot: number; forward: number; t: number; rate: number },
  gridPoints = 161,
): MarketRnd | null {
  const { t, rate, forward, spot } = opts;
  const pts = curve.points;
  const xs = pts.map((p) => p.strike);
  const ys = pts.map((p) => p.call);
  const spline = naturalCubicSpline(xs, ys);

  const lo = xs[0];
  const hi = xs[xs.length - 1];
  const npts = Math.max(41, gridPoints);
  const h = (hi - lo) / (npts - 1);
  const disc = Math.exp(rate * t);

  const strikes: number[] = new Array(npts);
  const raw: number[] = new Array(npts);
  let minDensity = Infinity;
  for (let i = 0; i < npts; i++) {
    const K = lo + i * h;
    strikes[i] = K;
    const q = disc * spline.second(K);
    raw[i] = q;
    if (q < minDensity) minDensity = q;
  }

  let posArea = 0;
  let negArea = 0;
  for (let i = 0; i < npts - 1; i++) {
    const a = raw[i], b = raw[i + 1];
    posArea += ((Math.max(a, 0) + Math.max(b, 0)) / 2) * h;
    negArea += ((Math.min(a, 0) + Math.min(b, 0)) / 2) * h; // negative or 0
  }
  if (!isFinite(posArea) || posArea <= 0) return null;

  const density = raw.map((v) => Math.max(0, v) / posArea);
  const cdf = new Array<number>(npts).fill(0);
  let acc = 0;
  for (let i = 1; i < npts; i++) {
    acc += ((density[i - 1] + density[i]) / 2) * h;
    cdf[i] = acc;
  }
  const total = cdf[npts - 1];
  const cdfNorm = total > 0 ? cdf.map((v) => v / total) : cdf;
  cdfNorm[0] = 0;
  cdfNorm[npts - 1] = 1;

  const clippedFraction =
    Math.abs(negArea) + posArea > 0
      ? Math.abs(negArea) / (posArea + Math.abs(negArea))
      : 0;

  return {
    strikes,
    density,
    cdf: cdfNorm,
    minDensity,
    rawArea: posArea,
    clippedFraction,
    nUsed: pts.length,
    nDirect: curve.nDirect,
    nParity: curve.nParity,
    dropped: curve.dropped,
    strikeMin: lo,
    strikeMax: hi,
    t,
    rate,
    forward,
    spot,
  };
}

/**
 * Price an arbitrary payoff on S_T by integrating against the density:
 *   price = e^{−rT} ∫ payoff(K) · q(K) dK   (trapezoid, midpoint payoff)
 */
export function pricePayoff(
  rnd: { strikes: number[]; density: number[] },
  payoff: (K: number) => number,
  t: number,
  rate: number,
): number {
  const ks = rnd.strikes;
  let acc = 0;
  for (let i = 1; i < ks.length; i++) {
    const h = ks[i] - ks[i - 1];
    const mid = (ks[i - 1] + ks[i]) / 2;
    acc += ((rnd.density[i - 1] + rnd.density[i]) / 2) * payoff(mid) * h;
  }
  return Math.exp(-rate * t) * acc;
}
