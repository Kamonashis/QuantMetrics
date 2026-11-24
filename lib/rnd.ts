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
//     → clip negative curvature (reported, never hidden)
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
  /** ascending fine-grid strikes, including every observed knot */
  strikes: number[];
  /** clipped, trapezoid-normalized density (integrates to ≈ 1) */
  density: number[];
  /** cumulative distribution, 0 → 1 */
  cdf: number[];
  /** raw minimum of q(K) before clipping — negative when the smile is not perfectly convex */
  minDensity: number;
  /** integral of the clipped density BEFORE renormalization — 1.00 is a clean smile */
  rawArea: number;
  /** Negative curvature mass / total absolute curvature — not missing wing mass. */
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
 * q(K) = e^{rT} · C″(K) on a fine grid over the observed strike range,
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

  const strikes = [...new Set([...xs, ...Array.from({ length: npts - 2 }, (_, i) => lo + (i + 1) * h)])].sort((a, b) => a - b);
  const raw = strikes.map((k) => disc * spline.second(k));

  const normalized = normalizeDensity(strikes, raw);
  if (!normalized) return null;

  return {
    strikes,
    density: normalized.density,
    cdf: normalized.cdf,
    minDensity: normalized.minDensity,
    rawArea: normalized.positiveArea,
    clippedFraction: normalized.clippedFraction,
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

export interface DensityGrid {
  strikes: number[];
  density: number[];
  cdf: number[];
}

/** Diagnostics are computed before any repair, using the same quadrature as the CDF. */
export function normalizeDensity(strikes: number[], raw: number[]) {
  if (strikes.length < 2 || strikes.length !== raw.length || strikes.some((k, i) =>
    !Number.isFinite(k) || k <= 0 || (i > 0 && k <= strikes[i - 1])) || raw.some((q) => !Number.isFinite(q))) return null;
  let positiveArea = 0;
  let negativeArea = 0;
  const cdf = [0];
  for (let i = 1; i < strikes.length; i++) {
    const a = raw[i - 1], b = raw[i], h = strikes[i] - strikes[i - 1];
    const signed = (a + b) * h / 2;
    const positive = (a >= 0 && b >= 0) ? signed : (a <= 0 && b <= 0) ? 0 :
      h * Math.max(a, b) ** 2 / (2 * Math.abs(a - b));
    positiveArea += positive;
    negativeArea += positive - signed;
    // Represent the repaired curve as linear between its nonnegative knots.
    cdf.push(cdf[i - 1] + h * (Math.max(0, a) + Math.max(0, b)) / 2);
  }
  const area = cdf[cdf.length - 1];
  if (!Number.isFinite(area) || area <= 0) return null;
  return {
    strikes, density: raw.map((q) => Math.max(0, q) / area), cdf: cdf.map((p) => p / area),
    integrated: positiveArea - negativeArea, positiveArea, clippedFraction: negativeArea / (positiveArea + negativeArea),
    minDensity: Math.min(...raw),
  };
}

function gridValue(grid: { strikes: number[]; density: number[] }, k: number): { index: number; density: number } {
  const { strikes, density } = grid;
  let lo = 0, hi = strikes.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (strikes[mid] <= k) lo = mid; else hi = mid;
  }
  const x = (k - strikes[lo]) / (strikes[hi] - strikes[lo]);
  return { index: lo, density: density[lo] + x * (density[hi] - density[lo]) };
}

export function densityCdf(grid: DensityGrid, k: number): number {
  if (k <= grid.strikes[0]) return 0;
  if (k >= grid.strikes[grid.strikes.length - 1]) return 1;
  const p = gridValue(grid, k);
  return grid.cdf[p.index] + (k - grid.strikes[p.index]) * (p.density + grid.density[p.index]) / 2;
}

/** Expectation under the piecewise-linear density (3-node Gaussian quadrature per cell). */
export function densityMean(grid: DensityGrid, f: (strike: number) => number, breaks: number[] = []): number {
  let value = 0;
  for (let i = 1; i < grid.strikes.length; i++) {
    const left = grid.strikes[i - 1], right = grid.strikes[i];
    const edges = [left, ...breaks.filter((k) => k > left && k < right).sort((a, b) => a - b), right];
    for (let j = 1; j < edges.length; j++) {
      const half = (edges[j] - edges[j - 1]) / 2, mid = edges[j - 1] + half;
      for (const [node, weight] of GAUSS_NODES) {
        const k = mid + half * node;
        const q = grid.density[i - 1] + (grid.density[i] - grid.density[i - 1]) * (k - left) / (right - left);
        value += half * weight * f(k) * q;
      }
    }
  }
  return value;
}

const GAUSS_NODES = [[-Math.sqrt(3 / 5), 5 / 9], [0, 8 / 9], [Math.sqrt(3 / 5), 5 / 9]];

/** Extend a clean, observed-support market estimate with explicitly modelled SVI wings. */
export function extendWithSviTails(market: MarketRnd, fitted: DensityGrid) {
  const lo = market.strikeMin, hi = market.strikeMax;
  if (lo <= fitted.strikes[0] || hi >= fitted.strikes[fitted.strikes.length - 1]) return null;
  const left = densityCdf(fitted, lo), right = 1 - densityCdf(fitted, hi);
  const strikes = [...new Set([...market.strikes, ...fitted.strikes])].sort((a, b) => a - b);
  const taper = (hi - lo) * 0.05;
  const raw = strikes.map((k) => {
    const model = gridValue(fitted, k).density;
    if (k <= lo || k >= hi) return model;
    const w = Math.min(1, (k - lo) / taper, (hi - k) / taper);
    return w * (1 - left - right) * gridValue(market, k).density + (1 - w) * model;
  });
  const grid = normalizeDensity(strikes, raw);
  if (!grid || Math.abs(densityMean(grid, (k) => k) / market.forward - 1) > 0.02) return null;
  return { ...grid, leftTailMass: densityCdf(grid, lo), rightTailMass: 1 - densityCdf(grid, hi) };
}
