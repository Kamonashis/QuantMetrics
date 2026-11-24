// Correlation lab engine — pure math, no React, no I/O.
//
// Design rules:
//  - Samples are joined on dates. A missing or too-short series is refused,
//    never replaced with simulated data or a default constant.
//  - Covariances are DAILY everywhere in this module; annualisation happens
//    exactly once, at the point a volatility is reported.
//  - Every tail statistic is compared with a normal benchmark at the SAME
//    correlation, so "worse than normal" means what it says.

import type { IntradayBar, PriceBar } from './quant-engine';
import { standardNormalQuantile } from './quant-engine';
import { chiSquareUpperP, incompleteGamma, invertSymmetric } from './factor-regression';

export const TRADING_DAYS = 252;
export const MIN_OBSERVATIONS = 60;
export const MAX_ASSETS = 12;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function mean(v: readonly number[]): number {
  let s = 0;
  for (const x of v) s += x;
  return v.length > 0 ? s / v.length : NaN;
}

function variance(v: readonly number[]): number {
  if (v.length < 2) return NaN;
  const m = mean(v);
  let s = 0;
  for (const x of v) s += (x - m) ** 2;
  return s / (v.length - 1);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function normPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/** Standard normal CDF via the regularized incomplete gamma function. */
export function normCdf(x: number): number {
  if (!Number.isFinite(x)) return x > 0 ? 1 : 0;
  const p = incompleteGamma(0.5, (x * x) / 2);
  return x >= 0 ? 0.5 * (1 + p) : 0.5 * (1 - p);
}

function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function quantile(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return NaN;
  const pos = clamp(q, 0, 1) * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Share of `history` at or below `value`, in [0, 1]. */
export function percentileRank(history: readonly number[], value: number): number {
  const finite = history.filter(Number.isFinite);
  if (finite.length === 0) return NaN;
  let below = 0;
  for (const h of finite) if (h <= value) below++;
  return below / finite.length;
}

// ---------------------------------------------------------------------------
// Sample construction
// ---------------------------------------------------------------------------

export interface SeriesAudit {
  ticker: string;
  barsInWindow: number;
  firstDate: string | null;
  lastDate: string | null;
  basis: 'adjusted' | 'close';
}

export interface CorrelationAudit {
  start: string | null;
  end: string | null;
  series: SeriesAudit[];
  missing: string[];
  commonPriceDates: number;
  returnObs: number;
  /** Sessions of the longest series that were dropped to align calendars. */
  droppedSessions: number;
  returnType: 'simple' | 'log';
}

export interface CorrelationSample {
  tickers: string[];
  dates: string[];
  /** returns[asset][t] */
  returns: number[][];
  returnType: 'simple' | 'log';
  audit: CorrelationAudit;
}

export type CorrelationSampleResult =
  | { ok: true; sample: CorrelationSample }
  | { ok: false; reason: string; audit: CorrelationAudit };

export function buildCorrelationSample(
  bars: Record<string, readonly PriceBar[] | undefined>,
  tickersIn: readonly string[],
  options: { startDate?: string; endDate?: string; returnType: 'simple' | 'log' }
): CorrelationSampleResult {
  const tickers = [...new Set(tickersIn.map((t) => t.trim().toUpperCase()).filter(Boolean))];
  const maps = tickers.map((t) => {
    const inWindow = (bars[t] ?? []).filter(
      (b) => (!options.startDate || b.date >= options.startDate) && (!options.endDate || b.date <= options.endDate)
    );
    const adjusted =
      inWindow.length > 0 &&
      inWindow.every((b) => {
        const adj = (b as PriceBar & { adjClose?: number }).adjClose;
        return typeof adj === 'number' && Number.isFinite(adj) && adj > 0;
      });
    const map = new Map<string, number>();
    for (const b of inWindow) {
      const px = adjusted ? (b as PriceBar & { adjClose?: number }).adjClose! : b.close;
      if (Number.isFinite(px) && px > 0) map.set(b.date, px);
    }
    return { ticker: t, map, basis: (adjusted ? 'adjusted' : 'close') as 'adjusted' | 'close', dates: [...map.keys()].sort() };
  });

  const missing = maps.filter((m) => m.dates.length < 2).map((m) => m.ticker);
  const longest = maps.reduce((a, m) => Math.max(a, m.dates.length), 0);
  const base = maps.reduce<string[] | null>((acc, m) => (acc === null || m.dates.length < acc.length ? m.dates : acc), null) ?? [];
  const common = missing.length > 0 ? [] : base.filter((d) => maps.every((m) => m.map.has(d)));

  const audit: CorrelationAudit = {
    start: common[0] ?? null,
    end: common[common.length - 1] ?? null,
    series: maps.map((m) => ({
      ticker: m.ticker,
      barsInWindow: m.dates.length,
      firstDate: m.dates[0] ?? null,
      lastDate: m.dates[m.dates.length - 1] ?? null,
      basis: m.basis,
    })),
    missing,
    commonPriceDates: common.length,
    returnObs: Math.max(0, common.length - 1),
    droppedSessions: Math.max(0, longest - common.length),
    returnType: options.returnType,
  };

  if (tickers.length < 2) return { ok: false, reason: 'Add at least two assets.', audit };
  if (tickers.length > MAX_ASSETS) return { ok: false, reason: `At most ${MAX_ASSETS} assets are supported.`, audit };
  if (missing.length > 0) return { ok: false, reason: `No price history in this window for ${missing.join(', ')}.`, audit };
  if (audit.returnObs < MIN_OBSERVATIONS) {
    return {
      ok: false,
      reason: `Only ${audit.returnObs} overlapping daily returns — at least ${MIN_OBSERVATIONS} are needed.`,
      audit,
    };
  }

  const returns = maps.map((m) => {
    const out: number[] = new Array(common.length - 1);
    for (let i = 1; i < common.length; i++) {
      const p0 = m.map.get(common[i - 1])!;
      const p1 = m.map.get(common[i])!;
      out[i - 1] = options.returnType === 'log' ? Math.log(p1 / p0) : p1 / p0 - 1;
    }
    return out;
  });

  return { ok: true, sample: { tickers, dates: common.slice(1), returns, returnType: options.returnType, audit } };
}

/** A sub-window of a sample (no re-alignment needed). */
function sliceReturns(returns: number[][], lo: number, hi: number): number[][] {
  return returns.map((r) => r.slice(lo, hi));
}

// ---------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------

export function pearson(x: readonly number[], y: readonly number[]): number {
  const n = Math.min(x.length, y.length);
  if (n < 3) return NaN;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += x[i];
    my += y[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  return sxx > 0 && syy > 0 ? clamp(sxy / Math.sqrt(sxx * syy), -1, 1) : NaN;
}

/** Daily sample covariance (n − 1). */
export function covarianceMatrix(returns: number[][]): number[][] {
  const k = returns.length;
  const n = returns[0]?.length ?? 0;
  const means = returns.map(mean);
  const cov = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let i = 0; i < k; i++) {
    for (let j = i; j < k; j++) {
      let s = 0;
      for (let t = 0; t < n; t++) s += (returns[i][t] - means[i]) * (returns[j][t] - means[j]);
      cov[i][j] = cov[j][i] = s / (n - 1);
    }
  }
  return cov;
}

export function covToCorr(cov: number[][]): number[][] {
  const sd = cov.map((row, i) => Math.sqrt(Math.max(0, row[i])));
  return cov.map((row, i) => row.map((v, j) => (i === j ? 1 : sd[i] > 0 && sd[j] > 0 ? clamp(v / (sd[i] * sd[j]), -1, 1) : 0)));
}

export function correlationMatrix(returns: number[][]): number[][] {
  return covToCorr(covarianceMatrix(returns));
}

function ranks(v: readonly number[]): number[] {
  const idx = v.map((x, i) => ({ x, i })).sort((a, b) => a.x - b.x);
  const out = new Array(v.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1].x === idx[i].x) j++;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) out[idx[k].i] = r;
    i = j + 1;
  }
  return out;
}

export function spearman(x: readonly number[], y: readonly number[]): number {
  return pearson(ranks(x), ranks(y));
}

export function spearmanMatrix(returns: number[][]): number[][] {
  const r = returns.map(ranks);
  return r.map((a, i) => r.map((b, j) => (i === j ? 1 : pearson(a, b))));
}

/** Correlation of each pair after controlling for every other asset. Null when singular. */
export function partialCorrelationMatrix(corr: number[][]): number[][] | null {
  const inv = invertSymmetric(corr);
  if (!inv) return null;
  const n = corr.length;
  for (let i = 0; i < n; i++) if (!(inv[i][i] > 0)) return null;
  return inv.map((row, i) => row.map((v, j) => (i === j ? 1 : clamp(-v / Math.sqrt(inv[i][i] * inv[j][j]), -1, 1))));
}

export interface ShrinkageResult {
  cov: number[][];
  corr: number[][];
  /** Shrinkage intensity in [0, 1]. */
  delta: number;
}

/** Ledoit–Wolf (2004) shrinkage of the daily covariance toward a constant-correlation target. */
export function ledoitWolf(returns: number[][]): ShrinkageResult {
  const N = returns.length;
  const T = returns[0]?.length ?? 0;
  const means = returns.map(mean);
  const X = returns.map((r, i) => r.map((v) => v - means[i]));
  const S = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = i; j < N; j++) {
      let s = 0;
      for (let t = 0; t < T; t++) s += X[i][t] * X[j][t];
      S[i][j] = S[j][i] = s / T;
    }
  }
  const sd = S.map((row, i) => Math.sqrt(row[i]));
  let rSum = 0;
  let rCount = 0;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      rSum += S[i][j] / (sd[i] * sd[j]);
      rCount++;
    }
  }
  const rBar = rCount > 0 ? rSum / rCount : 0;
  const F = S.map((row, i) => row.map((v, j) => (i === j ? v : rBar * sd[i] * sd[j])));

  let gamma = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) gamma += (F[i][j] - S[i][j]) ** 2;

  let delta = 0;
  if (gamma > 1e-30 && N > 2) {
    let pi = 0;
    let rho = 0;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let pij = 0;
        let thetaII = 0;
        let thetaJJ = 0;
        for (let t = 0; t < T; t++) {
          const c = X[i][t] * X[j][t] - S[i][j];
          pij += c * c;
          thetaII += (X[i][t] * X[i][t] - S[i][i]) * c;
          thetaJJ += (X[j][t] * X[j][t] - S[j][j]) * c;
        }
        pij /= T;
        pi += pij;
        if (i === j) rho += pij;
        else rho += (rBar / 2) * ((sd[j] / sd[i]) * (thetaII / T) + (sd[i] / sd[j]) * (thetaJJ / T));
      }
    }
    delta = clamp((pi - rho) / gamma / T, 0, 1);
  }

  const scale = T / (T - 1);
  const cov = S.map((row, i) => row.map((v, j) => scale * (delta * F[i][j] + (1 - delta) * v)));
  return { cov, corr: covToCorr(cov), delta };
}

/** Jacobi eigen-decomposition of a symmetric matrix; eigenvalues descending, vectors[k] is the k-th eigenvector. */
export function jacobiEigen(A: number[][]): { values: number[]; vectors: number[][] } {
  const n = A.length;
  const a = A.map((r) => r.slice());
  const v: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (__, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[j][j] - a[i][i]);
  return { values: order.map((i) => a[i][i]), vectors: order.map((i) => v.map((row) => row[i])) };
}

// ---------------------------------------------------------------------------
// Basket summary & factors
// ---------------------------------------------------------------------------

export interface PairRef {
  a: string;
  b: string;
  corr: number;
}

export interface BasketSummary {
  avgCorr: number;
  strongest: PairRef;
  weakest: PairRef;
  /**
   * Number of uncorrelated, equal-volatility assets that would diversify as
   * well as this basket held at equal risk: N² / Σᵢⱼ ρᵢⱼ = N / (1 + (N−1)·ρ̄).
   */
  effectiveBets: number;
  /** Share of total variance carried by the largest eigenvalue. */
  topFactorShare: number;
}

export function averageCorrelation(corr: number[][]): number {
  let s = 0;
  let c = 0;
  for (let i = 0; i < corr.length; i++) {
    for (let j = i + 1; j < corr.length; j++) {
      s += corr[i][j];
      c++;
    }
  }
  return c > 0 ? s / c : NaN;
}

export function effectiveBets(corr: number[][]): number {
  const n = corr.length;
  let total = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) total += corr[i][j];
  return total > n * n * 1e-3 ? (n * n) / total : n * n;
}

export function summarizeBasket(corr: number[][], tickers: readonly string[]): BasketSummary {
  let strongest: PairRef = { a: tickers[0], b: tickers[1], corr: -Infinity };
  let weakest: PairRef = { a: tickers[0], b: tickers[1], corr: Infinity };
  for (let i = 0; i < corr.length; i++) {
    for (let j = i + 1; j < corr.length; j++) {
      if (corr[i][j] > strongest.corr) strongest = { a: tickers[i], b: tickers[j], corr: corr[i][j] };
      if (corr[i][j] < weakest.corr) weakest = { a: tickers[i], b: tickers[j], corr: corr[i][j] };
    }
  }
  const eig = jacobiEigen(corr);
  return {
    avgCorr: averageCorrelation(corr),
    strongest,
    weakest,
    effectiveBets: effectiveBets(corr),
    topFactorShare: eig.values[0] / corr.length,
  };
}

export interface PrincipalFactor {
  index: number;
  eigenvalue: number;
  share: number;
  /** Above the Marchenko–Pastur noise edge (1 + √(N/T))². */
  aboveNoise: boolean;
  /** Correlation between each asset and the factor (eigenvector × √λ). */
  loadings: number[];
  sameSign: boolean;
}

export function principalFactors(corr: number[][], observations: number, count = 3): { factors: PrincipalFactor[]; noiseEdge: number } {
  const n = corr.length;
  const eig = jacobiEigen(corr);
  const noiseEdge = (1 + Math.sqrt(n / observations)) ** 2;
  const factors = eig.values.slice(0, Math.min(count, n)).map((lambda, k) => {
    let vec = eig.vectors[k];
    if (vec.reduce((a, b) => a + b, 0) < 0) vec = vec.map((x) => -x);
    const loadings = vec.map((x) => x * Math.sqrt(Math.max(0, lambda)));
    return {
      index: k + 1,
      eigenvalue: lambda,
      share: lambda / n,
      aboveNoise: lambda > noiseEdge,
      loadings,
      sameSign: loadings.every((l) => l > 0.05),
    };
  });
  return { factors, noiseEdge };
}

// ---------------------------------------------------------------------------
// Uncertainty for a pair
// ---------------------------------------------------------------------------

/** Fisher-z interval. Assumes independent observations; too narrow for fat-tailed daily returns. */
export function fisherInterval(r: number, n: number, z = 1.959964): [number, number] {
  if (!(n > 3) || !Number.isFinite(r)) return [NaN, NaN];
  const f = Math.atanh(clamp(r, -0.999999, 0.999999));
  const se = 1 / Math.sqrt(n - 3);
  return [Math.tanh(f - z * se), Math.tanh(f + z * se)];
}

/** Moving-block bootstrap percentile interval for a pair correlation (deterministic seed). */
export function blockBootstrapInterval(
  x: readonly number[],
  y: readonly number[],
  options: { reps?: number; seed?: number; blockLength?: number } = {}
): { low: number; high: number; blockLength: number } {
  const n = Math.min(x.length, y.length);
  const L = options.blockLength ?? Math.max(2, Math.ceil(Math.pow(n, 1 / 3)));
  const reps = options.reps ?? 400;
  const rand = seededRandom(options.seed ?? 20240611);
  const blocks = Math.ceil(n / L);
  const stats: number[] = [];
  const bx = new Array(blocks * L);
  const by = new Array(blocks * L);
  for (let r = 0; r < reps; r++) {
    for (let b = 0; b < blocks; b++) {
      const start = Math.floor(rand() * (n - L + 1));
      for (let k = 0; k < L; k++) {
        bx[b * L + k] = x[start + k];
        by[b * L + k] = y[start + k];
      }
    }
    const c = pearson(bx.slice(0, n), by.slice(0, n));
    if (Number.isFinite(c)) stats.push(c);
  }
  stats.sort((a, b) => a - b);
  return { low: quantile(stats, 0.025), high: quantile(stats, 0.975), blockLength: L };
}

// ---------------------------------------------------------------------------
// Return horizon
// ---------------------------------------------------------------------------

export interface HorizonRow {
  days: number;
  observations: number;
  avgCorr: number;
  matrix: number[][];
}

/** Correlation of non-overlapping multi-day returns (anchored at the latest date). */
export function horizonCorrelations(sample: CorrelationSample, horizons: readonly number[] = [1, 5, 21]): HorizonRow[] {
  const n = sample.dates.length;
  const rows: HorizonRow[] = [];
  for (const h of horizons) {
    const count = Math.floor(n / h);
    if (count < 30) continue;
    const offset = n - count * h;
    const agg = sample.returns.map((r) => {
      const out = new Array(count);
      for (let b = 0; b < count; b++) {
        let acc = sample.returnType === 'log' ? 0 : 1;
        for (let k = 0; k < h; k++) {
          const v = r[offset + b * h + k];
          acc = sample.returnType === 'log' ? acc + v : acc * (1 + v);
        }
        out[b] = sample.returnType === 'log' ? acc : acc - 1;
      }
      return out;
    });
    const matrix = correlationMatrix(agg);
    rows.push({ days: h, observations: count, avgCorr: averageCorrelation(matrix), matrix });
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Through time
// ---------------------------------------------------------------------------

export interface RollingBasket {
  window: number;
  dates: string[];
  avgCorr: number[];
  effectiveBets: number[];
  topFactorShare: number[];
}

export function rollingBasket(sample: CorrelationSample, window: number, step = 5): RollingBasket | null {
  const n = sample.dates.length;
  if (n < window + step) return null;
  const out: RollingBasket = { window, dates: [], avgCorr: [], effectiveBets: [], topFactorShare: [] };
  const ends: number[] = [];
  for (let end = n; end >= window; end -= step) ends.push(end);
  ends.reverse();
  for (const end of ends) {
    const c = correlationMatrix(sliceReturns(sample.returns, end - window, end));
    out.dates.push(sample.dates[end - 1]);
    out.avgCorr.push(averageCorrelation(c));
    out.effectiveBets.push(effectiveBets(c));
    out.topFactorShare.push(jacobiEigen(c).values[0] / c.length);
  }
  return out;
}

export interface RollingPair {
  dates: string[];
  corr: number[];
  low: number[];
  high: number[];
}

export function rollingPair(x: readonly number[], y: readonly number[], dates: readonly string[], window: number): RollingPair | null {
  const n = Math.min(x.length, y.length);
  if (n < window) return null;
  const out: RollingPair = { dates: [], corr: [], low: [], high: [] };
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (let t = 0; t < n; t++) {
    sx += x[t];
    sy += y[t];
    sxx += x[t] * x[t];
    syy += y[t] * y[t];
    sxy += x[t] * y[t];
    if (t >= window) {
      const k = t - window;
      sx -= x[k];
      sy -= y[k];
      sxx -= x[k] * x[k];
      syy -= y[k] * y[k];
      sxy -= x[k] * y[k];
    }
    if (t >= window - 1) {
      const vx = sxx - (sx * sx) / window;
      const vy = syy - (sy * sy) / window;
      const c = vx > 0 && vy > 0 ? clamp((sxy - (sx * sy) / window) / Math.sqrt(vx * vy), -1, 1) : NaN;
      const [lo, hi] = fisherInterval(c, window);
      out.dates.push(dates[t]);
      out.corr.push(c);
      out.low.push(lo);
      out.high.push(hi);
    }
  }
  return out;
}

/** RiskMetrics-style exponentially weighted correlation, seeded with the first `warmup` observations. */
export function ewmaPair(x: readonly number[], y: readonly number[], lambda = 0.94, warmup = 20): (number | null)[] {
  const n = Math.min(x.length, y.length);
  const out: (number | null)[] = new Array(n).fill(null);
  if (n <= warmup) return out;
  let vx = 0;
  let vy = 0;
  let cxy = 0;
  for (let t = 0; t < warmup; t++) {
    vx += x[t] * x[t];
    vy += y[t] * y[t];
    cxy += x[t] * y[t];
  }
  vx /= warmup;
  vy /= warmup;
  cxy /= warmup;
  for (let t = warmup; t < n; t++) {
    vx = lambda * vx + (1 - lambda) * x[t] * x[t];
    vy = lambda * vy + (1 - lambda) * y[t] * y[t];
    cxy = lambda * cxy + (1 - lambda) * x[t] * y[t];
    out[t] = vx > 0 && vy > 0 ? clamp(cxy / Math.sqrt(vx * vy), -1, 1) : null;
  }
  return out;
}

export interface MatrixDrift {
  window: number;
  dates: string[];
  /** Mean absolute change per pair between adjacent, non-overlapping windows. */
  drift: number[];
}

export function matrixDrift(sample: CorrelationSample, window: number, step = 5): MatrixDrift | null {
  const n = sample.dates.length;
  if (n < 2 * window + step) return null;
  const out: MatrixDrift = { window, dates: [], drift: [] };
  const ends: number[] = [];
  for (let end = n; end >= 2 * window; end -= step) ends.push(end);
  ends.reverse();
  const pairs = (sample.tickers.length * (sample.tickers.length - 1)) / 2;
  for (const end of ends) {
    const now = correlationMatrix(sliceReturns(sample.returns, end - window, end));
    const prev = correlationMatrix(sliceReturns(sample.returns, end - 2 * window, end - window));
    let s = 0;
    for (let i = 0; i < now.length; i++) for (let j = i + 1; j < now.length; j++) s += Math.abs(now[i][j] - prev[i][j]);
    out.dates.push(sample.dates[end - 1]);
    out.drift.push(s / pairs);
  }
  return out;
}

export interface MatrixTest {
  chi2: number;
  df: number;
  p: number;
}

/** Jennrich (1970) test that two correlation matrices are equal. Assumes normal returns, so it over-rejects with fat tails. */
export function jennrichTest(R1: number[][], n1: number, R2: number[][], n2: number): MatrixTest | null {
  const p = R1.length;
  if (p < 2 || n1 <= p || n2 <= p) return null;
  const Rbar = R1.map((row, i) => row.map((v, j) => (n1 * v + n2 * R2[i][j]) / (n1 + n2)));
  const inv = invertSymmetric(Rbar);
  if (!inv) return null;
  const c = (n1 * n2) / (n1 + n2);
  const Z = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (__, j) => {
      let s = 0;
      for (let k = 0; k < p; k++) s += inv[i][k] * (R1[k][j] - R2[k][j]);
      return Math.sqrt(c) * s;
    })
  );
  let trZ2 = 0;
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) trZ2 += Z[i][j] * Z[j][i];
  const S = Rbar.map((row, i) => row.map((v, j) => (i === j ? 1 : 0) + v * inv[i][j]));
  const Sinv = invertSymmetric(S);
  if (!Sinv) return null;
  let quad = 0;
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) quad += Z[i][i] * Sinv[i][j] * Z[j][j];
  const chi2 = Math.max(0, 0.5 * trZ2 - quad);
  const df = (p * (p - 1)) / 2;
  return { chi2, df, p: chiSquareUpperP(chi2, df) };
}

export interface PairChange {
  a: string;
  b: string;
  prior: number;
  recent: number;
  change: number;
  /** Two-sided p-value for the change (Fisher z, independent windows). */
  p: number;
}

export interface WindowComparison {
  window: number;
  recentStart: string;
  priorStart: string;
  priorEnd: string;
  recentAvg: number;
  priorAvg: number;
  test: MatrixTest | null;
  pairs: PairChange[];
}

/** Compare the most recent window with the one immediately before it. */
export function compareRecentToPrior(sample: CorrelationSample, window: number): WindowComparison | null {
  const n = sample.dates.length;
  if (n < 2 * window || window < 20) return null;
  const recent = correlationMatrix(sliceReturns(sample.returns, n - window, n));
  const prior = correlationMatrix(sliceReturns(sample.returns, n - 2 * window, n - window));
  const se = Math.sqrt(2 / (window - 3));
  const pairs: PairChange[] = [];
  for (let i = 0; i < recent.length; i++) {
    for (let j = i + 1; j < recent.length; j++) {
      const z = (Math.atanh(clamp(recent[i][j], -0.999999, 0.999999)) - Math.atanh(clamp(prior[i][j], -0.999999, 0.999999))) / se;
      pairs.push({
        a: sample.tickers[i],
        b: sample.tickers[j],
        prior: prior[i][j],
        recent: recent[i][j],
        change: recent[i][j] - prior[i][j],
        p: 2 * (1 - normCdf(Math.abs(z))),
      });
    }
  }
  pairs.sort((x, y) => Math.abs(y.change) - Math.abs(x.change));
  return {
    window,
    recentStart: sample.dates[n - window],
    priorStart: sample.dates[n - 2 * window],
    priorEnd: sample.dates[n - window - 1],
    recentAvg: averageCorrelation(recent),
    priorAvg: averageCorrelation(prior),
    test: jennrichTest(recent, window, prior, window),
    pairs,
  };
}

// ---------------------------------------------------------------------------
// DCC(1,1) for a pair — GARCH(1,1) margins, correlation step by likelihood
// ---------------------------------------------------------------------------

interface GarchFit {
  alpha: number;
  beta: number;
  sigma2: number[];
  residuals: number[];
}

/** Variance-targeted GARCH(1,1) by grid likelihood. */
function fitGarchTargeted(r: readonly number[]): GarchFit {
  const n = r.length;
  const m = mean(r);
  const e = r.map((v) => v - m);
  const s2 = e.reduce((a, b) => a + b * b, 0) / n;
  let best = { ll: -Infinity, alpha: 0.05, beta: 0.9 };
  for (let ai = 1; ai <= 12; ai++) {
    const alpha = ai * 0.02;
    for (let bi = 0; bi <= 16; bi++) {
      const beta = 0.66 + bi * 0.02;
      if (alpha + beta >= 0.999) continue;
      const omega = s2 * (1 - alpha - beta);
      let h = s2;
      let ll = 0;
      for (let t = 0; t < n; t++) {
        ll -= 0.5 * (Math.log(h) + (e[t] * e[t]) / h);
        h = omega + alpha * e[t] * e[t] + beta * h;
      }
      if (ll > best.ll) best = { ll, alpha, beta };
    }
  }
  const omega = s2 * (1 - best.alpha - best.beta);
  const sigma2 = new Array(n);
  let h = s2;
  for (let t = 0; t < n; t++) {
    sigma2[t] = h;
    h = omega + best.alpha * e[t] * e[t] + best.beta * h;
  }
  return { alpha: best.alpha, beta: best.beta, sigma2, residuals: e };
}

export interface DccFit {
  a: number;
  b: number;
  /** Long-run correlation of the volatility-standardised returns. */
  longRun: number;
  /** One-step-ahead conditional correlation for each date (uses data through t − 1). */
  path: number[];
  /** Conditional correlation for the next session. */
  next: number;
  halfLifeDays: number | null;
  /** a + b is at the stationarity limit: shifts in correlation barely decay. */
  nearIntegrated: boolean;
  /** Likelihood-ratio test of time-varying vs. constant correlation (χ², 2 df). */
  lrStat: number;
  lrP: number;
  garch: { alpha: number; beta: number }[];
}

export function fitDcc(x: readonly number[], y: readonly number[]): DccFit | null {
  const n = Math.min(x.length, y.length);
  if (n < 250) return null;
  const gx = fitGarchTargeted(x.slice(0, n));
  const gy = fitGarchTargeted(y.slice(0, n));
  let e1 = gx.residuals.map((v, t) => v / Math.sqrt(gx.sigma2[t]));
  let e2 = gy.residuals.map((v, t) => v / Math.sqrt(gy.sigma2[t]));
  const s1 = Math.sqrt(e1.reduce((a, b) => a + b * b, 0) / n);
  const s2 = Math.sqrt(e2.reduce((a, b) => a + b * b, 0) / n);
  e1 = e1.map((v) => v / s1);
  e2 = e2.map((v) => v / s2);
  const qbar = clamp(e1.reduce((a, v, t) => a + v * e2[t], 0) / n, -0.999, 0.999);

  const run = (a: number, b: number, keep: boolean) => {
    let q11 = 1;
    let q22 = 1;
    let q12 = qbar;
    let ll = 0;
    const path = keep ? new Array<number>(n) : null;
    for (let t = 0; t < n; t++) {
      const rho = clamp(q12 / Math.sqrt(q11 * q22), -0.999, 0.999);
      if (path) path[t] = rho;
      const d = 1 - rho * rho;
      ll -= 0.5 * (Math.log(d) + (e1[t] * e1[t] - 2 * rho * e1[t] * e2[t] + e2[t] * e2[t]) / d);
      const c = 1 - a - b;
      q11 = c + a * e1[t] * e1[t] + b * q11;
      q22 = c + a * e2[t] * e2[t] + b * q22;
      q12 = c * qbar + a * e1[t] * e2[t] + b * q12;
    }
    return { ll, path, next: clamp(q12 / Math.sqrt(q11 * q22), -0.999, 0.999) };
  };

  const constant = run(0, 0, false);
  let best = { ll: constant.ll, a: 0, b: 0 };
  const aGrid = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15];
  for (const a of aGrid) {
    for (let bi = 0; bi <= 49; bi++) {
      const b = 0.5 + bi * 0.01;
      if (a + b >= 0.999) continue;
      const { ll } = run(a, b, false);
      if (ll > best.ll) best = { ll, a, b };
    }
  }
  if (best.a > 0) {
    const coarse = { ...best };
    for (let da = -4; da <= 4; da++) {
      for (let db = -6; db <= 6; db++) {
        const a = coarse.a + da * 0.0025;
        const b = coarse.b + db * 0.0025;
        if (a <= 0 || b < 0 || a + b >= 0.999) continue;
        const { ll } = run(a, b, false);
        if (ll > best.ll) best = { ll, a, b };
      }
    }
  }
  const fit = run(best.a, best.b, true);
  const lrStat = Math.max(0, 2 * (best.ll - constant.ll));
  const persistence = best.a + best.b;
  return {
    a: best.a,
    b: best.b,
    nearIntegrated: persistence >= 0.995,
    longRun: qbar,
    path: fit.path!,
    next: fit.next,
    halfLifeDays: persistence > 0 && persistence < 1 ? Math.log(0.5) / Math.log(persistence) : null,
    lrStat,
    lrP: chiSquareUpperP(lrStat, 2),
    garch: [
      { alpha: gx.alpha, beta: gx.beta },
      { alpha: gy.alpha, beta: gy.beta },
    ],
  };
}

// ---------------------------------------------------------------------------
// Stress: exceedance correlation & tail dependence against a normal benchmark
// ---------------------------------------------------------------------------

/** P(X < a, Y < a) for a standard bivariate normal with correlation rho. */
export function bivariateNormalLowerOrthant(a: number, rho: number): number {
  const r = clamp(rho, -0.9999, 0.9999);
  const s = Math.sqrt(1 - r * r);
  const lo = Math.min(a, 0) - 9;
  const steps = 1500;
  const h = (a - lo) / steps;
  let total = 0;
  for (let i = 0; i < steps; i++) {
    const x = lo + (i + 0.5) * h;
    total += normPdf(x) * normCdf((a - r * x) / s);
  }
  return total * h;
}

const exceedanceCache = new Map<string, number>();

/**
 * Correlation of a standard bivariate normal pair, conditional on BOTH being
 * below their q-quantile. This is the benchmark for exceedance correlation:
 * even with a constant correlation, conditioning on a tail lowers it.
 */
export function normalExceedanceCorrelation(q: number, rho: number): number {
  const r = clamp(Math.round(rho * 200) / 200, -0.995, 0.995);
  const key = `${q}|${r}`;
  const hit = exceedanceCache.get(key);
  if (hit !== undefined) return hit;
  const a = standardNormalQuantile(q);
  const s = Math.sqrt(1 - r * r);
  const lo = Math.min(a, 0) - 9;
  const steps = 1500;
  const h = (a - lo) / steps;
  let m0 = 0;
  let mx = 0;
  let mxx = 0;
  let my = 0;
  let myy = 0;
  let mxy = 0;
  for (let i = 0; i < steps; i++) {
    const x = lo + (i + 0.5) * h;
    const mu = r * x;
    const beta = (a - mu) / s;
    const P = normCdf(beta);
    const pdf = normPdf(beta);
    const ey = mu * P - s * pdf;
    const eyy = (mu * mu + s * s) * P - s * (mu + a) * pdf;
    const w = normPdf(x) * h;
    m0 += w * P;
    mx += w * x * P;
    mxx += w * x * x * P;
    my += w * ey;
    myy += w * eyy;
    mxy += w * x * ey;
  }
  const ex = mx / m0;
  const eY = my / m0;
  const vx = mxx / m0 - ex * ex;
  const vy = myy / m0 - eY * eY;
  const out = vx > 0 && vy > 0 ? clamp((mxy / m0 - ex * eY) / Math.sqrt(vx * vy), -1, 1) : NaN;
  exceedanceCache.set(key, out);
  return out;
}

export interface ExceedancePoint {
  q: number;
  lower: number | null;
  lowerN: number;
  upper: number | null;
  upperN: number;
  normal: number;
}

const MIN_TAIL_OBS = 20;

function exceedanceAt(x: readonly number[], y: readonly number[], q: number): { lower: number | null; lowerN: number; upper: number | null; upperN: number } {
  const sx = [...x].sort((a, b) => a - b);
  const sy = [...y].sort((a, b) => a - b);
  const xl = quantile(sx, q);
  const yl = quantile(sy, q);
  const xu = quantile(sx, 1 - q);
  const yu = quantile(sy, 1 - q);
  const lx: number[] = [];
  const ly: number[] = [];
  const ux: number[] = [];
  const uy: number[] = [];
  for (let t = 0; t < x.length; t++) {
    if (x[t] <= xl && y[t] <= yl) {
      lx.push(x[t]);
      ly.push(y[t]);
    }
    if (x[t] >= xu && y[t] >= yu) {
      ux.push(x[t]);
      uy.push(y[t]);
    }
  }
  const lower = lx.length >= MIN_TAIL_OBS ? pearson(lx, ly) : NaN;
  const upper = ux.length >= MIN_TAIL_OBS ? pearson(ux, uy) : NaN;
  return {
    lower: Number.isFinite(lower) ? lower : null,
    lowerN: lx.length,
    upper: Number.isFinite(upper) ? upper : null,
    upperN: ux.length,
  };
}

/** Longin–Solnik / Ang–Chen exceedance correlations for one pair. */
export function exceedanceCorrelations(
  x: readonly number[],
  y: readonly number[],
  quantiles: readonly number[] = [0.1, 0.2, 0.3, 0.4, 0.5]
): { rho: number; points: ExceedancePoint[] } {
  const rho = pearson(x, y);
  return {
    rho,
    points: quantiles.map((q) => ({ q, ...exceedanceAt(x, y, q), normal: normalExceedanceCorrelation(q, rho) })),
  };
}

export interface BasketExceedance {
  q: number;
  /** Averages over pairs with enough joint-tail observations. */
  lowerAvg: number | null;
  upperAvg: number | null;
  normalAvg: number | null;
  pairsUsed: number;
  lower: (number | null)[][];
  upper: (number | null)[][];
  normal: number[][];
}

export function basketExceedance(sample: CorrelationSample, q = 0.2): BasketExceedance {
  const k = sample.tickers.length;
  const lower: (number | null)[][] = Array.from({ length: k }, () => new Array(k).fill(null));
  const upper: (number | null)[][] = Array.from({ length: k }, () => new Array(k).fill(null));
  const normal: number[][] = Array.from({ length: k }, () => new Array(k).fill(1));
  let ls = 0;
  let us = 0;
  let ns = 0;
  let used = 0;
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      const e = exceedanceAt(sample.returns[i], sample.returns[j], q);
      const bench = normalExceedanceCorrelation(q, pearson(sample.returns[i], sample.returns[j]));
      lower[i][j] = lower[j][i] = e.lower;
      upper[i][j] = upper[j][i] = e.upper;
      normal[i][j] = normal[j][i] = bench;
      if (e.lower !== null && e.upper !== null && Number.isFinite(bench)) {
        ls += e.lower;
        us += e.upper;
        ns += bench;
        used++;
      }
    }
  }
  return {
    q,
    lowerAvg: used > 0 ? ls / used : null,
    upperAvg: used > 0 ? us / used : null,
    normalAvg: used > 0 ? ns / used : null,
    pairsUsed: used,
    lower,
    upper,
    normal,
  };
}

export interface TailDependence {
  q: number;
  /** P(B in its worst q | A in its worst q), from ranks. */
  lower: number;
  upper: number;
  /** Same probability under a normal copula matched to the pair's rank correlation. */
  normal: number;
  /** Standard error of `lower`. */
  se: number;
  jointLower: number;
  expectedJointNormal: number;
}

export function tailDependence(x: readonly number[], y: readonly number[], q = 0.05): TailDependence {
  const n = Math.min(x.length, y.length);
  const rx = ranks(x);
  const ry = ranks(y);
  let lo = 0;
  let hi = 0;
  for (let t = 0; t < n; t++) {
    const u = rx[t] / (n + 1);
    const v = ry[t] / (n + 1);
    if (u < q && v < q) lo++;
    if (u > 1 - q && v > 1 - q) hi++;
  }
  const rhoS = pearson(rx, ry);
  const rhoG = 2 * Math.sin((Math.PI * rhoS) / 6);
  const joint = bivariateNormalLowerOrthant(standardNormalQuantile(q), rhoG);
  const pj = lo / n;
  return {
    q,
    lower: lo / (n * q),
    upper: hi / (n * q),
    normal: joint / q,
    se: Math.sqrt((pj * (1 - pj)) / n) / q,
    jointLower: lo,
    expectedJointNormal: joint * n,
  };
}

export interface TailPair extends TailDependence {
  a: string;
  b: string;
}

export function basketTailDependence(sample: CorrelationSample, q = 0.05): TailPair[] {
  const out: TailPair[] = [];
  for (let i = 0; i < sample.tickers.length; i++) {
    for (let j = i + 1; j < sample.tickers.length; j++) {
      out.push({ a: sample.tickers[i], b: sample.tickers[j], ...tailDependence(sample.returns[i], sample.returns[j], q) });
    }
  }
  return out.sort((x, y) => y.lower - y.normal - (x.lower - x.normal));
}

export interface WorstDay {
  date: string;
  basket: number;
  assets: number[];
  fell: number;
}

export function worstBasketDays(sample: CorrelationSample, weights: readonly number[], count = 10): WorstDay[] {
  const w = normalizeWeights(weights);
  const port = portfolioReturns(sample.returns, w);
  return port
    .map((r, t) => ({ r, t }))
    .sort((a, b) => a.r - b.r)
    .slice(0, count)
    .map(({ r, t }) => {
      const assets = sample.returns.map((s) => s[t]);
      return { date: sample.dates[t], basket: r, assets, fell: assets.filter((v) => v < 0).length };
    });
}

// ---------------------------------------------------------------------------
// Portfolio
// ---------------------------------------------------------------------------

export function normalizeWeights(weights: readonly number[]): number[] {
  const clean = weights.map((w) => (Number.isFinite(w) && w > 0 ? w : 0));
  const s = clean.reduce((a, b) => a + b, 0);
  return s > 0 ? clean.map((w) => w / s) : clean.map(() => 1 / clean.length);
}

export function portfolioReturns(returns: number[][], weights: readonly number[]): number[] {
  const n = returns[0]?.length ?? 0;
  const out = new Array(n).fill(0);
  for (let i = 0; i < returns.length; i++) {
    if (weights[i] === 0) continue;
    for (let t = 0; t < n; t++) out[t] += weights[i] * returns[i][t];
  }
  return out;
}

function quadForm(w: readonly number[], cov: number[][]): number {
  let v = 0;
  for (let i = 0; i < w.length; i++) for (let j = 0; j < w.length; j++) v += w[i] * cov[i][j] * w[j];
  return v;
}

export interface AssetRisk {
  weight: number;
  /** Annualised stand-alone volatility. */
  vol: number;
  /** Share of portfolio volatility, sums to 1. */
  riskShare: number;
  /** Annualised contribution to portfolio volatility, sums to portfolio vol. */
  contribution: number;
}

export interface RiskDecomposition {
  /** Annualised portfolio volatility. */
  vol: number;
  /** Weighted average of stand-alone volatilities (the volatility if every correlation were 1). */
  weightedAvgVol: number;
  diversificationRatio: number;
  /** Diversification ratio squared. */
  effectiveBets: number;
  assets: AssetRisk[];
}

/** Euler decomposition from a DAILY covariance matrix. Annualised exactly once here. */
export function riskDecomposition(weightsIn: readonly number[], covDaily: number[][]): RiskDecomposition {
  const w = normalizeWeights(weightsIn);
  const n = w.length;
  const sigmaW = covDaily.map((row) => row.reduce((a, v, j) => a + v * w[j], 0));
  const varP = w.reduce((a, wi, i) => a + wi * sigmaW[i], 0);
  const volDaily = Math.sqrt(Math.max(0, varP));
  const ann = Math.sqrt(TRADING_DAYS);
  const sd = covDaily.map((row, i) => Math.sqrt(Math.max(0, row[i])));
  const weightedAvgVol = w.reduce((a, wi, i) => a + wi * sd[i], 0);
  const dr = volDaily > 0 ? weightedAvgVol / volDaily : 1;
  return {
    vol: volDaily * ann,
    weightedAvgVol: weightedAvgVol * ann,
    diversificationRatio: dr,
    effectiveBets: dr * dr,
    assets: Array.from({ length: n }, (_, i) => {
      const contribDaily = volDaily > 0 ? (w[i] * sigmaW[i]) / volDaily : 0;
      return { weight: w[i], vol: sd[i] * ann, riskShare: volDaily > 0 ? contribDaily / volDaily : 0, contribution: contribDaily * ann };
    }),
  };
}

export interface TailRisk {
  level: number;
  /** One-day losses, as positive fractions. */
  parametricVaR: number;
  historicalVaR: number;
  expectedShortfall: number;
  tailDays: number;
  /** Sums to parametricVaR. */
  componentVaR: number[];
  /** Sums to expectedShortfall. */
  componentES: number[];
}

export function tailRisk(sample: CorrelationSample, weightsIn: readonly number[], covDaily: number[][], level = 0.95): TailRisk {
  const w = normalizeWeights(weightsIn);
  const port = portfolioReturns(sample.returns, w);
  const n = port.length;
  const z = -standardNormalQuantile(1 - level);
  const sigmaW = covDaily.map((row) => row.reduce((a, v, j) => a + v * w[j], 0));
  const volDaily = Math.sqrt(Math.max(0, w.reduce((a, wi, i) => a + wi * sigmaW[i], 0)));
  const order = port.map((r, t) => ({ r, t })).sort((a, b) => a.r - b.r);
  const tailDays = Math.max(1, Math.floor((1 - level) * n));
  const tail = order.slice(0, tailDays);
  const es = -mean(tail.map((d) => d.r));
  return {
    level,
    parametricVaR: z * volDaily,
    historicalVaR: -quantile(order.map((d) => d.r), 1 - level),
    expectedShortfall: es,
    tailDays,
    componentVaR: w.map((wi, i) => (volDaily > 0 ? (z * wi * sigmaW[i]) / volDaily : 0)),
    componentES: w.map((wi, i) => -wi * mean(tail.map((d) => sample.returns[i][d.t]))),
  };
}

export function inverseVolWeights(cov: number[][]): number[] {
  return normalizeWeights(cov.map((row, i) => (row[i] > 0 ? 1 / Math.sqrt(row[i]) : 0)));
}

/** Exact long-only minimum-variance weights by an active-set method. */
export function minVarianceLongOnly(cov: number[][]): number[] {
  const n = cov.length;
  let active = Array.from({ length: n }, (_, i) => i);
  let w = new Array(n).fill(1 / n);
  for (let iter = 0; iter < 6 * n; iter++) {
    const sub = active.map((i) => active.map((j) => cov[i][j]));
    const inv = invertSymmetric(sub);
    if (!inv) break;
    const x = inv.map((row) => row.reduce((a, b) => a + b, 0));
    const total = x.reduce((a, b) => a + b, 0);
    if (!(total > 0)) break;
    const ws = x.map((v) => v / total);
    let worst = -1;
    let worstVal = -1e-12;
    ws.forEach((v, k) => {
      if (v < worstVal) {
        worstVal = v;
        worst = k;
      }
    });
    if (worst >= 0 && active.length > 1) {
      active = active.filter((_, k) => k !== worst);
      continue;
    }
    w = new Array(n).fill(0);
    active.forEach((i, k) => (w[i] = ws[k]));
    const varP = quadForm(w, cov);
    let add = -1;
    let addVal = varP * (1 - 1e-9);
    for (let i = 0; i < n; i++) {
      if (active.includes(i)) continue;
      const g = cov[i].reduce((a, v, j) => a + v * w[j], 0);
      if (g < addVal) {
        addVal = g;
        add = i;
      }
    }
    if (add < 0) break;
    active = [...active, add].sort((a, b) => a - b);
  }
  return normalizeWeights(w);
}

/** Equal risk contribution by cyclical coordinate descent (Griveau-Billion et al.). */
export function equalRiskContribution(cov: number[][]): number[] {
  const n = cov.length;
  const b = 1 / n;
  let x = cov.map((row, i) => (row[i] > 0 ? 1 / Math.sqrt(row[i]) : 1));
  for (let iter = 0; iter < 2000; iter++) {
    let maxRel = 0;
    for (let i = 0; i < n; i++) {
      let off = 0;
      for (let j = 0; j < n; j++) if (j !== i) off += cov[i][j] * x[j];
      const next = (-off + Math.sqrt(off * off + 4 * cov[i][i] * b)) / (2 * cov[i][i]);
      maxRel = Math.max(maxRel, Math.abs(next - x[i]) / Math.max(1e-300, x[i]));
      x[i] = next;
    }
    if (maxRel < 1e-12) break;
  }
  x = x.map((v) => (Number.isFinite(v) && v > 0 ? v : 0));
  return normalizeWeights(x);
}

/** Long-only maximum-diversification weights (min variance on the correlation matrix, rescaled by volatility). */
export function maxDiversification(cov: number[][]): number[] {
  const sd = cov.map((row, i) => Math.sqrt(Math.max(0, row[i])));
  const y = minVarianceLongOnly(covToCorr(cov));
  return normalizeWeights(y.map((v, i) => (sd[i] > 0 ? v / sd[i] : 0)));
}

/** Leaf order from single-linkage clustering on the distance √((1 − ρ)/2). */
export function hierarchicalOrder(corr: number[][]): number[] {
  const n = corr.length;
  let clusters: number[][] = Array.from({ length: n }, (_, i) => [i]);
  const dist = (a: number[], b: number[]) => {
    let d = Infinity;
    for (const i of a) for (const j of b) d = Math.min(d, Math.sqrt(Math.max(0, (1 - corr[i][j]) / 2)));
    return d;
  };
  while (clusters.length > 1) {
    let bi = 0;
    let bj = 1;
    let bd = Infinity;
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const d = dist(clusters[i], clusters[j]);
        if (d < bd) {
          bd = d;
          bi = i;
          bj = j;
        }
      }
    }
    const merged = [...clusters[bi], ...clusters[bj]];
    clusters = clusters.filter((_, k) => k !== bi && k !== bj);
    clusters.push(merged);
  }
  return clusters[0] ?? [];
}

/** Hierarchical risk parity (López de Prado 2016). */
export function hierarchicalRiskParity(cov: number[][]): number[] {
  const n = cov.length;
  const order = hierarchicalOrder(covToCorr(cov));
  const w = new Array(n).fill(1);
  const clusterVar = (items: number[]) => {
    const ivp = normalizeWeights(items.map((i) => (cov[i][i] > 0 ? 1 / cov[i][i] : 0)));
    let v = 0;
    for (let a = 0; a < items.length; a++) for (let b = 0; b < items.length; b++) v += ivp[a] * cov[items[a]][items[b]] * ivp[b];
    return v;
  };
  const bisect = (items: number[]) => {
    if (items.length < 2) return;
    const mid = Math.floor(items.length / 2);
    const left = items.slice(0, mid);
    const right = items.slice(mid);
    const vl = clusterVar(left);
    const vr = clusterVar(right);
    const alpha = vl + vr > 0 ? 1 - vl / (vl + vr) : 0.5;
    left.forEach((i) => (w[i] *= alpha));
    right.forEach((i) => (w[i] *= 1 - alpha));
    bisect(left);
    bisect(right);
  };
  bisect(order);
  return normalizeWeights(w);
}

export type AllocationId = 'equal' | 'inverseVol' | 'erc' | 'minVar' | 'maxDiv' | 'hrp';

export const ALLOCATIONS: { id: AllocationId; label: string; description: string }[] = [
  { id: 'equal', label: 'Equal weight', description: 'Same capital in every asset.' },
  { id: 'inverseVol', label: 'Inverse volatility', description: 'Less capital in more volatile assets; ignores correlation.' },
  { id: 'erc', label: 'Equal risk', description: 'Every asset contributes the same share of portfolio risk.' },
  { id: 'hrp', label: 'Hierarchical risk parity', description: 'Splits risk across correlation clusters; no matrix inversion.' },
  { id: 'maxDiv', label: 'Max diversification', description: 'Maximises the diversification ratio, long-only.' },
  { id: 'minVar', label: 'Minimum variance', description: 'Lowest-volatility long-only mix; often concentrated.' },
];

export function allocationWeights(id: AllocationId, cov: number[][]): number[] {
  switch (id) {
    case 'equal':
      return cov.map(() => 1 / cov.length);
    case 'inverseVol':
      return inverseVolWeights(cov);
    case 'erc':
      return equalRiskContribution(cov);
    case 'minVar':
      return minVarianceLongOnly(cov);
    case 'maxDiv':
      return maxDiversification(cov);
    case 'hrp':
      return hierarchicalRiskParity(cov);
  }
}

export interface StressRow {
  label: string;
  avgCorr: number;
  /** Annualised portfolio volatility under the stressed correlations. */
  vol: number;
}

/**
 * Portfolio volatility if correlations rise. Each scenario blends the
 * correlation matrix toward all-ones (which keeps it valid) until the
 * average pairwise correlation is `shift` higher.
 */
export function correlationStress(weightsIn: readonly number[], covDaily: number[][], shifts: readonly number[] = [0, 0.1, 0.2, 0.3]): StressRow[] {
  const w = normalizeWeights(weightsIn);
  const sd = covDaily.map((row, i) => Math.sqrt(Math.max(0, row[i])));
  const R = covToCorr(covDaily);
  const rhoBar = averageCorrelation(R);
  const volAt = (lambda: number) => {
    let v = 0;
    for (let i = 0; i < w.length; i++) {
      for (let j = 0; j < w.length; j++) v += w[i] * w[j] * sd[i] * sd[j] * (i === j ? 1 : (1 - lambda) * R[i][j] + lambda);
    }
    return Math.sqrt(Math.max(0, v) * TRADING_DAYS);
  };
  const rows: StressRow[] = shifts
    .filter((d) => d === 0 || rhoBar + d < 1)
    .map((d) => {
      const lambda = d === 0 ? 0 : clamp(d / (1 - rhoBar), 0, 1);
      return { label: d === 0 ? 'Today' : `Correlations +${d.toFixed(1)}`, avgCorr: rhoBar + d, vol: volAt(lambda) };
    });
  rows.push({ label: 'Everything moves together', avgCorr: 1, vol: volAt(1) });
  return rows;
}

export interface HedgeRow {
  ticker: string;
  corr: number;
  /** Units of the hedge to short per unit of the asset. */
  hedgeRatio: number;
  /** Share of the asset's variance removed by the hedge (ρ²). */
  varianceReduction: number;
  vol: number;
  hedgedVol: number;
}

export function hedgeTable(sample: CorrelationSample, hedgeIndex: number): HedgeRow[] {
  const cov = covarianceMatrix(sample.returns);
  const varH = cov[hedgeIndex][hedgeIndex];
  const ann = Math.sqrt(TRADING_DAYS);
  return sample.tickers
    .map((ticker, i) => {
      const sdI = Math.sqrt(cov[i][i]);
      const corr = varH > 0 && sdI > 0 ? cov[i][hedgeIndex] / (sdI * Math.sqrt(varH)) : 0;
      return {
        ticker,
        corr,
        hedgeRatio: varH > 0 ? cov[i][hedgeIndex] / varH : 0,
        varianceReduction: corr * corr,
        vol: sdI * ann,
        hedgedVol: sdI * Math.sqrt(Math.max(0, 1 - corr * corr)) * ann,
      };
    })
    .filter((_, i) => i !== hedgeIndex);
}

export interface WalkForwardRow {
  id: AllocationId;
  label: string;
  /** Volatility the weights promised when they were set (in-sample). */
  expectedVol: number;
  /** Volatility actually realised over the following holding periods. */
  realizedVol: number;
}

export interface WalkForward {
  lookback: number;
  hold: number;
  rebalances: number;
  firstDate: string;
  rows: WalkForwardRow[];
}

/** Set weights on a trailing window, hold them, measure what happened next. No look-ahead. */
export function walkForwardAllocations(sample: CorrelationSample, lookback = 252, hold = 21): WalkForward | null {
  const n = sample.dates.length;
  if (n < lookback + 3 * hold) return null;
  const acc = ALLOCATIONS.map((a) => ({ id: a.id, label: a.label, expectedVar: 0, realized: [] as number[] }));
  let rebalances = 0;
  for (let t0 = lookback; t0 + hold <= n; t0 += hold) {
    const cov = ledoitWolf(sliceReturns(sample.returns, t0 - lookback, t0)).cov;
    const future = sliceReturns(sample.returns, t0, t0 + hold);
    for (const a of acc) {
      const w = allocationWeights(a.id, cov);
      a.expectedVar += quadForm(w, cov);
      a.realized.push(...portfolioReturns(future, w));
    }
    rebalances++;
  }
  return {
    lookback,
    hold,
    rebalances,
    firstDate: sample.dates[lookback],
    rows: acc.map((a) => ({
      id: a.id,
      label: a.label,
      expectedVol: Math.sqrt((a.expectedVar / rebalances) * TRADING_DAYS),
      realizedVol: Math.sqrt(variance(a.realized) * TRADING_DAYS),
    })),
  };
}

// ---------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------

export interface ResidualCorrelation {
  tickers: string[];
  matrix: number[][];
  /** What was removed. */
  removed: string;
}

function residualsOn(y: readonly number[], x: readonly number[]): number[] {
  const mx = mean(x);
  const my = mean(y);
  let sxy = 0;
  let sxx = 0;
  for (let t = 0; t < x.length; t++) {
    sxy += (x[t] - mx) * (y[t] - my);
    sxx += (x[t] - mx) ** 2;
  }
  const beta = sxx > 0 ? sxy / sxx : 0;
  return y.map((v, t) => v - my - beta * (x[t] - mx));
}

/**
 * Correlation left after removing the common driver: the chosen market asset
 * when one is in the basket, otherwise the first principal component.
 */
export function residualCorrelation(sample: CorrelationSample, marketIndex: number | null): ResidualCorrelation {
  if (marketIndex !== null && marketIndex >= 0 && marketIndex < sample.tickers.length) {
    const m = sample.returns[marketIndex];
    const keep = sample.tickers.map((_, i) => i).filter((i) => i !== marketIndex);
    return {
      tickers: keep.map((i) => sample.tickers[i]),
      matrix: correlationMatrix(keep.map((i) => residualsOn(sample.returns[i], m))),
      removed: sample.tickers[marketIndex],
    };
  }
  const corr = correlationMatrix(sample.returns);
  const vec = jacobiEigen(corr).vectors[0];
  const sd = sample.returns.map((r) => Math.sqrt(variance(r)));
  const means = sample.returns.map(mean);
  const n = sample.dates.length;
  const score = new Array(n).fill(0);
  for (let i = 0; i < sample.returns.length; i++) {
    for (let t = 0; t < n; t++) score[t] += (vec[i] * (sample.returns[i][t] - means[i])) / (sd[i] || 1);
  }
  return {
    tickers: [...sample.tickers],
    matrix: correlationMatrix(sample.returns.map((r) => residualsOn(r, score))),
    removed: 'the first common factor',
  };
}

export interface TreeEdge {
  a: string;
  b: string;
  corr: number;
}

/** Minimum spanning tree on the Mantegna distance √(2(1 − ρ)). */
export function minimumSpanningTree(corr: number[][], tickers: readonly string[]): { edges: TreeEdge[]; degree: number[] } {
  const n = tickers.length;
  const all: { i: number; j: number; corr: number }[] = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) all.push({ i, j, corr: corr[i][j] });
  all.sort((x, y) => y.corr - x.corr);
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  const edges: TreeEdge[] = [];
  const degree = new Array(n).fill(0);
  for (const e of all) {
    const ri = find(e.i);
    const rj = find(e.j);
    if (ri === rj) continue;
    parent[ri] = rj;
    edges.push({ a: tickers[e.i], b: tickers[e.j], corr: e.corr });
    degree[e.i]++;
    degree[e.j]++;
    if (edges.length === n - 1) break;
  }
  return { edges, degree };
}

export interface GrangerResult {
  /** Robust Wald statistic that all lags of the other asset are zero, given the asset's own lags. */
  wald: number;
  df: number;
  p: number;
  /** Lag with the largest robust t-statistic, and its coefficient. */
  peakLag: number;
  peakCoef: number;
}

/** Does `cause` help predict `effect` beyond `effect`'s own past? White-robust Wald test. */
export function grangerTest(cause: readonly number[], effect: readonly number[], lagCount: number): GrangerResult | null {
  const n = Math.min(cause.length, effect.length);
  const p = 1 + 2 * lagCount;
  const rows = n - lagCount;
  if (rows < p * 8) return null;
  const X: number[][] = [];
  const Y: number[] = [];
  for (let t = lagCount; t < n; t++) {
    const row = [1];
    for (let k = 1; k <= lagCount; k++) row.push(effect[t - k]);
    for (let k = 1; k <= lagCount; k++) row.push(cause[t - k]);
    X.push(row);
    Y.push(effect[t]);
  }
  const xtx = Array.from({ length: p }, () => new Array(p).fill(0));
  const xty = new Array(p).fill(0);
  for (let r = 0; r < rows; r++) {
    for (let i = 0; i < p; i++) {
      xty[i] += X[r][i] * Y[r];
      for (let j = 0; j < p; j++) xtx[i][j] += X[r][i] * X[r][j];
    }
  }
  const inv = invertSymmetric(xtx);
  if (!inv) return null;
  const beta = inv.map((row) => row.reduce((a, v, j) => a + v * xty[j], 0));
  const meat = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let r = 0; r < rows; r++) {
    let e = Y[r];
    for (let i = 0; i < p; i++) e -= X[r][i] * beta[i];
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) meat[i][j] += e * e * X[r][i] * X[r][j];
  }
  const mul = (A: number[][], B: number[][]) => A.map((row) => B[0].map((_, j) => row.reduce((a, v, k) => a + v * B[k][j], 0)));
  const cov = mul(mul(inv, meat), inv);
  const idx = Array.from({ length: lagCount }, (_, k) => 1 + lagCount + k);
  const sub = invertSymmetric(idx.map((i) => idx.map((j) => cov[i][j])));
  if (!sub) return null;
  let wald = 0;
  for (let a = 0; a < lagCount; a++) for (let b = 0; b < lagCount; b++) wald += beta[idx[a]] * sub[a][b] * beta[idx[b]];
  let peakLag = 1;
  let peakT = 0;
  idx.forEach((i, k) => {
    const t = cov[i][i] > 0 ? Math.abs(beta[i]) / Math.sqrt(cov[i][i]) : 0;
    if (t > peakT) {
      peakT = t;
      peakLag = k + 1;
    }
  });
  return { wald, df: lagCount, p: chiSquareUpperP(wald, lagCount), peakLag, peakCoef: beta[idx[peakLag - 1]] };
}

export interface LeadLag {
  /**
   * Descriptive cross-correlations. lag > 0: A today with B `lag` days earlier.
   * lag < 0: A `|lag|` days earlier with B today. `se` is robust to volatility clustering.
   */
  lags: { lag: number; corr: number; se: number }[];
  /** Predictive tests that control for each asset's own past. */
  aLeadsB: GrangerResult | null;
  bLeadsA: GrangerResult | null;
  /** Named only when exactly one direction is significant at 2.5% (two tests). */
  leader: 'A' | 'B' | 'both' | null;
}

export function leadLag(x: readonly number[], y: readonly number[], maxLag = 5): LeadLag | null {
  const n = Math.min(x.length, y.length);
  if (n < maxLag * 4 + 30) return null;
  const mx = mean(x.slice(0, n));
  const my = mean(y.slice(0, n));
  const lags: { lag: number; corr: number; se: number }[] = [];
  for (let k = -maxLag; k <= maxLag; k++) {
    let sab = 0;
    let saa = 0;
    let sbb = 0;
    let s4 = 0;
    for (let t = Math.abs(k); t < n; t++) {
      const a = (k >= 0 ? x[t] : x[t + k]) - mx;
      const b = (k >= 0 ? y[t - k] : y[t]) - my;
      sab += a * b;
      saa += a * a;
      sbb += b * b;
      s4 += a * a * b * b;
    }
    const denom = Math.sqrt(saa * sbb);
    lags.push({ lag: k, corr: denom > 0 ? sab / denom : 0, se: denom > 0 ? Math.sqrt(s4) / denom : Infinity });
  }
  const aLeadsB = grangerTest(x, y, maxLag);
  const bLeadsA = grangerTest(y, x, maxLag);
  const aSig = aLeadsB !== null && aLeadsB.p < 0.025;
  const bSig = bLeadsA !== null && bLeadsA.p < 0.025;
  return { lags, aLeadsB, bLeadsA, leader: aSig && bSig ? 'both' : aSig ? 'A' : bSig ? 'B' : null };
}

// ---------------------------------------------------------------------------
// Intraday: correlation by sampling interval (Epps effect)
// ---------------------------------------------------------------------------

export interface IntradaySamplingRow {
  minutes: number;
  corr: number | null;
  observations: number;
}

export interface IntradayCorrelation {
  days: number;
  overlappingBars: number;
  rows: IntradaySamplingRow[];
}

/**
 * Realized correlation of two assets from 1-minute bars matched on timestamp,
 * re-sampled at coarser intervals. Returns never cross the overnight gap.
 */
export function intradayCorrelationBySampling(
  barsA: readonly IntradayBar[],
  barsB: readonly IntradayBar[],
  intervals: readonly number[] = [1, 5, 15, 30]
): IntradayCorrelation {
  const key = (b: IntradayBar) => `${b.date}|${b.minuteOfDay}`;
  const mapB = new Map(barsB.map((b) => [key(b), b.close]));
  const byDay = new Map<string, { minute: number; a: number; b: number }[]>();
  let overlap = 0;
  for (const bar of barsA) {
    const other = mapB.get(key(bar));
    if (other === undefined || !(bar.close > 0) || !(other > 0)) continue;
    overlap++;
    const list = byDay.get(bar.date) ?? [];
    list.push({ minute: bar.minuteOfDay, a: bar.close, b: other });
    byDay.set(bar.date, list);
  }
  const rows = intervals.map((m) => {
    let sab = 0;
    let saa = 0;
    let sbb = 0;
    let count = 0;
    for (const list of byDay.values()) {
      const grid = list.filter((p) => p.minute % m === 0).sort((p, q) => p.minute - q.minute);
      for (let i = 1; i < grid.length; i++) {
        if (grid[i].minute - grid[i - 1].minute !== m) continue;
        const ra = Math.log(grid[i].a / grid[i - 1].a);
        const rb = Math.log(grid[i].b / grid[i - 1].b);
        sab += ra * rb;
        saa += ra * ra;
        sbb += rb * rb;
        count++;
      }
    }
    return { minutes: m, corr: count >= 30 && saa > 0 && sbb > 0 ? clamp(sab / Math.sqrt(saa * sbb), -1, 1) : null, observations: count };
  });
  return { days: byDay.size, overlappingBars: overlap, rows };
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

export function correlationCsv(tickers: readonly string[], matrix: (number | null)[][]): string {
  const rows = [['', ...tickers].join(',')];
  matrix.forEach((row, i) => rows.push([tickers[i], ...row.map((v) => (v === null || !Number.isFinite(v) ? '' : v.toFixed(4)))].join(',')));
  return rows.join('\n');
}

export function correlationLatex(tickers: readonly string[], matrix: (number | null)[][], caption = 'Correlation matrix'): string {
  const n = tickers.length;
  const lines = [
    '\\begin{table}[htbp]',
    '\\centering',
    `\\caption{${caption}}`,
    `\\begin{tabular}{l${'r'.repeat(n)}}`,
    '\\hline\\hline',
    ` & ${tickers.join(' & ')} \\\\`,
    '\\hline',
  ];
  matrix.forEach((row, i) => {
    const cells = row.map((v, j) => (j > i ? '' : v === null || !Number.isFinite(v) ? '--' : `$${v.toFixed(2)}$`));
    lines.push(`${tickers[i]} & ${cells.join(' & ')} \\\\`);
  });
  lines.push('\\hline\\hline', '\\end{tabular}', '\\end{table}');
  return lines.join('\n');
}
