// Vol Analytics: time-series statistics for the Implied Vol tab.
// Pure functions over daily close series. All series are plain number[] in
// chronological order unless noted. No React, no I/O.

import { incBeta, fSurvival, normalQuantile } from './options';

export interface DatedPoint {
  date: string;
  value: number;
}

/** Percentile rank (0–100) of `value` inside `sample`. */
export function percentileRank(sample: number[], value: number): number | null {
  if (sample.length === 0 || !isFinite(value)) return null;
  let below = 0;
  let equal = 0;
  for (const v of sample) {
    if (v < value) below++;
    else if (v === value) equal++;
  }
  return ((below + 0.5 * equal) / sample.length) * 100;
}

/** Sorted-quantile (q in [0,1]) with linear interpolation. */
export function quantileSorted(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = q * (sorted.length - 1);
  const base = Math.floor(pos);
  const rest = pos - base;
  if (base + 1 < sorted.length) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

/** Log returns from closes (first element omitted). */
export function logReturns(closes: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    if (closes[i - 1] > 0 && closes[i] > 0) out.push(Math.log(closes[i] / closes[i - 1]));
  }
  return out;
}

/**
 * Forward realized variance over horizon h starting at t+1 (0-based index of the
 * IV observation). Annualized by multiplying by 252/h and returning std, not var.
 * Returns null when the window exceeds available history.
 */
export function forwardRealizedVariance(returns: number[], t: number, h: number): number | null {
  if (t + h > returns.length) return null;
  let sum = 0;
  for (let i = t + 1; i <= t + h; i++) sum += returns[i - 1] * returns[i - 1];
  return (252 / h) * sum;
}

/** QLIKE loss between realized variance and a variance forecast (Hansen & Lunde 2005). */
export function qlike(rvVar: number, forecastVar: number): number {
  const rv = Math.max(rvVar, 1e-12);
  const fv = Math.max(forecastVar, 1e-12);
  return Math.log(fv) + rv / fv;
}

/** MSE on log-variance (robust alternative when both tails are noisy). */
export function mseLogVariance(rvVar: number, forecastVar: number): number {
  const rv = Math.max(rvVar, 1e-12);
  const fv = Math.max(forecastVar, 1e-12);
  const d = Math.log(fv) - Math.log(rv);
  return d * d;
}

/** Newey–West long-run variance of a series. */
export function neweyWestLrv(x: number[], lag: number): number {
  const n = x.length;
  if (n === 0) return 0;
  let mean = 0;
  for (const v of x) mean += v;
  mean /= n;
  const demeaned = x.map((v) => v - mean);
  let gamma0 = 0;
  for (const v of demeaned) gamma0 += v * v;
  gamma0 /= n;
  let lrv = gamma0;
  for (let l = 1; l <= lag; l++) {
    let gamma = 0;
    for (let i = l; i < n; i++) gamma += demeaned[i] * demeaned[i - l];
    gamma /= n;
    const weight = 1 - l / (lag + 1);
    lrv += 2 * weight * gamma;
  }
  return Math.max(0, lrv);
}

/**
 * Diebold–Mariano test on the loss differential d = lossA − lossB.
 * DM = mean(d) / sqrt(LRV/T). Two-sided p-value via normal approximation.
 */
export function dieboldMariano(lossA: number[], lossB: number[]): {
  dm: number;
  pValue: number;
  meanDiff: number;
  lrv: number;
} | null {
  const n = Math.min(lossA.length, lossB.length);
  if (n < 10) return null;
  const d: number[] = [];
  for (let i = 0; i < n; i++) d.push(lossA[i] - lossB[i]);
  let mean = 0;
  for (const v of d) mean += v;
  mean /= n;
  const lag = Math.max(1, Math.floor(Math.pow(n, 1 / 3)));
  const lrv = neweyWestLrv(d, lag);
  if (lrv <= 0) return { dm: 0, pValue: 1, meanDiff: mean, lrv: 0 };
  const dm = mean / Math.sqrt(lrv / n);
  // two-sided p via normal CDF
  const pValue = 2 * (1 - normCdfRef(Math.abs(dm)));
  return { dm, pValue: Math.max(0, Math.min(1, pValue)), meanDiff: mean, lrv };
}

function normCdfRef(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * ax);
  const erf =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-ax * ax);
  return 0.5 * (1 + sign * erf);
}

// ---------------------------------------------------------------------------
// OLS helper (small dense solve, Gaussian elimination with partial pivot)
// ---------------------------------------------------------------------------

export interface OlsResult {
  beta: number[];
  se: number[];
  tStat: number[];
  r2: number;
  sigma: number;      // residual std error
  ssr: number;
  fitted: number[];
  residuals: number[];
}

export function solveOls(X: number[][], y: number[]): OlsResult {
  const n = X.length;
  const k = X[0]?.length ?? 0;
  if (n === 0 || k === 0) {
    return { beta: [], se: [], tStat: [], r2: 0, sigma: 0, ssr: 0, fitted: [], residuals: [] };
  }
  // (X'X)^{-1} X'y via Gaussian elimination on the normal equations
  const XtX: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty: number[] = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < k; a++) {
      Xty[a] += X[i][a] * y[i];
      for (let b = 0; b <= a; b++) XtX[a][b] += X[i][a] * X[i][b];
    }
  }
  for (let a = 0; a < k; a++) for (let b = 0; b < a; b++) XtX[b][a] = XtX[a][b];

  // Solve XtX * beta = Xty with partial pivot
  const A: number[][] = XtX.map((row, i) => [...row, Xty[i]]);
  for (let col = 0; col < k; col++) {
    let piv = col;
    for (let row = col + 1; row < k; row++) {
      if (Math.abs(A[row][col]) > Math.abs(A[piv][col])) piv = row;
    }
    [A[col], A[piv]] = [A[piv], A[col]];
    const diag = A[col][col];
    if (Math.abs(diag) < 1e-14) continue;
    for (let row = 0; row < k; row++) {
      if (row === col) continue;
      const factor = A[row][col] / diag;
      for (let j = col; j <= k; j++) A[row][j] -= factor * A[col][j];
    }
  }
  const beta = new Array(k).fill(0);
  for (let i = 0; i < k; i++) {
    const diag = A[i][i];
    beta[i] = Math.abs(diag) > 1e-14 ? A[i][k] / diag : 0;
  }

  const fitted = new Array(n).fill(0);
  const residuals = new Array(n).fill(0);
  let ssr = 0;
  let yMean = 0;
  for (const v of y) yMean += v;
  yMean /= n;
  let tss = 0;
  for (let i = 0; i < n; i++) {
    let f = 0;
    for (let j = 0; j < k; j++) f += beta[j] * X[i][j];
    fitted[i] = f;
    residuals[i] = y[i] - f;
    ssr += residuals[i] * residuals[i];
    tss += (y[i] - yMean) * (y[i] - yMean);
  }
  const dof = Math.max(1, n - k);
  const sigma = Math.sqrt(ssr / dof);
  const B: number[][] = XtX.map((row, i) => [...row, ...new Array(k).fill(0).map((_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < k; col++) {
    let piv = col;
    for (let row = col + 1; row < k; row++) {
      if (Math.abs(B[row][col]) > Math.abs(B[piv][col])) piv = row;
    }
    [B[col], B[piv]] = [B[piv], B[col]];
    const diag = B[col][col];
    if (Math.abs(diag) < 1e-14) continue;
    for (let row = 0; row < k; row++) {
      if (row === col) continue;
      const factor = B[row][col] / diag;
      for (let j = col; j < 2 * k; j++) B[row][j] -= factor * B[col][j];
    }
  }
  const se = new Array(k).fill(sigma);
  for (let i = 0; i < k; i++) {
    const diag = B[i][i];
    if (Math.abs(diag) > 1e-14) {
      const xiix = B[i][k + i] / diag;
      se[i] = sigma * Math.sqrt(Math.max(0, xiix));
    }
  }
  const tStat = beta.map((b, i) => (se[i] > 1e-12 ? b / se[i] : 0));
  const r2 = tss > 0 ? Math.max(0, 1 - ssr / tss) : 0;

  return { beta, se, tStat, r2, sigma, ssr, fitted, residuals };
}

// ---------------------------------------------------------------------------
// Mincer–Zarnowitz: RV_{t+h} = a + b·IV_t + ε  (IV and RV both annualized σ² or σ — see param useVariance)
// ---------------------------------------------------------------------------

export interface MincerZarnowitzResult {
  n: number;
  intercept: number;
  slope: number;
  interceptSe: number;
  slopeSe: number;
  r2: number;
  jointF: number;
  jointP: number;
  horizon: number;
}

export function mincerZarnowitz(
  iv: number[],
  rv: number[],
  horizon: number
): MincerZarnowitzResult | null {
  const n = Math.min(iv.length, rv.length);
  if (n < 20) return null;
  const X: number[][] = [];
  const Y: number[] = [];
  for (let i = 0; i < n; i++) {
    X.push([1, iv[i]]);
    Y.push(rv[i]);
  }
  const ols = solveOls(X, Y);
  const ssr0 = Y.reduce((s, y, i) => s + (y - iv[i]) * (y - iv[i]), 0); // H0: a=0, b=1
  const d1 = 2;
  const d2 = n - 2;
  const jointF = d2 > 0 && ols.ssr > 0 ? ((ssr0 - ols.ssr) / d1) / (ols.ssr / d2) : 0;
  const jointP = fSurvival(Math.max(0, jointF), d1, d2);
  return {
    n,
    intercept: ols.beta[0],
    slope: ols.beta[1],
    interceptSe: ols.se[0],
    slopeSe: ols.se[1],
    r2: ols.r2,
    jointF: Math.max(0, jointF),
    jointP,
    horizon,
  };
}

/** Encompassing regression RV ~ a + w1·GARCH + w2·IV → variance weights (normalized). */
export function encompassingWeights(
  garchVar: number[],
  ivVar: number[],
  rvVar: number[]
): { wGarch: number; wIv: number; intercept: number; r2: number; n: number } | null {
  const n = Math.min(garchVar.length, ivVar.length, rvVar.length);
  if (n < 20) return null;
  const X: number[][] = [];
  const Y: number[] = [];
  for (let i = 0; i < n; i++) {
    X.push([1, garchVar[i], ivVar[i]]);
    Y.push(rvVar[i]);
  }
  const ols = solveOls(X, Y);
  const wG = ols.beta[1];
  const wI = ols.beta[2];
  const total = wG + wI;
  const normG = total > 0 ? wG / total : 0.5;
  const normI = total > 0 ? wI / total : 0.5;
  return { wGarch: normG, wIv: normI, intercept: ols.beta[0], r2: ols.r2, n };
}

/** IV beta: OLS of ΔIV on ΔVIX, plus correlation of ΔIV with returns. */
export function ivBeta(
  ivSeries: number[],
  vixSeries: number[],
  returns: number[]
): { beta: number; corrDeltaIvReturns: number; n: number } | null {
  const n = Math.min(ivSeries.length, vixSeries.length, returns.length);
  if (n < 30) return null;
  const dIv: number[] = [];
  const dVix: number[] = [];
  for (let i = 1; i < n; i++) {
    dIv.push(ivSeries[i] - ivSeries[i - 1]);
    dVix.push(vixSeries[i] - vixSeries[i - 1]);
  }
  const X = dVix.map((v) => [1, v]);
  const ols = solveOls(X, dIv);
  // corr(ΔIV, r)
  const rAligned = returns.slice(returns.length - dIv.length);
  const meanA = dIv.reduce((s, v) => s + v, 0) / dIv.length;
  const meanB = rAligned.reduce((s, v) => s + v, 0) / rAligned.length;
  let cov = 0, varA = 0, varB = 0;
  for (let i = 0; i < dIv.length; i++) {
    const da = dIv[i] - meanA;
    const db = rAligned[i] - meanB;
    cov += da * db;
    varA += da * da;
    varB += db * db;
  }
  const corr = varA > 0 && varB > 0 ? cov / Math.sqrt(varA * varB) : 0;
  return { beta: ols.beta[1], corrDeltaIvReturns: corr, n: dIv.length };
}

export type TermStructureVerdict = 'CONTANGO' | 'FLAT' | 'INVERTED';

export function termStructureVerdict(shortVol: number, longVol: number): TermStructureVerdict {
  const ratio = longVol > 0 ? shortVol / longVol : 1;
  if (ratio > 1.04) return 'INVERTED';   // short-dated IV richer than long-dated: stress/backwardation
  if (ratio < 0.96) return 'CONTANGO';
  return 'FLAT';
}

// ---------------------------------------------------------------------------
// GARCH-X: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ·IV²_{t-1}
// Mirrors the codebase's grid + coordinate-descent estimation.
// ivVariance is the daily exogenous variance series aligned to `returns`
// (same length; pass NaN for missing days — those dates are skipped in scoring).
// ---------------------------------------------------------------------------

export interface GarchXResult {
  omega: number;
  alpha: number;
  beta: number;
  gamma: number;
  persistence: number;
  logLikelihood: number;
  converged: boolean;
  conditionalVariance: number[];
  conditionalAnnualizedVol: number[];
}

export function fitGarchX(returns: number[], ivVariance: number[]): GarchXResult {
  const n = returns.length;
  const fillGamma = (gamma: number) => gamma;

  if (n < 60) {
    return {
      omega: 1e-5, alpha: 0.08, beta: 0.85, gamma: 0, persistence: 0.93,
      logLikelihood: 0, converged: false,
      conditionalVariance: new Array(n).fill(0.0002),
      conditionalAnnualizedVol: new Array(n).fill(0.22),
    };
  }

  let meanRet = 0;
  for (const r of returns) meanRet += r;
  meanRet /= n;
  const eps2 = returns.map((r) => (r - meanRet) ** 2);
  let sampleVar = 0;
  for (const e of eps2) sampleVar += e;
  sampleVar /= n;

  const ivOk = ivVariance.map((v) => (isFinite(v) && v > 0 ? v : null));
  const validCount = ivOk.filter((v) => v !== null).length;
  if (validCount < 40) {
    return {
      omega: 1e-5, alpha: 0.08, beta: 0.88, gamma: 0, persistence: 0.96,
      logLikelihood: 0, converged: false,
      conditionalVariance: new Array(n).fill(sampleVar),
      conditionalAnnualizedVol: new Array(n).fill(Math.sqrt(sampleVar) * Math.sqrt(252)),
    };
  }

  const negLL = (p: [number, number, number, number]): number => {
    const [omega, alpha, beta, gamma] = p;
    if (omega <= 1e-12 || alpha < 0 || beta < 0 || gamma < 0 || alpha + beta + gamma >= 0.9999) {
      return 1e12;
    }
    let ll = 0;
    let s2 = sampleVar;
    for (let t = 0; t < n; t++) {
      if (t > 0) {
        const exog = ivOk[t - 1] ?? 0;
        s2 = omega + alpha * eps2[t - 1] + beta * s2 + gamma * exog;
      }
      if (s2 <= 1e-12) return 1e12;
      ll += -0.5 * (Math.log(2 * Math.PI) + Math.log(s2) + eps2[t] / s2);
    }
    return -ll;
  };

  // Grid sweep around plausible values (gamma small: IV absorbs a fraction of clustering)
  let best: [number, number, number, number] = [0.02 * sampleVar, 0.06, 0.85, fillGamma(0.05)];
  let bestLL = negLL(best);
  for (const a of [0.03, 0.08, 0.13]) {
    for (const b of [0.78, 0.85, 0.9]) {
      for (const g of [0.02, 0.08, 0.15]) {
        if (a + b + g < 0.995) {
          const w = Math.max(1e-8, (1 - a - b - g) * sampleVar);
          const v = negLL([w, a, b, g]);
          if (v < bestLL) { bestLL = v; best = [w, a, b, g]; }
        }
      }
    }
  }

  let [w, a, b, g] = best;
  let stepW = w * 0.15, stepA = 0.01, stepB = 0.015, stepG = 0.02;
  let converged = false;
  for (let iter = 0; iter < 160; iter++) {
    let improved = false;
    const tryParam = (idx: 0 | 1 | 2 | 3, delta: number) => {
      const cand: [number, number, number, number] = [...best] as [number, number, number, number];
      cand[idx] += delta;
      const v = negLL(cand);
      if (v < bestLL) { bestLL = v; best = cand; [w, a, b, g] = cand; improved = true; }
    };
    tryParam(0, stepW); tryParam(0, -stepW);
    tryParam(1, stepA); tryParam(1, -stepA);
    tryParam(2, stepB); tryParam(2, -stepB);
    tryParam(3, stepG); tryParam(3, -stepG);
    if (!improved) {
      stepW *= 0.6; stepA *= 0.6; stepB *= 0.6; stepG *= 0.6;
      if (Math.max(stepA, stepB, stepG) < 1e-4 && stepW < 1e-8) { converged = true; break; }
    }
  }

  // Conditional variance path
  const cv: number[] = new Array(n).fill(sampleVar);
  let s2 = sampleVar;
  for (let t = 1; t < n; t++) {
    const exog = ivOk[t - 1] ?? 0;
    s2 = best[0] + best[1] * eps2[t - 1] + best[2] * s2 + best[3] * exog;
    if (s2 <= 1e-12) s2 = sampleVar;
    cv[t] = s2;
  }
  const factor = Math.sqrt(252);
  return {
    omega: best[0],
    alpha: best[1],
    beta: best[2],
    gamma: best[3],
    persistence: best[1] + best[2],
    logLikelihood: -bestLL,
    converged,
    conditionalVariance: cv,
    conditionalAnnualizedVol: cv.map((v) => Math.sqrt(v) * factor),
  };
}

/** Convenience: aligned IV² (daily variance) from an annualized vol series. */
export function annualizedIvToDailyVariance(ivAnnualized: number[]): number[] {
  return ivAnnualized.map((v) => {
    const d = v / Math.sqrt(252);
    return isFinite(d) && d >= 0 ? d * d : NaN;
  });
}

/** Cross-correlation of ΔVIX at lags k = −maxLag…+maxLag against ΔIV. Positive k = VIX leads. */
export function crossCorrelation(dIv: number[], dVix: number[], maxLag = 3): { lag: number; corr: number }[] {
  const n = Math.min(dIv.length, dVix.length);
  const a = dIv.slice(0, n);
  const b = dVix.slice(0, n);
  const out: { lag: number; corr: number }[] = [];
  for (let k = -maxLag; k <= maxLag; k++) {
    const pairs: [number, number][] = [];
    for (let i = 0; i < n; i++) {
      const j = i + k;
      if (j >= 0 && j < n) pairs.push([a[i], b[j]]);
    }
    if (pairs.length < 20) { out.push({ lag: k, corr: 0 }); continue; }
    const ma = pairs.reduce((s, p) => s + p[0], 0) / pairs.length;
    const mb = pairs.reduce((s, p) => s + p[1], 0) / pairs.length;
    let cov = 0, va = 0, vb = 0;
    for (const [x, y] of pairs) {
      const dx = x - ma, dy = y - mb;
      cov += dx * dy; va += dx * dx; vb += dy * dy;
    }
    out.push({ lag: k, corr: va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : 0 });
  }
  return out;
}

export { normalQuantile, incBeta };
