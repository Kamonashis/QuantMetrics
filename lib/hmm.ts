// Hidden Markov Models — Gaussian, full Baum-Welch (EM) in log-domain.
// Diagonal covariance on standardized features; states ordered calm→crisis
// by the rankFeature mean after fitting (label-switching resolution).
//
// Fits are deterministic given the same input. Fit failure returns null —
// callers must degrade gracefully, never fabricate a regime.

export interface HmmFit {
  k: number;
  featureCount: number;
  converged: boolean;
  iterations: number;
  logLikelihood: number;
  aic: number;
  bic: number;
  /** smoothed posteriors, [T][K], rows sum to 1 */
  posteriors: number[][];
  /** viterbi state sequence [T], ordered calm→crisis */
  viterbi: number[];
  /** stationary distribution [K] */
  stationary: number[];
  /** expected durations in periods, 1/(1 - a_ii) */
  expectedDurations: number[];
  /** fraction of posterior mass assigned to each state */
  occupancy: number[];
  /** transition matrix rows sum to 1 (ordered calm→crisis) */
  transition: number[][];
  /** state means in ORIGINAL units */
  stateMeans: number[][];
  /** state std devs in ORIGINAL units */
  stateStdDevs: number[][];
}

export interface HmmFitOptions {
  /** number of latent states (default 3) */
  k?: number;
  maxIters?: number;
  /** stop when |ΔLL| < tol (default 1e-7) */
  tol?: number;
  /** which feature orders states calm→crisis (default: last, the volatility feature) */
  rankFeature?: number;
}

const LOG_2PI = Math.log(2 * Math.PI);

function logSumExp(arr: number[]): number {
  let max = -Infinity;
  for (const v of arr) if (v > max) max = v;
  if (max === -Infinity) return -Infinity;
  let s = 0;
  for (const v of arr) s += Math.exp(v - max);
  return max + Math.log(s);
}

/** log N(x | mu, sigma) for diagonal features, no vector allocations */
function logEmission(x: number[], mean: number[], sd: number[]): number {
  let s = 0;
  for (let f = 0; f < x.length; f++) {
    const z = (x[f] - mean[f]) / sd[f];
    s += -0.5 * z * z - Math.log(sd[f]) - 0.5 * LOG_2PI;
  }
  return s;
}

/** Standardize columns: returns standardized matrix + means + stds (population). */
function standardize(X: number[][]): { Z: number[][]; mu: number[]; sd: number[] } {
  const n = X.length;
  const f = X[0].length;
  const mu: number[] = new Array(f).fill(0);
  const sd: number[] = new Array(f).fill(0);
  for (let j = 0; j < f; j++) {
    let s = 0;
    for (const row of X) s += row[j];
    mu[j] = s / n;
    let v = 0;
    for (const row of X) {
      const d = row[j] - mu[j];
      v += d * d;
    }
    sd[j] = Math.sqrt(v / n) || 1; // constant feature → sd 1 (no information, no crash)
  }
  const Z = X.map((row) => row.map((v, j) => (v - mu[j]) / sd[j]));
  return { Z, mu, sd };
}

/** k-means++ style initialization over the ranked feature: equal-count sorted segments. */
function initSegments(Z: number[][], k: number, rankFeature: number) {
  const T = Z.length;
  const sorted = [...Z].sort((a, b) => a[rankFeature] - b[rankFeature]);
  const f = Z[0].length;
  const segSize = Math.floor(T / k);
  const means: number[][] = [];
  const stdDevs: number[][] = [];
  for (let q = 0; q < k; q++) {
    const start = q * segSize;
    const end = q === k - 1 ? T : (q + 1) * segSize;
    const m = new Array(f).fill(0);
    for (let t = start; t < end; t++) for (let j = 0; j < f; j++) m[j] += sorted[t][j];
    for (let j = 0; j < f; j++) m[j] /= end - start;
    const sd = new Array(f).fill(0);
    for (let t = start; t < end; t++) for (let j = 0; j < f; j++) {
      const d = sorted[t][j] - m[j];
      sd[j] += d * d;
    }
    for (let j = 0; j < f; j++) sd[j] = Math.sqrt(sd[j] / Math.max(1, end - start)) + 1e-4;
    means.push(m);
    stdDevs.push(sd);
  }
  return { means, stdDevs };
}

function forwardBackward(
  logB: number[][], // [T][K] log emission
  logA: number[][], // [K][K] log transition (row-normalized)
  logPrior: number[] // [K] log prior
) {
  const T = logB.length;
  const K = logB[0].length;
  // Forward pass with scaling
  const alpha: number[][] = [];
  let scaleSum = 0;
  const a0 = new Array(K);
  for (let i = 0; i < K; i++) a0[i] = logPrior[i] + logB[0][i];
  const c0 = logSumExp(a0);
  scaleSum += c0;
  alpha.push(a0.map((v) => v - c0));
  for (let t = 1; t < T; t++) {
    const at: number[] = new Array(K);
    const prev = alpha[t - 1];
    for (let j = 0; j < K; j++) {
      const terms = new Array(K);
      for (let i = 0; i < K; i++) terms[i] = prev[i] + logA[i][j];
      at[j] = logB[t][j] + logSumExp(terms);
    }
    const c = logSumExp(at);
    scaleSum += c;
    alpha.push(at.map((v) => v - c));
  }
  const logLikelihood = scaleSum;

  // Backward pass — same normalization family
  const beta: number[][] = new Array(T);
  beta[T - 1] = new Array(K).fill(0);
  for (let t = T - 2; t >= 0; t--) {
    const bt: number[] = new Array(K);
    const next = beta[t + 1];
    for (let i = 0; i < K; i++) {
      const terms = new Array(K);
      for (let j = 0; j < K; j++) terms[j] = logA[i][j] + logB[t + 1][j] + next[j];
      bt[i] = logSumExp(terms);
    }
    // renormalize to match forward scaling
    const c = logSumExp(bt);
    beta[t] = bt.map((v) => v - c);
  }

  // Smoothed posteriors gamma
  const gamma: number[][] = new Array(T);
  for (let t = 0; t < T; t++) {
    const row = new Array(K);
    for (let i = 0; i < K; i++) row[i] = alpha[t][i] + beta[t][i];
    const c = logSumExp(row);
    gamma[t] = row.map((v) => Math.exp(v - c));
  }

  // xi terms for transition sufficient statistics
  const xiRowSums: number[][] = Array.from({ length: K }, () => new Array(K).fill(0));
  for (let t = 0; t < T - 1; t++) {
    const terms: number[][] = [];
    let tot = -Infinity;
    for (let i = 0; i < K; i++) {
      terms.push(new Array(K));
      for (let j = 0; j < K; j++) {
        const v = alpha[t][i] + logA[i][j] + logB[t + 1][j] + beta[t + 1][j];
        terms[i][j] = v;
        if (v > tot) tot = v;
      }
    }
    let sum = 0;
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < K; j++) terms[i][j] = Math.exp(terms[i][j] - tot);
      for (let j = 0; j < K; j++) sum += terms[i][j];
    }
    if (sum > 0) {
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) xiRowSums[i][j] += terms[i][j] / sum;
    }
  }

  return { logLikelihood, gamma, xiRowSums, alpha };
}

export function fitGaussianHmm(features: number[][], opts: HmmFitOptions = {}): HmmFit | null {
  const steps = fitGaussianHmmSteps(features, opts);
  let r = steps.next();
  while (!r.done) r = steps.next();
  return r.value;
}

/**
 * Same fit as `fitGaussianHmm`, but hands the main thread back (time-sliced,
 * ~12ms) between EM iterations. Identical math and result.
 */
export async function fitGaussianHmmAsync(
  features: number[][],
  opts: HmmFitOptions,
  yieldControl: () => Promise<void>
): Promise<HmmFit | null> {
  const nowMs = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const steps = fitGaussianHmmSteps(features, opts);
  let sliceStart = nowMs();
  let r = steps.next();
  while (!r.done) {
    if (nowMs() - sliceStart > 12) {
      await yieldControl();
      sliceStart = nowMs();
    }
    r = steps.next();
  }
  return r.value;
}

/** EM fit as a generator: yields once per EM iteration so callers may slice it. */
function* fitGaussianHmmSteps(features: number[][], opts: HmmFitOptions = {}): Generator<void, HmmFit | null, void> {
  const K = opts.k ?? 3;
  const maxIters = opts.maxIters ?? 200;
  const tol = opts.tol ?? 1e-7;
  const rankFeature = opts.rankFeature ?? Math.max(0, (features?.[0]?.length ?? 1) - 1);

  if (!features || features.length < 30) return null;
  const T = features.length;
  const f = features[0].length;
  for (const row of features) for (const v of row) if (!Number.isFinite(v)) return null;
  if (K < 2) return null;

  const { Z, mu, sd } = standardize(features);
  let { means, stdDevs } = initSegments(Z, K, rankFeature);

  // Persistent probability mass on staying — vol regimes are sticky
  let transition: number[][] = Array.from({ length: K }, (_, i) => {
    const row = new Array(K).fill(0.05 / (K - 1));
    row[i] = 0.9;
    return row;
  });
  let prior = new Array(K).fill(1 / K);

  const rowSums = (m: number[][]) => m.map((row) => {
    const s = row.reduce((a, b) => a + b, 0);
    return s > 0 ? row.map((v) => v / s) : row.map(() => 1 / row.length);
  });
  const logNorm = (m: number[][]) => m.map((row) => row.map((v) => Math.log(Math.max(v, 1e-300))));

  let prevLL = -Infinity;
  let converged = false;
  let itersUsed = 0;
  let logB: number[][] = [];
  let logA: number[][] = [];
  let logPrior: number[] = [];

  for (let iter = 0; iter < maxIters; iter++) {
    itersUsed = iter + 1;
    logB = Z.map((row) => means.map((m, k) => logEmission(row, m, stdDevs[k])));
    logA = logNorm(rowSums(transition));
    logPrior = prior.map((v) => Math.log(Math.max(v, 1e-300)));

    const { logLikelihood, gamma, xiRowSums } = forwardBackward(logB, logA, logPrior);

    // Check convergence against previous iteration
    if (Number.isFinite(prevLL) && Math.abs(logLikelihood - prevLL) < tol) {
      converged = true;
      prevLL = logLikelihood;
      break;
    }
    prevLL = logLikelihood;

    // M-step
    prior = gamma[0].slice();

    const gSums = new Array(K).fill(0);
    for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) gSums[k] += gamma[t][k];

    const prevTransition = transition.map((row) => row.slice());
    transition = Array.from({ length: K }, (_, i) => {
      const row = xiRowSums[i].slice();
      let s = row.reduce((a, b) => a + b, 0);
      if (s === 0) row.fill(1 / K);
      else for (let j = 0; j < K; j++) row[j] /= s;
      // keep a trace of inertia — helps weakly separated states stabilize
      for (let j = 0; j < K; j++) row[j] = 0.9 * row[j] + 0.1 * prevTransition[i][j];
      s = row.reduce((a, b) => a + b, 0);
      for (let j = 0; j < K; j++) row[j] /= s;
      return row;
    });

    for (let k = 0; k < K; k++) {
      if (gSums[k] < 1e-10) {
        means[k] = means[k].slice(); stdDevs[k] = stdDevs[k].map(() => 1);
        continue;
      }
      const m = new Array(f).fill(0);
      for (let t = 0; t < T; t++) for (let j = 0; j < f; j++) m[j] += gamma[t][k] * Z[t][j];
      for (let j = 0; j < f; j++) m[j] /= gSums[k];
      const sd = new Array(f);
      for (let j = 0; j < f; j++) {
        let v = 0;
        for (let t = 0; t < T; t++) {
          const d = Z[t][j] - m[j];
          v += gamma[t][k] * d * d;
        }
        sd[j] = Math.sqrt(Math.max(v / gSums[k], 1e-6));
      }
      means[k] = m;
      stdDevs[k] = sd;
    }
    yield;
  }

  // Final forward-backward on converged parameters
  logB = Z.map((row) => means.map((m, k) => logEmission(row, m, stdDevs[k])));
  const { gamma, logLikelihood } = forwardBackward(logB, logNorm(rowSums(transition)), prior.map((v) => Math.log(Math.max(v, 1e-300))));

  // Viterbi (log-domain)
  const viterbi: number[] = new Array(T);
  const delta: number[][] = [];
  const psi: number[][] = [];
  const d0 = new Array(K);
  for (let i = 0; i < K; i++) d0[i] = Math.log(Math.max(prior[i], 1e-300)) + logB[0][i];
  delta.push(d0);
  psi.push(new Array(K).fill(-1));
  for (let t = 1; t < T; t++) {
    const dt: number[] = new Array(K);
    const pt: number[] = new Array(K);
    for (let j = 0; j < K; j++) {
      let best = -Infinity;
      let bestI = 0;
      for (let i = 0; i < K; i++) {
        const v = delta[t - 1][i] + Math.log(Math.max(transition[i][j], 1e-300));
        if (v > best) { best = v; bestI = i; }
      }
      dt[j] = logB[t][j] + best;
      pt[j] = bestI;
    }
    delta.push(dt);
    psi.push(pt);
  }
  let bestLast = -Infinity;
  let lastState = 0;
  for (let k = 0; k < K; k++) {
    if (delta[T - 1][k] > bestLast) { bestLast = delta[T - 1][k]; lastState = k; }
  }
  viterbi[T - 1] = lastState;
  for (let t = T - 2; t >= 0; t--) {
    viterbi[t] = psi[t + 1][viterbi[t + 1]];
    if (viterbi[t] < 0) viterbi[t] = 0;
  }

  // Stationary distribution via power iteration
  let stationary = prior.slice();
  for (let it = 0; it < 200; it++) {
    const next = new Array(K).fill(0);
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) next[j] += stationary[i] * transition[i][j];
    const s = next.reduce((a, b) => a + b, 0);
    for (let j = 0; j < K; j++) next[j] /= s;
    const deltaPi = next.reduce((a, b, idx) => a + Math.abs(b - stationary[idx]), 0);
    stationary = next;
    if (deltaPi < 1e-12) break;
  }

  // State means/sds in original units
  const meansOrig = means.map((m) => m.map((v, j) => v * sd[j] + mu[j]));
  const stdsOrig = stdDevs.map((sv) => sv.map((v, j) => v * sd[j]));

  // Label switching resolution: rank calm→crisis by rankFeature mean
  const order = [...Array(K).keys()].sort((a, b) => meansOrig[a][rankFeature] - meansOrig[b][rankFeature]);
  const remap = (arr: number[]) => order.map((oldIdx) => arr[oldIdx]);
  const remap2d = (arr: number[][]) => order.map((oldIdx) => arr[oldIdx]);
  const remapMat = (m: number[][]) => {
    const out: number[][] = [];
    for (const r of order) {
      out.push(order.map((c) => m[r][c]));
    }
    return out;
  };
  const transitionOrdered = remapMat(transition);
  const posteriorsOrdered = gamma.map((row) => remap(row));
  const viterbiOrdered = viterbi.map((v) => order.indexOf(v));
  const stationaryOrdered = remap(stationary);
  const meansOrdered = remap2d(meansOrig);
  const stdsOrdered = remap2d(stdsOrig);

  const occupancy = new Array(K).fill(0);
  for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) occupancy[k] += posteriorsOrdered[t][k];
  for (let k = 0; k < K; k++) occupancy[k] /= T;

  const expectedDurations = transitionOrdered.map((row, k) => 1 / Math.max(1e-12, 1 - row[k]));

  // Diagnostics
  const numParams = (K - 1) + K * (K - 1) + 2 * K * f;
  const aic = 2 * numParams - 2 * logLikelihood;
  const bic = numParams * Math.log(T) - 2 * logLikelihood;

  if (!Number.isFinite(logLikelihood)) return null;

  return {
    k: K,
    featureCount: f,
    converged,
    iterations: itersUsed,
    logLikelihood,
    aic,
    bic,
    posteriors: posteriorsOrdered,
    viterbi: viterbiOrdered,
    stationary: stationaryOrdered,
    expectedDurations,
    occupancy,
    transition: transitionOrdered,
    stateMeans: meansOrdered,
    stateStdDevs: stdsOrdered,
  };
}

/** Convenience: posterior probability of being in the "crisis" (last) state at t. */
export function crisisProb(fit: HmmFit, t: number): number {
  const K = fit.k;
  return fit.posteriors[t]?.[K - 1] ?? 0;
}

export type HmmRegimeLabel = 'Calm' | 'Watchful' | 'Crisis';
export const HMM_REGIME_LABELS: readonly HmmRegimeLabel[] = ['Calm', 'Watchful', 'Crisis'] as const;

export interface CausalHmmSummary {
  /** Dominant current causal regime state (0 = Calm, 1 = Watchful, 2 = Crisis) */
  dominantState: 0 | 1 | 2;
  dominantLabel: HmmRegimeLabel;
  /** Strictly causal forward-filtered probabilities [P(Calm), P(Watchful), P(Crisis)] at latest bar */
  probs: [number, number, number];
  /** Number of training-prefix observations used to fit HMM parameters (zero holdout leakage) */
  trainSampleCount: number;
  converged: boolean;
}

export interface CausalHmmResult {
  fit: HmmFit;
  /** Strictly causal forward-filtered posteriors [T][3], each row sums to 1 */
  filteredPosteriors: number[][];
  summary: CausalHmmSummary;
}

/**
 * Builds canonical 3-column strictly causal volatility-regime features
 * [r_t, |r_t|, log(varProxy_t)] matching QuantMetrics' HMM convention.
 * When OHLC variance proxies are not supplied, blends r_t^2 with a causal
 * short-memory EWMA variance so flat zero-return sessions do not produce log(0) outliers.
 */
export function buildCausalReturnVolFeatures(
  returns: number[],
  varProxies?: number[]
): number[][] {
  const n = returns.length;
  const out: number[][] = new Array(n);
  const lambda = 0.72;
  let ewmaVar = n > 0 ? Math.max(1e-6, (returns[0] || 0) * (returns[0] || 0)) : 1e-4;

  for (let t = 0; t < n; t++) {
    const r = Number.isFinite(returns[t]) ? returns[t] : 0;
    const r2 = r * r;
    ewmaVar = t === 0 ? Math.max(1e-6, r2) : lambda * ewmaVar + (1 - lambda) * r2;
    const supplied = varProxies?.[t];
    const v =
      supplied !== undefined && Number.isFinite(supplied) && supplied > 0
        ? Math.max(1e-8, supplied)
        : Math.max(1e-8, 0.5 * r2 + 0.5 * ewmaVar);
    out[t] = [r, Math.abs(r), Math.log(v)];
  }
  return out;
}

/**
 * Pure log-domain forward filter (alpha_t recursion ONLY, zero backward pass).
 * Evaluates P(S_t = k | x_{0..t}; theta) in original feature units using fixed
 * parameters fitted on a training prefix — guarantees zero look-ahead bias.
 */
export function filterGaussianHmmForward(
  features: number[][],
  transition: number[][],
  stateMeans: number[][],
  stateStdDevs: number[][],
  initialPrior: number[]
): number[][] {
  const T = features.length;
  const K = stateMeans.length;
  if (T === 0 || K === 0) return [];

  const safeSd = stateStdDevs.map((row) => row.map((v) => Math.max(1e-6, Number.isFinite(v) ? v : 1)));
  const logA = transition.map((row) => {
    const sum = row.reduce((a, b) => a + (Number.isFinite(b) && b > 0 ? b : 0), 0);
    return row.map((v) => {
      const p = sum > 0 ? (Number.isFinite(v) && v > 0 ? v / sum : 0) : 1 / K;
      return Math.log(Math.max(p, 1e-300));
    });
  });
  const priorSum = initialPrior.reduce((a, b) => a + (Number.isFinite(b) && b > 0 ? b : 0), 0);
  const logPrior = initialPrior.map((v) => {
    const p = priorSum > 0 ? (Number.isFinite(v) && v > 0 ? v / priorSum : 0) : 1 / K;
    return Math.log(Math.max(p, 1e-300));
  });

  const filtered: number[][] = new Array(T);
  let prevLogAlpha = new Array<number>(K);

  // t = 0
  {
    const a0 = new Array<number>(K);
    for (let k = 0; k < K; k++) {
      a0[k] = logPrior[k] + logEmission(features[0], stateMeans[k], safeSd[k]);
    }
    const c0 = logSumExp(a0);
    for (let k = 0; k < K; k++) prevLogAlpha[k] = a0[k] - c0;
    filtered[0] = prevLogAlpha.map((v) => Math.exp(v));
  }

  // t = 1 .. T - 1 (strictly causal recursion)
  const terms = new Array<number>(K);
  for (let t = 1; t < T; t++) {
    const at = new Array<number>(K);
    const x = features[t];
    for (let j = 0; j < K; j++) {
      for (let i = 0; i < K; i++) {
        terms[i] = prevLogAlpha[i] + logA[i][j];
      }
      at[j] = logEmission(x, stateMeans[j], safeSd[j]) + logSumExp(terms);
    }
    const ct = logSumExp(at);
    const nextLogAlpha = new Array<number>(K);
    for (let j = 0; j < K; j++) nextLogAlpha[j] = at[j] - ct;
    filtered[t] = nextLogAlpha.map((v) => Math.exp(v));
    prevLogAlpha = nextLogAlpha;
  }

  return filtered;
}

/**
 * Fits a 3-state Gaussian HMM strictly on the training prefix features[0 .. trainEndIdx - 1]
 * (states ordered Calm → Watchful → Crisis by rankFeature mean) and computes strictly causal
 * forward-filtered regime probabilities across all rows 0 .. features.length - 1.
 *
 * Returns null if the training slice is too short (< 30 bars) or fails to fit cleanly.
 */
export function fitAndFilterCausalHmm(
  features: number[][],
  trainEndIdx: number,
  opts: HmmFitOptions = {}
): CausalHmmResult | null {
  if (!features || features.length < 30) return null;
  const end = Math.min(features.length, Math.max(0, Math.floor(trainEndIdx)));
  if (end < 30) return null;

  const trainSlice = features.slice(0, end);
  const rankFeature = opts.rankFeature ?? Math.max(0, (features[0]?.length ?? 1) - 1);
  const fit = fitGaussianHmm(trainSlice, {
    ...opts,
    k: 3,
    rankFeature,
  });
  return finishCausalHmm(features, end, fit);
}

/** Time-sliced twin of `fitAndFilterCausalHmm` (identical result). */
export async function fitAndFilterCausalHmmAsync(
  features: number[][],
  trainEndIdx: number,
  yieldControl: () => Promise<void>,
  opts: HmmFitOptions = {}
): Promise<CausalHmmResult | null> {
  if (!features || features.length < 30) return null;
  const end = Math.min(features.length, Math.max(0, Math.floor(trainEndIdx)));
  if (end < 30) return null;
  const trainSlice = features.slice(0, end);
  const rankFeature = opts.rankFeature ?? Math.max(0, (features[0]?.length ?? 1) - 1);
  const fit = await fitGaussianHmmAsync(trainSlice, { ...opts, k: 3, rankFeature }, yieldControl);
  return finishCausalHmm(features, end, fit);
}

function finishCausalHmm(features: number[][], end: number, fit: HmmFit | null): CausalHmmResult | null {
  if (!fit || fit.k !== 3) return null;

  const filteredPosteriors = filterGaussianHmmForward(
    features,
    fit.transition,
    fit.stateMeans,
    fit.stateStdDevs,
    fit.stationary
  );
  if (filteredPosteriors.length !== features.length) return null;

  const lastRow = filteredPosteriors[filteredPosteriors.length - 1] ?? [1 / 3, 1 / 3, 1 / 3];
  const probs: [number, number, number] = [
    lastRow[0] ?? 1 / 3,
    lastRow[1] ?? 1 / 3,
    lastRow[2] ?? 1 / 3,
  ];
  let dominantState: 0 | 1 | 2 = 0;
  if (probs[1] >= probs[0] && probs[1] >= probs[2]) dominantState = 1;
  else if (probs[2] >= probs[0] && probs[2] >= probs[1]) dominantState = 2;

  return {
    fit,
    filteredPosteriors,
    summary: {
      dominantState,
      dominantLabel: HMM_REGIME_LABELS[dominantState],
      probs,
      trainSampleCount: end,
      converged: fit.converged,
    },
  };
}

