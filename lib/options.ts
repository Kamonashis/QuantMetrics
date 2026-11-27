// Options Engine: Black-Scholes pricing, Newton-Raphson implied volatility,
// forward extraction via put-call parity, chain cleaning, SVI smile fitting,
// model-free implied variance (Carr–Madan / VIX methodology) and
// Breeden–Litzenberger risk-neutral densities.
//
// All functions are pure and dependency-free. Time to expiry (T) is in years,
// volatilities are decimals (0.25 = 25%), variances are per-day unless suffixed.
// Convention: European BSM approximation with continuous dividend yield q —
// equity options are American; treat every IV number as an approximation.

import { densityMean, normalizeDensity } from './rnd';

// ---------------------------------------------------------------------------
// Basic distribution helpers (Abramowitz & Stegun approximations, as used
// across the codebase)
// ---------------------------------------------------------------------------

export function normPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

export function normCdf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  // erf approximation (A&S 7.1.26)
  const t = 1 / (1 + 0.3275911 * ax);
  const erf =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-ax * ax);
  return 0.5 * (1 + sign * erf);
}

/** Regularized incomplete beta I_x(a, b) — continued fraction (Numerical Recipes). */
export function incBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const bt = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lbeta);

  const betacf = (x: number, a: number, b: number): number => {
    const MAXIT = 200;
    const EPS = 3e-14;
    const FPMIN = 1e-300;
    const qab = a + b;
    const qap = a + 1;
    const qam = a - 1;
    let c = 1;
    let d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1.0) < EPS) break;
    }
    return h;
  };

  if (x < (a + 1) / (a + b + 2)) {
    return (bt * betacf(x, a, b)) / a;
  }
  return 1 - (bt * betacf(1 - x, b, a)) / b;
}

function lgamma(z: number): number {
  // Lanczos approximation
  const cof = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.001208650973866179, -0.000005395239384953,
  ];
  const x = z - 1;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) {
    ser += cof[j] / (x + j + 1);
  }
  return -tmp + Math.log((2.5066282746310005 * ser) / z);
}

/** Two-sided p-value for F ~ F(d1, d2) at statistic f (upper tail ×2 for two-sided). */
export function fSurvival(f: number, d1: number, d2: number): number {
  if (f <= 0) return 1;
  const x = (d1 * f) / (d1 * f + d2);
  return Math.max(0, Math.min(1, 1 - incBeta(x, d1 / 2, d2 / 2)));
}

/** Standard normal quantile — re-export from quant-engine to avoid duplication drift; local copy kept for standalone use. */
export function normalQuantile(p: number): number {
  if (p <= 0) return -4.5;
  if (p >= 1) return 4.5;
  const a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637];
  const b = [-8.4735109309, 23.08336743743, -21.06224101826, 3.13082909833];
  const c = [
    0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
    0.02764388103308636, 0.0038405729373609, 0.0003951804142257,
    0.0000321767881768, 0.0000002888167364, 0.0000003960315187,
  ];
  const y = p - 0.5;
  if (Math.abs(y) < 0.42) {
    const r = y * y;
    const x =
      (y * ((((a[3] * r + a[2]) * r + a[1]) * r + a[0]) / ((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1)));
    return x;
  }
  let r = p;
  if (y > 0) r = 1 - p;
  r = Math.log(-Math.log(r));
  let x = c[0];
  for (let i = 1; i < 9; i++) x += c[i] * Math.pow(r, i);
  return y < 0 ? -x : x;
}

// ---------------------------------------------------------------------------
// Black–Scholes with continuous dividend yield
// ---------------------------------------------------------------------------

export interface BsmInput {
  spot: number;
  strike: number;
  t: number;         // years
  rate: number;      // continuously-compounded risk-free
  dividendYield: number; // continuous
  vol: number;       // decimal
  type: 'call' | 'put';
}

export function d1d2(S: number, K: number, T: number, r: number, q: number, sigma: number) {
  const sigSqrtT = sigma * Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigSqrtT;
  return { d1, d2: d1 - sigSqrtT };
}

export function bsPrice(input: BsmInput): number {
  const { spot: S, strike: K, t: T, rate: r, dividendYield: q, vol: sigma, type } = input;
  if (T <= 0) return type === 'call' ? Math.max(0, S - K) : Math.max(0, K - S);
  if (sigma <= 0) {
    const fwd = S * Math.exp(-q * T);
    const pvK = K * Math.exp(-r * T);
    return type === 'call' ? Math.max(0, fwd - pvK) : Math.max(0, pvK - fwd);
  }
  const { d1, d2 } = d1d2(S, K, T, r, q, sigma);
  const discS = S * Math.exp(-q * T);
  const discK = K * Math.exp(-r * T);
  if (type === 'call') return discS * normCdf(d1) - discK * normCdf(d2);
  return discK * normCdf(-d2) - discS * normCdf(-d1);
}

export function bsVega(S: number, K: number, T: number, r: number, q: number, sigma: number): number {
  if (T <= 0 || sigma <= 0) return 0;
  const { d1 } = d1d2(S, K, T, r, q, sigma);
  return S * Math.exp(-q * T) * Math.sqrt(T) * normPdf(d1);
}

export function bsDelta(S: number, K: number, T: number, r: number, q: number, sigma: number, type: 'call' | 'put'): number {
  if (T <= 0 || sigma <= 0) {
    const fwd = S * Math.exp(-q * T);
    if (type === 'call') return fwd > K * Math.exp(-r * T) ? Math.exp(-q * T) : 0;
    return fwd < K * Math.exp(-r * T) ? -Math.exp(-q * T) : 0;
  }
  const { d1 } = d1d2(S, K, T, r, q, sigma);
  return type === 'call' ? Math.exp(-q * T) * normCdf(d1) : Math.exp(-q * T) * (normCdf(d1) - 1);
}

export function bsGamma(S: number, K: number, T: number, r: number, q: number, sigma: number): number {
  if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) return 0;
  const { d1 } = d1d2(S, K, T, r, q, sigma);
  return (Math.exp(-q * T) * normPdf(d1)) / (S * sigma * Math.sqrt(T));
}

/**
 * European BSM Vanna (∂Δ/∂σ = ∂Vega/∂S), identical for calls and puts.
 * Units: change in option delta (shares per share) per +1.00 absolute change
 * in annualized decimal volatility (multiply by 0.01 for +1 vol pt).
 */
export function bsVanna(S: number, K: number, T: number, r: number, q: number, sigma: number): number {
  if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) return 0;
  const { d1, d2 } = d1d2(S, K, T, r, q, sigma);
  return (-Math.exp(-q * T) * normPdf(d1) * d2) / sigma;
}

/**
 * European BSM Charm (delta decay with calendar time passage: ∂Δ/∂t = -∂Δ/∂T).
 * Units: change in option delta per calendar year (yr⁻¹). Divide by 365 (or use
 * `bsCharmPerDay`) for delta change per calendar day.
 */
export function bsCharm(
  S: number,
  K: number,
  T: number,
  r: number,
  q: number,
  sigma: number,
  type: 'call' | 'put'
): number {
  if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) return 0;
  const sqrtT = Math.sqrt(T);
  const { d1, d2 } = d1d2(S, K, T, r, q, sigma);
  const discQ = Math.exp(-q * T);
  const common = (discQ * normPdf(d1) * (2 * (r - q) * T - d2 * sigma * sqrtT)) / (2 * T * sigma * sqrtT);
  return type === 'call' ? q * discQ * normCdf(d1) - common : -q * discQ * normCdf(-d1) - common;
}

/** Convenience helper: European BSM Charm expressed per calendar day (`bsCharm / 365`). */
export function bsCharmPerDay(
  S: number,
  K: number,
  T: number,
  r: number,
  q: number,
  sigma: number,
  type: 'call' | 'put'
): number {
  return bsCharm(S, K, T, r, q, sigma, type) / 365;
}

/**
 * European BSM Speed (∂Γ/∂S), identical for calls and puts.
 * Units: rate of change of gamma per unit change in spot price.
 */
export function bsSpeed(S: number, K: number, T: number, r: number, q: number, sigma: number): number {
  if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) return 0;
  const sqrtT = Math.sqrt(T);
  const { d1 } = d1d2(S, K, T, r, q, sigma);
  const gamma = (Math.exp(-q * T) * normPdf(d1)) / (S * sigma * sqrtT);
  return -(gamma / S) * (d1 / (sigma * sqrtT) + 1);
}

/**
 * Sanitize a raw option implied volatility for GEX / Greeks calculation.
 * Rejects non-finite, sub-floor (≤ minIv), extreme (> maxIv), and wide/zero-bid
 * outlier quotes relative to `fallbackIv`, returning a clean positive decimal IV.
 */
export function sanitizeIv(
  rawIv: number | null | undefined,
  fallbackIv: number,
  opts: {
    minIv?: number;
    maxIv?: number;
    bid?: number;
    ask?: number;
    maxRelSpread?: number;
  } = {}
): number {
  const minIv = opts.minIv ?? 0.02;
  const maxIv = opts.maxIv ?? 3.0;
  const maxRelSpread = opts.maxRelSpread ?? 0.5;
  const fallback =
    typeof fallbackIv === 'number' && Number.isFinite(fallbackIv) && fallbackIv > minIv && fallbackIv <= maxIv
      ? fallbackIv
      : 0.2;

  if (typeof rawIv !== 'number' || !Number.isFinite(rawIv) || rawIv <= minIv || rawIv > maxIv) {
    return fallback;
  }

  const bid = opts.bid ?? NaN;
  const ask = opts.ask ?? NaN;
  if (Number.isFinite(bid) || Number.isFinite(ask)) {
    const mid = bid > 0 && ask >= bid ? (bid + ask) / 2 : 0;
    const relSpread = mid > 0 ? (ask - bid) / mid : Infinity;
    const badQuote = !(bid > 0 && ask >= bid) || relSpread > maxRelSpread;
    // Wide/zero-bid wings on Yahoo frequently carry stale or extreme (>150-300%) IV prints.
    const outlierCap = Math.min(maxIv, Math.max(1.5, fallback * 4));
    if (badQuote && (rawIv > outlierCap || rawIv < fallback * 0.2)) {
      return fallback;
    }
  }
  return rawIv;
}

/**
 * Implied volatility from an option price. Newton from a BS-vega warm start,
 * bisection fallback. Returns null instead of a garbage number.
 */
export function impliedVol(
  price: number,
  S: number,
  K: number,
  T: number,
  r: number,
  q: number,
  type: 'call' | 'put'
): number | null {
  if (T <= 0 || S <= 0 || K <= 0 || price <= 0) return null;

  // No-arbitrage price bounds — outside these, IV does not exist.
  const undiscounted = S * Math.exp(-q * T);
  const pv = K * Math.exp(-r * T);
  const lower = type === 'call' ? Math.max(0, undiscounted - pv) : Math.max(0, pv - undiscounted);
  const upper = type === 'call' ? undiscounted : pv;
  if (price < lower - 1e-10 || price > upper + 1e-10) return null;

  // Warm start from Brenner-Subrahmanyam style approximation
  let sigma = Math.sqrt((2 * Math.PI) / T) * (price / S);
  if (!isFinite(sigma) || sigma < 0.005 || sigma > 3) sigma = 0.25;

  const MAX_ITER = 64;
  for (let i = 0; i < MAX_ITER; i++) {
    const err = bsPrice({ spot: S, strike: K, t: T, rate: r, dividendYield: q, vol: sigma, type }) - price;
    if (Math.abs(err) < 1e-9) return sigma;
    const vega = bsVega(S, K, T, r, q, sigma);
    if (vega < 1e-10) break; // fall through to bisection
    sigma = sigma - err / vega;
    if (!isFinite(sigma) || sigma <= 0.0005) sigma = 0.0005;
    if (sigma > 5) sigma = 5;
  }

  // Bisection fallback
  let lo = 1e-4;
  let hi = 5;
  let flo = bsPrice({ spot: S, strike: K, t: T, rate: r, dividendYield: q, vol: lo, type }) - price;
  const fhi = bsPrice({ spot: S, strike: K, t: T, rate: r, dividendYield: q, vol: hi, type }) - price;
  if (flo * fhi > 0) return null;
  for (let i = 0; i < 128; i++) {
    const mid = 0.5 * (lo + hi);
    const fm = bsPrice({ spot: S, strike: K, t: T, rate: r, dividendYield: q, vol: mid, type }) - price;
    if (Math.abs(fm) < 1e-10) return mid;
    if (flo * fm <= 0) {
      hi = mid;
    } else {
      lo = mid;
      flo = fm;
    }
  }
  return 0.5 * (lo + hi);
}

// ---------------------------------------------------------------------------
// Chain types & cleaning
// ---------------------------------------------------------------------------

export interface OptionContract {
  strike: number;
  bid: number;
  ask: number;
  lastPrice: number;
  openInterest: number;
  volume: number;
  impliedVolatility: number; // Yahoo's field — kept for cross-check only
  inTheMoney: boolean;
}

export interface CleanedContract extends OptionContract {
  mid: number;
  relSpread: number;
  iv: number;       // computed from mid
  vega: number;
  k: number;        // log-moneyness ln(K / F)
  w: number;        // total variance σ²T
  weight: number;   // fit weight (vega² × spread precision)
}

export interface CleanStats {
  used: number;
  total: number;
  medianRelSpread: number | null;
  droppedZeroBid: number;
  droppedWideSpread: number;
  droppedItm: number;
  droppedOi: number;
  droppedIvInvalid: number;
}

export interface ForwardParity {
  forward: number;
  strike: number;
  putCallDiff: number;
}

/** Extract the implied forward via put–call parity; median across paired strikes for robustness. */
export function impliedForward(
  calls: OptionContract[],
  puts: OptionContract[],
  T: number,
  r: number
): ForwardParity | null {
  if (T <= 0) return null;
  const disc = Math.exp(r * T);
  const putByStrike = new Map(puts.map((p) => [p.strike, p] as const));
  const forwards: ForwardParity[] = [];
  for (const c of calls) {
    const p = putByStrike.get(c.strike);
    if (!p) continue;
    const cmid = (c.bid + c.ask) / 2;
    const pmid = (p.bid + p.ask) / 2;
    if (cmid <= 0 || pmid <= 0) continue;
    const strikeFwd = c.strike + disc * (cmid - pmid);
    forwards.push({ forward: strikeFwd, strike: c.strike, putCallDiff: Math.abs(cmid - pmid) });
  }
  if (forwards.length === 0) return null;
  // Prefer the tightest-|C−P| pair (parity holds best at-the-money), then broaden to median.
  forwards.sort((a, b) => a.putCallDiff - b.putCallDiff);
  const tight = forwards[0];
  const sampleN = Math.max(1, Math.min(5, forwards.length));
  const tightFwds = forwards.slice(0, sampleN).map((f) => f.forward).sort((x, y) => x - y);
  const median = tightFwds[Math.floor(tightFwds.length / 2)];
  return {
    forward: median,
    strike: tight.strike,
    putCallDiff: tight.putCallDiff,
  };
}

/**
 * The filter that makes or breaks the feature:
 *  - drop zero-bid contracts
 *  - drop spread/mid > maxRelSpread
 *  - keep OTM only (calls above F, puts below F)
 *  - require OI ≥ minOi
 *  - computed IV must be finite and inside (0.01, 3)
 * Returns used contracts with fit weights, plus rejection counters.
 */
export function cleanChain(
  calls: OptionContract[],
  puts: OptionContract[],
  spot: number,
  forward: number,
  T: number,
  r: number,
  q: number,
  options: { maxRelSpread?: number; minOi?: number } = {}
): { calls: CleanedContract[]; puts: CleanedContract[]; stats: CleanStats } {
  const maxRelSpread = options.maxRelSpread ?? 0.25;
  const minOi = options.minOi ?? 10;

  const cleanSide = (contracts: OptionContract[], type: 'call' | 'put'): CleanedContract[] => {
    const out: CleanedContract[] = [];
    for (const c of contracts) {
      if (!(c.bid > 0) || !(c.ask >= c.bid)) continue;
      const mid = (c.bid + c.ask) / 2;
      if (mid <= 0) continue;
      const relSpread = (c.ask - c.bid) / mid;
      if (relSpread > maxRelSpread) continue;
      if (c.openInterest < minOi) continue;
      // OTM only
      if (type === 'call' && c.strike < forward) continue;
      if (type === 'put' && c.strike > forward) continue;

      const iv = impliedVol(mid, spot, c.strike, T, r, q, type);
      if (iv === null || !isFinite(iv) || iv < 0.01 || iv > 3) continue;
      const vega = bsVega(spot, c.strike, T, r, q, iv);
      const k = Math.log(c.strike / forward);
      const w = iv * iv * T;
      // Weight ∝ vega² × (1 / spreadRel²) — ATM liquid strikes dominate the fit.
      const weight = Math.max(1e-8, (vega * vega) / Math.pow(relSpread + 1e-3, 2));
      out.push({ ...c, mid, relSpread, iv, vega, k, w, weight });
    }
    return out;
  };

  const callsUsed = cleanSide(calls, 'call');
  const putsUsed = cleanSide(puts, 'put');

  const rejections = (all: OptionContract[], type: 'call' | 'put') => {
    let zeroBid = 0, wide = 0, itm = 0, oi = 0;
    for (const c of all) {
      if (!(c.bid > 0) || !(c.ask >= c.bid)) { zeroBid++; continue; }
      const mid = (c.bid + c.ask) / 2;
      const relSpread = mid > 0 ? (c.ask - c.bid) / mid : Infinity;
      if (relSpread > maxRelSpread) { wide++; continue; }
      if (c.openInterest < minOi) { oi++; continue; }
      if (type === 'call' && c.strike < forward) { itm++; continue; }
      if (type === 'put' && c.strike > forward) { itm++; continue; }
      // survived all pre-filters — may still fail the IV solve (counted as residual)
    }
    return { zeroBid, wide, oi, itm };
  };

  const rc = rejections(calls, 'call');
  const rp = rejections(puts, 'put');

  // Residual bucket: contracts that passed every pre-filter but failed the IV solve/range check.
  const otherRejections = rc.zeroBid + rc.wide + rc.oi + rc.itm + rp.zeroBid + rp.wide + rp.oi + rp.itm;
  const usedCount = callsUsed.length + putsUsed.length;
  const totalCount = calls.length + puts.length;
  const ivSolveFail = Math.max(0, totalCount - usedCount - otherRejections);

  const allUsed = [...callsUsed, ...putsUsed];
  const spreads = allUsed.map((c) => c.relSpread).sort((a, b) => a - b);
  const medianRelSpread =
    spreads.length > 0 ? spreads[Math.floor(spreads.length / 2)] : null;

  return {
    calls: callsUsed,
    puts: putsUsed,
    stats: {
      used: allUsed.length,
      total: calls.length + puts.length,
      medianRelSpread,
      droppedZeroBid: rc.zeroBid + rp.zeroBid,
      droppedWideSpread: rc.wide + rp.wide,
      droppedItm: rc.itm + rp.itm,
      droppedOi: rc.oi + rp.oi,
      droppedIvInvalid: ivSolveFail,
    },
  };
}

// ---------------------------------------------------------------------------
// SVI smile fit (raw SVI): w(k) = a + b·(ρ(k−m) + √((k−m)² + σ²))
// Weighted least squares via Nelder–Mead; arbitrage guard via penalty.
// ---------------------------------------------------------------------------

export interface SviParams {
  a: number;
  b: number;
  rho: number;
  m: number;
  sigma: number;
}

export interface SviFit extends SviParams {
  rmseW: number;       // RMSE in total variance
  rmseIv: number;      // RMSE in vol (decimal)
  nUsed: number;
  converged: boolean;
  arbitrageFlags: string[];
}

export function sviTotalVariance(p: SviParams, k: number): number {
  const dm = k - p.m;
  return p.a + p.b * (p.rho * dm + Math.sqrt(dm * dm + p.sigma * p.sigma));
}

export function sviObjective(p: SviParams, pts: CleanedContract[]): number {
  let err = 0;
  // Arbitrage penalties (hard-ish constraints)
  if (p.b < 0) err += 1e8 * p.b * p.b;
  const rhoAbs = Math.abs(p.rho);
  if (rhoAbs >= 0.95) err += 1e8 * (rhoAbs - 0.95) * (rhoAbs - 0.95);
  const lowerW = p.a + p.b * p.sigma * Math.sqrt(Math.max(0, 1 - Math.min(0.95, rhoAbs) * Math.min(0.95, rhoAbs)));
  if (lowerW < 0) err += 1e10 * lowerW * lowerW;
  if (p.sigma <= 1e-8) err += 1e8;

  for (const c of pts) {
    const model = sviTotalVariance(p, c.k);
    const d = model - c.w;
    err += c.weight * d * d;
  }
  return err;
}

/** Nelder–Mead simplex in 5-D on the SVI parameter vector. */
function nelderMead(
  init: SviParams,
  objective: (p: SviParams) => number,
  maxIterations = 400
): { p: SviParams; f: number; converged: boolean } {
  const keys: (keyof SviParams)[] = ['a', 'b', 'rho', 'm', 'sigma'];
  const step = [init.a * 0.3, 0.05, 0.1, 0.05, 0.1];
  const simplexParams: SviParams[] = [{ ...init }];
  for (let i = 0; i < 5; i++) {
    const p = { ...init };
    p[keys[i]] += step[i];
    simplexParams.push(p);
  }
  const simplex = simplexParams.map((p) => ({ p, f: objective(p) }));

  let converged = false;
  for (let iter = 0; iter < maxIterations; iter++) {
    simplex.sort((x, y) => x.f - y.f);
    const best = simplex[0];
    const worst = simplex[simplex.length - 1];
    const secondWorst = simplex[simplex.length - 2];
    if (Math.abs(worst.f - best.f) < 1e-14) { converged = true; break; }

    const centroid: SviParams = { a: 0, b: 0, rho: 0, m: 0, sigma: 0 };
    for (let i = 0; i < simplex.length - 1; i++) {
      centroid.a += simplex[i].p.a;
      centroid.b += simplex[i].p.b;
      centroid.rho += simplex[i].p.rho;
      centroid.m += simplex[i].p.m;
      centroid.sigma += simplex[i].p.sigma;
    }
    const inv = 1 / (simplex.length - 1);
    centroid.a *= inv; centroid.b *= inv; centroid.rho *= inv; centroid.m *= inv; centroid.sigma *= inv;

    const reflect: SviParams = {
      a: centroid.a + (centroid.a - worst.p.a),
      b: centroid.b + (centroid.b - worst.p.b),
      rho: centroid.rho + (centroid.rho - worst.p.rho),
      m: centroid.m + (centroid.m - worst.p.m),
      sigma: centroid.sigma + (centroid.sigma - worst.p.sigma),
    };
    const fRef = objective(reflect);

    if (fRef < best.f) {
      const expand: SviParams = {
        a: centroid.a + 2 * (centroid.a - worst.p.a),
        b: centroid.b + 2 * (centroid.b - worst.p.b),
        rho: centroid.rho + 2 * (centroid.rho - worst.p.rho),
        m: centroid.m + 2 * (centroid.m - worst.p.m),
        sigma: centroid.sigma + 2 * (centroid.sigma - worst.p.sigma),
      };
      const fExp = objective(expand);
      simplex[simplex.length - 1] = fExp < fRef ? { p: expand, f: fExp } : { p: reflect, f: fRef };
    } else if (fRef < secondWorst.f) {
      simplex[simplex.length - 1] = { p: reflect, f: fRef };
    } else {
      const contractOut = fRef < worst.f;
      const gamma = contractOut ? 0.5 : -0.5;
      const contract: SviParams = {
        a: centroid.a + gamma * (centroid.a - worst.p.a),
        b: centroid.b + gamma * (centroid.b - worst.p.b),
        rho: centroid.rho + gamma * (centroid.rho - worst.p.rho),
        m: centroid.m + gamma * (centroid.m - worst.p.m),
        sigma: centroid.sigma + gamma * (centroid.sigma - worst.p.sigma),
      };
      const fCon = objective(contract);
      if (fCon < (contractOut ? fRef : worst.f)) {
        simplex[simplex.length - 1] = { p: contract, f: fCon };
      } else {
        for (let i = 1; i < simplex.length; i++) {
          simplex[i].p = {
            a: best.p.a + 0.5 * (simplex[i].p.a - best.p.a),
            b: best.p.b + 0.5 * (simplex[i].p.b - best.p.b),
            rho: best.p.rho + 0.5 * (simplex[i].p.rho - best.p.rho),
            m: best.p.m + 0.5 * (simplex[i].p.m - best.p.m),
            sigma: best.p.sigma + 0.5 * (simplex[i].p.sigma - best.p.sigma),
          };
          simplex[i].f = objective(simplex[i].p);
        }
      }
    }
  }

  simplex.sort((x, y) => x.f - y.f);
  return { p: simplex[0].p, f: simplex[0].f, converged };
}

export function fitSvi(contracts: CleanedContract[], maxIterations = 400): SviFit | null {
  const pts = contracts.filter((c) => isFinite(c.w) && c.w > 0);
  if (pts.length < 5) return null;

  // Multi-start Nelder–Mead; equities skew negative, so weight the multi-start toward ρ<0.
  const ws = pts.map((p) => p.w);
  const minW = Math.min(...ws);
  const maxW = Math.max(...ws);
  const starts: SviParams[] = [
    { a: Math.max(1e-6, minW * 0.8), b: Math.max(1e-4, (maxW - minW) || minW), rho: -0.3, m: 0, sigma: 0.3 },
    { a: Math.max(1e-6, minW * 0.8), b: Math.max(1e-4, (maxW - minW) || minW), rho: -0.5, m: 0, sigma: 0.2 },
    { a: Math.max(1e-6, minW * 0.8), b: Math.max(1e-4, (maxW - minW) || minW), rho: -0.15, m: 0, sigma: 0.5 },
    { a: Math.max(1e-6, minW * 0.9), b: Math.max(1e-4, (maxW - minW) / 2 || minW), rho: -0.3, m: 0, sigma: 0.6 },
    { a: Math.max(1e-6, minW * 0.6), b: Math.max(1e-4, (maxW - minW) || minW), rho: 0.0, m: 0, sigma: 0.4 },
  ];

  let best: { p: SviParams; f: number } | null = null;
  for (const start of starts) {
    const res = nelderMead(start, (p) => sviObjective(p, pts), maxIterations);
    if (!best || res.f < best.f) best = res;
  }
  if (!best) return null;
  const fit = best.p;

  // Diagnostics
  // (fit already selected by multi-start)
  let sse = 0;
  for (const c of pts) {
    // For IV-space RMSE we need T; recover per contract from c.iv & c.w
    const t_i = c.w / (c.iv * c.iv);
    const fitIv = Math.sqrt(Math.max(1e-12, sviTotalVariance(fit, c.k)) / t_i);
    const d = fitIv - c.iv;
    sse += d * d;
  }
  const rmseIv = Math.sqrt(sse / pts.length);

  let sseW = 0;
  for (const c of pts) {
    const d = sviTotalVariance(fit, c.k) - c.w;
    sseW += d * d;
  }
  const rmseW = Math.sqrt(sseW / pts.length);

  // Arbitrage checks (calendar/butterfly via density sign on a grid)
  const arbitrageFlags: string[] = [];
  if (fit.b < 0) arbitrageFlags.push('b<0 (non-positive slope)');
  if (Math.abs(fit.rho) >= 1) arbitrageFlags.push('|ρ|≥1');
  const lowW = fit.a + fit.b * fit.sigma * Math.sqrt(Math.max(0, 1 - fit.rho * fit.rho));
  if (lowW < 0) arbitrageFlags.push('negative total variance region');

  return {
    ...fit,
    rmseW,
    rmseIv,
    nUsed: pts.length,
    converged: true,
    arbitrageFlags,
  };
}

/** Delta → strike on the fitted smile (bisection on the smile delta curve). */
export function deltaToStrike(
  fit: SviParams,
  targetDelta: number,
  type: 'call' | 'put',
  spot: number,
  forward: number,
  T: number,
  r: number,
  q: number,
  kMin: number,
  kMax: number
): number | null {
  const f = (k: number): number => {
    const iv = Math.sqrt(Math.max(1e-12, sviTotalVariance(fit, k)) / T);
    const K = Math.exp(k) * forward;
    return bsDelta(spot, K, T, r, q, iv, type) - targetDelta;
  };
  let lo = kMin;
  let hi = kMax;
  let flo = f(lo);
  const fhi = f(hi);
  if (flo * fhi > 0) return null; // target delta outside the fit's coverage
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi);
    const fm = f(mid);
    if (Math.abs(fm) < 1e-6) return Math.exp(mid) * forward;
    if (flo * fm <= 0) {
      hi = mid;
    } else {
      lo = mid;
      flo = fm;
    }
  }
  return Math.exp(0.5 * (lo + hi)) * forward;
}

/** 25Δ risk reversal / butterfly / 90-10 skew from the fitted smile. Equity convention:
 *  RR = σ(25Δ call) − σ(25Δ put) (usually negative); BF = (σ25ΔC + σ25ΔP)/2 − σATM;
 *  skew90_10 = σ(10Δ put) − σ(10Δ call) (positive = puts bid, crash fear). */
export function skewMetrics(
  fit: SviParams,
  spot: number,
  forward: number,
  T: number,
  r: number,
  q: number,
  kMin: number,
  kMax: number
): { rr25: number | null; bf25: number | null; skew9010: number | null; atmIv: number | null } {
  const ivAtK = (k: number) => Math.sqrt(Math.max(1e-12, sviTotalVariance(fit, k)) / T);
  const atmIv = ivAtK(0);

  const k25c = deltaToStrike(fit, 0.25, 'call', spot, forward, T, r, q, kMin, kMax);
  const k25p = deltaToStrike(fit, -0.25, 'put', spot, forward, T, r, q, kMin, kMax);
  const k10p = deltaToStrike(fit, -0.10, 'put', spot, forward, T, r, q, kMin, kMax);
  const k10c = deltaToStrike(fit, 0.10, 'call', spot, forward, T, r, q, kMin, kMax);

  const iv25c = k25c ? ivAtK(Math.log(k25c / forward)) : null;
  const iv25p = k25p ? ivAtK(Math.log(k25p / forward)) : null;
  const iv10p = k10p ? ivAtK(Math.log(k10p / forward)) : null;
  const iv10c = k10c ? ivAtK(Math.log(k10c / forward)) : null;

  return {
    rr25: iv25c !== null && iv25p !== null ? iv25c - iv25p : null,
    bf25: iv25c !== null && iv25p !== null && atmIv !== null ? (iv25c + iv25p) / 2 - atmIv : null,
    skew9010: iv10p !== null && iv10c !== null ? iv10p - iv10c : null,
    atmIv: isFinite(atmIv) ? atmIv : null,
  };
}

/** Interpolate total variance w = σ²T linearly in T between bracketing points; σ = √(w/T). Never interpolate σ directly. */
export function totalVarianceInterpolate(
  points: { t: number; iv: number }[],
  targetT: number
): number | null {
  const valid = points.filter((p) => p.t > 0 && isFinite(p.iv) && p.iv > 0).sort((a, b) => a.t - b.t);
  if (valid.length === 0) return null;
  if (valid.length === 1) return valid[0].iv;
  const w = (p: { t: number; iv: number }) => p.iv * p.iv * p.t;
  if (targetT <= valid[0].t) return valid[0].iv;
  if (targetT >= valid[valid.length - 1].t) return valid[valid.length - 1].iv;
  for (let i = 0; i < valid.length - 1; i++) {
    const a = valid[i];
    const b = valid[i + 1];
    if (targetT >= a.t && targetT <= b.t) {
      const wa = w(a);
      const wb = w(b);
      const frac = (targetT - a.t) / (b.t - a.t);
      const wTarget = wa + frac * (wb - wa);
      return Math.sqrt(Math.max(1e-12, wTarget / targetT));
    }
  }
  return null;
}

/**
 * Model-free implied variance for a single expiry (Carr–Madan / VIX methodology).
 * OTM puts below F, OTM calls above F, zero-bid contracts excluded.
 * Returns variance (not std). Pass cleaned OTM contracts with their mids and strikes.
 */
export function modelFreeImpliedVar(
  otmCalls: { strike: number; mid: number }[],
  otmPuts: { strike: number; mid: number }[],
  forward: number,
  T: number,
  rate: number
): number | null {
  if (T <= 0 || forward <= 0) return null;
  const disc = Math.exp(rate * T);

  // Merged, sorted strike set
  const callMap = new Map(otmCalls.map((c) => [c.strike, c.mid] as const));
  const putMap = new Map(otmPuts.map((p) => [p.strike, p.mid] as const));
  const strikes = Array.from(new Set([...callMap.keys(), ...putMap.keys()])).sort((a, b) => a - b);
  if (strikes.length < 5) return null;

  // K₀: strike nearest the forward (VIX uses minimal |C−P|, we approximate with nearest F)
  let k0 = strikes[0];
  let best = Infinity;
  for (const K of strikes) {
    const d = Math.abs(K - forward);
    if (d < best) { best = d; k0 = K; }
  }

  let sum = 0;
  for (let i = 0; i < strikes.length; i++) {
    const K = strikes[i];
    // ΔK: half the distance to neighbors at the ends, full spacing inside
    const left = i > 0 ? K - strikes[i - 1] : K - (strikes[1] ?? K);
    const right = i < strikes.length - 1 ? strikes[i + 1] - K : (strikes[i - 1] !== undefined ? K - strikes[i - 1] : K);
    const dK = i === 0 ? right : i === strikes.length - 1 ? left : (right + left) / 2;

    if (K === k0) continue;
    // Q(K): put below K₀, call above
    const q = K < k0 ? putMap.get(K) : callMap.get(K);
    if (q === undefined || q <= 0) continue; // zero-bid excluded
    sum += (dK / (K * K)) * disc * q;
  }

  const sqTerm = Math.pow(forward / k0 - 1, 2) / T;
  const mfVar = (2 * disc / T) * sum - sqTerm;
  if (!isFinite(mfVar)) return null;
  return Math.max(0, mfVar);
}

/** Breeden–Litzenberger risk-neutral density from the fitted smile (finite-difference on BS calls, uniform K grid). */
export function riskNeutralDensity(
  fit: SviParams,
  spot: number,
  forward: number,
  T: number,
  r: number,
  q: number,
  nPoints = 601,
) {
  const carryForward = spot * Math.exp((r - q) * T);
  const minVariance = fit.a + fit.b * fit.sigma * Math.sqrt(1 - fit.rho ** 2);
  if (![spot, forward, T, r, q, fit.a, fit.b, fit.rho, fit.m, fit.sigma].every(Number.isFinite) ||
      !(spot > 0 && forward > 0 && T > 0 && minVariance > 0 && fit.sigma > 0) ||
      fit.b < 0 || Math.abs(fit.rho) >= 1 || fit.b * (1 + Math.abs(fit.rho)) >= 2) return null;
  const halfWidth = Math.max(0.3, 9 * Math.sqrt(sviTotalVariance(fit, 0)));
  const wingWidth = Math.max(halfWidth, 9 * Math.sqrt(Math.max(sviTotalVariance(fit, -halfWidth), sviTotalVariance(fit, halfWidth))));
  const logWidth = Math.min(12, wingWidth);
  const strikes = Array.from({ length: nPoints }, (_, i) => forward * Math.exp(logWidth * (2 * i / (nPoints - 1) - 1)));

  const price = (K: number, type: 'call' | 'put') => {
    const k = Math.log(K / forward);
    const iv = Math.sqrt(Math.max(1e-12, sviTotalVariance(fit, k)) / T);
    return bsPrice({ spot, strike: K, t: T, rate: r, dividendYield: q, vol: iv, type });
  };

  const disc = Math.exp(r * T);
  const raw = strikes.map((K) => {
    const h = K * 0.0005;
    // Put/call parity is linear in K, so either side has the same curvature.
    const type = K < carryForward ? 'put' : 'call';
    return disc * (price(K + h, type) - 2 * price(K, type) + price(K - h, type)) / h ** 2;
  });
  const grid = normalizeDensity(strikes, raw);
  if (!grid) return null;
  const forwardError = Math.abs(densityMean(grid, (k) => k) / carryForward - 1);
  const lo = strikes[0], hi = strikes[strikes.length - 1];
  const slope = (k: number, type: 'call' | 'put') => disc *
    (price(k * 1.0005, type) - price(k * 0.9995, type)) / (k * 0.001);
  const edgeMass = Math.abs(slope(lo, 'put')) + Math.abs(slope(hi, 'call'));
  return { ...grid, forwardError, edgeMass };
}

/** Expected move: 1σ from ATM IV, and the straddle-implied ≈ 0.8 × (call + put ATM). */
export function expectedMove(
  spot: number,
  atmIv: number,
  T: number,
  atmCallMid: number | null,
  atmPutMid: number | null
): { oneSigmaPct: number; oneSigmaCurrency: number; straddleImpliedPct: number | null } {
  const oneSigmaCurrency = spot * atmIv * Math.sqrt(T);
  const straddle =
    atmCallMid !== null && atmPutMid !== null ? 0.8 * (atmCallMid + atmPutMid) : null;
  return {
    oneSigmaPct: (oneSigmaCurrency / spot) * 100,
    oneSigmaCurrency,
    straddleImpliedPct: straddle !== null ? (straddle / spot) * 100 : null,
  };
}

/** N(d2)-style risk-neutral probabilities from spot ATM IV: P(S_T > K) and touch probability. */
export function riskNeutralProbabilities(
  spot: number,
  strike: number,
  T: number,
  r: number,
  q: number,
  atmIv: number
): { probAbove: number; probTouchApprox: number } {
  const { d2 } = d1d2(spot, strike, T, r, q, atmIv);
  const probAbove = normCdf(d2);
  const probTouchApprox = Math.min(1, 2 * normCdf(-Math.abs(d2)));
  return { probAbove, probTouchApprox };
}

// =============================================================
// Local smile refit — natural cubic spline & SSVI (surface SVI)
// Used by the client to offer fitting alternatives to the
// server-side raw SVI fit. Methods clamp failures honestly and
// return null rather than fabricate a curve.
// =============================================================

export interface NaturalSplineSmileFit {
  model: 'spline';
  /** knot log-moneyness (sorted, unique) */
  xs: number[];
  /** fitted total variance at the knots */
  ys: number[];
  /** natural-spline second derivatives (knot arrays, endpoints = 0) */
  m: number[];
  rmseIv: number;
  nUsed: number;
}

export interface SsviSmileFit {
  model: 'ssvi';
  /** w(θ,k) = θ/2·(1 + ρ·φ(θ)·k + √((φ(θ)·k + ρ)² + 1 − ρ²)),  φ(θ) = η·θ^(−γ) */
  rho: number;
  eta: number;
  gamma: number;
  atmTotalVariance: number;
  rmseW: number;
  rmseIv: number;
  nUsed: number;
  converged: boolean;
}

export type LocalSmileFit = NaturalSplineSmileFit | SsviSmileFit;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Natural cubic spline through (k, w = iv²·T) points — total variance stays linear at the wings. */
export function fitNaturalSplineSmile(
  pts: Array<{ k: number; iv: number }>,
  T: number
): NaturalSplineSmileFit | null {
  const n0 = pts.length;
  if (n0 < 3 || !(T > 0)) return null;

  // Dedupe by k (average weights), then sort ascending
  const byK = new Map<number, { sumW: number; sumY: number }>();
  for (const p of pts) {
    if (!Number.isFinite(p.k) || !Number.isFinite(p.iv) || p.iv <= 0) continue;
    const w = p.iv * p.iv * T;
    const key = Math.round(p.k * 1e9);
    const acc = byK.get(key) ?? { sumW: 0, sumY: 0 };
    acc.sumW += 1;
    acc.sumY += w;
    byK.set(key, acc);
  }
  const knots = [...byK.entries()]
    .map(([key, acc]) => ({ k: key / 1e9, w: acc.sumY / acc.sumW }))
    .filter((q) => Number.isFinite(q.k) && Number.isFinite(q.w) && q.w > 0)
    .sort((a, b) => a.k - b.k);
  const n = knots.length;
  if (n < 3) return null;

  const xs = knots.map((q) => q.k);
  const ys = knots.map((q) => q.w);
  const h: number[] = [];
  for (let i = 0; i < n - 1; i++) h.push(xs[i + 1] - xs[i]);

  // Natural spline: solve tridiagonal system for second derivatives m[]
  const alpha: number[] = new Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    alpha[i] =
      3 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
  }
  const l: number[] = new Array(n).fill(0);
  const mu: number[] = new Array(n).fill(0);
  const z: number[] = new Array(n).fill(0);
  l[0] = 1;
  mu[0] = 0;
  z[0] = 0;
  for (let i = 1; i < n - 1; i++) {
    l[i] = 2 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
    if (Math.abs(l[i]) < 1e-12) return null;
    mu[i] = h[i] / l[i];
    z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
  }
  l[n - 1] = 1;
  z[n - 1] = 0;
  const m: number[] = new Array(n).fill(0);
  for (let j = n - 2; j >= 0; j--) {
    m[j] = z[j] - mu[j] * m[j + 1];
  }

  let sseIv = 0;
  for (const p of pts) {
    if (!Number.isFinite(p.k) || p.iv <= 0) continue;
    const wFit = splineTotalVariance({ xs, ys, m }, p.k);
    const ivFit = Math.sqrt(Math.max(1e-12, wFit / T));
    sseIv += (ivFit - p.iv) * (ivFit - p.iv);
  }

  return {
    model: 'spline',
    xs,
    ys,
    m,
    rmseIv: Math.sqrt(sseIv / pts.length),
    nUsed: pts.length,
  };
}

export function splineTotalVariance(
  fit: Pick<NaturalSplineSmileFit, 'xs' | 'ys' | 'm'>,
  k: number
): number {
  const { xs, ys, m } = fit;
  const n = xs.length;
  if (n < 2) return 0;
  if (k <= xs[0]) return ys[0];
  if (k >= xs[n - 1]) return ys[n - 1];
  // binary search for segment
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] > k) hi = mid;
    else lo = mid;
  }
  const i = lo;
  const h = xs[i + 1] - xs[i];
  const A = (xs[i + 1] - k) / h;
  const B = (k - xs[i]) / h;
  const w =
    m[i] * Math.pow(xs[i + 1] - k, 3) / (6 * h) +
    m[i + 1] * Math.pow(k - xs[i], 3) / (6 * h) +
    (ys[i] / h - (m[i] * h) / 6) * (xs[i + 1] - k) +
    (ys[i + 1] / h - (m[i + 1] * h) / 6) * (k - xs[i]);
  return Math.max(1e-9, w);
}

/** SSVI body: w(θ, k) given (θ, ρ, η, γ). Mathematically ≥ 0 for |ρ| < 1. */
function ssviCore(theta: number, rho: number, eta: number, gamma: number, k: number): number {
  const phi = eta * Math.pow(theta, -gamma);
  const t1 = 1 + rho * phi * k;
  const u = phi * k + rho;
  const t2 = Math.sqrt(u * u + 1 - rho * rho);
  return (theta / 2) * (t1 + t2);
}

function ssviObjectiveFactory(pts: Array<{ k: number; iv: number; weight: number }>, T: number) {
  return (p: number[]): number => {
    const [oTheta, oRho, oEta, oGamma] = p;
    const theta = Math.exp(oTheta);
    const rho = Math.tanh(oRho) * 0.98;
    const eta = Math.exp(oEta);
    const gamma = (1 / (1 + Math.exp(-oGamma))) * 0.5;
    let err = 0;
    for (const c of pts) {
      const model = ssviCore(theta, rho, eta, gamma, c.k);
      const target = c.iv * c.iv * T;
      const d = model - target;
      err += c.weight * d * d;
    }
    return err;
  };
}

export function fitSsvi(
  pts: Array<{ k: number; iv: number; weight: number }>,
  T: number
): SsviSmileFit | null {
  const good = pts.filter((c) => Number.isFinite(c.k) && c.iv > 0 && Number.isFinite(c.weight));
  if (good.length < 5 || !(T > 0)) return null;

  const wAtm = Math.min(...good.map((c) => c.iv * c.iv * T));
  const logTheta0 = Math.log(Math.max(1e-6, wAtm));
  const sigmoidInv = (x: number) => Math.log(x / (1 - x));

  // Multi-start Nelder–Mead over the 4-D unconstrained space
  const starts: Array<[number, number, number, number]> = [
    [logTheta0, Math.atanh(-0.4), Math.log(0.3), sigmoidInv(0.2)],
    [logTheta0, Math.atanh(-0.6), Math.log(0.15), sigmoidInv(0.35)],
    [logTheta0, Math.atanh(-0.2), Math.log(0.5), sigmoidInv(0.1)],
    [logTheta0, Math.atanh(0.1), Math.log(0.4), sigmoidInv(0.25)],
  ];

  let best: { p: number[]; f: number } | null = null;
  for (const start of starts) {
    const res = nelderMeadMulti(start, ssviObjectiveFactory(good, T), 250);
    if (!best || res.f < best.f) best = res;
  }
  if (!best) return null;

  const [oTheta, oRho, oEta, oGamma] = best.p as [number, number, number, number];
  const theta = Math.exp(oTheta);
  const rho = Math.tanh(oRho) * 0.98;
  const eta = Math.exp(oEta);
  const gamma = (1 / (1 + Math.exp(-oGamma))) * 0.5;

  let sseW = 0;
  let sseIv = 0;
  for (const c of good) {
    const wFit = ssviCore(theta, rho, eta, gamma, c.k);
    sseW += (wFit - c.iv * c.iv * T) * (wFit - c.iv * c.iv * T);
    const dIv = Math.sqrt(Math.max(1e-12, wFit / T)) - c.iv;
    sseIv += dIv * dIv;
  }

  return {
    model: 'ssvi',
    rho,
    eta,
    gamma,
    atmTotalVariance: theta,
    rmseW: Math.sqrt(sseW / good.length),
    rmseIv: Math.sqrt(sseIv / good.length),
    nUsed: good.length,
    converged: true,
  };
}

export function ssviTotalVariance(fit: SsviSmileFit, k: number): number {
  return ssviCore(fit.atmTotalVariance, fit.rho, fit.eta, fit.gamma, k);
}

/** Generic Nelder–Mead on a numeric vector (n-D). */
function nelderMeadMulti(
  init: number[],
  objective: (p: number[]) => number,
  maxIterations = 250
): { p: number[]; f: number } {
  const n = init.length;
  const step = init.map((v) => Math.max(0.05, Math.abs(v) * 0.25));
  const simplexPts: number[][] = [init.slice()];
  for (let i = 0; i < n; i++) {
    const p = init.slice();
    p[i] += step[i];
    simplexPts.push(p);
  }
  const simplex = simplexPts.map((p) => ({ p, f: objective(p) }));

  for (let iter = 0; iter < maxIterations; iter++) {
    simplex.sort((a, b) => a.f - b.f);
    const best = simplex[0];
    const worst = simplex[simplex.length - 1];
    const secondWorst = simplex[simplex.length - 2];
    if (Math.abs(worst.f - best.f) < 1e-16) break;

    const centroid = new Array(n).fill(0);
    for (let i = 0; i < simplex.length - 1; i++) {
      for (let j = 0; j < n; j++) centroid[j] += simplex[i].p[j];
    }
    for (let j = 0; j < n; j++) centroid[j] /= n;

    const reflect = centroid.map((c, j) => c + (c - worst.p[j]));
    const fRef = objective(reflect);

    if (fRef < best.f) {
      const expand = centroid.map((c, j) => c + 2 * (c - worst.p[j]));
      const fExp = objective(expand);
      simplex[simplex.length - 1] = fExp < fRef ? { p: expand, f: fExp } : { p: reflect, f: fRef };
    } else if (fRef < secondWorst.f) {
      simplex[simplex.length - 1] = { p: reflect, f: fRef };
    } else {
      const contractOut = fRef < worst.f;
      const gammaC = contractOut ? 0.5 : -0.5;
      const contract = centroid.map((c, j) => c + gammaC * (c - worst.p[j]));
      const fCon = objective(contract);
      if (fCon < (contractOut ? fRef : worst.f)) {
        simplex[simplex.length - 1] = { p: contract, f: fCon };
      } else {
        for (let i = 1; i < simplex.length; i++) {
          for (let j = 0; j < n; j++) {
            simplex[i].p[j] = best.p[j] + 0.5 * (simplex[i].p[j] - best.p[j]);
          }
          simplex[i].f = objective(simplex[i].p);
        }
      }
    }
  }

  simplex.sort((a, b) => a.f - b.f);
  return { p: simplex[0].p, f: simplex[0].f };
}

/** Dispatch wrapper: fit either local model on a slice's raw points. */
export function fitLocalSmile(
  pts: Array<{ k: number; iv: number; weight: number }>,
  T: number,
  method: 'spline' | 'ssvi'
): LocalSmileFit | null {
  if (method === 'spline') {
    const out = fitNaturalSplineSmile(
      pts.map((p) => ({ k: p.k, iv: p.iv })),
      T
    );
    return out;
  }
  return fitSsvi(pts, T);
}

export function evalLocalSmileIv(fit: LocalSmileFit, k: number, T: number): number {
  const w =
    fit.model === 'spline'
      ? splineTotalVariance(fit, k)
      : ssviTotalVariance(fit, k);
  return Math.sqrt(Math.max(1e-12, w / T));
}
