// GEX / Dealer-Positioning Engine — pure functions over a cleaned chain.
// Dealer positioning is UNOBSERVABLE; this module works off the public
// convention: dealers are long calls and short puts ("squeeze"), or the mirror
// ("inverse"). Every consumer must caption results with that assumption.
//
// Units:
// - GEX / Volume GEX: USD of delta re-hedge per 1% adverse move (Σ sign·Γ·N·100·S²·0.01)
// - Charm: USD change in dealer option-delta notional per 1 calendar day (Σ sign·(Charm_yr/365)·OI·100·S)
// - Vanna: USD change in dealer option-delta notional per +1 vol pt (+0.01 IV) (Σ sign·Vanna·OI·S)

import type { RawChainRow } from './options/providers/types';
import {
  bsCharmPerDay,
  bsGamma,
  bsVanna,
  impliedForward,
  impliedVol,
  sanitizeIv,
  type OptionContract,
} from './options';

export type GexConvention = 'squeeze' | 'inverse';
export type GammaRegime = 'long' | 'short' | 'unknown';
export type GexExpiryBucketId = '0dte' | 'weekly' | 'monthly' | 'far';

export const MIN_GEX_T = 0.5 / 365;

export interface GexChainContext {
  symbol: string;
  spot: number;
  /** Put-call parity forward (falls back to spot when parity fails) */
  forward: number;
  rate: number;
  dividendYield: number;
  /** ATM IV of the front slice (or iv30), decimal */
  atmIv: number;
  /** T of the front slice, years (floored at 0.5 / 365) */
  frontT: number;
  /** Whether 0-DTE contracts (dte === 0) are included in this context */
  include0Dte: boolean;
  /** Rows with OI > 0 within the strike band — full chain (ITM + ATM + OTM) */
  rows: RawChainRow[];
}

export interface StrikeBucket {
  strike: number;
  callOi: number;
  putOi: number;
  callVolume: number;
  putVolume: number;
  callGex: number;
  putGex: number;
  gex: number;
  callVolumeGex: number;
  putVolumeGex: number;
  volumeGex: number;
  callCharm: number;
  putCharm: number;
  charm: number;
  callVanna: number;
  putVanna: number;
  vanna: number;
}

export interface GammaProfileResult {
  byStrike: StrikeBucket[];
  totalCallGex: number;
  totalPutGex: number;
  totalGex: number;
  totalCallVolumeGex: number;
  totalPutVolumeGex: number;
  totalVolumeGex: number;
  totalCallCharm: number;
  totalPutCharm: number;
  totalCharm: number;
  totalCallVanna: number;
  totalPutVanna: number;
  totalVanna: number;
  maxAbsGex: number;
}

export interface FlipProfilePoint {
  spot: number;
  gex: number;
  callGex: number;
  putGex: number;
  gexSqueeze: number;
  gexInverse: number;
}

export interface ConventionFlipSummary {
  flip: number | null;
  regime: GammaRegime;
}

export interface GammaFlipResult {
  flip: number | null;
  profile: FlipProfilePoint[];
  regime: GammaRegime;
  byConvention: {
    squeeze: ConventionFlipSummary;
    inverse: ConventionFlipSummary;
  };
}

export interface ExpiryBucketSummary {
  bucket: GexExpiryBucketId;
  label: string;
  dteRange: string;
  oi: number;
  callOi: number;
  putOi: number;
  callGex: number;
  putGex: number;
  totalGex: number;
  absGex: number;
  shareOfAbsGex: number;
  totalVolumeGex: number;
  totalCharm: number;
  totalVanna: number;
  byStrike: StrikeBucket[];
}

export interface StrikeExpiryMatrix {
  strikes: number[];
  expiries: {
    expiry: string;
    dte: number;
    bucket: GexExpiryBucketId;
    totalGex: number;
    callGex: number;
    putGex: number;
    totalOi: number;
  }[];
  /** [strikeIdx][expiryIdx] */
  netGex: number[][];
  callGex: number[][];
  putGex: number[][];
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

const gexTimeYears = (dte: number): number => Math.max(MIN_GEX_T, dte / 365);

const signOfType = (type: 'call' | 'put', convention: GexConvention): number => {
  const base = type === 'call' ? 1 : -1;
  return convention === 'squeeze' ? base : -base;
};

const regimeFromGex = (gex: number): GammaRegime =>
  Math.abs(gex) < 1e-8 ? 'unknown' : gex > 0 ? 'long' : 'short';

/** Normalize rows for analysis: dte ≥ 0 (or ≥ 1 when include0Dte=false), oi > 0, strike within ±25% of spot. */
export function cleanRows(
  rows: RawChainRow[],
  spot: number,
  opts: { maxAbsMovePct?: number; include0Dte?: boolean } = {}
): RawChainRow[] {
  const pct = opts.maxAbsMovePct ?? 0.25;
  const minDte = (opts.include0Dte ?? true) ? 0 : 1;
  const lo = spot * (1 - pct);
  const hi = spot * (1 + pct);
  return rows.filter((r) => r.dte >= minDte && r.oi > 0 && r.strike >= lo && r.strike <= hi);
}

/** Put-call parity forward, median of the 5 strikes nearest ATM (preferring dte ≥ 1, falling back to 0-DTE). */
export function impliedForwardFromRows(
  rows: RawChainRow[],
  spot: number,
  rate: number
): number | null {
  const dte1Plus = rows.filter((r) => r.dte >= 1);
  const frontRows = dte1Plus.length > 0 ? dte1Plus : rows.filter((r) => r.dte >= 0);
  if (frontRows.length === 0) return null;
  const minDte = Math.min(...frontRows.map((r) => r.dte));
  const slice = frontRows.filter((r) => r.dte === minDte || r.dte === minDte + 1 || r.dte === minDte - 1);
  const t = gexTimeYears(minDte);
  const toContract = (r: RawChainRow): OptionContract => ({
    strike: r.strike,
    bid: r.bid,
    ask: r.ask,
    lastPrice: r.last,
    openInterest: r.oi,
    volume: r.volume,
    impliedVolatility: r.iv ?? 0,
    inTheMoney: r.type === 'call' ? r.strike < spot : r.strike > spot,
  });
  const calls = slice.filter((r) => r.type === 'call').map(toContract);
  const puts = slice.filter((r) => r.type === 'put').map(toContract);
  const res = impliedForward(calls, puts, t, rate);
  return res?.forward ?? null;
}

/** Build a chain context from raw rows. Returns null when unusable. */
export function buildContext(input: {
  symbol: string;
  asOf: string;
  spot: number;
  rows: RawChainRow[];
  rate: number;
  dividendYield: number;
  atmIv: number;
  frontT: number;
  include0Dte?: boolean;
}): GexChainContext | null {
  const { symbol, spot, rate, dividendYield, rows } = input;
  const include0Dte = input.include0Dte ?? true;
  if (!(spot > 0) || rows.length === 0) return null;
  const cleaned = cleanRows(rows, spot, { include0Dte });
  if (cleaned.length === 0) return null;
  const forward = impliedForwardFromRows(cleaned, spot, rate) ?? spot;
  const frontT = input.frontT > 0 ? Math.max(MIN_GEX_T, input.frontT) : 30 / 365;
  const atmIv = input.atmIv > 0 && Number.isFinite(input.atmIv) ? input.atmIv : 0.2;
  return { symbol, spot, forward, rate, dividendYield, atmIv, frontT, include0Dte, rows: cleaned };
}

/**
 * Resolve a clean implied volatility for one chain row:
 * accepts healthy raw Yahoo IV, tries Newton mid-quote IV when raw IV is an
 * outlier/missing on tight quotes, and falls back to `atmIv`.
 */
export function resolveGexRowIv(
  row: RawChainRow,
  spot: number,
  T: number,
  rate: number,
  dividendYield: number,
  atmIv: number
): number {
  const fallback = sanitizeIv(atmIv, 0.2);
  const sanitized = sanitizeIv(row.iv, fallback, { bid: row.bid, ask: row.ask });
  const rawAccepted = typeof row.iv === 'number' && Number.isFinite(row.iv) && sanitized === row.iv;

  const mid = row.bid > 0 && row.ask >= row.bid ? (row.bid + row.ask) / 2 : 0;
  const relSpread = mid > 0 ? (row.ask - row.bid) / mid : Infinity;
  // Cheap wing penny-tick outliers (e.g., 3.00 IV on a $0.05/$0.06 option) still need checking.
  const pennyExtreme =
    rawAccepted &&
    mid > 0 &&
    mid < 0.25 &&
    (row.iv as number) > Math.max(1.5, fallback * 4);

  if (rawAccepted && !pennyExtreme) return sanitized;

  if (mid > 0 && relSpread <= 0.4 && T > 0 && spot > 0 && row.strike > 0) {
    const solved = impliedVol(mid, spot, row.strike, T, rate, dividendYield, row.type);
    if (solved !== null && Number.isFinite(solved) && solved > 0.02 && solved <= 2.5) {
      return solved;
    }
  }
  return fallback;
}

interface RowExposure {
  gex: number;
  volumeGex: number;
  charm: number;
  vanna: number;
}

function computeRowExposure(
  row: RawChainRow,
  S: number,
  r: number,
  q: number,
  sigma: number,
  convention: GexConvention
): RowExposure {
  const T = gexTimeYears(row.dte);
  const sign = signOfType(row.type, convention);
  const gamma = bsGamma(S, row.strike, T, r, q, sigma);
  const scaleSpot = 100 * S * S * 0.01;
  const gex = gamma > 0 && Number.isFinite(gamma) ? sign * gamma * row.oi * scaleSpot : 0;
  const volumeGex = gamma > 0 && Number.isFinite(gamma) && row.volume > 0 ? sign * gamma * row.volume * scaleSpot : 0;

  const charmPerDay = bsCharmPerDay(S, row.strike, T, r, q, sigma, row.type);
  const charm = Number.isFinite(charmPerDay) ? sign * charmPerDay * row.oi * 100 * S : 0;

  const vannaPerUnit = bsVanna(S, row.strike, T, r, q, sigma);
  // 0.01 (per +1 vol pt) * 100 (shares/contract) = 1
  const vanna = Number.isFinite(vannaPerUnit) ? sign * vannaPerUnit * row.oi * S : 0;

  return { gex, volumeGex, charm, vanna };
}

function emptyStrikeBucket(strike: number): StrikeBucket {
  return {
    strike,
    callOi: 0,
    putOi: 0,
    callVolume: 0,
    putVolume: 0,
    callGex: 0,
    putGex: 0,
    gex: 0,
    callVolumeGex: 0,
    putVolumeGex: 0,
    volumeGex: 0,
    callCharm: 0,
    putCharm: 0,
    charm: 0,
    callVanna: 0,
    putVanna: 0,
    vanna: 0,
  };
}

function summarizeBuckets(buckets: Map<number, StrikeBucket>): GammaProfileResult {
  const byStrike = Array.from(buckets.values()).sort((a, b) => a.strike - b.strike);
  let totalCallGex = 0;
  let totalPutGex = 0;
  let totalCallVolumeGex = 0;
  let totalPutVolumeGex = 0;
  let totalCallCharm = 0;
  let totalPutCharm = 0;
  let totalCallVanna = 0;
  let totalPutVanna = 0;
  let maxAbsGex = 0;

  for (const b of byStrike) {
    totalCallGex += b.callGex;
    totalPutGex += b.putGex;
    totalCallVolumeGex += b.callVolumeGex;
    totalPutVolumeGex += b.putVolumeGex;
    totalCallCharm += b.callCharm;
    totalPutCharm += b.putCharm;
    totalCallVanna += b.callVanna;
    totalPutVanna += b.putVanna;
    const absG = Math.abs(b.gex);
    if (absG > maxAbsGex) maxAbsGex = absG;
  }

  return {
    byStrike,
    totalCallGex,
    totalPutGex,
    totalGex: totalCallGex + totalPutGex,
    totalCallVolumeGex,
    totalPutVolumeGex,
    totalVolumeGex: totalCallVolumeGex + totalPutVolumeGex,
    totalCallCharm,
    totalPutCharm,
    totalCharm: totalCallCharm + totalPutCharm,
    totalCallVanna,
    totalPutVanna,
    totalVanna: totalCallVanna + totalPutVanna,
    maxAbsGex,
  };
}

function aggregateRowsToProfile(
  rows: RawChainRow[],
  ctx: GexChainContext,
  convention: GexConvention,
  precomputedSigmas?: number[]
): GammaProfileResult {
  const buckets = new Map<number, StrikeBucket>();
  const S = ctx.spot;
  const r = ctx.rate;
  const q = ctx.dividendYield;

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const T = gexTimeYears(row.dte);
    const sigma = precomputedSigmas
      ? precomputedSigmas[i]
      : resolveGexRowIv(row, S, T, r, q, ctx.atmIv);
    const exp = computeRowExposure(row, S, r, q, sigma, convention);

    const b = buckets.get(row.strike) ?? emptyStrikeBucket(row.strike);
    if (row.type === 'call') {
      b.callOi += row.oi;
      b.callVolume += row.volume;
      b.callGex += exp.gex;
      b.callVolumeGex += exp.volumeGex;
      b.callCharm += exp.charm;
      b.callVanna += exp.vanna;
    } else {
      b.putOi += row.oi;
      b.putVolume += row.volume;
      b.putGex += exp.gex;
      b.putVolumeGex += exp.volumeGex;
      b.putCharm += exp.charm;
      b.putVanna += exp.vanna;
    }
    b.gex = b.callGex + b.putGex;
    b.volumeGex = b.callVolumeGex + b.putVolumeGex;
    b.charm = b.callCharm + b.putCharm;
    b.vanna = b.callVanna + b.putVanna;
    buckets.set(row.strike, b);
  }

  return summarizeBuckets(buckets);
}

/** Full-chain GEX-by-strike profile (calls + puts across all strikes ≤ maxDte). */
export function gammaProfile(
  ctx: GexChainContext,
  opts: { maxDte?: number; convention?: GexConvention } = {}
): GammaProfileResult {
  const maxDte = opts.maxDte ?? 45;
  const convention = opts.convention ?? 'squeeze';
  const rows = ctx.rows.filter((row) => row.dte <= maxDte);
  return aggregateRowsToProfile(rows, ctx, convention);
}

/** Aggregate $ of delta re-hedge to perform per 1% move, in a ±band around spot (unified full-chain). */
export function hedgeFlow(
  ctx: GexChainContext,
  opts: { bandPct?: number; maxDte?: number; convention?: GexConvention } = {}
): { perPctUp: number; perPctDown: number; net: number } {
  const bandPct = opts.bandPct ?? 0.02;
  const maxDte = opts.maxDte ?? 45;
  const convention = opts.convention ?? 'squeeze';
  const S = ctx.spot;
  const q = ctx.dividendYield;
  const r = ctx.rate;
  let perPctUp = 0;
  let perPctDown = 0;
  for (const row of ctx.rows) {
    if (row.dte > maxDte) continue;
    const dist = row.strike / S - 1;
    if (Math.abs(dist) > bandPct) continue;
    const T = gexTimeYears(row.dte);
    const sigma = resolveGexRowIv(row, S, T, r, q, ctx.atmIv);
    const { gex } = computeRowExposure(row, S, r, q, sigma, convention);
    if (gex === 0) continue;
    if (dist >= 0) perPctUp += gex;
    else perPctDown += gex;
  }
  return { perPctUp, perPctDown, net: perPctUp + perPctDown };
}

/**
 * Gamma flip — the spot at which aggregate GEX changes sign, found by recomputing
 * the whole profile across a grid of hypothetical spots (sticky-strike IV, dynamic
 * forward scaling F(S) = S * (ctx.forward / ctx.spot)).
 * Returns the first zero crossing, enriched profile, and both convention regimes.
 */
export function gammaFlip(
  ctx: GexChainContext,
  opts: { rangePct?: number; steps?: number; maxDte?: number; convention?: GexConvention } = {}
): GammaFlipResult {
  const rangePct = opts.rangePct ?? 0.15;
  const steps = opts.steps ?? 121;
  const maxDte = opts.maxDte ?? 45;
  const convention = opts.convention ?? 'squeeze';
  const S0 = ctx.spot;
  const forwardScale = S0 > 0 ? ctx.forward / S0 : 1;

  const activeRows = ctx.rows.filter((r) => r.dte <= maxDte);
  const sigmas = activeRows.map((row) =>
    resolveGexRowIv(row, S0, gexTimeYears(row.dte), ctx.rate, ctx.dividendYield, ctx.atmIv)
  );

  const profile: FlipProfilePoint[] = [];
  let flipSqueeze: number | null = null;
  let prevSpot = 0;
  let prevSqueeze = 0;

  for (let i = 0; i < steps; i++) {
    const frac = steps > 1 ? i / (steps - 1) : 0.5;
    const S = S0 * (1 - rangePct + 2 * rangePct * frac);
    const F = S * forwardScale;

    let squeezeCall = 0;
    let squeezePut = 0;
    for (let j = 0; j < activeRows.length; j++) {
      const row = activeRows[j];
      const T = gexTimeYears(row.dte);
      const gamma = bsGamma(S, row.strike, T, ctx.rate, ctx.dividendYield, sigmas[j]);
      if (!(gamma > 0) || !Number.isFinite(gamma)) continue;
      const mag = gamma * row.oi * 100 * S * S * 0.01;
      if (row.type === 'call') squeezeCall += mag;
      else squeezePut -= mag;
    }

    const gexSqueeze = squeezeCall + squeezePut;
    const gexInverse = -gexSqueeze;
    // Keep F referenced on hypo context if future extensions consume forward-linked skew
    void F;

    const callGex = convention === 'squeeze' ? squeezeCall : -squeezeCall;
    const putGex = convention === 'squeeze' ? squeezePut : -squeezePut;
    const gex = convention === 'squeeze' ? gexSqueeze : gexInverse;

    profile.push({ spot: S, gex, callGex, putGex, gexSqueeze, gexInverse });

    if (i > 0 && flipSqueeze === null && Math.sign(gexSqueeze) !== Math.sign(prevSqueeze) && prevSqueeze !== 0) {
      flipSqueeze = prevSpot + ((0 - prevSqueeze) * (S - prevSpot)) / (gexSqueeze - prevSqueeze);
    }
    prevSpot = S;
    prevSqueeze = gexSqueeze;
  }

  let currentSqueeze = 0;
  for (let j = 0; j < activeRows.length; j++) {
    const row = activeRows[j];
    const T = gexTimeYears(row.dte);
    const gamma = bsGamma(S0, row.strike, T, ctx.rate, ctx.dividendYield, sigmas[j]);
    if (!(gamma > 0) || !Number.isFinite(gamma)) continue;
    const sign = row.type === 'call' ? 1 : -1;
    currentSqueeze += sign * gamma * row.oi * 100 * S0 * S0 * 0.01;
  }

  const regimeSqueeze = regimeFromGex(currentSqueeze);
  const regimeInverse = regimeFromGex(-currentSqueeze);
  const flip = flipSqueeze;
  const regime = convention === 'squeeze' ? regimeSqueeze : regimeInverse;

  return {
    flip,
    profile,
    regime,
    byConvention: {
      squeeze: { flip: flipSqueeze, regime: regimeSqueeze },
      inverse: { flip: flipSqueeze, regime: regimeInverse },
    },
  };
}

// ---------------------------------------------------------------------------
// Walls (OI-weighted & Gamma-weighted)
// ---------------------------------------------------------------------------

export interface Wall {
  strike: number;
  oi: number;
  /** OI share within ±10% of spot (all types, ≤ maxDte) */
  share: number;
  /** (strike/spot − 1) */
  distPct: number;
  /** |ln(strike/F)| / (atmIv·√frontT) */
  distSigma: number;
}

export interface Walls {
  call: Wall | null;
  put: Wall | null;
  runnerUpCall: Wall | null;
  runnerUpPut: Wall | null;
  totalOiLocal: number;
}

export interface GammaWall {
  strike: number;
  oi: number;
  /** Signed GEX contribution on this side under the requested convention */
  gex: number;
  absGex: number;
  /** Share of total gross GEX (Σ |callGex| + |putGex| across strikes ≤ maxDte) */
  share: number;
  distPct: number;
  distSigma: number;
}

export interface GammaWalls {
  call: GammaWall | null;
  put: GammaWall | null;
  runnerUpCall: GammaWall | null;
  runnerUpPut: GammaWall | null;
  totalAbsGex: number;
}

/** Wall strikes: largest OI concentration per side, excluding the 2 strikes nearest spot. */
export function detectWalls(
  ctx: GexChainContext,
  opts: { maxDte?: number } = {}
): Walls {
  const maxDte = opts.maxDte ?? 45;
  const S = ctx.spot;
  const rows = ctx.rows.filter((r) => r.dte <= maxDte);
  if (rows.length === 0) {
    return { call: null, put: null, runnerUpCall: null, runnerUpPut: null, totalOiLocal: 0 };
  }

  const callOi = new Map<number, number>();
  const putOi = new Map<number, number>();
  for (const r of rows) {
    const map = r.type === 'call' ? callOi : putOi;
    map.set(r.strike, (map.get(r.strike) ?? 0) + r.oi);
  }

  const strikes = Array.from(new Set(rows.map((r) => r.strike))).sort((a, b) => a - b);
  const distFromSpot = (K: number) => Math.abs(K - S);
  const twoNearest = [...strikes].sort((a, b) => distFromSpot(a) - distFromSpot(b)).slice(0, 2);

  const totalOiLocal = rows.filter((r) => Math.abs(r.strike / S - 1) <= 0.1).reduce((s, r) => s + r.oi, 0);
  const denom = ctx.atmIv * Math.sqrt(Math.max(MIN_GEX_T, ctx.frontT));

  const toWall = (strike: number, oi: number): Wall => ({
    strike,
    oi,
    share: totalOiLocal > 0 ? oi / totalOiLocal : 0,
    distPct: strike / S - 1,
    distSigma: Math.abs(Math.log(strike / ctx.forward)) / denom,
  });

  const callCandidates = strikes.filter((K) => K > S && !twoNearest.includes(K));
  const putCandidates = strikes.filter((K) => K < S && !twoNearest.includes(K));

  const pick = (cands: number[], map: Map<number, number>) => {
    const ranked = cands
      .map((K) => ({ K, oi: map.get(K) ?? 0 }))
      .filter((x) => x.oi > 0)
      .sort((a, b) => b.oi - a.oi);
    return {
      wall: ranked[0] ? toWall(ranked[0].K, ranked[0].oi) : null,
      runnerUp: ranked[1] ? toWall(ranked[1].K, ranked[1].oi) : null,
    };
  };

  const c = pick(callCandidates, callOi);
  const p = pick(putCandidates, putOi);

  return {
    call: c.wall,
    put: p.wall,
    runnerUpCall: c.runnerUp,
    runnerUpPut: p.runnerUp,
    totalOiLocal,
  };
}

/**
 * Gamma-weighted walls: strikes with largest call GEX (positive under squeeze)
 * and largest put GEX (negative under squeeze) across the full ≤ maxDte chain.
 */
export function detectGammaWalls(
  ctx: GexChainContext,
  opts: { maxDte?: number; convention?: GexConvention } = {}
): GammaWalls {
  const maxDte = opts.maxDte ?? 45;
  const convention = opts.convention ?? 'squeeze';
  const S = ctx.spot;
  const { byStrike } = gammaProfile(ctx, { maxDte, convention });
  if (byStrike.length === 0) {
    return { call: null, put: null, runnerUpCall: null, runnerUpPut: null, totalAbsGex: 0 };
  }

  let totalAbsGex = 0;
  for (const b of byStrike) {
    totalAbsGex += Math.abs(b.callGex) + Math.abs(b.putGex);
  }
  const denom = ctx.atmIv * Math.sqrt(Math.max(MIN_GEX_T, ctx.frontT));

  const toGammaWall = (b: StrikeBucket, side: 'call' | 'put'): GammaWall => {
    const gex = side === 'call' ? b.callGex : b.putGex;
    const absGex = Math.abs(gex);
    return {
      strike: b.strike,
      oi: side === 'call' ? b.callOi : b.putOi,
      gex,
      absGex,
      share: totalAbsGex > 0 ? absGex / totalAbsGex : 0,
      distPct: b.strike / S - 1,
      distSigma: Math.abs(Math.log(b.strike / ctx.forward)) / denom,
    };
  };

  const callRanked = byStrike
    .filter((b) => Math.abs(b.callGex) > 0)
    .sort((a, b) => Math.abs(b.callGex) - Math.abs(a.callGex));
  const putRanked = byStrike
    .filter((b) => Math.abs(b.putGex) > 0)
    .sort((a, b) => Math.abs(b.putGex) - Math.abs(a.putGex));

  return {
    call: callRanked[0] ? toGammaWall(callRanked[0], 'call') : null,
    put: putRanked[0] ? toGammaWall(putRanked[0], 'put') : null,
    runnerUpCall: callRanked[1] ? toGammaWall(callRanked[1], 'call') : null,
    runnerUpPut: putRanked[1] ? toGammaWall(putRanked[1], 'put') : null,
    totalAbsGex,
  };
}

// ---------------------------------------------------------------------------
// Expiry decomposition & Strike × Expiry GEX matrix
// ---------------------------------------------------------------------------

function isThirdFridayDate(ymd: string): boolean {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(ymd);
  if (!m) return false;
  const d = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
  return d.getUTCDay() === 5 && d.getUTCDate() >= 15 && d.getUTCDate() <= 21;
}

export function classifyExpiryBucket(dte: number, expiry: string): GexExpiryBucketId {
  if (dte <= 0) return '0dte';
  if ((dte > 14 && dte <= 45) || (dte <= 45 && isThirdFridayDate(expiry))) return 'monthly';
  if (dte <= 14) return 'weekly';
  return 'far';
}

const BUCKET_META: { id: GexExpiryBucketId; label: string; dteRange: string }[] = [
  { id: '0dte', label: '0-DTE', dteRange: '0d' },
  { id: 'weekly', label: 'Weekly', dteRange: '1–14d' },
  { id: 'monthly', label: 'Monthly / OPEX', dteRange: '15–45d & 3rd Fri' },
  { id: 'far', label: 'Far', dteRange: '46–90d' },
];

/** Expiry-bucketed GEX summary + per-strike breakdown across `'0dte' | 'weekly' | 'monthly' | 'far'`. */
export function expiryBuckets(
  ctx: GexChainContext,
  opts: { maxDte?: number; convention?: GexConvention } = {}
): ExpiryBucketSummary[] {
  const maxDte = opts.maxDte ?? 90;
  const convention = opts.convention ?? 'squeeze';
  const grouped = new Map<GexExpiryBucketId, RawChainRow[]>();
  for (const m of BUCKET_META) grouped.set(m.id, []);

  for (const row of ctx.rows) {
    if (row.dte > maxDte) continue;
    grouped.get(classifyExpiryBucket(row.dte, row.expiry))?.push(row);
  }

  const rawSummaries = BUCKET_META.map((meta) => {
    const rows = grouped.get(meta.id) ?? [];
    const prof = aggregateRowsToProfile(rows, ctx, convention);
    let callOi = 0;
    let putOi = 0;
    for (const r of rows) {
      if (r.type === 'call') callOi += r.oi;
      else putOi += r.oi;
    }
    const absGex = Math.abs(prof.totalCallGex) + Math.abs(prof.totalPutGex);
    return {
      bucket: meta.id,
      label: meta.label,
      dteRange: meta.dteRange,
      oi: callOi + putOi,
      callOi,
      putOi,
      callGex: prof.totalCallGex,
      putGex: prof.totalPutGex,
      totalGex: prof.totalGex,
      absGex,
      shareOfAbsGex: 0,
      totalVolumeGex: prof.totalVolumeGex,
      totalCharm: prof.totalCharm,
      totalVanna: prof.totalVanna,
      byStrike: prof.byStrike,
    };
  });

  const totalGross = rawSummaries.reduce((s, b) => s + b.absGex, 0);
  if (totalGross > 0) {
    for (const s of rawSummaries) {
      s.shareOfAbsGex = s.absGex / totalGross;
    }
  }
  return rawSummaries;
}

/** 2D Strike × Expiry GEX matrix across the top expiries ≤ maxDte (ranked by gross GEX for strike selection). */
export function strikeExpiryMatrix(
  ctx: GexChainContext,
  opts: { maxDte?: number; maxExpiries?: number; maxStrikes?: number; convention?: GexConvention } = {}
): StrikeExpiryMatrix {
  const maxDte = opts.maxDte ?? 90;
  const maxExpiries = opts.maxExpiries ?? 12;
  const maxStrikes = opts.maxStrikes ?? 28;
  const convention = opts.convention ?? 'squeeze';
  const S = ctx.spot;
  const r = ctx.rate;
  const q = ctx.dividendYield;

  const byExp = new Map<string, { expiry: string; dte: number; rows: RawChainRow[] }>();
  for (const row of ctx.rows) {
    if (row.dte > maxDte) continue;
    const entry = byExp.get(row.expiry) ?? { expiry: row.expiry, dte: row.dte, rows: [] };
    if (row.dte < entry.dte) entry.dte = row.dte;
    entry.rows.push(row);
    byExp.set(row.expiry, entry);
  }

  const expiriesRaw = Array.from(byExp.values())
    .sort((a, b) => a.dte - b.dte || a.expiry.localeCompare(b.expiry))
    .slice(0, maxExpiries);

  if (expiriesRaw.length === 0) {
    return { strikes: [], expiries: [], netGex: [], callGex: [], putGex: [] };
  }

  const strikeGross = new Map<number, number>();
  const cellCall = new Map<string, number>();
  const cellPut = new Map<string, number>();
  const expiriesOut: StrikeExpiryMatrix['expiries'] = [];

  for (const exp of expiriesRaw) {
    let callGex = 0;
    let putGex = 0;
    let totalOi = 0;
    for (const row of exp.rows) {
      const T = gexTimeYears(row.dte);
      const sigma = resolveGexRowIv(row, S, T, r, q, ctx.atmIv);
      const { gex } = computeRowExposure(row, S, r, q, sigma, convention);
      totalOi += row.oi;
      const cellKey = `${row.strike}|${exp.expiry}`;
      if (row.type === 'call') {
        callGex += gex;
        cellCall.set(cellKey, (cellCall.get(cellKey) ?? 0) + gex);
      } else {
        putGex += gex;
        cellPut.set(cellKey, (cellPut.get(cellKey) ?? 0) + gex);
      }
      strikeGross.set(row.strike, (strikeGross.get(row.strike) ?? 0) + Math.abs(gex));
    }
    expiriesOut.push({
      expiry: exp.expiry,
      dte: exp.dte,
      bucket: classifyExpiryBucket(exp.dte, exp.expiry),
      totalGex: callGex + putGex,
      callGex,
      putGex,
      totalOi,
    });
  }

  const topStrikes = Array.from(strikeGross.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxStrikes)
    .map(([strike]) => strike)
    .sort((a, b) => a - b);

  const netGex: number[][] = [];
  const callMatrix: number[][] = [];
  const putMatrix: number[][] = [];
  for (const strike of topStrikes) {
    const netRow: number[] = [];
    const callRow: number[] = [];
    const putRow: number[] = [];
    for (const exp of expiriesOut) {
      const key = `${strike}|${exp.expiry}`;
      const c = cellCall.get(key) ?? 0;
      const p = cellPut.get(key) ?? 0;
      callRow.push(c);
      putRow.push(p);
      netRow.push(c + p);
    }
    netGex.push(netRow);
    callMatrix.push(callRow);
    putMatrix.push(putRow);
  }

  return {
    strikes: topStrikes,
    expiries: expiriesOut,
    netGex,
    callGex: callMatrix,
    putGex: putMatrix,
  };
}

/** Share of total OI (≤ maxDte) expiring within 7 DTE — short-dated flow dominance. */
export function expiryConcentration7d(ctx: GexChainContext, maxDte = 90): number {
  let total = 0;
  let short = 0;
  for (const r of ctx.rows) {
    if (r.dte > maxDte) continue;
    total += r.oi;
    if (r.dte <= 7) short += r.oi;
  }
  return total > 0 ? short / total : 0;
}

// ---------------------------------------------------------------------------
// Trap score & state machine (deterministic, bounded 0–100)
// ---------------------------------------------------------------------------

export type TrapState = 'idle' | 'building' | 'armed' | 'triggered' | 'exhausted';
export type TrapDirection = 'up' | 'down' | 'none';

export interface TrapScoreInput {
  walls: Walls;
  flip: { flip: number | null; regime: GammaRegime };
  spot: number;
  atmIv: number;
  frontT: number;
  /** Δ OI at the trapped wall, 1 session back (call or put side) */
  oiChange1d: number | null;
  /** Δ OI at the trapped wall, 5 sessions back */
  oiChange5d: number | null;
  /** z-score of oiChange5d vs trailing distribution (≥ 20 samples) */
  oiChange5dZ: number | null;
  realizedVol20: number | null;
  ret5d: number | null;
  expiryConcentration7d: number;
  /** true if the spot closed beyond the wall on the trap side within 3 sessions */
  closedBeyondWallRecently: boolean;
  previousState?: TrapState;
}

export interface TrapScoreResult {
  score: number;
  state: TrapState;
  direction: TrapDirection;
  components: { wall: number; proximity: number; regime: number; velocity: number; coil: number };
  velocityDegraded: boolean;
  regime: GammaRegime;
  nearWall: 'call' | 'put' | 'none';
}

export function trapScore(inp: TrapScoreInput): TrapScoreResult {
  const { walls, flip, spot, atmIv, frontT } = inp;

  const candidates = [walls.call, walls.put].filter((w): w is Wall => w !== null);
  if (candidates.length === 0 || !(atmIv > 0)) {
    return {
      score: 0, state: 'idle', direction: 'none',
      components: { wall: 0, proximity: 0, regime: 0, velocity: 0, coil: 0 },
      velocityDegraded: inp.oiChange5dZ === null, regime: flip.regime, nearWall: 'none',
    };
  }

  // Direction: the wall being approached / just crossed — nearest by σ wins
  const frontWing = atmIv * Math.sqrt(Math.max(MIN_GEX_T, frontT));
  const near = [...candidates].sort((a, b) => a.distSigma - b.distSigma)[0];
  let direction: TrapDirection;
  if (walls.call && near.strike === walls.call.strike) direction = 'up';
  else if (walls.put && near.strike === walls.put.strike) direction = 'down';
  else direction = near.strike >= spot ? 'up' : 'down';

  // --- Component 1: wall concentration (saturates ~25% local OI at one strike)
  const share = near.share;
  const wallComp = 25 * clamp(Math.log1p(share / 0.05) / Math.log1p(0.25 / 0.05), 0, 1);

  // --- Component 2: proximity (decays over ~2 expected moves)
  const distLog = Math.abs(Math.log(near.strike / spot));
  const proxComp = 25 * Math.exp(-distLog / (0.5 * frontWing));

  // --- Component 3: regime (short gamma & spot on the trapping side of the flip)
  let regimeComp = 0;
  if (flip.regime === 'short' && flip.flip !== null) {
    const flipDistSigma = Math.abs(Math.log(spot / flip.flip)) / frontWing;
    regimeComp = 25 * clamp(1 - flipDistSigma / 1.5, 0, 1);
  }

  // --- Component 4: velocity (5d ΔOI on the trapped side — positive build feeds pressure)
  let velocityComp = 0;
  let velocityDegraded = false;
  if (inp.oiChange5dZ === null) {
    velocityDegraded = true;
  } else {
    const z = inp.oiChange5dZ;
    const wallGrowing = (direction === 'up' || direction === 'down') && (inp.oiChange5d ?? 0) > 0;
    const mag = clamp((z - 0.5) / 1.5, 0, 1) * 15;
    velocityComp = wallGrowing ? mag : 0;
  }

  // --- Component 5: coil (a grind, not a spike)
  let coilComp = 0;
  if (inp.realizedVol20 !== null && inp.realizedVol20 > 0 && inp.ret5d !== null) {
    const rangeProxy = Math.abs(inp.ret5d) / (inp.realizedVol20 * Math.sqrt(5));
    const grind = clamp((rangeProxy - 0.5) / 1.5, 0, 1);
    const volVsIv = 1 - clamp((inp.realizedVol20 / atmIv - 0.8) / 0.6, 0, 1);
    coilComp = 10 * grind * volVsIv;
  }

  const score = clamp(Math.round(wallComp + proxComp + regimeComp + velocityComp + coilComp), 0, 100);

  // --- State machine (transitions via previousState)
  const isShortGamma = flip.regime === 'short';
  let state: TrapState;
  if (score >= 70 && isShortGamma && inp.closedBeyondWallRecently) {
    state = 'triggered';
  } else if (score >= 55 && near.distSigma <= 0.75 && isShortGamma) {
    state = 'armed';
  } else if (score >= 30 || (share >= 0.12 && (inp.oiChange5d ?? 0) > 0)) {
    state = 'building';
  } else if (inp.previousState === 'triggered') {
    state = 'exhausted';
  } else {
    state = 'idle';
  }

  return {
    score,
    state,
    direction: state === 'triggered' || state === 'armed' || state === 'building' ? direction : 'none',
    components: {
      wall: Math.round(wallComp * 10) / 10,
      proximity: Math.round(proxComp * 10) / 10,
      regime: Math.round(regimeComp * 10) / 10,
      velocity: Math.round(velocityComp * 10) / 10,
      coil: Math.round(coilComp * 10) / 10,
    },
    velocityDegraded,
    regime: flip.regime,
    nearWall: near.strike >= spot ? 'call' : 'put',
  };
}
