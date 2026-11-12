// GEX / Dealer-Positioning Engine — pure functions over a cleaned chain.
// Dealer positioning is UNOBSERVABLE; this module works off the public
// convention: dealers are long calls and short puts ("squeeze"), or the mirror
// ("inverse"). Every consumer must caption results with that assumption.
//
// Units: GEX is quoted in USD of delta re-hedge per 1% adverse move
// (Σ Γ·OI·100·S²·0.01) — the same scale desks use for "how much does the
// market move when the hedge flips".

import type { RawChainRow } from './options/providers/types';
import { bsGamma, impliedForward, type OptionContract } from './options';

export type GexConvention = 'squeeze' | 'inverse';

export interface GexChainContext {
  symbol: string;
  spot: number;
  /** used only for distSigma in wall detection — falls back to spot when parity fails */
  forward: number;
  rate: number;
  dividendYield: number;
  /** ATM IV of the front slice (or iv30), decimal */
  atmIv: number;
  /** T of the front slice, years */
  frontT: number;
  /** rows with OI > 0 — unfiltered otherwise (ITM included; walls need them) */
  rows: RawChainRow[];
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Normalize rows for analysis: dte ≥ 1, oi > 0, strike within ±25% of spot. */
export function cleanRows(
  rows: RawChainRow[],
  spot: number,
  opts: { maxAbsMovePct?: number } = {}
): RawChainRow[] {
  const pct = opts.maxAbsMovePct ?? 0.25;
  const lo = spot * (1 - pct);
  const hi = spot * (1 + pct);
  return rows.filter((r) => r.dte >= 1 && r.oi > 0 && r.strike >= lo && r.strike <= hi);
}

/** Put-call parity forward, median of the 5 strikes nearest ATM (tightest C−P wins inside the slice pair selection). */
export function impliedForwardFromRows(
  rows: RawChainRow[],
  spot: number,
  rate: number
): number | null {
  // Map rows to OptionContract shape expected by impliedForward (per expiry separately for correctness — take the front slice)
  const frontRows = rows.filter((r) => r.dte >= 1);
  if (frontRows.length === 0) return null;
  const minDte = Math.min(...frontRows.map((r) => r.dte));
  const slice = frontRows.filter((r) => r.dte === minDte || r.dte === minDte + 1 || r.dte === minDte - 1);
  const t = Math.max(1e-6, minDte / 365);
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

const signOf = (row: RawChainRow, convention: GexConvention): number => {
  const base = row.type === 'call' ? 1 : -1;
  return convention === 'squeeze' ? base : -base;
};

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
}): GexChainContext | null {
  const { symbol, spot, rate, dividendYield, rows } = input;
  if (!(spot > 0) || rows.length === 0) return null;
  const cleaned = cleanRows(rows, spot);
  if (cleaned.length === 0) return null;
  const forward = impliedForwardFromRows(cleaned, spot, rate) ?? spot;
  const frontT = input.frontT > 0 ? input.frontT : 30 / 365;
  const atmIv = input.atmIv > 0 ? input.atmIv : 0.2;
  return { symbol, spot, forward, rate, dividendYield, atmIv, frontT, rows: cleaned };
}

interface StrikeBucket {
  strike: number;
  callOi: number;
  putOi: number;
  gex: number;
}

function bucketByStrike(
  ctx: GexChainContext,
  maxDte: number,
  convention: GexConvention
): Map<number, StrikeBucket> {
  const buckets = new Map<number, StrikeBucket>();
  const q = ctx.dividendYield;
  const r = ctx.rate;
  const S = ctx.spot;
  for (const row of ctx.rows) {
    if (row.dte > maxDte) continue;
    // OTM only for gamma work
    if (row.type === 'call' && row.strike < ctx.forward) continue;
    if (row.type === 'put' && row.strike > ctx.forward) continue;
    const T = Math.max(1e-6, row.dte / 365);
    const sigma = row.iv && row.iv > 0.01 && row.iv < 4 ? row.iv : ctx.atmIv;
    const gamma = bsGamma(S, row.strike, T, r, q, sigma);
    if (!(gamma > 0)) continue;
    const gex = signOf(row, convention) * gamma * row.oi * 100 * S * S * 0.01;
    const b = buckets.get(row.strike) ?? { strike: row.strike, callOi: 0, putOi: 0, gex: 0 };
    if (row.type === 'call') b.callOi += row.oi;
    else b.putOi += row.oi;
    b.gex += gex;
    buckets.set(row.strike, b);
  }
  return buckets;
}

/** GEX-by-strike profile (bars within ±band for display/hedging) + aggregate. */
export function gammaProfile(
  ctx: GexChainContext,
  opts: { maxDte?: number; convention?: GexConvention } = {}
): { byStrike: StrikeBucket[]; totalGex: number; maxAbsGex: number } {
  const maxDte = opts.maxDte ?? 45;
  const convention = opts.convention ?? 'squeeze';
  const byStrike = Array.from(bucketByStrike(ctx, maxDte, convention).values()).sort(
    (a, b) => a.strike - b.strike
  );
  const totalGex = byStrike.reduce((s, b) => s + b.gex, 0);
  const maxAbsGex = byStrike.reduce((m, b) => Math.max(m, Math.abs(b.gex)), 0);
  return { byStrike, totalGex, maxAbsGex };
}

/** Aggregate $ of delta re-hedge to perform per 1% move, in a ±band around spot. */
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
    const T = Math.max(1e-6, row.dte / 365);
    const sigma = row.iv && row.iv > 0.01 && row.iv < 4 ? row.iv : ctx.atmIv;
    const gamma = bsGamma(S, row.strike, T, r, q, sigma);
    if (!(gamma > 0)) continue;
    const gex = signOf(row, convention) * gamma * row.oi * 100 * S * S * 0.01;
    const dist = row.strike / S - 1;
    if (Math.abs(dist) > bandPct) continue;
    if (dist >= 0) perPctUp += gex;
    else perPctDown += gex;
  }
  return { perPctUp, perPctDown, net: perPctUp + perPctDown };
}

/**
 * Gamma flip — the spot at which aggregate GEX changes sign, found by recomputing
 * the whole profile across a grid of hypothetical spots (sticky-strike IV).
 * Returns the first zero crossing scanning upward; null when none exists.
 */
export function gammaFlip(
  ctx: GexChainContext,
  opts: { rangePct?: number; steps?: number; maxDte?: number; convention?: GexConvention } = {}
): { flip: number | null; profile: Array<{ spot: number; gex: number }>; regime: 'long' | 'short' | 'unknown' } {
  const rangePct = opts.rangePct ?? 0.15;
  const steps = opts.steps ?? 121;
  const maxDte = opts.maxDte ?? 45;
  const convention = opts.convention ?? 'squeeze';
  const S0 = ctx.spot;

  const profile: { spot: number; gex: number }[] = [];
  let flip: number | null = null;
  let prevSpot = 0;
  let prevGex = 0;

  for (let i = 0; i < steps; i++) {
    const frac = i / (steps - 1);
    const S = S0 * (1 - rangePct + 2 * rangePct * frac);
    // Rebuild the profile at this hypothetical spot (sticky IV, sticky forward)
    const hypo: GexChainContext = { ...ctx, spot: S };
    const { totalGex } = gammaProfile(hypo, { maxDte, convention });
    profile.push({ spot: S, gex: totalGex });
    if (i > 0 && flip === null && Math.sign(totalGex) !== Math.sign(prevGex) && prevGex !== 0) {
      // Linear interpolation between the two grid points bracketing the zero
      flip = prevSpot + ((0 - prevGex) * (S - prevSpot)) / (totalGex - prevGex);
    }
    prevSpot = S;
    prevGex = totalGex;
  }

  const atCurrent = gammaProfile(ctx, { maxDte, convention }).totalGex;
  const regime: 'long' | 'short' | 'unknown' =
    Math.abs(atCurrent) < 1e-8 ? 'unknown' : atCurrent > 0 ? 'long' : 'short';

  return { flip, profile, regime };
}

// ---------------------------------------------------------------------------
// Walls
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

  // OI by strike per side
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

  const toWall = (strike: number, oi: number): Wall => ({
    strike,
    oi,
    share: totalOiLocal > 0 ? oi / totalOiLocal : 0,
    distPct: strike / S - 1,
    distSigma: Math.abs(Math.log(strike / ctx.forward)) / (ctx.atmIv * Math.sqrt(Math.max(1e-6, ctx.frontT))),
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
  flip: { flip: number | null; regime: 'long' | 'short' | 'unknown' };
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
  regime: 'long' | 'short' | 'unknown';
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
  const frontWing = atmIv * Math.sqrt(Math.max(1e-6, frontT));
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

  // --- Component 4: velocity (5d ΔOI z-score; sign-scaled)
  let velocityComp = 0;
  let velocityDegraded = false;
  if (inp.oiChange5dZ === null) {
    velocityDegraded = true;
  } else {
    const z = inp.oiChange5dZ;
    const signMatch = (direction === 'up' ? (inp.oiChange5d ?? 0) > 0 : (inp.oiChange5d ?? 0) > 0);
    const mag = clamp((z - 0.5) / 1.5, 0, 1) * 15;
    velocityComp = signMatch ? mag : 0;
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
  } else if (inp.previousState === 'triggered' && (!isShortGamma || spot <= (walls.call?.strike ?? Infinity))) {
    state = 'exhausted';
  } else {
    state = inp.previousState === 'triggered' ? 'exhausted' : 'idle';
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
