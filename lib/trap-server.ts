// Server-side trap bundle computation: pulls a provider chain + daily closes,
// derives the full dealer-positioning state, and exposes it for the snapshot
// store and the API. Pure orchestration over lib/gex.ts — no HTTP of its own
// beyond the closes fetch.
import {
  buildContext,
  gammaProfile,
  gammaFlip,
  detectWalls,
  expiryConcentration7d,
  hedgeFlow,
  trapScore,
  type GexConvention,
  type TrapScoreResult,
  type Walls,
  type TrapState,
  type TrapDirection,
} from './gex';
import type { ProviderChainResult, RawChainRow } from './options/providers/types';
import { fetchDailyCloses } from './vol-indices';
import { readHistory, readPanel, oiAtStrikeByDate } from './iv-store';

export interface TrapBundle {
  spot: number;
  forward: number;
  atmIv: number;
  frontT: number;
  headlineDte: number;
  atmIv30: number | null;
  totalGex: number;
  gammaFlip: { flip: number | null; regime: 'long' | 'short' | 'unknown'; profile: { spot: number; gex: number }[] };
  byStrike: { strike: number; callOi: number; putOi: number; gex: number }[];
  maxAbsGex: number;
  walls: Walls;
  flow: { perPctUp: number; perPctDown: number; net: number };
  expiryConcentration7d: number;
  expectedMove30: number | null;
  rv20: number | null;
  ret5d: number | null;
  oiChange1d: number | null;
  oiChange5d: number | null;
  oiChange5dZ: number | null;
  trap: TrapScoreResult;
  convention: GexConvention;
  previousState: TrapState | null;
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0);
const stdev = (xs: number[]) => {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
};

async function closesStats(
  symbol: string
): Promise<{ rv20: number | null; ret5d: number | null; closes: { date: string; close: number }[] }> {
  const closes = await fetchDailyCloses(symbol, '3mo');
  if (!closes || closes.length < 25) return { rv20: null, ret5d: null, closes: closes ?? [] };
  const rets: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    const a = closes[i - 1].close;
    const b = closes[i].close;
    if (a > 0 && b > 0) rets.push(Math.log(b / a));
  }
  const last20 = rets.slice(-20);
  const rv20 = Math.sqrt(mean(last20.map((r) => r * r)) * 252);
  const ret5d = rets.slice(-5).reduce((s, r) => s + r, 0);
  return { rv20, ret5d, closes };
}

/**
 * Compute the full trap bundle from a provider chain.
 * `iv30` (if known) feeds ATM IV and expected move; falls back to the headline slice's ATM IV.
 */
export async function computeTrapBundle(
  chain: ProviderChainResult,
  opts: { iv30?: number | null; convention?: GexConvention } = {}
): Promise<TrapBundle | null> {
  const convention = opts.convention ?? 'squeeze';
  const rows = chain.rows;
  if (!rows.length) return null;

  // Headline slice: nearest expiry ≥ 21 DTE, else the front slice
  const dteSet = [...new Set(rows.map((r) => r.dte))].sort((a, b) => a - b);
  const headlineDte = dteSet.find((d) => d >= 21) ?? dteSet[0] ?? 30;
  const headlineRows = rows.filter((r) => r.dte === headlineDte);
  const atmIvFromChain = (() => {
    const atm = headlineRows.reduce<RawChainRow | null>((best, r) => {
      if (!r.iv || r.iv <= 0) return best;
      if (!best) return r;
      return Math.abs(r.strike - chain.spot) < Math.abs(best.strike - chain.spot) ? r : best;
    }, null);
    return atm?.iv && atm.iv > 0 ? atm.iv : null;
  })();
  const atmIv = opts.iv30 ?? atmIvFromChain ?? 0.2;
  const frontT = Math.max(1e-6, headlineDte / 365);

  const ctx = buildContext({
    symbol: chain.symbol,
    asOf: chain.asOf,
    spot: chain.spot,
    rows,
    rate: chain.riskFreeRate,
    dividendYield: chain.dividendYield,
    atmIv,
    frontT,
  });
  if (!ctx) return null;

  const { byStrike, totalGex, maxAbsGex } = gammaProfile(ctx, { maxDte: 45, convention });
  const { flip, profile, regime } = gammaFlip(ctx, { convention });
  const walls = detectWalls(ctx, { maxDte: 45 });
  const flow = hedgeFlow(ctx, { bandPct: 0.02, convention });
  const conc7d = expiryConcentration7d(ctx);
  const atmIv30 = opts.iv30 ?? null;
  const expectedMove30 = atmIv30 !== null ? chain.spot * atmIv30 * Math.sqrt(30 / 365) : null;

  const { rv20, ret5d, closes } = await closesStats(chain.symbol);

  // Δ OI at the trapped wall strike (nearer of call/put by σ), 1d/5d back
  const prior = readHistory(chain.symbol).filter((s) => s.rows && s.rows.length > 0);
  const trappedWall = [walls.call, walls.put]
    .filter((w): w is NonNullable<typeof w> => w !== null)
    .sort((a, b) => a.distSigma - b.distSigma)[0] ?? null;
  const trappedSide: 'call' | 'put' = trappedWall
    ? walls.call && trappedWall.strike === walls.call.strike
      ? 'call'
      : 'put'
    : 'call';
  let oiChange1d: number | null = null;
  let oiChange5d: number | null = null;
  let oiChange5dZ: number | null = null;
  if (trappedWall) {
    const todayOi = trappedWall.oi;
    const priorOi = oiAtStrikeByDate(prior, trappedWall.strike, trappedSide, 120);
    if (priorOi.length >= 1) oiChange1d = todayOi - priorOi[priorOi.length - 1].oi;
    if (priorOi.length >= 5) oiChange5d = todayOi - priorOi[priorOi.length - 5].oi;
    // z-score of the 5d change vs trailing 5d changes at this strike
    const series = priorOi.map((p) => p.oi);
    const deltas: number[] = [];
    for (let i = 5; i < series.length; i++) deltas.push(series[i] - series[i - 5]);
    if (oiChange5d !== null && deltas.length >= 20) {
      const sd = stdev(deltas);
      oiChange5dZ = sd > 0 ? (oiChange5d - mean(deltas)) / sd : null;
    }
  }

  // Closed beyond the trapped wall within the last 3 sessions?
  let closedBeyondWallRecently = false;
  if (closes.length > 0 && trappedWall) {
    const last3 = closes.slice(-3).map((c) => c.close);
    if (trappedSide === 'call') {
      closedBeyondWallRecently = last3.some((c) => c >= trappedWall.strike);
    } else {
      closedBeyondWallRecently = last3.some((c) => c <= trappedWall.strike);
    }
  }

  // Previous daily state (for real state transitions)
  const panelHistory = readPanel(chain.symbol);
  const previousState = (panelHistory[panelHistory.length - 1]?.trapState ?? undefined) as TrapState | undefined;

  const trap = trapScore({
    walls,
    flip: { flip, regime },
    spot: chain.spot,
    atmIv,
    frontT,
    oiChange1d,
    oiChange5d,
    oiChange5dZ,
    realizedVol20: rv20,
    ret5d,
    expiryConcentration7d: conc7d,
    closedBeyondWallRecently,
    previousState,
  });

  return {
    spot: chain.spot,
    forward: ctx.forward,
    atmIv,
    frontT,
    headlineDte,
    atmIv30,
    totalGex,
    gammaFlip: { flip, regime, profile },
    byStrike,
    maxAbsGex,
    walls,
    flow,
    expiryConcentration7d: conc7d,
    expectedMove30,
    rv20,
    ret5d,
    oiChange1d,
    oiChange5d,
    oiChange5dZ,
    trap,
    convention,
    previousState: previousState ?? null,
  };
}
