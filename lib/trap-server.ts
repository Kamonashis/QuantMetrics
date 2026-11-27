// Server-side trap bundle computation: pulls a provider chain + daily bars,
// derives the full dealer-positioning state, and exposes it for the snapshot
// store and the API. Pure orchestration over lib/gex.ts — no HTTP of its own
// beyond the live daily-bar fetch when bars are not injected.
import {
  MIN_GEX_T,
  buildContext,
  detectGammaWalls,
  detectWalls,
  expiryBuckets,
  expiryConcentration7d,
  gammaFlip,
  gammaProfile,
  hedgeFlow,
  strikeExpiryMatrix,
  trapScore,
  type ExpiryBucketSummary,
  type GammaRegime,
  type GammaWalls,
  type GexConvention,
  type StrikeBucket,
  type StrikeExpiryMatrix,
  type TrapDirection,
  type TrapScoreResult,
  type TrapState,
  type Walls,
  type FlipProfilePoint,
} from './gex';
import { impliedVol, sanitizeIv } from './options';
import type { ProviderChainResult, RawChainRow } from './options/providers/types';
import { fetchLiveStockData } from './market-service';
import {
  oiAtStrikeByDate,
  readHistory,
  readPanel,
  type IvSnapshot,
  type PanelRow,
} from './iv-store';

export interface ConventionTrapSummary {
  totalGex: number;
  totalCallGex: number;
  totalPutGex: number;
  totalVolumeGex: number;
  totalCharm: number;
  totalVanna: number;
  flip: number | null;
  regime: GammaRegime;
  flow: { perPctUp: number; perPctDown: number; net: number };
  score: number;
  state: TrapState;
  direction: TrapDirection;
  components: { wall: number; proximity: number; regime: number; velocity: number; coil: number };
  nearWall: 'call' | 'put' | 'none';
}

export interface TrapBundle {
  spot: number;
  forward: number;
  atmIv: number;
  frontT: number;
  headlineDte: number;
  atmIv30: number | null;
  include0Dte: boolean;
  totalGex: number;
  totalCallGex: number;
  totalPutGex: number;
  totalCallVolumeGex: number;
  totalPutVolumeGex: number;
  totalVolumeGex: number;
  totalCallCharm: number;
  totalPutCharm: number;
  totalCharm: number;
  totalCallVanna: number;
  totalPutVanna: number;
  totalVanna: number;
  gammaFlip: { flip: number | null; regime: GammaRegime; profile: FlipProfilePoint[] };
  gammaFlipInverse: { flip: number | null; regime: GammaRegime };
  flipDistPct: number | null;
  flipDistSigma: number | null;
  byStrike: StrikeBucket[];
  maxAbsGex: number;
  walls: Walls;
  gammaWalls: GammaWalls;
  expiryBuckets: ExpiryBucketSummary[];
  strikeExpiryMatrix: StrikeExpiryMatrix;
  flow: { perPctUp: number; perPctDown: number; net: number };
  expiryConcentration7d: number;
  zeroDteShare: number;
  expectedMove30: number | null;
  rv20: number | null;
  ret5d: number | null;
  adv20Dollar: number | null;
  gexPerAdv: number | null;
  oiChange1d: number | null;
  oiChange5d: number | null;
  oiChange5dZ: number | null;
  trap: TrapScoreResult;
  convention: GexConvention;
  byConvention: {
    squeeze: ConventionTrapSummary;
    inverse: ConventionTrapSummary;
  };
  previousState: TrapState | null;
}

export interface TrapBundleOptions {
  iv30?: number | null;
  convention?: GexConvention;
  include0Dte?: boolean;
  /** Optional injected daily bars (used in tests or callers that already hold live bars) */
  dailyBars?: { date: string; close: number; volume?: number }[] | null;
  /** Optional injected OI snapshots (avoids disk read in unit tests) */
  priorSnapshots?: IvSnapshot[];
  /** Optional injected panel history (avoids disk read in unit tests) */
  panelHistory?: PanelRow[];
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0);
const stdev = (xs: number[]) => {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
};

export function computeDailyStats(
  bars: { date: string; close: number; volume?: number }[] | null
): {
  rv20: number | null;
  ret5d: number | null;
  adv20Dollar: number | null;
  closes: { date: string; close: number }[];
} {
  if (!bars || bars.length === 0) {
    return { rv20: null, ret5d: null, adv20Dollar: null, closes: [] };
  }
  const closes = bars
    .filter((b) => typeof b.close === 'number' && Number.isFinite(b.close) && b.close > 0)
    .map((b) => ({ date: b.date, close: b.close }));

  const recent20 = bars.slice(-20);
  const dollarVols: number[] = [];
  for (const b of recent20) {
    if (
      typeof b.close === 'number' &&
      Number.isFinite(b.close) &&
      b.close > 0 &&
      typeof b.volume === 'number' &&
      Number.isFinite(b.volume) &&
      b.volume > 0
    ) {
      dollarVols.push(b.close * b.volume);
    }
  }
  const adv20Dollar = dollarVols.length >= 15 ? mean(dollarVols) : null;

  if (closes.length < 21) {
    return { rv20: null, ret5d: null, adv20Dollar, closes };
  }
  const rets: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    const a = closes[i - 1].close;
    const b = closes[i].close;
    if (a > 0 && b > 0) rets.push(Math.log(b / a));
  }
  if (rets.length < 20) {
    return { rv20: null, ret5d: null, adv20Dollar, closes };
  }
  const last20 = rets.slice(-20);
  const rv20 = Math.sqrt(mean(last20.map((r) => r * r)) * 252);
  const ret5d = rets.length >= 5 ? rets.slice(-5).reduce((s, r) => s + r, 0) : null;
  return { rv20, ret5d, adv20Dollar, closes };
}

async function fetchLiveStats(
  symbol: string,
  injectedBars?: { date: string; close: number; volume?: number }[] | null
) {
  if (injectedBars !== undefined) {
    return computeDailyStats(injectedBars);
  }
  try {
    const res = await fetchLiveStockData(symbol, '3mo');
    // Never fall back to synthetic GBM bars for GEX/ADV20/RV20 calculations
    if (res.source !== 'live_market') {
      return computeDailyStats(null);
    }
    return computeDailyStats(res.bars);
  } catch {
    return computeDailyStats(null);
  }
}

function pickHeadlineDte(rows: RawChainRow[]): number {
  const dteSet = [...new Set(rows.map((r) => r.dte).filter((d) => Number.isFinite(d) && d >= 0))].sort(
    (a, b) => a - b
  );
  return (
    dteSet.find((d) => d >= 21) ??
    dteSet.find((d) => d >= 7) ??
    dteSet.find((d) => d >= 1) ??
    dteSet[0] ??
    30
  );
}

function estimateAtmIvFromSlice(
  headlineRows: RawChainRow[],
  spot: number,
  frontT: number,
  rate: number,
  dividendYield: number
): number | null {
  const sorted = [...headlineRows].sort((a, b) => Math.abs(a.strike - spot) - Math.abs(b.strike - spot));
  for (const r of sorted) {
    if (r.bid > 0 && r.ask >= r.bid) {
      const mid = (r.bid + r.ask) / 2;
      const relSpread = mid > 0 ? (r.ask - r.bid) / mid : Infinity;
      if (mid > 0 && relSpread <= 0.4) {
        const solved = impliedVol(mid, spot, r.strike, frontT, rate, dividendYield, r.type);
        if (solved !== null && Number.isFinite(solved) && solved > 0.02 && solved <= 2.5) {
          return solved;
        }
      }
    }
    if (typeof r.iv === 'number' && Number.isFinite(r.iv) && r.iv > 0.02 && r.iv <= 2.5) {
      return sanitizeIv(r.iv, 0.2, { bid: r.bid, ask: r.ask });
    }
  }
  return null;
}

/**
 * Compute the full trap bundle from a provider chain.
 * `iv30` (if known) feeds ATM IV and expected move; falls back to the headline slice's ATM IV.
 */
export async function computeTrapBundle(
  chain: ProviderChainResult,
  opts: TrapBundleOptions = {}
): Promise<TrapBundle | null> {
  const convention = opts.convention ?? 'squeeze';
  const include0Dte = opts.include0Dte ?? true;
  const rows = chain.rows;
  if (!rows.length) return null;

  const activeInputRows = include0Dte ? rows : rows.filter((r) => r.dte >= 1);
  if (!activeInputRows.length) return null;

  // Prefer ≥21, then ≥7, then ≥1 DTE and floor frontT at 0.5/365 so 0-DTE never collapses σ distance
  const headlineDte = pickHeadlineDte(activeInputRows);
  const frontT = Math.max(MIN_GEX_T, headlineDte / 365);
  const headlineRows = activeInputRows.filter((r) => r.dte === headlineDte);
  const atmIvFromChain = estimateAtmIvFromSlice(
    headlineRows,
    chain.spot,
    frontT,
    chain.riskFreeRate,
    chain.dividendYield
  );
  const atmIv30 =
    typeof opts.iv30 === 'number' && Number.isFinite(opts.iv30) && opts.iv30 > 0 ? opts.iv30 : null;
  const atmIv = atmIv30 ?? atmIvFromChain ?? 0.2;

  const ctx = buildContext({
    symbol: chain.symbol,
    asOf: chain.asOf,
    spot: chain.spot,
    rows: activeInputRows,
    rate: chain.riskFreeRate,
    dividendYield: chain.dividendYield,
    atmIv,
    frontT,
    include0Dte,
  });
  if (!ctx) return null;

  const profileRes = gammaProfile(ctx, { maxDte: 45, convention });
  const flipRes = gammaFlip(ctx, { convention });
  const walls = detectWalls(ctx, { maxDte: 45 });
  const gammaWalls = detectGammaWalls(ctx, { maxDte: 45, convention });
  const flow = hedgeFlow(ctx, { bandPct: 0.02, maxDte: 45, convention });
  const conc7d = expiryConcentration7d(ctx, 90);
  const buckets = expiryBuckets(ctx, { maxDte: 90, convention });
  const matrix = strikeExpiryMatrix(ctx, { maxDte: 90, maxExpiries: 12, maxStrikes: 28, convention });
  const zeroDteShare = include0Dte ? (buckets.find((b) => b.bucket === '0dte')?.shareOfAbsGex ?? 0) : 0;
  const expectedMove30 = atmIv30 !== null ? chain.spot * atmIv30 * Math.sqrt(30 / 365) : null;

  const { rv20, ret5d, adv20Dollar, closes } = await fetchLiveStats(chain.symbol, opts.dailyBars);
  const gexPerAdv = adv20Dollar !== null && adv20Dollar > 0 ? profileRes.totalGex / adv20Dollar : null;

  const flipDistPct =
    flipRes.flip !== null && chain.spot > 0 ? flipRes.flip / chain.spot - 1 : null;
  const flipDistSigma =
    flipRes.flip !== null && chain.spot > 0 && atmIv > 0
      ? Math.abs(Math.log(flipRes.flip / chain.spot)) / (atmIv * Math.sqrt(frontT))
      : null;

  // Δ OI at the trapped wall strike (nearer of call/put by σ), 1d/5d back
  const prior = (opts.priorSnapshots ?? readHistory(chain.symbol)).filter((s) => s.rows && s.rows.length > 0);
  const trappedWall =
    [walls.call, walls.put]
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
    const series = priorOi.map((p) => p.oi);
    const deltas: number[] = [];
    for (let i = 5; i < series.length; i++) deltas.push(series[i] - series[i - 5]);
    if (oiChange5d !== null && deltas.length >= 20) {
      const sd = stdev(deltas);
      oiChange5dZ = sd > 0 ? (oiChange5d - mean(deltas)) / sd : null;
    }
  }

  let closedBeyondWallRecently = false;
  if (closes.length > 0 && trappedWall) {
    const last3 = closes.slice(-3).map((c) => c.close);
    closedBeyondWallRecently =
      trappedSide === 'call'
        ? last3.some((c) => c >= trappedWall.strike)
        : last3.some((c) => c <= trappedWall.strike);
  }

  const panelHistory = opts.panelHistory ?? readPanel(chain.symbol);
  const previousState = (panelHistory[panelHistory.length - 1]?.trapState ?? undefined) as TrapState | undefined;

  const baseScoreArgs = {
    walls,
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
  };

  const trap = trapScore({
    ...baseScoreArgs,
    flip: { flip: flipRes.flip, regime: flipRes.regime },
  });

  const signFor = (targetConv: GexConvention) => (targetConv === convention ? 1 : -1);
  const buildConvSummary = (targetConv: GexConvention): ConventionTrapSummary => {
    const mult = signFor(targetConv);
    const convFlip = flipRes.byConvention[targetConv];
    const convTrap =
      targetConv === convention
        ? trap
        : trapScore({ ...baseScoreArgs, flip: { flip: convFlip.flip, regime: convFlip.regime } });
    return {
      totalGex: profileRes.totalGex * mult,
      totalCallGex: profileRes.totalCallGex * mult,
      totalPutGex: profileRes.totalPutGex * mult,
      totalVolumeGex: profileRes.totalVolumeGex * mult,
      totalCharm: profileRes.totalCharm * mult,
      totalVanna: profileRes.totalVanna * mult,
      flip: convFlip.flip,
      regime: convFlip.regime,
      flow: {
        perPctUp: flow.perPctUp * mult,
        perPctDown: flow.perPctDown * mult,
        net: flow.net * mult,
      },
      score: convTrap.score,
      state: convTrap.state,
      direction: convTrap.direction,
      components: convTrap.components,
      nearWall: convTrap.nearWall,
    };
  };

  return {
    spot: chain.spot,
    forward: ctx.forward,
    atmIv,
    frontT,
    headlineDte,
    atmIv30,
    include0Dte,
    totalGex: profileRes.totalGex,
    totalCallGex: profileRes.totalCallGex,
    totalPutGex: profileRes.totalPutGex,
    totalCallVolumeGex: profileRes.totalCallVolumeGex,
    totalPutVolumeGex: profileRes.totalPutVolumeGex,
    totalVolumeGex: profileRes.totalVolumeGex,
    totalCallCharm: profileRes.totalCallCharm,
    totalPutCharm: profileRes.totalPutCharm,
    totalCharm: profileRes.totalCharm,
    totalCallVanna: profileRes.totalCallVanna,
    totalPutVanna: profileRes.totalPutVanna,
    totalVanna: profileRes.totalVanna,
    gammaFlip: { flip: flipRes.flip, regime: flipRes.regime, profile: flipRes.profile },
    gammaFlipInverse: flipRes.byConvention.inverse,
    flipDistPct,
    flipDistSigma,
    byStrike: profileRes.byStrike,
    maxAbsGex: profileRes.maxAbsGex,
    walls,
    gammaWalls,
    expiryBuckets: buckets,
    strikeExpiryMatrix: matrix,
    flow,
    expiryConcentration7d: conc7d,
    zeroDteShare,
    expectedMove30,
    rv20,
    ret5d,
    adv20Dollar,
    gexPerAdv,
    oiChange1d,
    oiChange5d,
    oiChange5dZ,
    trap,
    convention,
    byConvention: {
      squeeze: buildConvSummary('squeeze'),
      inverse: buildConvSummary('inverse'),
    },
    previousState: previousState ?? null,
  };
}

/** Shared serializer from `TrapBundle` to client `TrapPayload` (used by both `/api/options` and `/api/fno`). */
export function toTrapPayload(bundle: TrapBundle) {
  return {
    available: true as const,
    spot: bundle.spot,
    forward: bundle.forward,
    atmIv: bundle.atmIv,
    headlineDte: bundle.headlineDte,
    atmIv30: bundle.atmIv30,
    include0Dte: bundle.include0Dte,
    totalGex: bundle.totalGex,
    totalCallGex: bundle.totalCallGex,
    totalPutGex: bundle.totalPutGex,
    totalCallVolumeGex: bundle.totalCallVolumeGex,
    totalPutVolumeGex: bundle.totalPutVolumeGex,
    totalVolumeGex: bundle.totalVolumeGex,
    totalCallCharm: bundle.totalCallCharm,
    totalPutCharm: bundle.totalPutCharm,
    totalCharm: bundle.totalCharm,
    totalCallVanna: bundle.totalCallVanna,
    totalPutVanna: bundle.totalPutVanna,
    totalVanna: bundle.totalVanna,
    regime: bundle.gammaFlip.regime,
    gammaFlip: { flip: bundle.gammaFlip.flip, regime: bundle.gammaFlip.regime },
    gammaFlipInverse: bundle.gammaFlipInverse,
    flipDistPct: bundle.flipDistPct,
    flipDistSigma: bundle.flipDistSigma,
    flipProfile: bundle.gammaFlip.profile,
    byStrike: bundle.byStrike,
    maxAbsGex: bundle.maxAbsGex,
    walls: bundle.walls,
    gammaWalls: bundle.gammaWalls,
    expiryBuckets: bundle.expiryBuckets,
    strikeExpiryMatrix: bundle.strikeExpiryMatrix,
    flow: bundle.flow,
    expiryConcentration7d: bundle.expiryConcentration7d,
    zeroDteShare: bundle.zeroDteShare,
    expectedMove30: bundle.expectedMove30,
    rv20: bundle.rv20,
    ret5d: bundle.ret5d,
    adv20Dollar: bundle.adv20Dollar,
    gexPerAdv: bundle.gexPerAdv,
    oiChange1d: bundle.oiChange1d,
    oiChange5d: bundle.oiChange5d,
    oiChange5dZ: bundle.oiChange5dZ,
    convention: bundle.convention,
    byConvention: bundle.byConvention,
    previousState: bundle.previousState,
    score: bundle.trap.score,
    state: bundle.trap.state,
    direction: bundle.trap.direction,
    components: bundle.trap.components,
    velocityDegraded: bundle.trap.velocityDegraded,
    nearWall: bundle.trap.nearWall,
  };
}

/** Shared serializer from `TrapBundle` to daily `PanelRow` persisted in `.cache/iv/{SYM}/panel.json`. */
export function toTrapPanelRow(dateUtc: string, bundle: TrapBundle, atmIv30: number | null): PanelRow {
  return {
    date: dateUtc,
    spot: bundle.spot,
    atmIv30,
    expectedMove30: bundle.expectedMove30,
    totalGex: bundle.totalGex,
    gammaFlip: bundle.gammaFlip.flip,
    regime: bundle.gammaFlip.regime,
    callWall: bundle.walls.call
      ? {
          strike: bundle.walls.call.strike,
          oi: bundle.walls.call.oi,
          share: bundle.walls.call.share,
          distPct: bundle.walls.call.distPct,
          distSigma: bundle.walls.call.distSigma,
        }
      : null,
    putWall: bundle.walls.put
      ? {
          strike: bundle.walls.put.strike,
          oi: bundle.walls.put.oi,
          share: bundle.walls.put.share,
          distPct: bundle.walls.put.distPct,
          distSigma: bundle.walls.put.distSigma,
        }
      : null,
    expiryConcentration7d: bundle.expiryConcentration7d,
    oiWallChange1d: bundle.oiChange1d,
    oiWallChange5d: bundle.oiChange5d,
    oiWallChangeZ: bundle.oiChange5dZ,
    trapScore: bundle.trap.score,
    trapState: bundle.trap.state,
    trapDirection: bundle.trap.direction,
    rv20: bundle.rv20,
    ret5d: bundle.ret5d,
    computedAt: new Date().toISOString(),
  };
}
