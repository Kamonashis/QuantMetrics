// Options API: server-side chain ingestion (Yahoo v7 with crumb session),
// vol-index family (v8 chart), and the snapshot store. Every external fetch is
// timeout-bounded and degrades to { available: false, reason } — never a 500.

import { NextRequest, NextResponse } from 'next/server';
import {
  impliedForward,
  cleanChain,
  fitSvi,
  skewMetrics,
  modelFreeImpliedVar,
  totalVarianceInterpolate,
  expectedMove,
  riskNeutralDensity,
  riskNeutralProbabilities,
  sviTotalVariance,
  type CleanedContract,
  type OptionContract,
} from '@/lib/options';
import { fetchVolIndexFamily } from '@/lib/vol-indices';
import {
  fetchYahooChain,
  fetchEarningsDates,
  resolveSymbol,
  type ProviderExpiry,
  type ProviderChainResult,
} from '@/lib/options/providers/yahoo';
import type { RawChainRow } from '@/lib/options/providers/types';
import { computeTrapBundle, type TrapBundle } from '@/lib/trap-server';
import {
  writeSnapshot,
  writePanelRow,
  readHistory,
  historyStats,
  readPanel,
  type PanelRow,
} from '@/lib/iv-store';
import type { TrapState, TrapDirection } from '@/lib/gex';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// ---------------------------------------------------------------------------
// Per-expiry computation (SVI pipeline)
// ---------------------------------------------------------------------------

export interface SlicePoint {
  strike: number;
  k: number;
  iv: number;
  mid: number;
  relSpread: number;
  weight: number;
  type: 'call' | 'put';
}

export interface ExpirySliceOut {
  expirationDate: string;
  dte: number;
  t: number;
  forward: number;
  atmIv: number | null;
  rr25: number | null;
  bf25: number | null;
  skew9010: number | null;
  mfivAnnualized: number | null;
  expectedMove: { oneSigmaPct: number; oneSigmaCurrency: number; straddleImpliedPct: number | null } | null;
  svi: { a: number; b: number; rho: number; m: number; sigma: number; rmseIv: number; nUsed: number; arbitrageFlags: string[] } | null;
  clean: { used: number; total: number; medianRelSpread: number | null };
  points: SlicePoint[];
}

function rowToContract(r: RawChainRow, spot: number): OptionContract {
  return {
    strike: r.strike,
    bid: r.bid,
    ask: r.ask,
    lastPrice: r.last,
    openInterest: r.oi,
    volume: r.volume,
    impliedVolatility: r.iv ?? 0,
    inTheMoney: r.type === 'call' ? r.strike < spot : r.strike > spot,
  };
}

function computeExpirySlice(
  spot: number,
  rate: number,
  dividendYield: number,
  expiry: ProviderExpiry
): ExpirySliceOut {
  const expEpoch = expiry.expirationEpoch;
  const now = Date.now() / 1000;
  const t = Math.max(1e-6, (expEpoch - now) / (365 * 86400));
  const dte = t * 365;

  const callsIn = expiry.calls.map((r: RawChainRow) => rowToContract(r, spot));
  const putsIn = expiry.puts.map((r: RawChainRow) => rowToContract(r, spot));

  const parity = impliedForward(callsIn, putsIn, t, rate);
  const forward = parity?.forward ?? spot;

  const cleaned = cleanChain(callsIn, putsIn, spot, forward, t, rate, dividendYield);
  const usedAll = [...cleaned.calls, ...cleaned.puts];

  const fit = fitSvi(usedAll);
  let atmIv: number | null = null;
  let rr25: number | null = null;
  let bf25: number | null = null;
  let skew9010: number | null = null;
  if (fit) {
    atmIv = Math.sqrt(Math.max(1e-12, sviTotalVariance(fit, 0)) / t);
    if (usedAll.length > 0) {
      const kMin = Math.min(...usedAll.map((c) => c.k));
      const kMax = Math.max(...usedAll.map((c) => c.k));
      const sk = skewMetrics(fit, spot, forward, t, rate, dividendYield, Math.max(-0.9, kMin), Math.min(0.9, kMax));
      rr25 = sk.rr25;
      bf25 = sk.bf25;
      skew9010 = sk.skew9010;
      if (atmIv === null) atmIv = sk.atmIv;
    }
  } else if (usedAll.length > 0) {
    const nearest = usedAll.reduce((a, b) => (Math.abs(b.k) < Math.abs(a.k) ? b : a));
    atmIv = nearest.iv;
  }

  const otmCalls = cleaned.calls.map((c: CleanedContract) => ({ strike: c.strike, mid: c.mid }));
  const otmPuts = cleaned.puts.map((c: CleanedContract) => ({ strike: c.strike, mid: c.mid }));
  const mfVar = modelFreeImpliedVar(otmCalls, otmPuts, forward, t, rate);
  const mfivAnnualized = mfVar !== null ? Math.sqrt(mfVar) : null;

  // Straddle-implied expected move: use the pair nearest the forward
  let straCall: number | null = null;
  let straPut: number | null = null;
  let bestAbs = Infinity;
  const pairMap = new Map<number, { c?: CleanedContract; p?: CleanedContract }>();
  for (const c of cleaned.calls) pairMap.set(c.strike, { ...pairMap.get(c.strike), c });
  for (const p of cleaned.puts) pairMap.set(p.strike, { ...pairMap.get(p.strike), p });
  for (const { c, p } of pairMap.values()) {
    if (c && p) {
      const d = Math.abs(c.strike - forward);
      if (d < bestAbs) {
        bestAbs = d;
        straCall = c.mid;
        straPut = p.mid;
      }
    }
  }
  const em = atmIv !== null ? expectedMove(spot, atmIv, t, straCall, straPut) : null;

  const points: SlicePoint[] = usedAll.map((c) => ({
    strike: c.strike,
    k: c.k,
    iv: c.iv,
    mid: c.mid,
    relSpread: c.relSpread,
    weight: c.weight,
    type: cleaned.calls.includes(c) ? 'call' : 'put',
  }));

  return {
    expirationDate: new Date(expEpoch * 1000).toISOString().split('T')[0],
    dte: Math.round(dte),
    t,
    forward,
    atmIv,
    rr25,
    bf25,
    skew9010,
    mfivAnnualized,
    expectedMove: em,
    svi: fit
      ? {
          a: fit.a,
          b: fit.b,
          rho: fit.rho,
          m: fit.m,
          sigma: fit.sigma,
          rmseIv: fit.rmseIv,
          nUsed: fit.nUsed,
          arbitrageFlags: fit.arbitrageFlags,
        }
      : null,
    clean: {
      used: cleaned.stats.used,
      total: cleaned.stats.total,
      medianRelSpread: cleaned.stats.medianRelSpread,
    },
    points,
  };
}

// ---------------------------------------------------------------------------
// Trap bundle → panel row (server-persisted daily facts)
// ---------------------------------------------------------------------------

function todayUtc(): string {
  return new Date().toISOString().split('T')[0];
}

function trapPanelRow(dateUtc: string, bundle: TrapBundle, atmIv30: number | null): PanelRow {
  return {
    date: dateUtc,
    spot: bundle.spot,
    atmIv30,
    expectedMove30: bundle.expectedMove30,
    totalGex: bundle.totalGex,
    gammaFlip: bundle.gammaFlip.flip,
    regime: bundle.gammaFlip.regime,
    callWall: bundle.walls.call
      ? { strike: bundle.walls.call.strike, oi: bundle.walls.call.oi, share: bundle.walls.call.share, distPct: bundle.walls.call.distPct, distSigma: bundle.walls.call.distSigma }
      : null,
    putWall: bundle.walls.put
      ? { strike: bundle.walls.put.strike, oi: bundle.walls.put.oi, share: bundle.walls.put.share, distPct: bundle.walls.put.distPct, distSigma: bundle.walls.put.distSigma }
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

function trapPayload(bundle: TrapBundle) {
  return {
    available: true as const,
    spot: bundle.spot,
    forward: bundle.forward,
    atmIv: bundle.atmIv,
    headlineDte: bundle.headlineDte,
    atmIv30: bundle.atmIv30,
    totalGex: bundle.totalGex,
    regime: bundle.gammaFlip.regime,
    gammaFlip: { flip: bundle.gammaFlip.flip, regime: bundle.gammaFlip.regime },
    flipProfile: bundle.gammaFlip.profile,
    byStrike: bundle.byStrike,
    maxAbsGex: bundle.maxAbsGex,
    walls: bundle.walls,
    flow: bundle.flow,
    expiryConcentration7d: bundle.expiryConcentration7d,
    expectedMove30: bundle.expectedMove30,
    rv20: bundle.rv20,
    ret5d: bundle.ret5d,
    oiChange1d: bundle.oiChange1d,
    oiChange5d: bundle.oiChange5d,
    oiChange5dZ: bundle.oiChange5dZ,
    convention: bundle.convention,
    previousState: bundle.previousState,
    score: bundle.trap.score,
    state: bundle.trap.state,
    direction: bundle.trap.direction,
    components: bundle.trap.components,
    velocityDegraded: bundle.trap.velocityDegraded,
    nearWall: bundle.trap.nearWall,
  };
}

/** Persist today's snapshot (with rows) + derived panel row. Best effort; never throws. */
function persistAccrual(
  chain: ProviderChainResult,
  symbol: string,
  iv30: number | null,
  frontRr25: number | null,
  nContractsUsed: number,
  bundle: TrapBundle | null
) {
  const dateUtc = todayUtc();
  const snapshotViable = iv30 !== null || bundle !== null;
  const snapResult = snapshotViable
    ? writeSnapshot({
        symbol,
        date: dateUtc,
        fetchedAt: new Date().toISOString(),
        spot: chain.spot,
        rate: chain.riskFreeRate,
        iv30,
        atmIv: iv30,
        skewRr25: frontRr25,
        totalVariance: iv30 !== null ? iv30 * iv30 * (30 / 365) : null,
        nContractsUsed,
        minDte: Math.min(...chain.rows.map((r: RawChainRow) => r.dte)),
        source: 'live_chain',
        rows: chain.rows,
      })
    : { ok: false, error: 'no usable IV in fetched chain — snapshot skipped' };

  if (bundle) {
    try {
      writePanelRow(symbol, trapPanelRow(dateUtc, bundle, iv30));
    } catch {
      /* panel persistence is best effort */
    }
  }
  return snapResult;
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const mode = searchParams.get('mode') || 'chain';

  // ------------------------------------------------------------------ indices
  if (mode === 'indices') {
    try {
      const range = searchParams.get('range') || '2y';
      const family = await fetchVolIndexFamily(range);
      const source = Object.values(family).every((s) => s.source === 'live_market')
        ? 'live_market'
        : 'partial_or_fallback';
      return NextResponse.json(
        { available: true, mode, fetchedAt: new Date().toISOString(), source, series: family },
        { headers: { 'Cache-Control': 'public, s-maxage=900, stale-while-revalidate=3600' } }
      );
    } catch (err) {
      return NextResponse.json(
        { available: false, mode, reason: err instanceof Error ? err.message : 'indices fetch failed' },
        { status: 200 }
      );
    }
  }

  // ------------------------------------------------------------------ earnings
  if (mode === 'earnings') {
    const symbol = (searchParams.get('symbol') || '').trim().toUpperCase();
    if (!symbol) {
      return NextResponse.json(
        { available: false, mode, reason: 'symbol is required' },
        { status: 200 }
      );
    }
    const { nextEarningsDate, allDates, isEstimate, asOf } =
      (await fetchEarningsDates(symbol)) ?? {
        nextEarningsDate: null,
        allDates: [] as string[],
        isEstimate: true,
        asOf: new Date().toISOString(),
      };
    // Null-degrading: Yahoo's earnings surface fails often; "no data" is the designed answer.
    const available = nextEarningsDate !== null;
    return NextResponse.json(
      available
        ? { available: true, mode, symbol, nextEarningsDate, allDates, isEstimate, source: 'yahoo_quotesum_v10', asOf }
        : { available: false, mode, symbol, reason: 'Yahoo quoteSummary calendarEvents unavailable (crumb/rate-limit/coverage)' },
      { headers: { 'Cache-Control': 'public, s-maxage=900, stale-while-revalidate=3600' } }
    );
  }

  // ------------------------------------------------------------------ history
  if (mode === 'history') {
    const raw = searchParams.get('symbol') || 'AAPL';
    const { symbol } = resolveSymbol(raw);
    const hist = readHistory(symbol);
    return NextResponse.json({
      available: true,
      mode,
      symbol,
      count: hist.length,
      firstDate: hist[0]?.date ?? null,
      lastDate: hist[hist.length - 1]?.date ?? null,
      history: hist,
    });
  }

  // -------------------------------------------------------------------- panel
  if (mode === 'panel') {
    const raw = searchParams.get('symbol') || 'AAPL';
    const { symbol } = resolveSymbol(raw);
    const panel = readPanel(symbol).slice(-500);
    const stats = historyStats(symbol);
    return NextResponse.json({
      available: true,
      mode,
      symbol,
      count: panel.length,
      panel,
      degraded: panel.length < 20
        ? { reason: 'insufficient_oi_history', days: panel.length, required: 20 }
        : null,
      snapshots: stats,
    });
  }

  // --------------------------------------------------------------- trap scan
  if (mode === 'trap-scan') {
    const raw = searchParams.get('symbols') || 'AAPL,MSFT,GOOGL,NVDA,TSLA,SPY';
    const symbols = raw.split(',').map((s) => s.trim().toUpperCase()).filter(Boolean).slice(0, 6);
    const rows: Array<Record<string, unknown>> = [];
    for (const sym of symbols) {
      try {
        const chain = await fetchYahooChain(sym, { maxExpiries: 8, maxDte: 90 });
        if (!chain) {
          rows.push({ symbol: sym, available: false, reason: 'chain unavailable' });
          continue;
        }
        const bundle = await computeTrapBundle(chain, {});
        if (!bundle) {
          rows.push({ symbol: sym, available: false, reason: 'trap computation failed' });
          continue;
        }
        rows.push({
          symbol: chain.symbol,
          available: true,
          substituted: chain.substitution,
          spot: chain.spot,
          score: bundle.trap.score,
          state: bundle.trap.state,
          direction: bundle.trap.direction,
          regime: bundle.gammaFlip.regime,
          gammaFlip: bundle.gammaFlip.flip,
          callWall: bundle.walls.call ? { strike: bundle.walls.call.strike, share: bundle.walls.call.share, distSigma: bundle.walls.call.distSigma } : null,
          putWall: bundle.walls.put ? { strike: bundle.walls.put.strike, share: bundle.walls.put.share, distSigma: bundle.walls.put.distSigma } : null,
          oiChange5d: bundle.oiChange5d,
          expiryConcentration7d: bundle.expiryConcentration7d,
        });
      } catch (err) {
        rows.push({ symbol: sym, available: false, reason: err instanceof Error ? err.message : 'scan failed' });
      }
    }
    return NextResponse.json(
      { available: true, mode, rows, fetchedAt: new Date().toISOString() },
      { headers: { 'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=600' } }
    );
  }

  // -------------------------------------------------------------------- chain
  // Snapshot-only options factors for a small cross-section. No accrual here —
  // callers must label these as current reads, not a historical factor panel.
  if (mode === 'factor-scan') {
    const raw = searchParams.get('symbols') || searchParams.get('symbol') || 'AAPL';
    const symbols = [...new Set(raw.split(',').map((s) => s.trim().toUpperCase()).filter(Boolean))].slice(0, 5);
    const nowSec = Date.now() / 1000;
    const result: Record<string, unknown> = {};
    for (const rawSymbol of symbols) {
      const { symbol, substitution } = resolveSymbol(rawSymbol);
      const chain = await fetchYahooChain(symbol, { maxExpiries: 6, maxDte: 90 });
      if (!chain) {
        result[rawSymbol] = { available: false, symbol, substitution, reason: 'chain_unavailable' };
        continue;
      }
      const usable = chain.expiries.filter(
        (e) => (e.expirationEpoch - nowSec) / (365 * 86400) >= 7 / 365
      );
      const slices: ExpirySliceOut[] = [];
      for (const exp of usable) {
        try {
          slices.push(computeExpirySlice(chain.spot, chain.riskFreeRate, chain.dividendYield, exp));
        } catch {
          // skip a failed expiry
        }
      }
      const termPoints = slices
        .filter((s) => s.atmIv !== null && s.atmIv > 0)
        .map((s) => ({ t: s.t, iv: s.atmIv as number }));
      const iv30 = totalVarianceInterpolate(termPoints, 30 / 365);
      const front = slices.find((s) => s.dte >= 21 && s.dte <= 45) ?? slices[0];
      const putOi = chain.rows
        .filter((r) => r.type === 'put' && r.dte <= 45)
        .reduce((sum, r) => sum + (r.oi > 0 ? r.oi : 0), 0);
      const callOi = chain.rows
        .filter((r) => r.type === 'call' && r.dte <= 45)
        .reduce((sum, r) => sum + (r.oi > 0 ? r.oi : 0), 0);
      const put25Iv = front?.points
        ?.filter((p) => p.type === 'put')
        .reduce<{ iv: number; dist: number } | null>((best, p) => {
          const dist = Math.abs(Math.abs(p.k) - 0.25);
          return !best || dist < best.dist ? { iv: p.iv, dist } : best;
        }, null)?.iv ?? null;
      result[rawSymbol] = {
        available: slices.length > 0,
        mode: 'factor-scan',
        symbol,
        substitution,
        date: new Date().toISOString().slice(0, 10),
        iv30,
        atmIv: front?.atmIv ?? null,
        put25Iv,
        putCallOiRatio: callOi > 0 ? putOi / callOi : null,
        source: 'snapshot_only',
      };
    }
    return NextResponse.json(
      { available: true, mode: 'factor-scan', asOf: new Date().toISOString(), data: result },
      { headers: { 'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=600' } }
    );
  }

  if (mode === 'chain') {
    const raw = searchParams.get('symbol') || 'AAPL';
    const { symbol, substitution } = resolveSymbol(raw);
    const chain = await fetchYahooChain(symbol, { maxExpiries: 8, maxDte: 90 });
    if (!chain) {
      return NextResponse.json({
        available: false,
        mode,
        symbol,
        substitution,
        reason: 'Chain unavailable (Yahoo rejected or timed out). Try again shortly.',
      });
    }
    const spot = chain.spot;
    const rate = chain.riskFreeRate;
    const dividendYield = chain.dividendYield;

    // Slices with T >= 7/365 — the front week is noise-dominated for the smile
    const now = Date.now() / 1000;
    const usable = chain.expiries.filter((e) => (e.expirationEpoch - now) / (365 * 86400) >= 7 / 365);

    const slices: ExpirySliceOut[] = [];
    for (const expOpts of usable) {
      try {
        slices.push(computeExpirySlice(spot, rate, dividendYield, expOpts));
      } catch {
        // Skip a failed expiry; partial data is more useful than none
      }
    }

    if (slices.length === 0) {
      return NextResponse.json({
        available: false,
        mode,
        symbol,
        substitution,
        reason: 'All expiry fetches failed or produced no usable contracts.',
      });
    }

    // Term structure + IV30 via total-variance interpolation
    const termPoints = slices
      .filter((s) => s.atmIv !== null && s.atmIv > 0)
      .map((s) => ({ t: s.t, iv: s.atmIv as number }));
    const iv30 = totalVarianceInterpolate(termPoints, 30 / 365);

    // Headline slice for RND: nearest ≥ 21 DTE slice, else the first
    const headline = slices.find((s) => s.dte >= 21) ?? slices[0];
    let rnd: { strikes: number[]; density: number[]; minDensity: number; integrated: number } | null = null;
    let pricedBand: { oneSigmaPct: number; probDrawdown15: number; probTouch10: number } | null = null;
    if (headline.svi) {
      const density = riskNeutralDensity(headline.svi, spot, headline.forward, headline.t, rate, dividendYield, 81);
      rnd = density;
      const rn = riskNeutralProbabilities(spot, spot * 0.85, headline.t, rate, dividendYield, headline.atmIv ?? 0.2);
      const rnTouch = riskNeutralProbabilities(spot, spot * 0.9, headline.t, rate, dividendYield, headline.atmIv ?? 0.2);
      const sigAtm = headline.atmIv ?? 0.2;
      pricedBand = {
        oneSigmaPct: sigAtm * Math.sqrt(headline.t) * 100,
        probDrawdown15: (1 - rn.probAbove) * 100,
        probTouch10: rnTouch.probTouchApprox * 100,
      };
    }

    // Trap bundle (GEX, walls, score) — computed for the snapshot and the payload
    const bundle = await computeTrapBundle(chain, { iv30 });

    // Snapshot accrual (best effort)
    const frontRr25 = slices.find((s) => s.dte >= 21 && s.dte <= 45)?.rr25 ?? slices[0].rr25 ?? null;
    const totalUsed = slices.reduce((s, x) => s + x.clean.used, 0);
    const snapResult = persistAccrual(chain, symbol, iv30, frontRr25, totalUsed, bundle);
    const history = historyStats(symbol);

    const ivHistory30 = readHistory(symbol)
      .map((h) => h.iv30)
      .filter((v): v is number => typeof v === 'number' && isFinite(v));

    return NextResponse.json(
      {
        available: true,
        mode,
        symbol,
        requested: raw.trim().toUpperCase(),
        substitution,
        asOf: new Date().toISOString(),
        quote: {
          price: spot,
          currency: chain.currency,
          name: chain.name,
          changePct: chain.changePct,
          marketCap: chain.marketCap,
        },
        rate,
        dividendYield,
        iv30,
        iv30Source: termPoints.length >= 2 ? 'term-structure interpolation (w = σ²T)' : 'nearest-expiry ATM IV',
        termStructure: slices.map((s) => ({ dte: s.dte, t: s.t, iv: s.atmIv })),
        slices,
        rnd,
        pricedBand,
        trap: bundle ? trapPayload(bundle) : null,
        snapshot: { ...history, iv30HistoryCount: ivHistory30.length, writeOk: snapResult.ok, snapshotError: snapResult.ok ? null : (snapResult as { error?: string }).error ?? null },
        ivHistory30,
        warnings: substitution ? [substitution] : [],
        fetchedAt: new Date().toISOString(),
      },
      { headers: { 'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=900' } }
    );
  }

  return NextResponse.json({ available: false, reason: `Unknown mode: ${mode}` }, { status: 400 });
}
