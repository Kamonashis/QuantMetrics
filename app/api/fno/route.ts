// F&O (US universe) API: single route with ?mode= — follows the repo convention.
// Modes: panel (summary across expiries), chain (strike table for one expiry), status.
// All reads are served from a live Yahoo snapshot (server-cached ~15 min by the provider);
// OI history accrues only via app usage (.cache/iv) — it can never be backfilled.

import { NextRequest, NextResponse } from 'next/server';
import { totalVarianceInterpolate, impliedVol, bsVega, bsDelta, type OptionContract } from '@/lib/options';
import {
  fetchYahooChain,
  resolveSymbol,
  type ProviderChainResult,
} from '@/lib/options/providers/yahoo';
import { buildExpiryLadder } from '@/lib/fno/expiry-ladder';
import { oiLadder, pcr, maxPain } from '@/lib/fno/positioning';
import { buildTermStructure } from '@/lib/fno/term-structure';
import { expectedMoves } from '@/lib/fno/expected-move';
import { readHistory, historyStats, writeSnapshot, writePanelRow } from '@/lib/iv-store';
import { computeTrapBundle } from '@/lib/trap-server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const todayUtc = () => new Date().toISOString().split('T')[0];

function buildLadderWithRows(chain: ProviderChainResult) {
  const spot = chain.spot;
  const rate = chain.riskFreeRate;
  return buildExpiryLadder(chain.rows, spot, rate, chain.dividendYield).map((g) => ({
    ...g,
    rows: chain.rows.filter((r) => r.expiry === g.expiry),
  }));
}

async function persistAccrual(
  chain: ProviderChainResult,
  symbol: string,
  iv30: number | null,
  frontRr25: number | null,
  totalUsed: number
) {
  // Short-circuit: if today's snapshot is fresh (< 20 min), skip the heavy trap bundle recompute.
  const history = readHistory(symbol);
  const fresh = history.some((h) => h.date === todayUtc() && (Date.now() - new Date(h.fetchedAt).getTime()) < 20 * 60 * 1000);
  if (fresh && iv30 !== null) return { ok: true, reused: true };

  const bundle = await computeTrapBundle(chain, { iv30 });
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
        nContractsUsed: totalUsed,
        minDte: Math.min(...chain.rows.map((r) => r.dte)),
        source: 'live_chain',
        rows: chain.rows,
      })
    : { ok: false, error: 'no usable IV in fetched chain — snapshot skipped' };

  if (bundle) {
    try {
      writePanelRow(symbol, {
        date: dateUtc,
        spot: bundle.spot,
        atmIv30: iv30,
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
      });
    } catch {
      /* panel persistence is best effort */
    }
  }
  return snapResult;
}

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const mode = searchParams.get('mode') || 'panel';

  // ------------------------------------------------------------------ status
  if (mode === 'status') {
    const raw = searchParams.get('symbol') || 'AAPL';
    const { symbol } = resolveSymbol(raw);
    const stats = historyStats(symbol);
    const staleDays = stats.lastDate
      ? Math.floor((Date.now() - new Date(stats.lastDate + 'T00:00:00Z').getTime()) / 86400000)
      : null;
    return NextResponse.json({
      available: true,
      mode,
      symbol,
      sessions: stats.count,
      firstDate: stats.firstDate,
      lastDate: stats.lastDate,
      staleDays,
      accrualNote: stats.count < 20 ? `Accruing — ${stats.count}/20 sessions for full OI percentile coverage` : null,
    });
  }

  // ------------------------------------------------------------------ panel
  if (mode === 'panel') {
    const raw = searchParams.get('symbol') || 'AAPL';
    const { symbol, substitution } = resolveSymbol(raw);
    const chain = await fetchYahooChain(symbol, { maxExpiries: 12, maxDte: 90 });
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
    const q = chain.dividendYield;
    const rows = chain.rows;

    const ladder = buildLadderWithRows(chain);
    const ivPoints = ladder.filter((l) => l.atmIv !== null && l.atmIv > 0).map((l) => ({ t: l.t, iv: l.atmIv as number }));
    const iv30 = totalVarianceInterpolate(ivPoints, 30 / 365);
    const termStructure = buildTermStructure(ladder, spot, rate, q);
    const ladderRows = oiLadder(rows, spot);
    const pcrBundle = pcr(rows, spot);
    const pain = maxPain(rows, spot);
    const moves = expectedMoves(
      ladder.slice(0, 8).map((l) => ({ dte: l.dte, expiry: l.expiry, t: l.t, atmIv: l.atmIv, rows: l.rows })),
      spot
    );
    const totalOi = rows.reduce((s, r) => s + r.oi, 0);
    const frontOi = ladder.filter((l) => l.dte <= 7).reduce((s, l) => s + l.totalOi, 0);
    const concentration7d = totalOi > 0 ? frontOi / totalOi : 0;
    const strikesTotal = rows.length;
    const strikesUsed = rows.filter((r) => r.bid > 0).length;

    // Persist accrual (best effort) — keeps both F&O and IV/Trap features learning from the same snapshot
    const frontRr25 = termStructure.points.find((p) => p.dte >= 15 && p.dte <= 45)?.rr25 ?? null;
    persistAccrual(chain, symbol, iv30, frontRr25, strikesUsed).catch(() => undefined);

    const stats = historyStats(symbol);
    const staleDays = stats.lastDate
      ? Math.floor((Date.now() - new Date(stats.lastDate + 'T00:00:00Z').getTime()) / 86400000)
      : null;

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
          prevClose: chain.prevClose ?? null,
          asOf: chain.quoteAsOf ?? null,
          source: 'yahoo_regular_market',
          currency: chain.currency,
          name: chain.name,
          changePct: chain.changePct,
          marketCap: chain.marketCap,
        },
        rate,
        dividendYield: q,
        iv30,
        stale: staleDays !== null && staleDays > 3,
        staleDays,
        sessions: stats.count,
        firstDate: stats.firstDate,
        lastDate: stats.lastDate,
        expiryLadder: ladder.map((l) => ({
          expiry: l.expiry,
          dte: l.dte,
          forward: l.forward,
          atmStrike: l.atmStrike,
          atmIv: l.atmIv,
          callOi: l.callOi,
          putOi: l.putOi,
          totalOi: l.totalOi,
          share7d: l.share7d,
        })),
        oiLadder: ladderRows,
        pcr: pcrBundle,
        maxPain: pain,
        termStructure,
        expectedMoves: moves,
        concentration7d,
        strikesUsed,
        strikesTotal,
        fetchedAt: new Date().toISOString(),
      },
      { headers: { 'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=900' } }
    );
  }

  // ------------------------------------------------------------------ chain
  if (mode === 'chain') {
    const raw = searchParams.get('symbol') || 'AAPL';
    const { symbol } = resolveSymbol(raw);
    const expiryParam = searchParams.get('expiry') || undefined;
    const chain = await fetchYahooChain(symbol, { maxExpiries: 12, maxDte: 90 });
    if (!chain) {
      return NextResponse.json({ available: false, mode, symbol, reason: 'Chain unavailable' });
    }
    const spot = chain.spot;
    const rate = chain.riskFreeRate;
    const q = chain.dividendYield;

    const targetRows = expiryParam
      ? chain.rows.filter((r) => r.expiry === expiryParam)
      : chain.rows.filter((r) => r.dte === Math.min(...chain.rows.map((x) => x.dte)));
    if (targetRows.length === 0) {
      return NextResponse.json({ available: false, mode, symbol, reason: 'No contracts for requested expiry' });
    }
    const dte = targetRows[0].dte;

    const strikes = Array.from(new Set(targetRows.map((r) => r.strike)));
    const atmStrike = strikes.reduce((best, s) => (Math.abs(s - spot) < Math.abs(best - spot) ? s : best), strikes[0]);
    const atmIdx = strikes.sort((a, b) => a - b).indexOf(atmStrike);
    const lo = Math.max(0, atmIdx - 10);
    const hi = Math.min(strikes.length, atmIdx + 11);
    const windowStrikes = new Set(strikes.slice(lo, hi));
    const windowRows = targetRows.filter((r) => windowStrikes.has(r.strike));

    // Today's expiry hasn't settled yet — floor T at half a session, never 0.
    const t = Math.max(0.5 / 365, dte / 365);
    const strikesOut = strikes.slice(lo, hi).map((strike) => {
      const side = (type: 'call' | 'put') => {
        const r = windowRows.find((x) => x.strike === strike && x.type === type);
        if (!r) return null;
        const mid = (r.bid + r.ask) / 2;
        const iv = impliedVol(mid, spot, strike, t, rate, q, type);
        const ivValid = iv !== null && isFinite(iv) && iv > 0.001 && iv < 4;
        const vega = ivValid ? bsVega(spot, strike, t, rate, q, iv as number) : null;
        const delta = ivValid ? bsDelta(spot, strike, t, rate, q, iv as number, type) : null;
        const spreadPct = mid > 0 ? ((r.ask - r.bid) / mid) * 100 : null;
        return {
          bid: r.bid,
          ask: r.ask,
          mid: Number(mid.toFixed(4)),
          iv: ivValid ? Number((iv as number * 100).toFixed(2)) : null,
          oi: r.oi,
          volume: r.volume,
          vega: vega !== null ? Number((vega / 100).toFixed(4)) : null,
          delta: delta !== null ? Number(delta.toFixed(4)) : null,
          spreadPct: spreadPct !== null ? Number(spreadPct.toFixed(2)) : null,
        };
      };
      return { strike, call: side('call'), put: side('put') };
    });

    return NextResponse.json(
      {
        available: true,
        mode,
        symbol,
        expiry: targetRows[0].expiry,
        dte,
        asOf: new Date().toISOString(),
        quote: { price: spot },
        atmStrike,
        windowStrikes: strikes.slice(lo, hi),
        rows: strikesOut,
        fetchedAt: new Date().toISOString(),
      },
      { headers: { 'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=600' } }
    );
  }

  return NextResponse.json({ available: false, reason: `Unknown mode: ${mode}` }, { status: 400 });
}
