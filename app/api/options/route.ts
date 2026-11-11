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
} from '@/lib/options';
import { fetchVolIndexFamily } from '@/lib/vol-indices';
import { writeSnapshot, readHistory, historyStats } from '@/lib/iv-store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

// ---------------------------------------------------------------------------
// Crumb session (module scope — crumb rotates; cookie jar stays in memory)
// ---------------------------------------------------------------------------

interface CrumbSession {
  cookie: string;
  crumb: string;
  fetchedAt: number;
}

let session: CrumbSession | null = null;
let sessionPromise: Promise<CrumbSession> | null = null;

async function getSession(forceRefresh = false): Promise<CrumbSession> {
  if (!forceRefresh && session && Date.now() - session.fetchedAt < 45 * 60 * 1000) {
    return session;
  }
  if (sessionPromise) return sessionPromise;
  sessionPromise = (async () => {
    const fc = await fetch('https://fc.yahoo.com', {
      headers: { 'User-Agent': UA },
      cache: 'no-store',
    }).catch(() => null as Response | null);
    const rawCookies: string[] =
      (fc?.headers as unknown as { getSetCookie?: () => string[] })?.getSetCookie?.() ??
      (fc?.headers.get('set-cookie') ? [fc.headers.get('set-cookie') as string] : []);
    const cookie = rawCookies.map((c) => c.split(';')[0]).join('; ');
    const crumbRes = await fetch('https://query1.finance.yahoo.com/v1/test/getcrumb', {
      headers: { 'User-Agent': UA, Cookie: cookie },
      cache: 'no-store',
    });
    const crumb = await crumbRes.text();
    if (!crumbRes.ok || !crumb || crumb.includes('<') || crumb.length > 64) {
      throw new Error('Failed to obtain Yahoo crumb');
    }
    const s = { cookie, crumb, fetchedAt: Date.now() };
    session = s;
    return s;
  })();
  try {
    return await sessionPromise;
  } finally {
    sessionPromise = null;
  }
}

// ---------------------------------------------------------------------------
// Symbol resolution: indices have no chains → explicit ETF proxies with badges
// ---------------------------------------------------------------------------

const INDEX_PROXIES: Record<string, { proxy: string; note: string }> = {
  '^IXIC': { proxy: 'QQQ', note: 'NASDAQ Composite has no options chain — using QQQ proxy (NASDAQ-100 ETF)' },
  '^NDX': { proxy: 'QQQ', note: 'NASDAQ-100 index has no reliable chain — using QQQ proxy' },
  '^SPX': { proxy: 'SPY', note: 'S&P 500 index chain is spotty on Yahoo — using SPY proxy' },
  '^DJI': { proxy: 'DIA', note: 'Dow Jones index has no options chain — using DIA proxy' },
  '^RUT': { proxy: 'IWM', note: 'Russell 2000 index chain is spotty — using IWM proxy' },
  '^VIX': { proxy: 'SPY', note: 'VIX has no listed options surface — using SPY as reference underlier' },
  '^VXN': { proxy: 'QQQ', note: 'VXN has no listed options surface — using QQQ as reference underlier' },
};

function resolveSymbol(raw: string): { symbol: string; substitution: string | null } {
  const sym = raw.trim().toUpperCase();
  const proxy = INDEX_PROXIES[sym];
  if (proxy) return { symbol: proxy.proxy, substitution: proxy.note };
  return { symbol: sym, substitution: null };
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

async function fetchJson(url: string, timeoutMs: number, cookie?: string): Promise<unknown | null> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      headers: { 'User-Agent': UA, ...(cookie ? { Cookie: cookie } : {}) },
      cache: 'no-store',
    });
    clearTimeout(timeoutId);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    clearTimeout(timeoutId);
    return null;
  }
}

interface RawContract {
  strike: number;
  bid: number;
  ask: number;
  lastPrice: number;
  openInterest: number;
  volume: number;
  impliedVolatility: number;
  inTheMoney: boolean;
}

interface ChainResult {
  quote: {
    regularMarketPrice?: number;
    currency?: string;
    shortName?: string;
    quoteType?: string;
    trailingAnnualDividendYield?: number | null;
    regularMarketChangePercent?: number;
    marketCap?: number;
  };
  expirationDates: number[];
  options: { expirationDate: number | string; calls: RawContract[]; puts: RawContract[] }[];
}

async function fetchChain(symbol: string, dateEpoch?: number): Promise<ChainResult | null> {
  const s = await getSession();
  const base = `https://query2.finance.yahoo.com/v7/finance/options/${encodeURIComponent(symbol)}`;
  const url = dateEpoch
    ? `${base}?date=${dateEpoch}&crumb=${encodeURIComponent(s.crumb)}`
    : `${base}?crumb=${encodeURIComponent(s.crumb)}`;
  let json = (await fetchJson(url, 8000, s.cookie)) as { optionChain?: { result?: ChainResult[] } } | null;
  let result = json?.optionChain?.result?.[0];
  if (!result) {
    // One forced crumb refresh + retry on failure (rotates)
    const s2 = await getSession(true);
    const url2 = dateEpoch
      ? `${base}?date=${dateEpoch}&crumb=${encodeURIComponent(s2.crumb)}`
      : `${base}?crumb=${encodeURIComponent(s2.crumb)}`;
    json = (await fetchJson(url2, 8000, s2.cookie)) as { optionChain?: { result?: ChainResult[] } } | null;
    result = json?.optionChain?.result?.[0];
  }
  return result ?? null;
}

async function fetchRiskFreeRate(): Promise<number> {
  try {
    const url = 'https://query1.finance.yahoo.com/v8/finance/chart/%5EIRX?range=5d&interval=1d&includePrePost=false';
    const json = (await fetchJson(url, 5000)) as {
      chart?: { result?: { indicators?: { quote?: { close?: (number | null)[] }[] } }[] };
    } | null;
    const closes = json?.chart?.result?.[0]?.indicators?.quote?.[0]?.close ?? [];
    const valid = closes.filter((v): v is number => typeof v === 'number' && isFinite(v));
    const last = valid[valid.length - 1];
    if (typeof last === 'number' && last > 0) return last / 100;
  } catch {
    /* fall through to default */
  }
  return 0.04; // conservative default — labelled in UI as fallback when used
}

// ---------------------------------------------------------------------------
// Per-expiry computation
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

function computeExpirySlice(
  spot: number,
  rate: number,
  dividendYield: number,
  expiry: ChainResult['options'][number]
): ExpirySliceOut {
  const expRaw = expiry.expirationDate as unknown as number | string;
  const expEpoch =
    typeof expRaw === 'number'
      ? expRaw
      : Math.floor(new Date(expRaw).getTime() / 1000);
  const now = Date.now() / 1000;
  const t = Math.max(1e-6, (expEpoch - now) / (365 * 86400));
  const dte = t * 365;

  const callsIn = (expiry.calls ?? []).map((c) => ({ ...c }));
  const putsIn = (expiry.puts ?? []).map((c) => ({ ...c }));

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
    // No fit: fall back to nearest-ATM raw IVs
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
// Route handlers
// ---------------------------------------------------------------------------

function todayUtc(): string {
  return new Date().toISOString().split('T')[0];
}

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

  // -------------------------------------------------------------------- chain
  if (mode === 'chain') {
    const raw = searchParams.get('symbol') || 'AAPL';
    const { symbol, substitution } = resolveSymbol(raw);
    const core = await fetchChain(symbol);
    if (!core) {
      return NextResponse.json({
        available: false,
        mode,
        symbol,
        substitution,
        reason: 'Chain unavailable (Yahoo rejected or timed out). Try again shortly.',
      });
    }

    const quote = core.quote ?? {};
    const spot = quote.regularMarketPrice ?? 0;
    const quoteType = quote.quoteType ?? '';
    if (!(spot > 0)) {
      return NextResponse.json({ available: false, mode, symbol, reason: 'No spot price in chain payload', substitution });
    }
    if (quoteType && !['EQUITY', 'ETF'].includes(quoteType)) {
      return NextResponse.json({
        available: false,
        mode,
        symbol,
        substitution,
        reason: `Options chains are only supported for equities/ETFs (got ${quoteType}).`,
      });
    }

    const rate = await fetchRiskFreeRate();
    const dividendYield =
      typeof quote.trailingAnnualDividendYield === 'number' && isFinite(quote.trailingAnnualDividendYield)
        ? quote.trailingAnnualDividendYield
        : 0;

    // Pick up to 8 expiries with T >= 7/365
    const now = Date.now() / 1000;
    const candidates = (core.expirationDates ?? [])
      .map((epoch) => ({ epoch, t: (epoch - now) / (365 * 86400) }))
      .filter((e) => e.t >= 7 / 365)
      .sort((a, b) => a.t - b.t)
      .slice(0, 8);

    // Fetch each expiry sequentially (they cannot be batched)
    const slices: ExpirySliceOut[] = [];
    for (const cand of candidates) {
      const expResult = await fetchChain(symbol, cand.epoch);
      const expOpts = expResult?.options?.[0];
      if (!expOpts) continue;
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

    // Snapshot accrual (best effort) — only persist when at least one slice produced an ATM IV
    const dateUtc = todayUtc();
    const frontRr25 = slices.find((s) => s.dte >= 21 && s.dte <= 45)?.rr25 ?? slices[0].rr25 ?? null;
    const totalUsed = slices.reduce((s, x) => s + x.clean.used, 0);
    const snapshotViable = iv30 !== null || slices.some((s) => s.atmIv !== null);
    const snapResult = snapshotViable
      ? writeSnapshot({
          symbol,
          date: dateUtc,
          fetchedAt: new Date().toISOString(),
          spot,
          rate,
          iv30,
          atmIv: iv30,
          skewRr25: frontRr25,
          totalVariance: iv30 !== null ? iv30 * iv30 * (30 / 365) : null,
          nContractsUsed: totalUsed,
          minDte: slices[0]?.dte ?? 0,
          source: 'live_chain',
        })
      : { ok: false, error: 'no usable IV in fetched chain — snapshot skipped' };
    const history = historyStats(symbol);

    const nearest30 = slices.find((s) => s.dte >= 21) ?? slices[0];
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
          currency: quote.currency ?? 'USD',
          name: quote.shortName ?? symbol,
          changePct: quote.regularMarketChangePercent ?? null,
          marketCap: typeof quote.marketCap === 'number' && isFinite(quote.marketCap) ? quote.marketCap : null,
        },
        rate,
        dividendYield,
        iv30,
        iv30Source: termPoints.length >= 2 ? 'term-structure interpolation (w = σ²T)' : 'nearest-expiry ATM IV',
        termStructure: slices.map((s) => ({ dte: s.dte, t: s.t, iv: s.atmIv })),
        slices,
        rnd,
        pricedBand,
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
