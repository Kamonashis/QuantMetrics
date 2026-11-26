// Fundamentals API: Yahoo v10 quoteSummary (crumb session) for factor research.
// Every module group is timeout-bounded and degrades to { available: false, reason }.
// Snapshots are accrued daily; no point-in-time filing dates exist on this surface.

import { NextRequest, NextResponse } from 'next/server';
import { getSession, fetchJson } from '@/lib/options/providers/yahoo';
import { parseFundamentalSnapshot, parseAnalystSnapshot } from '@/lib/fundamentals';
import { persistFundamentalSnapshot, readFundamentalHistory } from '@/lib/fundamental-store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const FUNDAMENTALS_MODULES = [
  'assetProfile',
  'price',
  'defaultKeyStatistics',
  'incomeStatementHistory',
  'incomeStatementHistoryQuarterly',
  'balanceSheetHistory',
  'balanceSheetHistoryQuarterly',
  'cashflowStatementHistory',
  'cashflowStatementHistoryQuarterly',
].join(',');

const ANALYST_MODULES = [
  'earningsHistory',
  'earningsTrend',
  'epsTrend',
  'recommendationTrend',
].join(',');

const CACHE_TTL_MS = 15 * 60 * 1000;
const cache = new Map<string, { at: number; payload: Record<string, unknown> }>();
const MAX_SYMBOLS = 10;

function symbolOf(raw: string | null): string | null {
  if (!raw) return null;
  const sym = raw.trim().toUpperCase().replace(/[^A-Z0-9^.=/-]/g, '');
  return sym.length > 0 && sym.length <= 12 ? sym : null;
}

async function fetchModules(symbol: string, modules: string, cookie: string, crumb: string) {
  const url =
    `https://query1.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(symbol)}` +
    `?modules=${encodeURIComponent(modules)}&crumb=${encodeURIComponent(crumb)}`;
  const first = await fetchJson(url, 6500, cookie);
  if (first !== null) return first;
  const forced = await getSession(true).catch(() => null);
  if (!forced) return null;
  return fetchJson(
    `https://query2.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(symbol)}` +
      `?modules=${encodeURIComponent(modules)}&crumb=${encodeURIComponent(forced.crumb)}`,
    6500,
    forced.cookie
  );
}

export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  const rawSymbols = params.get('symbols') ?? params.get('symbol') ?? 'NVDA';
  const symbols = [...new Set(rawSymbols.split(',').map(symbolOf).filter((s): s is string => s !== null))].slice(
    0,
    MAX_SYMBOLS
  );
  if (symbols.length === 0) {
    return NextResponse.json({ success: false, error: 'No valid symbols' }, { status: 400 });
  }

  const now = new Date();
  const observedAt = now.toISOString().slice(0, 10);
  const todayKey = `${observedAt}|${symbols.join(',')}`;
  const cached = cache.get(todayKey);
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) {
    return NextResponse.json(
      { success: true, timestamp: now.toISOString(), cached: true, source: 'yahoo_quotesum_v10', data: cached.payload },
      { headers: { 'Cache-Control': 'public, s-maxage=900, stale-while-revalidate=300' } }
    );
  }

  const session = await getSession().catch(() => null);
  const data: Record<string, unknown> = {};

  await Promise.all(
    symbols.map(async (symbol) => {
      let fundamentalJson: unknown = null;
      let analystJson: unknown = null;
      let reason: string | null = null;
      if (!session) {
        reason = 'yahoo_crumb_unavailable';
      } else {
        const [a, b] = await Promise.all([
          fetchModules(symbol, FUNDAMENTALS_MODULES, session.cookie, session.crumb),
          fetchModules(symbol, ANALYST_MODULES, session.cookie, session.crumb),
        ]);
        fundamentalJson = a;
        analystJson = b;
        if (a === null && b === null) reason = 'yahoo_quotesum_unavailable';
      }

      const snapshot = parseFundamentalSnapshot(symbol, observedAt, fundamentalJson);
      const analyst = parseAnalystSnapshot(symbol, observedAt, analystJson);
      if (snapshot || analyst) {
        persistFundamentalSnapshot(symbol, observedAt, snapshot, analyst);
      }
      const history = readFundamentalHistory(symbol);
      data[symbol] = {
        available: snapshot !== null || analyst !== null || history.sessions > 0,
        reason: reason ?? (sectorMissing(snapshot) ? 'sector_unavailable' : undefined),
        sector: snapshot?.sector ?? null,
        industry: snapshot?.industry ?? null,
        marketCap: snapshot?.marketCap ?? null,
        sessions: history.sessions,
        firstDate: history.firstDate,
        lastDate: history.lastDate,
        degraded: history.degraded,
        snapshot,
        analyst,
      };
    })
  );

  const payload = { success: true, timestamp: now.toISOString(), source: 'yahoo_quotesum_v10', data };
  cache.set(todayKey, { at: Date.now(), payload: data });
  return NextResponse.json(payload, {
    headers: { 'Cache-Control': 'public, s-maxage=900, stale-while-revalidate=300' },
  });
}

function sectorMissing(snapshot: { sector: string | null } | null): boolean {
  return snapshot !== null && snapshot.sector === null;
}
