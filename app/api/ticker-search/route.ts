import { NextRequest, NextResponse } from 'next/server';

const UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

const MAX_Q_LENGTH = 32;
const CACHE_TTL_MS = 5 * 60 * 1000;
const CACHE_MAX_ENTRIES = 200;
const ALLOWED_TYPES = new Set(['EQUITY', 'ETF', 'INDEX']);
const MAX_RESULTS = 8;

export interface TickerSuggestion {
  symbol: string;
  name: string;
  exchange: string;
  quoteType: string;
}

// Module-scope short-lived cache — Yahoo gets hit once per query per 5 minutes.
const cache = new Map<string, { at: number; quotes: TickerSuggestion[] }>();

function pruneCache() {
  if (cache.size <= CACHE_MAX_ENTRIES) return;
  const byAge = [...cache.entries()].sort((a, b) => a[1].at - b[1].at);
  for (let i = 0; i < byAge.length - CACHE_MAX_ENTRIES; i++) {
    cache.delete(byAge[i][0]);
  }
}

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const q = (searchParams.get('q') || '').trim().slice(0, MAX_Q_LENGTH);

  if (!q) {
    return NextResponse.json({ quotes: [], source: 'empty' });
  }

  const key = q.toLowerCase();
  const cached = cache.get(key);
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) {
    return NextResponse.json({ quotes: cached.quotes, source: 'yahoo', cached: true });
  }

  try {
    const url =
      `https://query1.finance.yahoo.com/v1/finance/search` +
      `?q=${encodeURIComponent(q)}&quotesCount=10&newsCount=0&listsCount=0` +
      `&enableFuzzyQuery=false&enableNavLinks=false&enableResearchReports=false`;

    const res = await fetch(url, {
      signal: AbortSignal.timeout(4000),
      headers: { 'User-Agent': UA },
      next: { revalidate: 300 },
    });
    if (!res.ok) {
      throw new Error(`Yahoo search responded ${res.status}`);
    }

    const data: unknown = await res.json();
    const rawQuotes =
      data && typeof data === 'object' && Array.isArray((data as { quotes?: unknown }).quotes)
        ? ((data as { quotes: Array<Record<string, unknown>> }).quotes ?? [])
        : [];

    const seen = new Set<string>();
    const quotes: TickerSuggestion[] = [];
    for (const raw of rawQuotes) {
      const symbol =
        typeof raw?.symbol === 'string' ? raw.symbol.trim().toUpperCase() : '';
      const quoteType =
        typeof raw?.quoteType === 'string' ? raw.quoteType.trim().toUpperCase() : '';
      if (!symbol || !ALLOWED_TYPES.has(quoteType) || seen.has(symbol)) continue;
      seen.add(symbol);
      const name =
        (typeof raw?.shortname === 'string' && raw.shortname) ||
        (typeof raw?.longname === 'string' && raw.longname) ||
        symbol;
      const exchange =
        (typeof raw?.exchDisp === 'string' && raw.exchDisp) ||
        (typeof raw?.exchange === 'string' && raw.exchange) ||
        '';
      quotes.push({ symbol, name, exchange, quoteType });
      if (quotes.length >= MAX_RESULTS) break;
    }

    if (quotes.length > 0) {
      cache.set(key, { at: Date.now(), quotes });
      pruneCache();
    }
    return NextResponse.json({ quotes, source: 'yahoo' });
  } catch (err) {
    const reason = err instanceof Error ? err.message : 'search unavailable';
    return NextResponse.json({ quotes: [], source: 'unavailable', reason });
  }
}
