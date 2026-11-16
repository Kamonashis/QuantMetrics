// Yahoo Finance chain adapter — currently the only live provider.
// query2.finance.yahoo.com/v7/finance/options/{SYM} with a crumb session held in
// module scope (crumb rotates; cookie jar stays in memory). ~15-min delayed.
// A CBOE delayed-chain adapter or NSE bhavcopy adapter would slot in as siblings.

import {
  type RawChainRow,
  type ProviderChainResult,
  type ProviderExpiry,
  type ChainProviderOptions,
  type ChainProvider,
} from './types';
import { parseQuoteSummaryEarnings } from '../../earnings';

const UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

// ---------------------------------------------------------------------------
// Crumb session
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

export const YAHOO_UA = UA;

// ---------------------------------------------------------------------------
// Symbol resolution: indices have no chains → explicit ETF proxies with badges
// ---------------------------------------------------------------------------

export const INDEX_PROXIES: Record<string, { proxy: string; note: string }> = {
  '^IXIC': { proxy: 'QQQ', note: 'NASDAQ Composite has no options chain — using QQQ proxy (NASDAQ-100 ETF)' },
  '^NDX': { proxy: 'QQQ', note: 'NASDAQ-100 index has no reliable chain — using QQQ proxy' },
  '^SPX': { proxy: 'SPY', note: 'S&P 500 index chain is spotty on Yahoo — using SPY proxy' },
  '^DJI': { proxy: 'DIA', note: 'Dow Jones index has no options chain — using DIA proxy' },
  '^RUT': { proxy: 'IWM', note: 'Russell 2000 index chain is spotty — using IWM proxy' },
  '^VIX': { proxy: 'SPY', note: 'VIX has no listed options surface — using SPY as reference underlier' },
  '^VXN': { proxy: 'QQQ', note: 'VXN has no listed options surface — using QQQ as reference underlier' },
};

export function resolveSymbol(raw: string): { symbol: string; substitution: string | null } {
  const sym = raw.trim().toUpperCase();
  const proxy = INDEX_PROXIES[sym];
  if (proxy) return { symbol: proxy.proxy, substitution: proxy.note };
  return { symbol: sym, substitution: null };
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

export async function fetchJson(
  url: string,
  timeoutMs: number,
  cookie?: string
): Promise<unknown | null> {
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

export interface RawContract {
  strike: number;
  bid: number;
  ask: number;
  lastPrice: number;
  openInterest: number;
  volume: number;
  impliedVolatility: number;
  inTheMoney: boolean;
  contractSymbol?: string;
}

export interface FullChainResult {
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

export async function fetchFullChain(symbol: string, dateEpoch?: number): Promise<FullChainResult | null> {
  const s = await getSession();
  const base = `https://query2.finance.yahoo.com/v7/finance/options/${encodeURIComponent(symbol)}`;
  const url = dateEpoch
    ? `${base}?date=${dateEpoch}&crumb=${encodeURIComponent(s.crumb)}`
    : `${base}?crumb=${encodeURIComponent(s.crumb)}`;
  let json = (await fetchJson(url, 8000, s.cookie)) as { optionChain?: { result?: FullChainResult[] } } | null;
  let result = json?.optionChain?.result?.[0];
  if (!result) {
    // One forced crumb refresh + retry on failure (rotates)
    const s2 = await getSession(true);
    const url2 = dateEpoch
      ? `${base}?date=${dateEpoch}&crumb=${encodeURIComponent(s2.crumb)}`
      : `${base}?crumb=${encodeURIComponent(s2.crumb)}`;
    json = (await fetchJson(url2, 8000, s2.cookie)) as { optionChain?: { result?: FullChainResult[] } } | null;
    result = json?.optionChain?.result?.[0];
  }
  return result ?? null;
}

export async function fetchRiskFreeRate(): Promise<number> {
  const url = 'https://query1.finance.yahoo.com/v8/finance/chart/%5EIRX?range=5d&interval=1d&includePrePost=false';
  try {
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
// Earnings calendar — same crumb session, quoteSummary v10
// ---------------------------------------------------------------------------

export interface EarningsDates {
  /** Earliest earnings date on/after today (UTC), YYYY-MM-DD; null if none parsed. */
  nextEarningsDate: string | null;
  allDates: string[];
  isEstimate: boolean;
  asOf: string;
}

const earningsCache = new Map<string, { fetchedAt: number; data: EarningsDates | null }>();
const EARNINGS_TTL_MS = 15 * 60 * 1000; // match the ~15-min chain cache idiom

/** quoteSummary v10 → calendarEvents. 6s timeout, one forced crumb refresh + retry, 15-min in-memory cache. Never throws — null on any failure. */
export async function fetchEarningsDates(symbol: string): Promise<EarningsDates | null> {
  const sym = resolveSymbol(symbol).symbol; // reuse the ETF-proxy map for index tickers
  const cached = earningsCache.get(sym);
  if (cached && Date.now() - cached.fetchedAt < EARNINGS_TTL_MS) return cached.data;

  const todayYmd = new Date().toISOString().slice(0, 10);
  const url = (crumb: string) =>
    `https://query2.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(sym)}?modules=calendarEvents&crumb=${encodeURIComponent(crumb)}`;

  let result: EarningsDates | null = null;
  for (const force of [false, true]) {
    try {
      const s = await getSession(force);
      const json = await fetchJson(url(s.crumb), 6000, s.cookie);
      const parsed = parseQuoteSummaryEarnings(json, todayYmd);
      if (parsed) {
        result = { ...parsed, asOf: new Date().toISOString() };
        break;
      }
    } catch {
      /* fall through to the retry / null */
    }
  }
  earningsCache.set(sym, { fetchedAt: Date.now(), data: result });
  return result;
}

export function toRows(
  expiry: { expirationDate: number | string },
  contracts: RawContract[],
  type: 'call' | 'put',
  dte: number
): RawChainRow[] {
  const expRaw = expiry.expirationDate as unknown as number | string;
  const expEpoch = typeof expRaw === 'number' ? expRaw : Math.floor(new Date(expRaw).getTime() / 1000);
  const expiryDate = new Date(expEpoch * 1000).toISOString().split('T')[0];
  const rows: RawChainRow[] = [];
  for (const c of contracts) {
    const oi = c.openInterest ?? 0;
    const volume = c.volume ?? 0;
    // Keep OI>0 (positioning) OR volume>0 (flow) rows — ATM strikes often have OI=0
    // on names whose weeklies are flow-driven; without them the straddle can't be priced.
    if (!(oi > 0 || volume > 0)) continue;
    rows.push({
      expiry: expiryDate,
      dte,
      strike: c.strike,
      type,
      oi,
      volume: c.volume ?? 0,
      bid: c.bid ?? 0,
      ask: c.ask ?? 0,
      last: c.lastPrice ?? 0,
      iv: typeof c.impliedVolatility === 'number' && isFinite(c.impliedVolatility) ? c.impliedVolatility : null,
      contractSymbol: c.contractSymbol ?? `${type === 'call' ? 'C' : 'P'}${c.strike}`,
    });
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Provider implementation
// ---------------------------------------------------------------------------

async function fetchYahooChain(
  rawSymbol: string,
  opts: ChainProviderOptions = {}
): Promise<ProviderChainResult | null> {
  const { symbol, substitution } = resolveSymbol(rawSymbol);
  const maxExpiries = opts.maxExpiries ?? 12;
  const maxDte = opts.maxDte ?? 90;

  const core = await fetchFullChain(symbol);
  if (!core) return null;

  const quote = core.quote ?? {};
  const spot = quote.regularMarketPrice ?? 0;
  if (!(spot > 0)) return null;
  const quoteType = quote.quoteType ?? '';
  if (quoteType && !['EQUITY', 'ETF'].includes(quoteType)) return null;

  const rate = await fetchRiskFreeRate();
  const dividendYield =
    typeof quote.trailingAnnualDividendYield === 'number' && isFinite(quote.trailingAnnualDividendYield)
      ? quote.trailingAnnualDividendYield
      : 0;

  const now = Date.now() / 1000;
  const candidates = (core.expirationDates ?? [])
    .map((epoch) => ({ epoch, t: (epoch - now) / (365 * 86400) }))
    .filter((e) => e.t > 0)
    .sort((a, b) => a.t - b.t)
    .slice(0, maxExpiries);

  // Sequential per-expiry fetches — they cannot be batched
  const expiries: ProviderExpiry[] = [];
  const flatRows: RawChainRow[] = [];
  for (const cand of candidates) {
    const expResult = await fetchFullChain(symbol, cand.epoch);
    const expOpts = expResult?.options?.[0];
    if (!expOpts) continue;
    const dte = Math.round(cand.t * 365);
    if (dte > maxDte) continue;
    const calls = toRows(expOpts, expOpts.calls ?? [], 'call', dte);
    const puts = toRows(expOpts, expOpts.puts ?? [], 'put', dte);
    const expRaw = expOpts.expirationDate as unknown as number | string;
    const expEpoch = typeof expRaw === 'number' ? expRaw : Math.floor(new Date(expRaw).getTime() / 1000);
    const expiryDate = new Date(expEpoch * 1000).toISOString().split('T')[0];
    if (calls.length + puts.length === 0) continue;
    expiries.push({ expirationEpoch: expEpoch, expirationDate: expiryDate, calls, puts });
    flatRows.push(...calls, ...puts);
  }

  if (flatRows.length === 0) return null;

  return {
    symbol,
    substitution,
    asOf: new Date().toISOString(),
    spot,
    currency: quote.currency ?? 'USD',
    name: quote.shortName ?? symbol,
    changePct: quote.regularMarketChangePercent ?? null,
    marketCap: typeof quote.marketCap === 'number' && isFinite(quote.marketCap) ? quote.marketCap : null,
    quoteType,
    riskFreeRate: rate,
    dividendYield,
    delayedMinutes: 15,
    provider: 'yahoo',
    expirationDates: (core.expirationDates ?? []).filter((e) => e > now).sort((a, b) => a - b),
    expiries,
    rows: flatRows,
  };
}

export const yahooProvider: ChainProvider = {
  id: 'yahoo',
  supports: (symbol: string) => {
    const { symbol: resolved } = resolveSymbol(symbol);
    return /^[A-Z0-9.\-]{1,12}$/.test(resolved);
  },
  fetchChain: fetchYahooChain,
};

export { fetchYahooChain, type ProviderExpiry, type ProviderChainResult };
