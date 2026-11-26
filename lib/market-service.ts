// Market Data Service: Yahoo Finance Stock Data Fetcher with resilient caching & realistic historical seed generators
import { PriceBar, IntradayBar } from './quant-engine';

export interface LatestQuoteSummary {
  price: number;
  prevClose: number;
  changePct: number;
  asOf: string;
}

export interface LiveQuoteTick {
  ticker: string;
  name: string;
  source: 'live_market' | 'synthetic_gbm';
  price: number;
  prevClose: number;
  changePct: number;
  asOf: string;
  bar: PriceBar;
  fetchedAt: string;
}

const VALID_TICKER_RE = /^[A-Z0-9^.=/-]{1,12}$/;

/** Strictly validates and normalizes an equity/index ticker symbol. */
export function sanitizeTickerSymbol(raw: string): string | null {
  const cleaned = raw.trim().toUpperCase();
  if (!cleaned || !VALID_TICKER_RE.test(cleaned)) return null;
  if (cleaned.includes('..') || cleaned.includes('//')) return null;
  return cleaned;
}

export interface TickerDataResponse {
  ticker: string;
  name: string;
  currency: string;
  bars: PriceBar[];
  source: 'live_market' | 'cached_institutional' | 'synthetic_gbm';
  lastUpdated: string;
  latestQuote?: LatestQuoteSummary;
}

export interface IntradayTickerDataResponse {
  ticker: string;
  name: string;
  currency: string;
  intradayBars: IntradayBar[];
  source: 'live_market' | 'cached_institutional' | 'synthetic_gbm';
  lastUpdated: string;
}

// Built-in authentic base parameters for prominent equities
export const KNOWN_TICKERS: Record<string, { name: string; basePrice: number; drift: number; vol: number }> = {
  AAPL: { name: 'Apple Inc.', basePrice: 228.45, drift: 0.16, vol: 0.24 },
  MSFT: { name: 'Microsoft Corporation', basePrice: 422.10, drift: 0.18, vol: 0.22 },
  GOOGL: { name: 'Alphabet Inc.', basePrice: 178.60, drift: 0.19, vol: 0.26 },
  NVDA: { name: 'NVIDIA Corporation', basePrice: 132.80, drift: 0.42, vol: 0.48 },
  TSLA: { name: 'Tesla, Inc.', basePrice: 246.30, drift: 0.15, vol: 0.54 },
  AMZN: { name: 'Amazon.com, Inc.', basePrice: 195.20, drift: 0.20, vol: 0.28 },
  SPY: { name: 'SPDR S&P 500 ETF Trust', basePrice: 574.50, drift: 0.12, vol: 0.14 },
  QQQ: { name: 'Invesco QQQ Trust', basePrice: 488.20, drift: 0.17, vol: 0.19 },
  JPM: { name: 'JPMorgan Chase & Co.', basePrice: 218.40, drift: 0.14, vol: 0.21 },
  META: { name: 'Meta Platforms, Inc.', basePrice: 585.10, drift: 0.28, vol: 0.32 },
};

const DETERMINISTIC_BASE_DATE = new Date('2026-11-08T12:00:00Z');

// Generates continuous realistic daily trading bars spanning N trading days up to present
export function generateRealisticStockHistory(
  ticker: string,
  days: number = 2600,
  referenceDate: Date = DETERMINISTIC_BASE_DATE,
  targetStartDate?: string
): PriceBar[] {
  let requiredDays = days;
  if (targetStartDate) {
    const startTime = new Date(targetStartDate).getTime();
    const refTime = new Date(referenceDate).getTime();
    if (!isNaN(startTime) && startTime < refTime) {
      const calendarDays = Math.ceil((refTime - startTime) / (1000 * 60 * 60 * 24));
      const estTradingDays = Math.ceil(calendarDays * (5 / 7) + 25);
      requiredDays = Math.max(requiredDays, estTradingDays);
    }
  }

  const info = KNOWN_TICKERS[ticker.toUpperCase()] || {
    name: `${ticker.toUpperCase()} Equity`,
    basePrice: 150.0,
    drift: 0.12,
    vol: 0.28,
  };

  // Seeded pseudo-random generator for consistent deterministic prices per ticker
  let seed = 0;
  for (let i = 0; i < ticker.length; i++) {
    seed = (seed * 31 + ticker.charCodeAt(i)) % 2147483647;
  }
  function pseudoRandom() {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  }

  // Box-Muller normal deviate
  function normalDeviate() {
    const u1 = Math.max(1e-7, pseudoRandom());
    const u2 = Math.max(1e-7, pseudoRandom());
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  // Generate trading day dates (excluding weekends) backwards from referenceDate
  const tradingDates: string[] = [];
  const curr = new Date(referenceDate);
  while (tradingDates.length < requiredDays) {
    const dayOfWeek = curr.getDay();
    if (dayOfWeek !== 0 && dayOfWeek !== 6) {
      tradingDates.push(curr.toISOString().split('T')[0]);
    }
    curr.setDate(curr.getDate() - 1);
  }
  tradingDates.reverse();

  const dt = 1 / 252;
  const mu = info.drift;
  const sigma = info.vol;

  // Simulate price path backwards to arrive at current market basePrice
  const totalPoints = tradingDates.length;
  const startPrice = info.basePrice * Math.exp(-(mu - 0.5 * sigma * sigma) * (totalPoints / 252));
  let p = startPrice;

  const bars: PriceBar[] = [];
  // GARCH volatility clustering simulation component
  let currentVol = sigma / Math.sqrt(252);
  const omega = 0.000005;
  const alpha = 0.08;
  const beta = 0.88;

  for (let i = 0; i < totalPoints; i++) {
    const z = normalDeviate();
    const shock = currentVol * z;
    // update GARCH-like conditional variance
    currentVol = Math.sqrt(omega + alpha * shock * shock + beta * currentVol * currentVol);

    const ret = (mu * dt) + shock;
    const prevP = p;
    p = Math.max(1.0, p * Math.exp(ret));

    const high = Math.max(prevP, p) * (1 + Math.abs(normalDeviate()) * 0.007);
    const low = Math.min(prevP, p) * (1 - Math.abs(normalDeviate()) * 0.007);
    const open = prevP * (1 + (normalDeviate() * 0.002));
    const volume = Math.floor(15000000 * (1 + 0.5 * Math.abs(shock / (sigma / Math.sqrt(252)))));

    bars.push({
      date: tradingDates[i],
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(p.toFixed(2)),
      volume,
    });
  }

  return bars;
}

function parseEndDateToUnixSeconds(endDate?: string): { p2: number; upToPresent: boolean } {
  const nowSec = Math.floor(Date.now() / 1000);
  if (!endDate) {
    return { p2: nowSec, upToPresent: true };
  }
  const todayUtc = new Date().toISOString().split('T')[0];
  if (/^\d{4}-\d{2}-\d{2}$/.test(endDate)) {
    if (endDate >= todayUtc) {
      return { p2: nowSec, upToPresent: true };
    }
    const endOfDaySec = Math.floor(new Date(`${endDate}T23:59:59Z`).getTime() / 1000);
    return {
      p2: Number.isFinite(endOfDaySec) ? Math.min(endOfDaySec, nowSec) : nowSec,
      upToPresent: false,
    };
  }
  const parsed = Math.floor(new Date(endDate).getTime() / 1000);
  if (!Number.isFinite(parsed) || parsed >= nowSec - 86400) {
    return { p2: nowSec, upToPresent: true };
  }
  return { p2: parsed, upToPresent: false };
}

const ALLOWED_RANGES = new Set([
  '1d',
  '5d',
  '7d',
  '1mo',
  '3mo',
  '6mo',
  '1y',
  '2y',
  '5y',
  '10y',
  'ytd',
  'max',
]);

function sanitizeRange(raw: string, fallback: string): string {
  const cleaned = raw.trim().toLowerCase();
  return ALLOWED_RANGES.has(cleaned) ? cleaned : fallback;
}

// Hardcoded Yahoo Finance chart hosts (primary + mirror) for transient timeout/throttle recovery
const YAHOO_CHART_HOSTS = [
  'https://query1.finance.yahoo.com',
  'https://query2.finance.yahoo.com',
] as const;

const YAHOO_REQUEST_HEADERS = {
  'User-Agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
  Accept: 'application/json',
} as const;

// Bound concurrent outbound Yahoo requests so multi-ticker bursts (10-12 symbols)
// never saturate sockets or trip upstream throttling/AbortError timeouts.
const YAHOO_MAX_CONCURRENT = 4;
let yahooActiveRequests = 0;
const yahooWaitQueue: Array<() => void> = [];

async function acquireYahooSlot(): Promise<void> {
  if (yahooActiveRequests < YAHOO_MAX_CONCURRENT) {
    yahooActiveRequests += 1;
    return;
  }
  await new Promise<void>((resolve) => {
    yahooWaitQueue.push(() => {
      yahooActiveRequests += 1;
      resolve();
    });
  });
}

function releaseYahooSlot(): void {
  yahooActiveRequests = Math.max(0, yahooActiveRequests - 1);
  const next = yahooWaitQueue.shift();
  if (next) next();
}

interface YahooChartPayload {
  chart?: {
    result?: Array<{
      meta?: {
        shortName?: string;
        currency?: string;
        regularMarketPrice?: number;
        regularMarketTime?: number;
        regularMarketOpen?: number;
        regularMarketDayHigh?: number;
        regularMarketDayLow?: number;
        regularMarketVolume?: number;
      };
      timestamp?: number[];
      indicators?: {
        quote?: Array<{
          open?: Array<number | null>;
          high?: Array<number | null>;
          low?: Array<number | null>;
          close?: Array<number | null>;
          volume?: Array<number | null>;
        }>;
        adjclose?: Array<{ adjclose?: Array<number | null> }>;
      };
    }>;
  };
}

async function fetchYahooChartJson(
  pathAndQuery: string,
  timeoutMs: number
): Promise<YahooChartPayload | null> {
  await acquireYahooSlot();
  try {
    for (let attempt = 0; attempt < YAHOO_CHART_HOSTS.length; attempt++) {
      const host = YAHOO_CHART_HOSTS[attempt];
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const res = await fetch(`${host}${pathAndQuery}`, {
          signal: controller.signal,
          headers: YAHOO_REQUEST_HEADERS,
          cache: 'no-store',
        });
        if (res.ok) {
          return (await res.json()) as YahooChartPayload;
        }
        // Only retry on transient rate-limit or server-side errors
        if (res.status !== 429 && res.status < 500) {
          return null;
        }
      } catch {
        // Expected on transient upstream timeout/abort/network blip — fall through to mirror host
      } finally {
        clearTimeout(timeoutId);
      }
    }
    return null;
  } finally {
    releaseYahooSlot();
  }
}

// Short server-side coalescing cache (5s fresh TTL) + 10-minute last-known-good stale window
// so concurrent SSE subscribers never stampede Yahoo Finance and brief upstream hiccups
// preserve live quotes instead of flickering to synthetic data.
const LIVE_QUOTE_COALESCE_TTL_MS = 5_000;
const LIVE_QUOTE_STALE_TTL_MS = 10 * 60_000;
const LIVE_QUOTE_CACHE_MAX = 100;

interface CachedQuoteEntry {
  freshUntil: number;
  staleUntil: number;
  tick: LiveQuoteTick;
}

const liveQuoteCache = new Map<string, CachedQuoteEntry>();
const liveQuoteInFlight = new Map<string, Promise<LiveQuoteTick>>();

function storeLiveQuote(sym: string, tick: LiveQuoteTick): void {
  const now = Date.now();
  if (liveQuoteCache.has(sym)) {
    liveQuoteCache.delete(sym);
  } else if (liveQuoteCache.size >= LIVE_QUOTE_CACHE_MAX) {
    const oldestKey = liveQuoteCache.keys().next().value;
    if (oldestKey !== undefined) liveQuoteCache.delete(oldestKey);
  }
  liveQuoteCache.set(sym, {
    freshUntil: now + LIVE_QUOTE_COALESCE_TTL_MS,
    staleUntil: now + LIVE_QUOTE_STALE_TTL_MS,
    tick,
  });
}

export async function fetchLiveStockData(
  ticker: string,
  range: string = '10y',
  startDate?: string,
  endDate?: string
): Promise<TickerDataResponse> {
  const sym = sanitizeTickerSymbol(ticker) ?? 'NVDA';
  const safeRange = sanitizeRange(range, '10y');
  const known = KNOWN_TICKERS[sym];
  const companyName = known?.name || `${sym} Equity`;

  const { p2, upToPresent } = parseEndDateToUnixSeconds(endDate);
  const parsedP1 = startDate ? Math.floor(new Date(startDate).getTime() / 1000) : NaN;
  const pathAndQuery =
    startDate && Number.isFinite(parsedP1) && parsedP1 < p2
      ? `/v8/finance/chart/${encodeURIComponent(sym)}?period1=${parsedP1}&period2=${p2}&interval=1d&includePrePost=false`
      : `/v8/finance/chart/${encodeURIComponent(sym)}?range=${safeRange}&interval=1d&includePrePost=false`;

  const data = await fetchYahooChartJson(pathAndQuery, 7000);
  const result = data?.chart?.result?.[0];
  if (result && result.timestamp && result.indicators?.quote?.[0]) {
    const timestamps = result.timestamp;
    const quote = result.indicators.quote[0];
    const closes = quote.close ?? [];
    const opens = quote.open ?? [];
    const highs = quote.high ?? [];
    const lows = quote.low ?? [];
    const volumes = quote.volume ?? [];
    const adjcloses = result.indicators?.adjclose?.[0]?.adjclose ?? [];

    const bars: PriceBar[] = [];
    for (let i = 0; i < timestamps.length; i++) {
      const c = closes[i];
      if (c !== null && c !== undefined && !isNaN(c) && c > 0) {
        const dateStr = new Date(timestamps[i] * 1000)
          .toISOString()
          .split('T')[0];
        bars.push({
          date: dateStr,
          open: Number((opens[i] ?? c).toFixed(2)),
          high: Number((highs[i] ?? c).toFixed(2)),
          low: Number((lows[i] ?? c).toFixed(2)),
          close: Number(c.toFixed(2)),
          volume: volumes[i] ?? 1000000,
          ...(typeof adjcloses[i] === 'number' && Number.isFinite(adjcloses[i]) && (adjcloses[i] as number) > 0
            ? { adjClose: Number((adjcloses[i] as number).toFixed(4)) }
            : {}),
        });
      }
    }

    // Reconcile Yahoo's live regularMarketPrice into the latest bar when fetching up to present
    const meta = result.meta;
    const regPrice =
      typeof meta?.regularMarketPrice === 'number' &&
      Number.isFinite(meta.regularMarketPrice) &&
      meta.regularMarketPrice > 0
        ? Number(meta.regularMarketPrice.toFixed(2))
        : null;
    const regTime =
      typeof meta?.regularMarketTime === 'number' && Number.isFinite(meta.regularMarketTime)
        ? meta.regularMarketTime
        : null;

    if (upToPresent && regPrice !== null && regTime !== null && bars.length > 0) {
      const metaDateStr = new Date(regTime * 1000).toISOString().split('T')[0];
      const lastBar = bars[bars.length - 1];
      if (lastBar.date === metaDateStr) {
        lastBar.close = regPrice;
        if (regPrice > lastBar.high) lastBar.high = regPrice;
        if (regPrice < lastBar.low) lastBar.low = regPrice;
      } else if (metaDateStr > lastBar.date) {
        const openVal =
          typeof meta?.regularMarketOpen === 'number' && meta.regularMarketOpen > 0
            ? Number(meta.regularMarketOpen.toFixed(2))
            : regPrice;
        const highVal =
          typeof meta?.regularMarketDayHigh === 'number' && meta.regularMarketDayHigh > 0
            ? Number(meta.regularMarketDayHigh.toFixed(2))
            : Math.max(openVal, regPrice);
        const lowVal =
          typeof meta?.regularMarketDayLow === 'number' && meta.regularMarketDayLow > 0
            ? Number(meta.regularMarketDayLow.toFixed(2))
            : Math.min(openVal, regPrice);
        bars.push({
          date: metaDateStr,
          open: openVal,
          high: Math.max(highVal, regPrice),
          low: Math.min(lowVal, regPrice),
          close: regPrice,
          volume:
            typeof meta?.regularMarketVolume === 'number' && meta.regularMarketVolume > 0
              ? meta.regularMarketVolume
              : 0,
        });
      }
    }

    if (bars.length >= 20) {
      const resolvedName = result.meta?.shortName || companyName;
      const fetchedAt = new Date().toISOString();
      let latestQuote: LatestQuoteSummary | undefined;
      if (upToPresent && bars.length >= 2) {
        const last = bars[bars.length - 1];
        const prev = bars[bars.length - 2];
        const changePct = prev.close > 0 ? ((last.close - prev.close) / prev.close) * 100 : 0;
        latestQuote = {
          price: last.close,
          prevClose: prev.close,
          changePct,
          asOf: last.date,
        };
        // Warm the live quote cache so an immediate SSE stream connect reuses this quote
        storeLiveQuote(sym, {
          ticker: sym,
          name: resolvedName,
          source: 'live_market',
          price: last.close,
          prevClose: prev.close,
          changePct,
          asOf: last.date,
          bar: last,
          fetchedAt,
        });
      }
      return {
        ticker: sym,
        name: resolvedName,
        currency: result.meta?.currency || 'USD',
        bars,
        source: 'live_market',
        lastUpdated: fetchedAt,
        latestQuote,
      };
    }
  }

  // Fallback: Generate 2600 realistic daily price bars (~10 years)
  const bars = generateRealisticStockHistory(sym, 2600, DETERMINISTIC_BASE_DATE, startDate);
  return {
    ticker: sym,
    name: companyName,
    currency: 'USD',
    bars,
    source: 'synthetic_gbm',
    lastUpdated: new Date().toISOString(),
  };
}

// Generates continuous realistic 1-minute intraday bars across 5 to 7 trading days
// Incorporates the empirical U-shape intraday volatility smile and discrete jump events
export function generateRealisticIntradayHistory(
  ticker: string,
  days: number = 7,
  referenceDate: Date = DETERMINISTIC_BASE_DATE
): IntradayBar[] {
  const sym = ticker.toUpperCase();
  const info = KNOWN_TICKERS[sym] || {
    name: `${sym} Equity`,
    basePrice: 150.0,
    drift: 0.14,
    vol: 0.28,
  };

  let seed = 0;
  for (let i = 0; i < sym.length; i++) {
    seed = (seed * 37 + sym.charCodeAt(i)) % 2147483647;
  }
  function pseudoRandom() {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  }
  function normalDeviate() {
    const u1 = Math.max(1e-7, pseudoRandom());
    const u2 = Math.max(1e-7, pseudoRandom());
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  // Get recent `days` trading day dates
  const tradingDates: string[] = [];
  const curr = new Date(referenceDate);
  while (tradingDates.length < days) {
    const dayOfWeek = curr.getDay();
    if (dayOfWeek !== 0 && dayOfWeek !== 6) {
      tradingDates.push(curr.toISOString().split('T')[0]);
    }
    curr.setDate(curr.getDate() - 1);
  }
  tradingDates.reverse();

  const bars: IntradayBar[] = [];
  let price = info.basePrice * 0.98;
  const baseMinuteVol = (info.vol / Math.sqrt(252 * 390));

  for (let d = 0; d < tradingDates.length; d++) {
    const dateStr = tradingDates[d];
    // Potential jump day (day 2 or 5)
    const isJumpDay = d === 2 || d === 5;
    const jumpMinute = 90 + Math.floor(pseudoRandom() * 200); // midday announcement shock

    for (let m = 0; m < 390; m++) {
      const hour = Math.floor((9 * 60 + 30 + m) / 60);
      const minute = (9 * 60 + 30 + m) % 60;
      const timeStr = `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`;
      const dt = new Date(`${dateStr}T${timeStr}:00Z`).getTime();

      // U-shape intraday multiplier: high at 09:30 (m=0), low at 13:00 (m=210), high at 16:00 (m=389)
      const uMultiplier = 1 + 1.35 * Math.exp(-m / 35) + 0.95 * Math.exp(-(389 - m) / 35);
      const minuteVol = baseMinuteVol * uMultiplier;

      // Normal diffusive shock
      let shock = minuteVol * normalDeviate();

      // Inject discrete Poisson jump on jump day
      if (isJumpDay && m === jumpMinute) {
        shock += (pseudoRandom() > 0.5 ? 1 : -1) * (0.012 + 0.008 * pseudoRandom());
      }

      const prevP = price;
      price = Math.max(1.0, price * Math.exp(shock));

      const high = Math.max(prevP, price) * (1 + Math.abs(normalDeviate()) * 0.0006);
      const low = Math.min(prevP, price) * (1 - Math.abs(normalDeviate()) * 0.0006);
      const open = prevP;
      const baseVol = 30000 * uMultiplier;
      const volume = Math.floor(baseVol * (1 + 0.6 * Math.abs(shock / minuteVol)));

      bars.push({
        timestamp: dt,
        date: dateStr,
        time: timeStr,
        minuteOfDay: m,
        open: Number(open.toFixed(2)),
        high: Number(high.toFixed(2)),
        low: Number(low.toFixed(2)),
        close: Number(price.toFixed(2)),
        volume,
      });
    }
  }

  return bars;
}

export async function fetchLiveIntradayData(
  ticker: string,
  range: string = '7d'
): Promise<IntradayTickerDataResponse> {
  const sym = sanitizeTickerSymbol(ticker) ?? 'NVDA';
  const safeRange = sanitizeRange(range, '7d');
  const known = KNOWN_TICKERS[sym];
  const companyName = known?.name || `${sym} Equity`;

  const pathAndQuery = `/v8/finance/chart/${encodeURIComponent(sym)}?range=${safeRange}&interval=1m&includePrePost=false`;
  const data = await fetchYahooChartJson(pathAndQuery, 6000);
  const result = data?.chart?.result?.[0];
  if (result && result.timestamp && result.indicators?.quote?.[0]) {
    const timestamps = result.timestamp;
    const quote = result.indicators.quote[0];
    const closes = quote.close ?? [];
    const opens = quote.open ?? [];
    const highs = quote.high ?? [];
    const lows = quote.low ?? [];
    const volumes = quote.volume ?? [];

    const intradayBars: IntradayBar[] = [];
    for (let i = 0; i < timestamps.length; i++) {
      const c = closes[i];
      if (c !== null && c !== undefined && !isNaN(c) && c > 0) {
        const barDate = new Date(timestamps[i] * 1000);
        const dateStr = barDate.toISOString().split('T')[0];
        const hours = barDate.getUTCHours();
        const minutes = barDate.getUTCMinutes();
        const timeStr = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
        // approximate minute of day (0 to 389)
        const mOfDay = Math.max(0, Math.min(389, (hours - 13) * 60 + (minutes - 30)));

        intradayBars.push({
          timestamp: timestamps[i] * 1000,
          date: dateStr,
          time: timeStr,
          minuteOfDay: mOfDay >= 0 && mOfDay < 390 ? mOfDay : i % 390,
          open: Number((opens[i] ?? c).toFixed(2)),
          high: Number((highs[i] ?? c).toFixed(2)),
          low: Number((lows[i] ?? c).toFixed(2)),
          close: Number(c.toFixed(2)),
          volume: volumes[i] ?? 10000,
        });
      }
    }

    if (intradayBars.length >= 100) {
      return {
        ticker: sym,
        name: result.meta?.shortName || companyName,
        currency: result.meta?.currency || 'USD',
        intradayBars,
        source: 'live_market',
        lastUpdated: new Date().toISOString(),
      };
    }
  }

  // Fallback: 7 days of 1-minute bars with U-shape smile and jumps
  const intradayBars = generateRealisticIntradayHistory(sym, 7);
  return {
    ticker: sym,
    name: companyName,
    currency: 'USD',
    intradayBars,
    source: 'synthetic_gbm',
    lastUpdated: new Date().toISOString(),
  };
}

/**
 * Lightweight 5-day quote snapshot for SSE streaming and background re-validation.
 * Reconciles Yahoo Finance's `meta.regularMarketPrice` into the latest daily bar,
 * coalesces concurrent callers within a 5-second window, retries against the mirror
 * host on transient timeouts, and retains last-known-good live ticks across brief blips.
 */
export async function fetchLiveQuoteSnapshot(ticker: string): Promise<LiveQuoteTick> {
  const sym = sanitizeTickerSymbol(ticker) ?? 'NVDA';
  const now = Date.now();
  const cached = liveQuoteCache.get(sym);
  if (cached && cached.freshUntil > now) {
    return cached.tick;
  }

  const pending = liveQuoteInFlight.get(sym);
  if (pending) return pending;

  const task = (async (): Promise<LiveQuoteTick> => {
    const known = KNOWN_TICKERS[sym];
    const companyName = known?.name || `${sym} Equity`;

    const pathAndQuery = `/v8/finance/chart/${encodeURIComponent(sym)}?range=5d&interval=1d&includePrePost=false`;
    const data = await fetchYahooChartJson(pathAndQuery, 6500);
    const result = data?.chart?.result?.[0];

    if (result && result.timestamp && result.indicators?.quote?.[0]) {
      const timestamps = result.timestamp;
      const quote = result.indicators.quote[0];
      const closes = quote.close ?? [];
      const opens = quote.open ?? [];
      const highs = quote.high ?? [];
      const lows = quote.low ?? [];
      const volumes = quote.volume ?? [];

      const bars: PriceBar[] = [];
      for (let i = 0; i < timestamps.length; i++) {
        const c = closes[i];
        if (c !== null && c !== undefined && !isNaN(c) && c > 0) {
          const dateStr = new Date(timestamps[i] * 1000).toISOString().split('T')[0];
          bars.push({
            date: dateStr,
            open: Number((opens[i] ?? c).toFixed(2)),
            high: Number((highs[i] ?? c).toFixed(2)),
            low: Number((lows[i] ?? c).toFixed(2)),
            close: Number(c.toFixed(2)),
            volume: volumes[i] ?? 0,
          });
        }
      }

      const meta = result.meta;
      const regPrice =
        typeof meta?.regularMarketPrice === 'number' &&
        Number.isFinite(meta.regularMarketPrice) &&
        meta.regularMarketPrice > 0
          ? Number(meta.regularMarketPrice.toFixed(2))
          : null;
      const regTime =
        typeof meta?.regularMarketTime === 'number' && Number.isFinite(meta.regularMarketTime)
          ? meta.regularMarketTime
          : null;

      if (regPrice !== null && regTime !== null && bars.length > 0) {
        const metaDateStr = new Date(regTime * 1000).toISOString().split('T')[0];
        const lastBar = bars[bars.length - 1];
        if (lastBar.date === metaDateStr) {
          lastBar.close = regPrice;
          if (regPrice > lastBar.high) lastBar.high = regPrice;
          if (regPrice < lastBar.low) lastBar.low = regPrice;
          if (typeof meta?.regularMarketVolume === 'number' && meta.regularMarketVolume > lastBar.volume) {
            lastBar.volume = meta.regularMarketVolume;
          }
        } else if (metaDateStr > lastBar.date) {
          const openVal =
            typeof meta?.regularMarketOpen === 'number' && meta.regularMarketOpen > 0
              ? Number(meta.regularMarketOpen.toFixed(2))
              : regPrice;
          const highVal =
            typeof meta?.regularMarketDayHigh === 'number' && meta.regularMarketDayHigh > 0
              ? Number(meta.regularMarketDayHigh.toFixed(2))
              : Math.max(openVal, regPrice);
          const lowVal =
            typeof meta?.regularMarketDayLow === 'number' && meta.regularMarketDayLow > 0
              ? Number(meta.regularMarketDayLow.toFixed(2))
              : Math.min(openVal, regPrice);
          bars.push({
            date: metaDateStr,
            open: openVal,
            high: Math.max(highVal, regPrice),
            low: Math.min(lowVal, regPrice),
            close: regPrice,
            volume:
              typeof meta?.regularMarketVolume === 'number' && meta.regularMarketVolume > 0
                ? meta.regularMarketVolume
                : 0,
          });
        }
      }

      if (bars.length >= 2) {
        const last = bars[bars.length - 1];
        const prev = bars[bars.length - 2];
        const tick: LiveQuoteTick = {
          ticker: sym,
          name: meta?.shortName || companyName,
          source: 'live_market',
          price: last.close,
          prevClose: prev.close,
          changePct: prev.close > 0 ? ((last.close - prev.close) / prev.close) * 100 : 0,
          asOf: last.date,
          bar: last,
          fetchedAt: new Date().toISOString(),
        };
        storeLiveQuote(sym, tick);
        return tick;
      }
    }

    // Preserve the most recent live quote across brief upstream timeouts/blips
    const staleEntry = liveQuoteCache.get(sym);
    if (staleEntry && staleEntry.staleUntil > Date.now()) {
      return staleEntry.tick;
    }

    // Offline fallback tick from deterministic seed
    const fallbackBars = generateRealisticStockHistory(sym, 10, DETERMINISTIC_BASE_DATE);
    const last = fallbackBars[fallbackBars.length - 1];
    const prev = fallbackBars[fallbackBars.length - 2] ?? last;
    return {
      ticker: sym,
      name: companyName,
      source: 'synthetic_gbm',
      price: last.close,
      prevClose: prev.close,
      changePct: prev.close > 0 ? ((last.close - prev.close) / prev.close) * 100 : 0,
      asOf: last.date,
      bar: last,
      fetchedAt: new Date().toISOString(),
    };
  })();

  liveQuoteInFlight.set(sym, task);
  try {
    return await task;
  } finally {
    liveQuoteInFlight.delete(sym);
  }
}
