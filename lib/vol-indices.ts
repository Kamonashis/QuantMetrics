// Vol Index service: CBOE volatility family from Yahoo's chart endpoint.
// Same fetch pattern as lib/market-service.ts (timeout + revalidate + graceful
// fallback), but deterministic across the full family and marked with provenance.

import { PriceBar } from './quant-engine';

export type VolIndexKey =
  | 'VIX'    // 30d S&P 500 IV
  | 'VIX9D'  // 9d short end
  | 'VIX3M'  // 3m long end
  | 'VIX6M'  // 6m curve
  | 'VXN'    // NASDAQ-100 IV
  | 'VVIX'   // vol-of-vol
  | 'SKEW'   // tail-risk pricing
  | 'IRX';   // 13-week T-bill discount rate (risk-free reference)

export const VOL_INDEX_YAHOO: Record<VolIndexKey, string> = {
  VIX: '^VIX',
  VIX9D: '^VIX9D',
  VIX3M: '^VIX3M',
  VIX6M: '^VIX6M',
  VXN: '^VXN',
  VVIX: '^VVIX',
  SKEW: '^SKEW',
  IRX: '^IRX',
};

export interface VolIndexSeries {
  symbol: string;        // ^VIX etc.
  key: VolIndexKey;
  bars: PriceBar[];
  source: 'live_market' | 'synthetic_fallback';
}

export const VOL_INDEX_NAMES: Record<VolIndexKey, string> = {
  VIX: 'CBOE Volatility Index',
  VIX9D: 'CBOE 9-Day Volatility',
  VIX3M: 'CBOE 3-Month Volatility',
  VIX6M: 'CBOE 6-Month Volatility',
  VXN: 'CBOE NASDAQ-100 Volatility',
  VVIX: 'CBOE Vol-of-Vol',
  SKEW: 'CBOE Skew Index',
  IRX: '13-Week T-Bill',
};

// Fallback levels (used only when Yahoo is unreachable) — deterministic, seeded,
// and clearly labelled synthetic. Values approximate late-2026 regime.
const FALLBACK_LEVELS: Record<VolIndexKey, { base: number; vol: number; drift: number }> = {
  VIX: { base: 17.4, vol: 0.32, drift: 0 },
  VIX9D: { base: 15.1, vol: 0.36, drift: 0 },
  VIX3M: { base: 17.9, vol: 0.27, drift: 0 },
  VIX6M: { base: 18.6, vol: 0.24, drift: 0 },
  VXN: { base: 22.3, vol: 0.34, drift: 0 },
  VVIX: { base: 91.2, vol: 0.22, drift: 0 },
  SKEW: { base: 138.0, vol: 0.10, drift: 0 },
  IRX: { base: 4.12, vol: 0.01, drift: 0 },
};

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateSyntheticVolIndex(key: VolIndexKey, days = 504): PriceBar[] {
  const cfg = FALLBACK_LEVELS[key];
  const rand = mulberry32(0x9e3779b9 ^ key.length ^ key.charCodeAt(0) * 131);
  const bars: PriceBar[] = [];
  const today = new Date('2026-11-08T12:00:00Z');
  const dates: string[] = [];
  const cursor = new Date(today);
  while (dates.length < days) {
    const dow = cursor.getDay();
    if (dow !== 0 && dow !== 6) dates.push(cursor.toISOString().split('T')[0]);
    cursor.setDate(cursor.getDate() - 1);
  }
  dates.reverse();

  // Mean-reverting lognormal walk around the base level
  const meanLevel = Math.log(cfg.base);
  let x = meanLevel;
  for (let i = 0; i < dates.length; i++) {
    const shock = (rand() * 2 - 1) * cfg.vol * 0.16;
    x += 0.06 * (meanLevel - x) + shock;
    const level = Math.max(cfg.base * 0.45, Math.exp(x));
    bars.push({
      date: dates[i],
      open: Number(level.toFixed(2)),
      high: Number((level * 1.04).toFixed(2)),
      low: Number((level * 0.96).toFixed(2)),
      close: Number(level.toFixed(2)),
      volume: 0,
    });
  }
  return bars;
}

async function fetchChartBars(symbol: string, range: string): Promise<PriceBar[] | null> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000);
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(
      symbol
    )}?range=${range}&interval=1d&includePrePost=false`;
    const res = await fetch(url, {
      signal: controller.signal,
      headers: {
        'User-Agent':
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      },
      next: { revalidate: 3600 },
    });
    clearTimeout(timeoutId);
    if (!res.ok) return null;
    const data = await res.json();
    const result = data?.chart?.result?.[0];
    const timestamps: number[] | undefined = result?.timestamp;
    const quote = result?.indicators?.quote?.[0];
    const closes: (number | null)[] | undefined = quote?.close;
    if (!timestamps || !closes) return null;
    const bars: PriceBar[] = [];
    for (let i = 0; i < timestamps.length; i++) {
      const c = closes[i];
      if (c !== null && c !== undefined && !isNaN(c) && c > 0) {
        bars.push({
          date: new Date(timestamps[i] * 1000).toISOString().split('T')[0],
          open: c,
          high: c,
          low: c,
          close: Number(c.toFixed(3)),
          volume: 0,
        });
      }
    }
    return bars.length > 20 ? bars : null;
  } catch {
    return null;
  }
}

export async function fetchVolIndexFamily(range = '2y'): Promise<Record<VolIndexKey, VolIndexSeries>> {
  const keys = Object.keys(VOL_INDEX_YAHOO) as VolIndexKey[];
  const fetched = await Promise.all(keys.map((key) => fetchChartBars(VOL_INDEX_YAHOO[key], range)));
  const out = {} as Record<VolIndexKey, VolIndexSeries>;
  keys.forEach((key, i) => {
    const bars = fetched[i];
    if (bars) {
      out[key] = { symbol: VOL_INDEX_YAHOO[key], key, bars, source: 'live_market' };
    } else {
      out[key] = { symbol: VOL_INDEX_YAHOO[key], key, bars: generateSyntheticVolIndex(key), source: 'synthetic_fallback' };
    }
  });
  return out;
}

/** Latest close of a series, or null. */
export function latestClose(series: VolIndexSeries): number | null {
  const v = series.bars[series.bars.length - 1]?.close;
  return typeof v === 'number' && isFinite(v) ? v : null;
}
