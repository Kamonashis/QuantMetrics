// Pure TypeScript cache reconciliation helpers for propagating live market stream
// updates cleanly into both quote maps and daily OHLCV bar caches without clobbering
// historical backtest slices or causing redundant React re-renders.

import type { PriceBar } from './quant-engine';
import type { LiveQuoteTick } from './market-service';

const VALID_TICKER_RE = /^[A-Z0-9^.=/-]{1,12}$/;

/**
 * Returns true if the active endDate represents the current/latest market horizon
 * (either empty, matching the SSR deterministic placeholder, or >= today's ISO date).
 */
export function isEndDateUpToPresent(
  endDate: string | undefined | null,
  todayIso: string,
  placeholderDate: string = '2026-11-08'
): boolean {
  if (!endDate) return true;
  if (endDate === placeholderDate) return true;
  return endDate >= todayIso;
}

/**
 * Builds the deduplicated, sanitized ticker subscription list for the live SSE stream,
 * ensuring the user's active ticker is always prioritized alongside the default universe.
 */
export function buildStreamTickerList(
  defaultUniverse: readonly string[],
  activeTicker: string,
  extraTickers: readonly string[] = [],
  maxTickers: number = 12
): string[] {
  const seen = new Set<string>();
  const out: string[] = [];

  const push = (raw: string) => {
    if (out.length >= maxTickers) return;
    const sym = raw.trim().toUpperCase();
    if (!sym || !VALID_TICKER_RE.test(sym) || sym.includes('..') || sym.includes('//')) return;
    if (seen.has(sym)) return;
    seen.add(sym);
    out.push(sym);
  };

  push(activeTicker);
  for (const sym of defaultUniverse) push(sym);
  for (const sym of extraTickers) push(sym);

  return out;
}

/**
 * Reconciles an incoming live quote tick into a cached daily OHLCV bar array.
 *
 * Guarantees:
 * - Never mutates existing bars in place (immutable updates for React/memo propagation).
 * - Never patches or appends when `allowPatchOrAppend` is false (preserves historical endDate slices).
 * - Never injects synthetic fallback ticks over cached history.
 * - Returns `{ bars: existingBars, changed: false }` (exact reference equality) when the latest bar
 *   is already identical, avoiding unnecessary downstream GARCH/TCN/chart re-computations.
 */
export function reconcileBarsWithLiveTick(
  existingBars: PriceBar[] | undefined,
  tick: LiveQuoteTick,
  options: { allowPatchOrAppend: boolean; existingSource?: string | null }
): { bars: PriceBar[] | undefined; changed: boolean } {
  if (!options.allowPatchOrAppend) {
    return { bars: existingBars, changed: false };
  }
  if (tick.source !== 'live_market') {
    return { bars: existingBars, changed: false };
  }
  // Only patch/append into an already-live historical series so we never splice a single
  // live bar onto the tail of a synthetic GBM path (which would create an artificial price jump).
  if (options.existingSource !== 'live_market') {
    return { bars: existingBars, changed: false };
  }
  if (!existingBars || existingBars.length === 0) {
    return { bars: existingBars, changed: false };
  }

  const lastIdx = existingBars.length - 1;
  const last = existingBars[lastIdx];
  const incoming = tick.bar;

  if (incoming.date < last.date) {
    return { bars: existingBars, changed: false };
  }

  if (incoming.date === last.date) {
    const mergedHigh = Math.max(last.high, incoming.high, incoming.close);
    const mergedLow = Math.min(last.low, incoming.low, incoming.close);
    const mergedVol = Math.max(last.volume, incoming.volume);
    const mergedOpen = incoming.open > 0 ? incoming.open : last.open;
    const mergedClose = incoming.close;

    if (
      last.open === mergedOpen &&
      last.high === mergedHigh &&
      last.low === mergedLow &&
      last.close === mergedClose &&
      last.volume === mergedVol
    ) {
      return { bars: existingBars, changed: false };
    }

    const next = existingBars.slice();
    next[lastIdx] = {
      ...last,
      open: mergedOpen,
      high: mergedHigh,
      low: mergedLow,
      close: mergedClose,
      volume: mergedVol,
    };
    return { bars: next, changed: true };
  }

  // Strictly newer trading session date -> append cleanly
  return {
    bars: [...existingBars, { ...incoming }],
    changed: true,
  };
}
