// Pre-registered outcome resolution for the signal journal. Pure: records in,
// bars in, hit rates out. No refitting, no lookahead beyond the recorded date,
// and no fabricated outcome when the forward window is not yet available.

import type { PriceBar } from './quant-engine';

export interface SignalJournalRecord {
  date: string;
  symbol: string;
  setup: string;
  state: string;
  iv30: number | null;
  breakeven: number | null;
  spot: number | null;
  direction: 'up' | 'down' | 'none';
  wall: number | null;
  flip: number | null;
  impliedPct: number | null;
  horizon: number;
  edgeEstimate: string | null;
  pairLong: string | null;
  pairShort: string | null;
  recordedAt: string;
}

export type SignalOutcome = 'WIN' | 'LOSS' | 'UNRESOLVED';

export interface ResolvedSignal extends SignalJournalRecord {
  outcome: SignalOutcome;
  detail: string;
  forwardValue: number | null;
}

export interface HitRate {
  setup: string;
  wins: number;
  losses: number;
  unresolved: number;
  /** Sample behind the rate — null below MIN_SAMPLES, never a guessed rate. */
  n: number;
  hitRate: number | null;
}

export const MIN_SAMPLES = 10;

const logReturns = (closes: number[]): number[] => {
  const out: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    if (closes[i - 1] > 0 && closes[i] > 0) out.push(Math.log(closes[i] / closes[i - 1]));
  }
  return out;
};

/** Annualized realized vol of the `window` sessions starting at `start` (return index). */
function forwardRv(bars: PriceBar[], start: number, window: number): number | null {
  const rets = logReturns(bars.map((b) => b.close));
  if (start < 0 || start + window > rets.length) return null;
  let sum = 0;
  for (let i = start; i < start + window; i++) sum += rets[i] * rets[i];
  return Math.sqrt((sum / window) * 252);
}

function entryIndex(bars: PriceBar[], date: string): number {
  return bars.findIndex((b) => b.date >= date);
}

function unresolved(record: SignalJournalRecord, detail: string): ResolvedSignal {
  return { ...record, outcome: 'UNRESOLVED', detail, forwardValue: null };
}

/** Resolve one journal record against subsequent bars. No output for missing forward data. */
export function resolveSignal(
  record: SignalJournalRecord,
  barsBySymbol: Record<string, PriceBar[]>
): ResolvedSignal {
  const bars = barsBySymbol[record.symbol] ?? [];
  const i = entryIndex(bars, record.date);

  if (record.setup === 'cross-sectional') {
    const longBars = barsBySymbol[record.pairLong ?? ''] ?? [];
    const shortBars = barsBySymbol[record.pairShort ?? ''] ?? [];
    const li = entryIndex(longBars, record.date);
    const si = entryIndex(shortBars, record.date);
    const lv = forwardRv(longBars, li, 20);
    const sv = forwardRv(shortBars, si, 20);
    if (lv === null || sv === null) return unresolved(record, 'Pairwise forward RV not yet available');
    const win = lv > sv;
    return { ...record, outcome: win ? 'WIN' : 'LOSS', detail: `long RV20 ${(lv * 100).toFixed(1)}% vs short ${(sv * 100).toFixed(1)}%`, forwardValue: lv - sv };
  }

  if (i < 0) return unresolved(record, 'Entry date not in bar history');

  if (record.setup === 'premium-harvest') {
    const rv = forwardRv(bars, i, 20);
    if (rv === null || record.iv30 === null) return unresolved(record, 'Forward RV20 not yet available');
    const win = rv < record.iv30;
    return { ...record, outcome: win ? 'WIN' : 'LOSS', detail: `forward RV20 ${(rv * 100).toFixed(1)}% vs IV30 ${(record.iv30 * 100).toFixed(1)}% at entry`, forwardValue: rv - record.iv30 };
  }

  if (record.setup === 'long-gamma') {
    const rv = forwardRv(bars, i, 5);
    if (rv === null || record.breakeven === null) return unresolved(record, 'Forward RV5 not yet available');
    const win = rv > record.breakeven;
    return { ...record, outcome: win ? 'WIN' : 'LOSS', detail: `forward RV5 ${(rv * 100).toFixed(1)}% vs breakeven ${(record.breakeven * 100).toFixed(1)}%`, forwardValue: rv - record.breakeven };
  }

  if (record.setup === 'directional-trap') {
    const horizon = Math.min(10, record.horizon || 10);
    if (i + horizon >= bars.length || record.wall === null || record.direction === 'none') {
      return unresolved(record, 'Wall/flip window not yet complete');
    }
    for (let k = 1; k <= horizon; k++) {
      const bar = bars[i + k];
      if (record.direction === 'up') {
        if (bar.high >= record.wall) return { ...record, outcome: 'WIN', detail: `call wall ${record.wall.toFixed(2)} hit on session +${k}`, forwardValue: bar.high - record.wall };
        if (record.flip !== null && bar.low <= record.flip) return { ...record, outcome: 'LOSS', detail: `gamma flip ${record.flip.toFixed(2)} hit first on session +${k}`, forwardValue: bar.low - record.flip };
      } else {
        if (bar.low <= record.wall) return { ...record, outcome: 'WIN', detail: `put wall ${record.wall.toFixed(2)} hit on session +${k}`, forwardValue: record.wall - bar.low };
        if (record.flip !== null && bar.high >= record.flip) return { ...record, outcome: 'LOSS', detail: `gamma flip ${record.flip.toFixed(2)} hit first on session +${k}`, forwardValue: record.flip - bar.high };
      }
    }
    return unresolved(record, 'Neither wall nor flip reached in 10 sessions');
  }

  if (record.setup === 'event-straddle') {
    const h = Math.max(1, record.horizon || 21);
    if (i + h >= bars.length || record.impliedPct === null) return unresolved(record, 'Event horizon not yet complete');
    const move = Math.abs(Math.log(bars[i + h].close / bars[i].close)) * 100;
    const win = move >= record.impliedPct;
    return { ...record, outcome: win ? 'WIN' : 'LOSS', detail: `realized |move| ${move.toFixed(2)}% vs implied ${record.impliedPct.toFixed(2)}%`, forwardValue: move - record.impliedPct };
  }

  if (record.setup === 'skew-tail') {
    const rvOrMove = forwardRv(bars, i, 10);
    if (rvOrMove === null || record.iv30 === null) return unresolved(record, 'Forward 10d move not yet available');
    const threshold = record.iv30 * Math.sqrt(10 / 252);
    const win = rvOrMove > threshold;
    return { ...record, outcome: win ? 'WIN' : 'LOSS', detail: `forward 10d RV ${(rvOrMove * 100).toFixed(1)}% vs 1σ ${(threshold * 100).toFixed(1)}%`, forwardValue: rvOrMove - threshold };
  }

  return unresolved(record, `No pre-registered resolver for ${record.setup}`);
}

/** Aggregate hit rates by setup. Rates below MIN_SAMPLES are withheld. */
export function summarizeHitRates(records: SignalJournalRecord[], barsBySymbol: Record<string, PriceBar[]>): HitRate[] {
  const bySetup = new Map<string, ResolvedSignal[]>();
  for (const r of records) {
    const resolved = resolveSignal(r, barsBySymbol);
    const list = bySetup.get(r.setup) ?? [];
    list.push(resolved);
    bySetup.set(r.setup, list);
  }
  return [...bySetup.entries()].map(([setup, list]) => {
    const wins = list.filter((r) => r.outcome === 'WIN').length;
    const losses = list.filter((r) => r.outcome === 'LOSS').length;
    const unresolved = list.length - wins - losses;
    const n = wins + losses;
    return { setup, wins, losses, unresolved, n, hitRate: n >= MIN_SAMPLES ? wins / n : null };
  }).sort((a, b) => a.setup.localeCompare(b.setup));
}
