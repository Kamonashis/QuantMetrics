// Trap validation: event study over the accrued panel history.
// Every number here exists to answer exactly one question: "when the trap was
// triggered, did the trapped direction actually follow through?"
// No fitting, no extrapolation — descriptive stats, a base rate, and an
// honest verdict string. The arrow stays off unless the numbers say yes.
import type { PanelRow } from './iv-store';
import { neweyWestLrv } from './vol-analytics';

export type VerdictKind = 'above_base_rate' | 'no_reliable_follow_through' | 'insufficient_history';

export interface TrapEvent {
  date: string;
  direction: 'up' | 'down';
  spot: number;
  score: number;
  fwd1d: number | null;
  fwd3d: number | null;
  fwd5d: number | null;
  fwd10d: number | null;
  mfe10d: number | null;
  mae10d: number | null;
}

export interface TrapBacktest {
  verdict: VerdictKind;
  verdictLine: string;
  nDedupedEvents: number;
  nRawTriggered: number;
  hitRate5d: number | null;
  baseRate5d: number | null;
  meanReturn5d: number | null;
  tStat5d: number | null;
  wilsonLo: number | null;
  wilsonHi: number | null;
  medianMfeMae: number | null;
  fwdMean: { h1: number | null; h3: number | null; h5: number | null; h10: number | null };
  decileTable: { decile: number; meanScore: number; mean5d: number | null; n: number }[];
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0);
const stdev = (xs: number[]) => {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
};
const median = (xs: number[]) => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};

function wilson(k: number, n: number): { lo: number; hi: number } {
  if (n === 0) return { lo: 0, hi: 0 };
  const z = 1.959964;
  const p = k / n;
  const denom = 1 + (z * z) / n;
  const centre = p + (z * z) / (2 * n);
  const margin = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return { lo: (centre - margin) / denom, hi: (centre + margin) / denom };
}

/** Forward log return over h sessions starting after dateIndex i. */
function fwdReturn(closes: number[], i: number, h: number): number | null {
  if (i + h >= closes.length) return null;
  const a = closes[i];
  const b = closes[i + h];
  if (!(a > 0 && b > 0)) return null;
  return Math.log(b / a);
}

/** Max favourable / adverse excursion over the following h sessions (log-return units). */
function excursions(closes: number[], i: number, h: number): { mfe: number | null; mae: number | null } {
  if (i + h >= closes.length) return { mfe: null, mae: null };
  const entry = closes[i];
  const path = closes.slice(i + 1, i + h + 1);
  let mfe = -Infinity;
  let mae = Infinity;
  for (const p of path) {
    const r = Math.log(p / entry);
    if (r > mfe) mfe = r;
    if (r < mae) mae = r;
  }
  return { mfe: mfe === -Infinity ? null : mfe, mae: mae === Infinity ? null : mae };
}

export function runTrapBacktest(panel: PanelRow[], bars: { date: string; close: number }[], minEvents = 20): TrapBacktest {
  const empty: TrapBacktest = {
    verdict: 'insufficient_history',
    verdictLine: `Insufficient history (n=0) — the trap indicator accrues evidence daily; the arrow stays disabled.`,
    nDedupedEvents: 0,
    nRawTriggered: 0,
    hitRate5d: null,
    baseRate5d: null,
    meanReturn5d: null,
    tStat5d: null,
    wilsonLo: null,
    wilsonHi: null,
    medianMfeMae: null,
    fwdMean: { h1: null, h3: null, h5: null, h10: null },
    decileTable: [],
  };
  if (panel.length === 0 || bars.length < 30) return empty;

  const closes = bars.map((b) => b.close);
  const dateIndex = new Map<string, number>();
  bars.forEach((b, i) => dateIndex.set(b.date, i));

  // Collect raw triggered events
  const raw: TrapEvent[] = [];
  for (const row of panel) {
    if (row.trapState !== 'triggered' || !row.trapDirection || row.trapDirection === 'none') continue;
    const i = dateIndex.get(row.date);
    if (i === undefined || i + 1 >= closes.length) continue;
    const dir = row.trapDirection === 'up' ? 1 : -1;
    const fwd1d = fwdReturn(closes, i, 1);
    const fwd3d = fwdReturn(closes, i, 3);
    const fwd5d = fwdReturn(closes, i, 5);
    const fwd10d = fwdReturn(closes, i, 10);
    const { mfe, mae } = excursions(closes, i, 10);
    raw.push({
      date: row.date,
      direction: row.trapDirection as 'up' | 'down',
      spot: row.spot,
      score: row.trapScore ?? 0,
      fwd1d: fwd1d === null ? null : dir * fwd1d,
      fwd3d: fwd3d === null ? null : dir * fwd3d,
      fwd5d: fwd5d === null ? null : dir * fwd5d,
      fwd10d: fwd10d === null ? null : dir * fwd10d,
      mfe10d: mfe === null ? null : Math.abs(mfe),
      mae10d: mae === null ? null : Math.abs(mae),
    });
  }

  // Dedupe: consecutive triggered rows within 10 sessions are one event
  const deduped: TrapEvent[] = [];
  for (const e of raw) {
    const last = deduped[deduped.length - 1];
    if (!last) { deduped.push(e); continue; }
    const gap = (dateIndex.get(e.date) ?? 0) - (dateIndex.get(last.date) ?? 0);
    if (gap >= 10) deduped.push(e);
  }

  const n = deduped.length;
  if (n < minEvents) {
    if (n === 0) return empty;
    return {
      ...empty,
      verdict: 'insufficient_history',
      verdictLine: `Insufficient history (n=${n}) — need ≥ ${minEvents} deduplicated triggered events before any directional claim is made.`,
      nDedupedEvents: n,
      nRawTriggered: raw.length,
    };
  }

  const fwd5 = deduped.map((e) => e.fwd5d).filter((v): v is number => v !== null);
  const hits = fwd5.filter((v) => v > 0).length;
  const hitRate = fwd5.length > 0 ? (hits / fwd5.length) * 100 : null;
  const mean5 = fwd5.length > 0 ? mean(fwd5) : null;
  // Newey-West t-stat for the mean of the differentiated series (2 lags for overlap)
  const lrv = neweyWestLrv(fwd5, 2);
  const tStat = lrv > 0 && fwd5.length > 0 ? (mean5 ?? 0) / Math.sqrt(lrv / fwd5.length) : null;

  // Unconditional base rate: fraction of all 5d windows that move in the same direction as the event
  const baseRates: number[] = [];
  for (const e of deduped) {
    const sign = e.direction === 'up' ? 1 : -1;
    let wins = 0;
    let total = 0;
    for (let i = 0; i < closes.length - 5; i++) {
      const r = Math.log(closes[i + 5] / closes[i]);
      if (sign * r > 0) wins++;
      total++;
    }
    baseRates.push(total > 0 ? wins / total : 0.5);
  }
  const baseRate = mean(baseRates) * 100;

  const wl = fwd5.length > 0 ? wilson(hits, fwd5.length) : null;

  // Median MFE/MAE ratio
  const ratios = deduped
    .filter((e) => e.mfe10d !== null && e.mae10d !== null && (e.mae10d as number) > 0)
    .map((e) => (e.mfe10d as number) / (e.mae10d as number));
  const medianMfeMae = ratios.length > 0 ? median(ratios) : null;

  const fwdMean = {
    h1: meanOrNull(deduped.map((e) => e.fwd1d)),
    h3: meanOrNull(deduped.map((e) => e.fwd3d)),
    h5: meanOrNull(deduped.map((e) => e.fwd5d)),
    h10: meanOrNull(deduped.map((e) => e.fwd10d)),
  };

  // Decile monotonicity by trap score
  const scored = deduped.filter((e) => e.fwd5d !== null).sort((a, b) => a.score - b.score);
  const decileTable: TrapBacktest['decileTable'] = [];
  if (scored.length >= 10) {
    const per = Math.floor(scored.length / 10);
    for (let d = 0; d < 10; d++) {
      const slice = scored.slice(d * per, d === 9 ? scored.length : (d + 1) * per);
      decileTable.push({
        decile: d + 1,
        meanScore: Math.round(mean(slice.map((s) => s.score))),
        mean5d: mean(slice.map((s) => s.fwd5d as number)) * 100,
        n: slice.length,
      });
    }
  }

  const significant = tStat !== null && Math.abs(tStat) > 1.96;
  const aboveBase = hitRate !== null && hitRate > baseRate && significant;

  let verdict: VerdictKind;
  if (aboveBase) {
    verdict = 'above_base_rate';
  } else {
    verdict = 'no_reliable_follow_through';
  }

  const sign = (v: number | null) => (v === null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%`);
  const verdictLine =
    verdict === 'above_base_rate'
      ? `Follow-through above base rate (n=${n}, ${sign(fwdMean.h5)}/5d, t=${tStat?.toFixed(1)}) vs base rate ${baseRate.toFixed(0)}%`
      : `No reliable follow-through (n=${n}, ${sign(fwdMean.h5)}/5d, t=${tStat?.toFixed(1)})`;

  return {
    verdict,
    verdictLine,
    nDedupedEvents: n,
    nRawTriggered: raw.length,
    hitRate5d: hitRate,
    baseRate5d: baseRate,
    meanReturn5d: mean5,
    tStat5d: tStat,
    wilsonLo: wl ? wl.lo * 100 : null,
    wilsonHi: wl ? wl.hi * 100 : null,
    medianMfeMae,
    fwdMean,
    decileTable,
  };
}

function meanOrNull(xs: (number | null)[]): number | null {
  const v = xs.filter((x): x is number => x !== null && isFinite(x));
  return v.length > 0 ? mean(v) : null;
}
