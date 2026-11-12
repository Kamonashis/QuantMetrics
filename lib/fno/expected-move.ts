// Expected-move analytics: the option chain's straddle-implied and sigma-implied
// move versus the distribution of realized moves on the same horizon.
// Historical IV is not freely available — this is today's pricing vs realized history,
// not a straddle backtest. Rendered as such everywhere it's shown.
import type { RawChainRow } from './types';
import { calculateReturns } from '../quant-engine';
import type { PriceBar } from '../quant-engine';
import type { ExpectedMovePoint } from './types';

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Implied moves per expiry group. `steps` are trading-day equivalents (DTE). */
export function expectedMoves(
  ladder: { expiry: string; dte: number; t: number; atmIv: number | null; rows: RawChainRow[] }[],
  spot: number
): ExpectedMovePoint[] {
  return ladder.map((l) => {
    const atmRows = l.rows
      .filter((r) => r.bid > 0 && r.ask >= r.bid)
      .map((r) => {
        const mid = (r.bid + r.ask) / 2;
        return { r, mid, absK: Math.abs(Math.log(r.strike / spot)) };
      })
      .sort((a, b) => a.absK - b.absK);
    const callRow = atmRows.find((x) => x.r.type === 'call');
    const putRow = atmRows.find((x) => x.r.type === 'put');
    const atmCall = callRow?.mid ?? null;
    const atmPut = putRow?.mid ?? null;
    const straddleMove =
      atmCall !== null && atmPut !== null && spot > 0
        ? ((atmCall + atmPut) * 0.8 / spot) * 100
        : null;
    const sigmaMove = l.atmIv !== null && isFinite(l.atmIv) ? l.atmIv * Math.sqrt(l.t) * 100 : null;
    return {
      dte: l.dte,
      expiry: l.expiry,
      atmCall,
      atmPut,
      straddleMove,
      sigmaMove,
      atmIv: l.atmIv,
    };
  });
}

/** Rolling absolute log-return over `horizon` sessions — the realized-move distribution sample. */
export function historicalAbsMoves(bars: PriceBar[], horizon: number): number[] {
  if (!bars || bars.length <= horizon) return [];
  const closes = bars.map((b) => b.close);
  const rets = calculateReturns(closes, 'log');
  const out: number[] = [];
  const h = Math.max(1, Math.floor(horizon));
  for (let i = 0; i + h <= rets.length; i++) {
    let sum = 0;
    for (let j = i; j < i + h; j++) sum += rets[j];
    out.push(Math.abs(sum));
  }
  return out;
}

/** Percentile of `value` within `samples` (0–100). Returns null when samples are empty. */
export function percentileRankLocal(samples: number[], value: number): number | null {
  if (samples.length === 0 || !isFinite(value)) return null;
  let below = 0, equal = 0;
  for (const s of samples) {
    if (s < value) below++;
    else if (s === value) equal++;
  }
  return ((below + 0.5 * equal) / samples.length) * 100;
}

export type MoveVerdict = 'CHEAP' | 'FAIR' | 'RICH';

export function moveVerdict(impliedMovePct: number | null, samples: number[]): MoveVerdict | null {
  if (impliedMovePct === null || samples.length === 0) return null;
  const pctile = percentileRankLocal(samples, impliedMovePct / 100);
  if (pctile === null) return null;
  if (pctile > 75) return 'RICH';
  if (pctile < 25) return 'CHEAP';
  return 'FAIR';
}

/** Distribution summary for rendering — median, P75, P90, count. */
export function moveDistributionSummary(samples: number[]) {
  if (samples.length === 0) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  const pick = (q: number) => sorted[Math.floor((sorted.length - 1) * q)];
  return {
    n: sorted.length,
    median: pick(0.5) * 100,
    p75: pick(0.75) * 100,
    p90: pick(0.9) * 100,
  };
}
