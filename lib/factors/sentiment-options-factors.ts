// Analyst-sentiment and options-implied factors. Pure TS.
// Analyst and options surfaces are current snapshots only (no point-in-time
// history), so callers must label any backtest using them as non-historical.

import type { FactorDefinition } from './types';

export interface EarningsSurprise {
  periodEnd: string;
  surprisePercent: number | null;
  epsDifference: number | null;
}

export interface AnalystSnapshot {
  symbol: string;
  observedAt: string;
  earningsHistory: EarningsSurprise[];
  revisions: {
    upLast30days: number | null;
    downLast30days: number | null;
    currentMean: number | null;
    mean90daysAgo: number | null;
  };
  estimates: { high: number | null; low: number | null; average: number | null }[];
  recommendations: {
    date: string;
    strongBuy: number | null;
    buy: number | null;
    hold: number | null;
    sell: number | null;
    strongSell: number | null;
  }[];
}

export interface OptionsFactorSnapshot {
  symbol: string;
  date: string;
  /** Annualized decimals (e.g. 0.24 = 24%). */
  iv30: number | null;
  rv20: number | null;
  atmIv: number | null;
  put25Iv: number | null;
  call25Iv: number | null;
  putCallOiRatio: number | null;
}

function finite(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export const ANALYST_FACTORS: readonly FactorDefinition[] = [
  { id: 'sent:PEAD_SUE', family: 'analyst', label: 'Post-Earnings Drift / SUE', formula: 'mean(surprisePercent) over recent quarters', rationale: 'Scores recent earnings beats; standardized unexpected earnings.', direction: 1, requires: 'analyst', minObservations: 1 },
  { id: 'sent:REVISION_BREADTH', family: 'analyst', label: 'Earnings Revision Breadth', formula: '(Up − Down) / (Up + Down), last 30 days', rationale: 'Net breadth of analyst estimate revisions.', direction: 1, requires: 'analyst', minObservations: 1 },
  { id: 'sent:ESTIMATE_DRIFT', family: 'analyst', label: 'Consensus Estimate Drift', formula: '(Current − 90d ago) / |90d ago|', rationale: 'Direction and pace of consensus revisions.', direction: 1, requires: 'analyst', minObservations: 1 },
  { id: 'sent:DISPERSION', family: 'analyst', label: 'Forecast Dispersion', formula: '(High − Low) / |Average|', rationale: 'Lower analyst disagreement historically accompanies steadier returns.', direction: -1, requires: 'analyst', minObservations: 1 },
  { id: 'sent:RATING_CHANGE', family: 'analyst', label: 'Consensus Rating Change', formula: 'Weighted rating score now − 30d ago', rationale: 'Captures shifts across strongBuy…strongSell counts.', direction: 1, requires: 'analyst', minObservations: 2 },
] as const;

export const OPTIONS_FACTORS: readonly FactorDefinition[] = [
  { id: 'opt:VRP', family: 'options', label: 'Volatility Risk Premium', formula: 'IV(30d ATM) − RV(30d)', rationale: 'Market-implied volatility above realized volatility.', direction: 1, requires: 'options', minObservations: 1 },
  { id: 'opt:SKEW', family: 'options', label: 'Volatility Skew', formula: 'IV(25d put) − IV(ATM)', rationale: 'Market-priced crash risk.', direction: -1, requires: 'options', minObservations: 1 },
  { id: 'opt:PCR', family: 'options', label: 'Put-Call Open Interest Ratio', formula: 'Put OI / Call OI, near-term expiries', rationale: 'Hedging-pressure and sentiment proxy.', direction: -1, requires: 'options', minObservations: 1 },
] as const;

export function computeAnalystFactors(
  snapshot: AnalystSnapshot | null | undefined
): Record<string, number | null> {
  const ids = ANALYST_FACTORS.map((f) => f.id);
  const out: Record<string, number | null> = {};
  for (const id of ids) out[id] = null;
  if (!snapshot) return out;

  const surprises = (snapshot.earningsHistory ?? [])
    .slice(0, 4)
    .map((e) => e.surprisePercent)
    .filter(finite);
  if (surprises.length > 0) out['sent:PEAD_SUE'] = mean(surprises);

  const { upLast30days: up, downLast30days: down } = snapshot.revisions;
  if (finite(up) && finite(down) && up + down > 0) {
    out['sent:REVISION_BREADTH'] = (up - down) / (up + down);
  }
  const current = snapshot.revisions.currentMean;
  const past = snapshot.revisions.mean90daysAgo;
  if (finite(current) && finite(past) && Math.abs(past) > 1e-9) {
    out['sent:ESTIMATE_DRIFT'] = (current - past) / Math.abs(past);
  }

  const estimates = (snapshot.estimates ?? []).filter(
    (e) => finite(e.high) && finite(e.low) && finite(e.average) && Math.abs(e.average as number) > 1e-9
  );
  if (estimates.length > 0) {
    out['sent:DISPERSION'] = mean(
      estimates.map((e) => ((e.high as number) - (e.low as number)) / Math.abs(e.average as number))
    );
  }

  const rated = [...(snapshot.recommendations ?? [])].sort((a, b) => b.date.localeCompare(a.date));
  if (rated.length >= 2) {
    const ratingScore = (r: (typeof rated)[number]): number | null => {
      const weights: [keyof typeof r, number][] = [
        ['strongBuy', 1],
        ['buy', 0.5],
        ['hold', 0],
        ['sell', -0.5],
        ['strongSell', -1],
      ];
      let total = 0;
      let count = 0;
      for (const [key, weight] of weights) {
        const v = r[key];
        if (finite(v)) {
          total += (v as number) * weight;
          count += v as number;
        }
      }
      return count > 0 ? total / count : null;
    };
    const now = ratingScore(rated[0]);
    const pastRating = ratingScore(rated[rated.length - 1]);
    if (now !== null && pastRating !== null) out['sent:RATING_CHANGE'] = now - pastRating;
  }

  for (const id of ids) {
    const v = out[id];
    if (v !== null && !finite(v)) out[id] = null;
  }
  return out;
}

export function computeOptionsFactors(
  snapshot: OptionsFactorSnapshot | null | undefined
): Record<string, number | null> {
  const ids = OPTIONS_FACTORS.map((f) => f.id);
  const out: Record<string, number | null> = {};
  for (const id of ids) out[id] = null;
  if (!snapshot) return out;

  if (finite(snapshot.iv30) && finite(snapshot.rv20)) {
    out['opt:VRP'] = (snapshot.iv30 as number) - (snapshot.rv20 as number);
  }
  if (finite(snapshot.put25Iv) && finite(snapshot.atmIv)) {
    out['opt:SKEW'] = (snapshot.put25Iv as number) - (snapshot.atmIv as number);
  }
  if (finite(snapshot.putCallOiRatio)) out['opt:PCR'] = snapshot.putCallOiRatio;

  for (const id of ids) {
    const v = out[id];
    if (v !== null && !finite(v)) out[id] = null;
  }
  return out;
}
