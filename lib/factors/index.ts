// Factor registry and panel assembly. Pure TS.

import type { PriceBar } from '../quant-engine';
import type { FactorDefinition, FactorPanelRow } from './types';
import { computePriceFactorSeries, PRICE_FACTORS } from './price-factors';
import {
  computeFundamentalFactors,
  FUNDAMENTAL_FACTORS,
  type FundamentalSnapshot,
} from './fundamental-factors';
import {
  ANALYST_FACTORS,
  OPTIONS_FACTORS,
  computeAnalystFactors,
  computeOptionsFactors,
  type AnalystSnapshot,
  type OptionsFactorSnapshot,
} from './sentiment-options-factors';
import { scoreCrossSection, type FactorRawRow } from './cross-section';
import type { CrossSectionOptions } from './types';

export * from './types';
export * from './price-factors';
export * from './fundamental-factors';
export * from './sentiment-options-factors';
export * from './cross-section';

export const ALL_FACTORS: readonly FactorDefinition[] = [
  ...PRICE_FACTORS,
  ...FUNDAMENTAL_FACTORS,
  ...ANALYST_FACTORS,
  ...OPTIONS_FACTORS,
];

export const FACTOR_BY_ID: Record<string, FactorDefinition> = Object.fromEntries(
  ALL_FACTORS.map((f) => [f.id, f])
);

export const FACTOR_FAMILIES = [
  'momentum',
  'reversal',
  'low_volatility',
  'low_beta',
  'downside_risk',
  'liquidity',
  'volatility_structure',
  'value',
  'quality',
  'investment',
  'size',
  'analyst',
  'options',
] as const;

function adjClose(bar: PriceBar): number {
  const withAdj = bar as PriceBar & { adjClose?: number };
  return Number.isFinite(withAdj.adjClose) ? (withAdj.adjClose as number) : bar.close;
}

export interface BuildFactorPanelInput {
  symbols: readonly string[];
  barsBySymbol: Record<string, PriceBar[]>;
  benchmarkBars: PriceBar[];
  /** Daily fundamental snapshots per symbol, ascending by observedAt. */
  fundamentalsBySymbol?: Record<string, FundamentalSnapshot[]>;
  /** Analyst snapshot is a current-only surface. */
  analystBySymbol?: Record<string, AnalystSnapshot | null>;
  /** Accrued daily options snapshots per symbol, ascending by date. */
  optionsBySymbol?: Record<string, OptionsFactorSnapshot[]>;
  sectors?: Record<string, string>;
  /** Date grid (YYYY-MM-DD) on which to score the cross-section. */
  dates: readonly string[];
  /** Forward-return horizon in trading sessions. */
  horizon?: 1 | 5 | 22;
  /** Score only the last N grid dates. */
  maxPoints?: number;
}

function latestAt<T extends { observedAt?: string; date?: string }>(
  snapshots: readonly T[] | undefined,
  date: string,
  key: 'observedAt' | 'date'
): T | null {
  if (!Array.isArray(snapshots)) return null;
  let best: T | null = null;
  for (const snap of snapshots) {
    const d = snap[key];
    if (typeof d === 'string' && d <= date && (!best || d > (best[key] as string))) best = snap;
  }
  return best;
}

/** Assemble raw, date-aligned factor rows for the cross-sectional panel. */
export function buildFactorPanel(input: BuildFactorPanelInput): FactorRawRow[] {
  const horizon = input.horizon ?? 22;
  const dates = input.dates;
  const maxPoints = Number.isFinite(input.maxPoints) ? Math.max(0, Math.floor(input.maxPoints as number)) : dates.length;
  const from = Math.max(0, dates.length - maxPoints);
  const rows: FactorRawRow[] = [];

  for (const symbol of input.symbols) {
    const bars = input.barsBySymbol?.[symbol] ?? [];
    if (bars.length < 2) continue;
    const priceSeries = computePriceFactorSeries(bars, input.benchmarkBars, { maxPoints: dates.length });
    const byDate = new Map(bars.map((b, i) => [b.date, i]));
    const fundamentals = input.fundamentalsBySymbol?.[symbol] ?? [];
    const analyst = input.analystBySymbol?.[symbol] ?? null;
    const options = input.optionsBySymbol?.[symbol] ?? [];

    for (let g = from; g < dates.length; g++) {
      const date = dates[g];
      const barIdx = byDate.get(date);
      if (barIdx === undefined) continue;
      const raw: Record<string, number | null> = {};
      for (const def of PRICE_FACTORS) {
        raw[def.id] = priceSeries[def.id]?.[barIdx] ?? null;
      }

      const fundamental = latestAt(fundamentals, date, 'observedAt');
      Object.assign(raw, computeFundamentalFactors(fundamental, date));

      // Analyst data has no history: only expose it on/after first observation.
      if (analyst && (!analyst.observedAt || analyst.observedAt <= date)) {
        Object.assign(raw, computeAnalystFactors(analyst));
      } else {
        for (const def of ANALYST_FACTORS) raw[def.id] = null;
      }

      const optionsSnap = latestAt(options, date, 'date');
      Object.assign(raw, computeOptionsFactors(optionsSnap ?? null));

      const futureIdx = barIdx + horizon;
      const future = futureIdx < bars.length ? adjClose(bars[futureIdx]) : null;
      const current = adjClose(bars[barIdx]);
      const forwardReturn =
        future !== null && Number.isFinite(future) && Number.isFinite(current) && current > 0
          ? future / current - 1
          : null;

      rows.push({
        date,
        symbol,
        sector: input.sectors?.[symbol] ?? 'Unclassified',
        raw,
        forwardReturn,
      });
    }
  }
  return rows;
}

export function buildScoredFactorPanel(
  input: BuildFactorPanelInput,
  definitions: readonly FactorDefinition[],
  options?: CrossSectionOptions
): FactorPanelRow[] {
  return scoreCrossSection(buildFactorPanel(input), definitions, options);
}

