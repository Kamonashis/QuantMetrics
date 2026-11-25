// Series builders for the Analysis → Overview structural decomposition + TCN
// next-day forecast. Pure, no React. Each builder returns the modelled values,
// their dates, and the asset closes aligned 1:1 to those dates (the TCN's
// causal HMM regime channel always reads the asset's own price returns).

import type { PriceBar } from './quant-engine';
import type { TcnSeriesKind } from './tcn-attention';

export type DecompositionSeriesKind = TcnSeriesKind;

export interface DecompositionSeries {
  kind: DecompositionSeriesKind;
  dates: string[];
  values: number[];
  /** Asset closes aligned 1:1 to `dates` (regime input for the TCN). */
  closes: number[];
  /** 'returns' only: log (true) or simple (false) returns. */
  returnsLog?: boolean;
}

export interface DecompositionSeriesMeta {
  label: string;
  shortLabel: string;
  /** Multiplicative needs a strictly positive series. */
  allowsMultiplicative: boolean;
  /** What the naive baseline is for this series. */
  baselineLabel: string;
  /** Whether a trading backtest is meaningful. */
  hasTradingBacktest: boolean;
  description: string;
}

export const DECOMPOSITION_SERIES_META: Record<DecompositionSeriesKind, DecompositionSeriesMeta> = {
  price: {
    label: 'Price',
    shortLabel: 'Price',
    allowsMultiplicative: true,
    baselineLabel: 'random walk (Ŷ = Yₜ₋₁)',
    hasTradingBacktest: true,
    description: 'Daily close.',
  },
  vol: {
    label: 'Realized Volatility',
    shortLabel: 'Realized Vol',
    allowsMultiplicative: true,
    baselineLabel: 'random walk (σ̂ = σₜ₋₁)',
    hasTradingBacktest: false,
    description:
      'Daily Garman–Klass realized volatility from OHLC, annualized √252 (vol points). Non-overlapping by construction — a rolling-window RV would overlap the next-day target and flatter the forecast.',
  },
  returns: {
    label: 'Returns',
    shortLabel: 'Returns',
    allowsMultiplicative: false,
    baselineLabel: 'zero-return forecast (r̂ = 0)',
    hasTradingBacktest: true,
    description:
      'Daily returns. Returns can be negative, so the decomposition is additive only. Skill is measured against a zero-return forecast, not a random walk.',
  },
};

/** Close series (the original decomposition input). */
export function buildPriceSeries(bars: PriceBar[]): DecompositionSeries {
  const dates: string[] = [];
  const values: number[] = [];
  for (const b of bars) {
    if (!Number.isFinite(b.close) || b.close <= 0) continue;
    dates.push(b.date);
    values.push(b.close);
  }
  return { kind: 'price', dates, values, closes: values.slice() };
}

/**
 * Daily Garman–Klass variance:
 *   σ²_GK = ½·ln(H/L)² − (2·ln2 − 1)·ln(C/O)²
 * annualized as σ = √(252·σ²_GK), reported in vol points (×100). Bars with
 * missing/invalid OHLC or a non-positive estimate are skipped (no fill, no
 * fabrication); the dates of the remaining bars carry the series.
 */
export function garmanKlassAnnualizedVol(bar: PriceBar): number | null {
  const { open, high, low, close } = bar;
  if (!(open > 0 && high > 0 && low > 0 && close > 0) || high < low) return null;
  const hl = Math.log(high / low);
  const co = Math.log(close / open);
  const v = 0.5 * hl * hl - (2 * Math.LN2 - 1) * co * co;
  if (!Number.isFinite(v) || v <= 0) return null;
  return Math.sqrt(252 * v) * 100;
}

export function buildRealizedVolSeries(bars: PriceBar[]): DecompositionSeries {
  const dates: string[] = [];
  const values: number[] = [];
  const closes: number[] = [];
  for (const b of bars) {
    const vol = garmanKlassAnnualizedVol(b);
    if (vol === null) continue;
    dates.push(b.date);
    values.push(vol);
    closes.push(b.close);
  }
  return { kind: 'vol', dates, values, closes };
}

/**
 * Daily returns dated at the later bar (r_t = f(C_t, C_{t-1})); the aligned
 * close is C_t. Follows the Analysis view's log/simple choice.
 */
export function buildReturnsSeries(bars: PriceBar[], returnType: 'simple' | 'log'): DecompositionSeries {
  const dates: string[] = [];
  const values: number[] = [];
  const closes: number[] = [];
  for (let i = 1; i < bars.length; i++) {
    const prev = bars[i - 1].close;
    const curr = bars[i].close;
    if (!(prev > 0 && curr > 0)) continue;
    dates.push(bars[i].date);
    values.push(returnType === 'log' ? Math.log(curr / prev) : curr / prev - 1);
    closes.push(curr);
  }
  return { kind: 'returns', dates, values, closes, returnsLog: returnType === 'log' };
}

export function buildDecompositionSeries(
  kind: DecompositionSeriesKind,
  bars: PriceBar[],
  returnType: 'simple' | 'log'
): DecompositionSeries {
  if (kind === 'vol') return buildRealizedVolSeries(bars);
  if (kind === 'returns') return buildReturnsSeries(bars, returnType);
  return buildPriceSeries(bars);
}

/** Model actually used for a series (returns are forced additive). */
export function effectiveDecompositionModel(
  kind: DecompositionSeriesKind,
  requested: 'additive' | 'multiplicative'
): 'additive' | 'multiplicative' {
  return DECOMPOSITION_SERIES_META[kind].allowsMultiplicative ? requested : 'additive';
}

/** Display formatting per series (native units in, label out). */
export function formatSeriesValue(kind: DecompositionSeriesKind, v: number, returnsLog?: boolean): string {
  if (!Number.isFinite(v)) return '—';
  if (kind === 'vol') return `${v.toFixed(2)} vol pts`;
  if (kind === 'returns') return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(3)}%${returnsLog ? ' (log)' : ''}`;
  return `$${v.toFixed(2)}`;
}

/** Signed change in native units (Ŷ − Y). */
export function formatSeriesDelta(kind: DecompositionSeriesKind, v: number): string {
  if (!Number.isFinite(v)) return '—';
  const sign = v >= 0 ? '+' : '−';
  if (kind === 'vol') return `${sign}${Math.abs(v).toFixed(2)} vol pts`;
  if (kind === 'returns') return `${sign}${Math.abs(v * 100).toFixed(3)} pp`;
  return `${sign}$${Math.abs(v).toFixed(2)}`;
}

/** Compact numeric (no unit word) for dense tables/axes. */
export function formatSeriesCompact(kind: DecompositionSeriesKind, v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (kind === 'vol') return v.toFixed(2);
  if (kind === 'returns') return `${(v * 100).toFixed(3)}%`;
  return `$${v.toFixed(2)}`;
}
