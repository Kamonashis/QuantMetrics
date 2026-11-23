// Pure price/volume factor construction from daily OHLCV bars.
// Raw values are stored unsigned: `direction` in the registry decides whether
// a high or low raw value is attractive. No React, no I/O.

import type { PriceBar } from '../quant-engine';
import type { FactorDefinition, FactorSeries } from './types';

const TRADING_DAYS = 252;
const EPS = 1e-12;

function adjClose(bar: PriceBar): number {
  const withAdj = bar as PriceBar & { adjClose?: number };
  return Number.isFinite(withAdj.adjClose) ? (withAdj.adjClose as number) : bar.close;
}

function finite(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function mean(values: number[]): number {
  if (values.length === 0) return NaN;
  let total = 0;
  for (const v of values) total += v;
  return total / values.length;
}

function stdev(values: number[]): number {
  if (values.length < 2) return NaN;
  const m = mean(values);
  let sum = 0;
  for (const v of values) sum += (v - m) ** 2;
  return Math.sqrt(sum / (values.length - 1));
}

function ewmaVariance(values: number[], lambda = 0.94): number {
  if (values.length === 0) return NaN;
  let v = values[0] ** 2;
  for (let i = 1; i < values.length; i++) {
    v = lambda * v + (1 - lambda) * values[i] ** 2;
  }
  return v;
}

interface OlsFit {
  alpha: number;
  beta: number;
  residuals: number[];
}

function ols(y: number[], x: number[]): OlsFit | null {
  const n = Math.min(y.length, x.length);
  if (n < 2) return null;
  const mx = mean(x.slice(0, n));
  const my = mean(y.slice(0, n));
  let cov = 0;
  let varX = 0;
  for (let i = 0; i < n; i++) {
    cov += (x[i] - mx) * (y[i] - my);
    varX += (x[i] - mx) ** 2;
  }
  if (varX <= EPS) return null;
  const beta = cov / varX;
  const alpha = my - beta * mx;
  const residuals: number[] = [];
  for (let i = 0; i < n; i++) residuals.push(y[i] - alpha - beta * x[i]);
  return { alpha, beta, residuals };
}

export const PRICE_FACTORS: readonly FactorDefinition[] = [
  {
    id: 'price:MOM_12_1',
    family: 'momentum',
    label: 'Classic 12–1 Momentum',
    formula: 'P(t−21) / P(t−252) − 1',
    rationale: 'Intermediate-term trend persistence while skipping the most recent month of microstructure reversal.',
    direction: 1,
    requires: 'price',
    minObservations: 252,
  },
  {
    id: 'price:MOM_6_1',
    family: 'momentum',
    label: '6–1 Momentum',
    formula: 'P(t−21) / P(t−126) − 1',
    rationale: 'Shorter-horizon confirmation of the intermediate trend.',
    direction: 1,
    requires: 'price',
    minObservations: 126,
  },
  {
    id: 'price:MOM_3_1',
    family: 'momentum',
    label: '3–1 Momentum',
    formula: 'P(t−21) / P(t−63) − 1',
    rationale: 'Quarterly trend with the final month skipped.',
    direction: 1,
    requires: 'price',
    minObservations: 63,
  },
  {
    id: 'price:HIGH_52W',
    family: 'momentum',
    label: '52-Week High Proximity',
    formula: 'P(t) / max(P(t−252…t))',
    rationale: 'Investor anchoring near prior price highs.',
    direction: 1,
    requires: 'price',
    minObservations: 252,
  },
  {
    id: 'price:RESID_MOM',
    family: 'momentum',
    label: 'Residual (Idiosyncratic) Momentum',
    formula: 'Σ ε(i,t) over t−252…t−21, ε from rolling market regression',
    rationale: 'Strips market and sector beta whipsaws out of momentum during regime shifts.',
    direction: 1,
    requires: 'price',
    minObservations: 252,
  },
  {
    id: 'price:REV_1M',
    family: 'reversal',
    label: '1-Month Reversal',
    formula: '−(P(t) / P(t−21) − 1)',
    rationale: 'Liquidity provision and temporary one-month overreaction.',
    direction: 1,
    requires: 'price',
    minObservations: 21,
  },
  {
    id: 'price:REV_5D',
    family: 'reversal',
    label: '5-Day Reversal',
    formula: '−(P(t) / P(t−5) − 1)',
    rationale: 'Short-horizon mean reversion in liquidity-driven dislocations.',
    direction: 1,
    requires: 'price',
    minObservations: 5,
  },
  {
    id: 'price:RV60',
    family: 'low_volatility',
    label: 'Realized Volatility (60d)',
    formula: 'σ̂₆₀(r) · √252',
    rationale: 'Low-volatility anomaly: lower-risk equities deliver superior risk-adjusted returns.',
    direction: -1,
    requires: 'price',
    minObservations: 60,
  },
  {
    id: 'price:RV252',
    family: 'low_volatility',
    label: 'Realized Volatility (252d)',
    formula: 'σ̂₂₅₂(r) · √252',
    rationale: 'Long-horizon low-volatility exposure.',
    direction: -1,
    requires: 'price',
    minObservations: 252,
  },
  {
    id: 'price:EWMA_VOL',
    family: 'low_volatility',
    label: 'EWMA Conditional Volatility',
    formula: '√(EWMA variance, λ = 0.94) · √252',
    rationale: 'Responsive low-vol signal that adapts to volatility regime shifts.',
    direction: -1,
    requires: 'price',
    minObservations: 60,
  },
  {
    id: 'price:BAB_BETA',
    family: 'low_beta',
    label: 'Betting Against Beta',
    formula: 'β(i) = Cov(r(i), r(m)) / Var(r(m)), 252d',
    rationale: 'Leverage-constrained investors bid up high-beta names, compressing their forward alpha.',
    direction: -1,
    requires: 'price',
    minObservations: 252,
  },
  {
    id: 'price:IVOL',
    family: 'downside_risk',
    label: 'Idiosyncratic Volatility',
    formula: 'std(ε(i)) from 252d CAPM residuals',
    rationale: 'Penalizes lottery-like residual variance.',
    direction: -1,
    requires: 'price',
    minObservations: 252,
  },
  {
    id: 'price:DOWNSIDE_BETA',
    family: 'downside_risk',
    label: 'Downside Beta',
    formula: 'β(i) conditional on r(m) < 0, 252d',
    rationale: 'Penalizes asymmetric crash exposure.',
    direction: -1,
    requires: 'price',
    minObservations: 252,
  },
  {
    id: 'price:AMIHUD',
    family: 'liquidity',
    label: 'Amihud Illiquidity Ratio',
    formula: '(1/N) Σ |r(t)| / (Close(t) · Volume(t)), 60d',
    rationale: 'Illiquidity premium and tradability filter.',
    direction: 1,
    requires: 'price',
    minObservations: 60,
  },
  {
    id: 'price:GK_VOL',
    family: 'volatility_structure',
    label: 'Garman–Klass Range Volatility',
    formula: '√(½·ln(H/L)² − (2ln2−1)·ln(C/O)²) · √252, 21d mean',
    rationale: 'More statistically efficient than close-to-close volatility.',
    direction: -1,
    requires: 'price',
    minObservations: 21,
  },
  {
    id: 'price:PARKINSON_VOL',
    family: 'volatility_structure',
    label: 'Parkinson Range Volatility',
    formula: '√(ln(H/L)² / (4ln2)) · √252, 21d mean',
    rationale: 'High/low range estimator with higher efficiency than close-only volatility.',
    direction: -1,
    requires: 'price',
    minObservations: 21,
  },
] as const;

export interface PriceFactorOptions {
  /** Only emit values for thefinal `maxPoints` dates (full history is still used as lookback). */
  maxPoints?: number;
}

/**
 * Compute every price factor as a date-aligned series.
 * `benchmarkBars` supplies market returns for CAPM-based factors; when absent,
 * beta/residual factors are null but the remaining factors still compute.
 */
export function computePriceFactorSeries(
  bars: readonly PriceBar[] | null | undefined,
  benchmarkBars: readonly PriceBar[] | null | undefined,
  options: PriceFactorOptions = {}
): Record<string, (number | null)[]> {
  const list = Array.isArray(bars) ? bars : [];
  const ids = PRICE_FACTORS.map((f) => f.id);
  const empty: Record<string, (number | null)[]> = {};
  for (const id of ids) empty[id] = new Array(list.length).fill(null);
  if (list.length < 2) return empty;

  const closes = list.map(adjClose);
  const returns: (number | null)[] = list.map((_, i) =>
    i === 0 || !finite(closes[i]) || !finite(closes[i - 1]) || closes[i - 1] <= 0
      ? null
      : Math.log(closes[i] / closes[i - 1])
  );

  const benchByDate = new Map<string, number>();
  const bench = Array.isArray(benchmarkBars) ? benchmarkBars : [];
  for (let i = 1; i < bench.length; i++) {
    const a = adjClose(bench[i - 1]);
    const b = adjClose(bench[i]);
    if (finite(a) && finite(b) && a > 0) benchByDate.set(bench[i].date, Math.log(b / a));
  }
  const benchReturns = list.map((bar) => benchByDate.get(bar.date) ?? null);

  const gkVariance = list.map((bar) => {
    if (bar.high <= 0 || bar.low <= 0 || bar.open <= 0 || bar.close <= 0 || bar.high < bar.low) return NaN;
    const hl = Math.log(bar.high / bar.low);
    const co = Math.log(bar.close / bar.open);
    return Math.max(0, 0.5 * hl * hl - (2 * Math.LN2 - 1) * co * co);
  });
  const parkinsonVariance = list.map((bar) => {
    if (bar.high <= 0 || bar.low <= 0 || bar.high < bar.low) return NaN;
    const hl = Math.log(bar.high / bar.low);
    return (hl * hl) / (4 * Math.LN2);
  });

  const maxPoints = Number.isFinite(options.maxPoints)
    ? Math.max(0, Math.floor(options.maxPoints as number))
    : list.length;
  const from = Math.max(0, list.length - maxPoints);

  const windowReturns = (t: number, n: number): number[] | null => {
    const out: number[] = [];
    for (let i = t - n + 1; i <= t; i++) {
      const r = returns[i];
      if (i < 0 || r === null || !finite(r)) return null;
      out.push(r);
    }
    return out;
  };
  const alignedPairs = (t: number, n: number): { y: number[]; x: number[] } | null => {
    const y: number[] = [];
    const x: number[] = [];
    for (let i = t - n + 1; i <= t; i++) {
      const r = returns[i];
      const m = benchReturns[i];
      if (i < 0 || r === null || m === null || !finite(r) || !finite(m)) return null;
      y.push(r);
      x.push(m);
    }
    return { y, x };
  };

  for (let t = from; t < list.length; t++) {
    const out: Record<string, number | null> = {};
    for (const id of ids) out[id] = null;

    if (t >= 252 && finite(closes[t - 252]) && closes[t - 252] > 0 && finite(closes[t - 21])) {
      out['price:MOM_12_1'] = closes[t - 21] / closes[t - 252] - 1;
    }
    if (t >= 126 && finite(closes[t - 126]) && closes[t - 126] > 0 && finite(closes[t - 21])) {
      out['price:MOM_6_1'] = closes[t - 21] / closes[t - 126] - 1;
    }
    if (t >= 63 && finite(closes[t - 63]) && closes[t - 63] > 0 && finite(closes[t - 21])) {
      out['price:MOM_3_1'] = closes[t - 21] / closes[t - 63] - 1;
    }
    if (t >= 252 && finite(closes[t])) {
      let peak = -Infinity;
      for (let i = t - 252; i <= t; i++) if (finite(closes[i])) peak = Math.max(peak, closes[i]);
      if (peak > 0) out['price:HIGH_52W'] = closes[t] / peak;
    }
    if (t >= 21 && finite(closes[t - 21]) && closes[t - 21] > 0 && finite(closes[t])) {
      out['price:REV_1M'] = -(closes[t] / closes[t - 21] - 1);
    }
    if (t >= 5 && finite(closes[t - 5]) && closes[t - 5] > 0 && finite(closes[t])) {
      out['price:REV_5D'] = -(closes[t] / closes[t - 5] - 1);
    }

    const r60 = windowReturns(t, 60);
    if (r60) {
      const s = stdev(r60);
      if (finite(s)) out['price:RV60'] = s * Math.sqrt(TRADING_DAYS);
    }
    const r252 = windowReturns(t, 252);
    if (r252) {
      const s = stdev(r252);
      if (finite(s)) out['price:RV252'] = s * Math.sqrt(TRADING_DAYS);
    }
    const rEwma = windowReturns(t, 252);
    if (rEwma) {
      const v = ewmaVariance(rEwma);
      if (finite(v) && v >= 0) out['price:EWMA_VOL'] = Math.sqrt(v) * Math.sqrt(TRADING_DAYS);
    }

    const pairs = alignedPairs(t, 252);
    if (pairs) {
      const fit = ols(pairs.y, pairs.x);
      if (fit) {
        out['price:BAB_BETA'] = fit.beta;
        const residualVol = stdev(fit.residuals);
        if (finite(residualVol)) out['price:IVOL'] = residualVol;
      }
      const downY: number[] = [];
      const downX: number[] = [];
      for (let i = 0; i < pairs.x.length; i++) {
        if (pairs.x[i] < 0) {
          downY.push(pairs.y[i]);
          downX.push(pairs.x[i]);
        }
      }
      if (downX.length >= 30) {
        const downFit = ols(downY, downX);
        if (downFit) out['price:DOWNSIDE_BETA'] = downFit.beta;
      }
    }

    // Residual momentum: fit on t−252…t−21, then sum residuals over that window.
    if (t >= 252) {
      const residPairs = alignedPairs(t - 21, 231);
      if (residPairs) {
        const fit = ols(residPairs.y, residPairs.x);
        if (fit) {
          let cumulative = 0;
          for (const r of fit.residuals) cumulative += r;
          out['price:RESID_MOM'] = cumulative;
        }
      }
    }

    if (r60) {
      let total = 0;
      let count = 0;
      for (let i = t - 59; i <= t; i++) {
        const bar = list[i];
        const r = returns[i];
        if (!bar || r === null || !finite(r) || !finite(bar.volume) || bar.volume <= 0 || !finite(bar.close) || bar.close <= 0) {
          continue;
        }
        total += Math.abs(r) / (bar.close * bar.volume);
        count++;
      }
      if (count > 0) out['price:AMIHUD'] = (total / count) * 1e6;
    }

    const gkSlice = gkVariance.slice(Math.max(0, t - 20), t + 1).filter(finite);
    if (gkSlice.length >= 21) {
      out['price:GK_VOL'] = Math.sqrt(mean(gkSlice)) * Math.sqrt(TRADING_DAYS);
    }
    const pkSlice = parkinsonVariance.slice(Math.max(0, t - 20), t + 1).filter(finite);
    if (pkSlice.length >= 21) {
      out['price:PARKINSON_VOL'] = Math.sqrt(mean(pkSlice)) * Math.sqrt(TRADING_DAYS);
    }

    for (const id of ids) empty[id][t] = finite(out[id]) ? (out[id] as number) : null;
  }

  return empty;
}

/** Latest-factor convenience used by the regression's factor regressor columns. */
export function computeLatestPriceFactors(
  bars: readonly PriceBar[] | null | undefined,
  benchmarkBars: readonly PriceBar[] | null | undefined
): Record<string, number | null> {
  const ids = PRICE_FACTORS.map((f) => f.id);
  const series = computePriceFactorSeries(bars, benchmarkBars, { maxPoints: 1 });
  const latest: Record<string, number | null> = {};
  for (const id of ids) latest[id] = series[id]?.[series[id].length - 1] ?? null;
  return latest;
}
