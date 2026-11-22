// Pure TypeScript computations for the Home landing surface:
// - Strictly causal 3-state HMM volatility regime series for the active hero chart
// - Compact per-symbol cards (60-session sparkline, RV20, causal HMM regime)
//   for the 10-ticker live market watchboard
// Zero React/DOM dependencies — relative imports only for node --test compatibility.

import { calculateReturns, type PriceBar } from './quant-engine';
import {
  buildCausalReturnVolFeatures,
  fitAndFilterCausalHmm,
  HMM_REGIME_LABELS,
  type HmmRegimeLabel,
} from './hmm';
import { extractDominantHmmSeries, type HmmStateIdx } from './hmm-regime-overlay';

export interface HomeUniverseSymbolMeta {
  ticker: string;
  name: string;
  sector: string;
}

export const HOME_UNIVERSE_SYMBOLS: readonly HomeUniverseSymbolMeta[] = [
  { ticker: 'AAPL', name: 'Apple Inc.', sector: 'Tech' },
  { ticker: 'MSFT', name: 'Microsoft Corp.', sector: 'Tech' },
  { ticker: 'GOOGL', name: 'Alphabet Inc.', sector: 'Tech' },
  { ticker: 'NVDA', name: 'NVIDIA Corp.', sector: 'Semis' },
  { ticker: 'TSLA', name: 'Tesla Inc.', sector: 'Auto' },
  { ticker: 'SPY', name: 'SPDR S&P 500 ETF', sector: 'Index' },
  { ticker: 'QQQ', name: 'Invesco QQQ Trust', sector: 'Index' },
  { ticker: 'AMZN', name: 'Amazon.com Inc.', sector: 'Tech' },
  { ticker: 'META', name: 'Meta Platforms', sector: 'Tech' },
  { ticker: 'JPM', name: 'JPMorgan Chase', sector: 'Banks' },
] as const;

export interface ActiveHmmPoint {
  date: string;
  close: number;
  retPct: number;
  state: HmmStateIdx;
  prob: number;
}

export interface ActiveHmmSeriesResult {
  points: ActiveHmmPoint[];
  latestState: HmmStateIdx;
  latestLabel: HmmRegimeLabel;
  latestProb: number;
  probs: [number, number, number];
}

/**
 * Fits a 3-state Gaussian HMM on a strictly historical training prefix of the
 * trailing daily window (up to `maxHistoryBars`, default 504 sessions / ~2Y)
 * and forward-filters strictly causal regime probabilities across the window.
 * Returns null if fewer than 45 valid bars are available.
 */
export function computeActiveCausalHmmSeries(
  bars: PriceBar[] | null | undefined,
  maxHistoryBars = 504
): ActiveHmmSeriesResult | null {
  if (!bars || bars.length < 45) return null;

  const slice = bars.length > maxHistoryBars + 1 ? bars.slice(-(maxHistoryBars + 1)) : bars;
  const closes = slice.map((b) => b.close);
  const rets = calculateReturns(closes, 'log');
  if (rets.length < 40) return null;

  const features = buildCausalReturnVolFeatures(rets);
  const trainEndIdx = Math.max(35, Math.min(features.length - 5, Math.floor(features.length * 0.6)));
  const causal = fitAndFilterCausalHmm(features, trainEndIdx, { maxIters: 40 });
  if (!causal) return null;

  const { states, probs: dominantProbs } = extractDominantHmmSeries(causal.filteredPosteriors);
  if (states.length !== rets.length) return null;

  const points: ActiveHmmPoint[] = new Array(rets.length);
  for (let i = 0; i < rets.length; i++) {
    const bar = slice[i + 1];
    const prevClose = slice[i].close;
    const pct = prevClose > 0 ? ((bar.close - prevClose) / prevClose) * 100 : 0;
    points[i] = {
      date: bar.date,
      close: bar.close,
      retPct: Number.isFinite(pct) ? pct : 0,
      state: states[i],
      prob: dominantProbs[i],
    };
  }

  return {
    points,
    latestState: causal.summary.dominantState,
    latestLabel: causal.summary.dominantLabel,
    latestProb: dominantProbs[dominantProbs.length - 1] ?? causal.summary.probs[causal.summary.dominantState],
    probs: causal.summary.probs,
  };
}

/**
 * Computes sample standard deviation of the trailing `window` log returns,
 * annualized by sqrt(252) (direct RV convention matching TickerSnapshotProvider).
 */
export function computeTrailingRv(bars: PriceBar[] | null | undefined, window = 20): number | null {
  if (!bars || bars.length < window + 1) return null;
  const tail = bars.slice(-(window + 1));
  const rets = calculateReturns(
    tail.map((b) => b.close),
    'log'
  );
  if (rets.length < window) return null;
  let mean = 0;
  for (let i = 0; i < rets.length; i++) mean += rets[i];
  mean /= rets.length;
  let sq = 0;
  for (let i = 0; i < rets.length; i++) {
    const d = rets[i] - mean;
    sq += d * d;
  }
  const variance = sq / Math.max(1, rets.length - 1);
  const rv = Math.sqrt(Math.max(0, variance) * 252);
  return Number.isFinite(rv) ? rv : null;
}

export interface HomeUniverseCardData {
  ticker: string;
  name: string;
  sector: string;
  price: number | null;
  changePct: number | null;
  pending: boolean;
  sparkline: number[];
  sparklinePositive: boolean;
  rv20: number | null;
  regimeState: HmmStateIdx | null;
  regimeLabel: HmmRegimeLabel | null;
  regimeProb: number | null;
}

export interface LiveQuoteInput {
  price: number;
  changePct: number;
  asOf?: string;
}

export interface TapeItemInput {
  symbol: string;
  price: number;
  changePct: number;
  pending?: boolean;
}

/**
 * Produces one watchboard card summary for a universe symbol.
 * Uses trailing <= 252 bars for a fast causal 3-state HMM fit and trailing 60
 * closes for the sparkline. Respects live quotes and `pending` honesty so
 * synthetic fallback closes are never displayed as live prices during initial load.
 */
export function computeHomeUniverseCard(
  meta: HomeUniverseSymbolMeta,
  bars: PriceBar[] | null | undefined,
  liveQuote?: LiveQuoteInput | null,
  tapeItem?: TapeItemInput | null
): HomeUniverseCardData {
  const safeBars = bars && bars.length > 0 ? bars : [];
  const sparkSlice = safeBars.length > 60 ? safeBars.slice(-60) : safeBars;
  const sparkline = sparkSlice.map((b) => b.close).filter((v) => Number.isFinite(v) && v > 0);
  const sparklinePositive =
    sparkline.length >= 2 ? sparkline[sparkline.length - 1] >= sparkline[0] : true;

  let price: number | null = null;
  let changePct: number | null = null;
  let pending = false;

  if (liveQuote && Number.isFinite(liveQuote.price) && liveQuote.price > 0) {
    price = liveQuote.price;
    changePct = Number.isFinite(liveQuote.changePct) ? liveQuote.changePct : 0;
    pending = false;
  } else if (tapeItem) {
    price = Number.isFinite(tapeItem.price) ? tapeItem.price : null;
    changePct = Number.isFinite(tapeItem.changePct) ? tapeItem.changePct : null;
    pending = Boolean(tapeItem.pending);
  } else if (safeBars.length >= 2) {
    const last = safeBars[safeBars.length - 1].close;
    const prev = safeBars[safeBars.length - 2].close;
    price = Number.isFinite(last) ? last : null;
    changePct = prev > 0 && Number.isFinite(last) ? ((last - prev) / prev) * 100 : null;
    pending = false;
  }

  const rv20 = computeTrailingRv(safeBars, 20);
  const hmm = computeActiveCausalHmmSeries(safeBars, 252);

  return {
    ticker: meta.ticker,
    name: meta.name,
    sector: meta.sector,
    price,
    changePct,
    pending,
    sparkline,
    sparklinePositive,
    rv20,
    regimeState: hmm ? hmm.latestState : null,
    regimeLabel: hmm ? HMM_REGIME_LABELS[hmm.latestState] : null,
    regimeProb: hmm ? hmm.latestProb : null,
  };
}

/**
 * Generates normalized SVG line + area paths for a compact sparkline.
 */
export function buildSparklineSvgPath(
  values: readonly number[],
  width: number,
  height: number,
  padY = 3
): { linePath: string; areaPath: string } {
  if (values.length < 2) {
    return { linePath: '', areaPath: '' };
  }
  let min = values[0];
  let max = values[0];
  for (let i = 1; i < values.length; i++) {
    if (values[i] < min) min = values[i];
    if (values[i] > max) max = values[i];
  }
  const span = Math.max(max - min, 1e-6);
  const innerH = Math.max(1, height - padY * 2);
  const stepX = width / Math.max(1, values.length - 1);

  const coords: string[] = new Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const x = i * stepX;
    const y = padY + (1 - (values[i] - min) / span) * innerH;
    coords[i] = `${x.toFixed(1)},${y.toFixed(1)}`;
  }

  const linePath = `M${coords.join(' L')}`;
  const areaPath = `${linePath} L${width.toFixed(1)},${height.toFixed(1)} L0,${height.toFixed(1)} Z`;
  return { linePath, areaPath };
}
