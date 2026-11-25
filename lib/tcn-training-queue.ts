// One sequential trainer for every TCN multi-domain head-trigger — the
// unified dashboard's retrain buttons and the app-level bootstrap trainer
// (hooks/use-tcn-bootstrap.ts). Head order, yields between heads, per-head
// cancellation, regression-input gating and the volatility forecast horizon
// live here once, so the two callers can never drift. Pure library code: no
// React, events only via callbacks, plus a tiny subscription surface that
// lets mounted view hooks re-hydrate when the bootstrap writes fresh
// artifacts into the shared engine caches.

import type { DecompositionResult, PriceBar } from './quant-engine';
import { calculateReturns, decomposeTimeSeries, DEFAULT_DECOMPOSITION_CYCLE_BAND } from './quant-engine';
import {
  DEFAULT_TCN_CONFIG,
  MIN_BARS_FOR_TCN,
  buildTcnCacheKey,
  getCachedTcnArtifact,
  hydrateTcnCacheFromSessionStorage,
  runTcnDecompositionForecast,
  type TcnDecompositionForecast,
} from './tcn-attention';
import { selectDefaultRegressionFactors } from './tcn-diagnostics';
import type { TcnTrainableHead } from './tcn-bootstrap-policy';
import {
  runTcnMultiAssetRegression,
  runTcnRiskVarForecast,
  runTcnVolatilityForecast,
  type MultiDomainRunOptions,
  type TcnMultiAssetRegressionResult,
  type TcnMultiDomainConfig,
  type TcnRiskVarForecast,
  type TcnVolatilityForecast,
} from './tcn-multi-domain';

export type TcnHeadResult =
  | TcnVolatilityForecast
  | TcnRiskVarForecast
  | TcnMultiAssetRegressionResult
  | null;

export interface TcnHeadSequenceContext {
  ticker: string;
  bars: PriceBar[];
  allStockBars: Record<string, PriceBar[]>;
  /** Manual retrains force a fresh fit; the bootstrap passes false and lets
   * each run's own cache decide (a full cache completes the boot instantly). */
  forceRetrain: boolean;
  yieldControl: () => Promise<void>;
  isCancelled: () => boolean;
  config?: Partial<TcnMultiDomainConfig>;
  volatilityForecastDays?: number;
}

export interface TcnHeadSequenceEvents {
  /** Fires when a head's fit starts (cache miss) or is about to be answered
   * from cache. Never fires for a head after cancellation. */
  onHeadStart?: (head: TcnTrainableHead) => void;
  /** result === null means skipped or insufficient (e.g. regression without
   * peer-series alignment) — reported, never fabricated. */
  onHeadDone?: (head: TcnTrainableHead, result: TcnHeadResult) => void;
}

/** Fit the given heads in order. Cancels cleanly between heads; a cancelled
 * head produces no onHeadDone (its result is discarded, not surfaced). */
export async function runTcnHeadSequence(
  heads: readonly TcnTrainableHead[],
  ctx: TcnHeadSequenceContext,
  events: TcnHeadSequenceEvents = {},
): Promise<void> {
  for (const head of heads) {
    if (ctx.isCancelled()) return;
    events.onHeadStart?.(head);
    await ctx.yieldControl();
    if (ctx.isCancelled()) return;
    const result = await runHead(head, ctx);
    if (ctx.isCancelled()) return;
    events.onHeadDone?.(head, result);
    await ctx.yieldControl();
  }
}

async function runHead(head: TcnTrainableHead, ctx: TcnHeadSequenceContext): Promise<TcnHeadResult> {
  const options: MultiDomainRunOptions = {
    config: ctx.config,
    forceRetrain: ctx.forceRetrain,
    yieldControl: ctx.yieldControl,
    isCancelled: ctx.isCancelled,
  };
  if (head === 'volatility') {
    return runTcnVolatilityForecast(ctx.ticker, ctx.bars, {
      ...options,
      forecastDays: ctx.volatilityForecastDays ?? 30,
    });
  }
  if (head === 'risk_var') {
    const returns = calculateReturns(ctx.bars.map((b) => b.close), 'log');
    if (returns.length < 70) return null;
    const retDates = ctx.bars.slice(1).map((b) => b.date);
    return runTcnRiskVarForecast(ctx.ticker, returns, retDates, options);
  }
  // regression — runs only when peer series align on dates; a missing
  // alignment is a null result (deferred to manual retrain), never an error.
  const regInputs = selectDefaultRegressionFactors(ctx.ticker, ctx.bars, ctx.allStockBars);
  if (!regInputs) return null;
  return runTcnMultiAssetRegression(
    regInputs.dependentTicker,
    regInputs.independentTickers,
    regInputs.returnsByTicker,
    regInputs.dates,
    options
  );
}

// ---------------------------------------------------------------------------
// TCN decomposition (Analysis → Overview) — app-load cache lookup / default fit
// ---------------------------------------------------------------------------

/** Decomposition inputs the Analysis view uses (app-level state). */
export interface TcnDecompositionBootParams {
  startDate?: string;
  endDate?: string;
  period: number;
  model: 'additive' | 'multiplicative';
  cycleBand?: { min: number; max: number };
}

/**
 * Builds the exact decomposition the Analysis view renders for these params:
 * same date filter, same `decomposeTimeSeries` call, same default cycle band.
 * Identical input ⇒ identical TCN cache key, so a bootstrap fit is the one the
 * chart looks up.
 */
export function buildBootDecomposition(
  bars: PriceBar[],
  params: TcnDecompositionBootParams
): DecompositionResult | null {
  const filtered = bars.filter(
    (b) => !(params.startDate && b.date < params.startDate) && !(params.endDate && b.date > params.endDate)
  );
  if (filtered.length < MIN_BARS_FOR_TCN) return null;
  const band = params.cycleBand ?? DEFAULT_DECOMPOSITION_CYCLE_BAND;
  return decomposeTimeSeries(
    filtered.map((b) => b.date),
    filtered.map((b) => b.close),
    params.period,
    params.model,
    band.min,
    band.max
  );
}

export interface TcnDecompositionBootResult {
  forecast: TcnDecompositionForecast | null;
  /** true ⇒ answered from the (session-hydrated) cache without training. */
  fromCache: boolean;
}

/**
 * Cache first: hydrate the session cache and look up the default-config
 * artifact for this decomposition. Only on a miss is the model trained with
 * DEFAULT_TCN_CONFIG (cached by the engine). A cancelled fit returns null and
 * caches nothing.
 */
export async function runTcnDecompositionBootstrap(ctx: {
  ticker: string;
  bars: PriceBar[];
  params: TcnDecompositionBootParams;
  yieldControl: () => Promise<void>;
  isCancelled: () => boolean;
}): Promise<TcnDecompositionBootResult> {
  const decomposition = buildBootDecomposition(ctx.bars, ctx.params);
  if (!decomposition) return { forecast: null, fromCache: false };
  hydrateTcnCacheFromSessionStorage();
  const cached = getCachedTcnArtifact(buildTcnCacheKey(ctx.ticker, decomposition, DEFAULT_TCN_CONFIG));
  if (cached) {
    // Re-run through the engine so its data-signature index points at this artifact.
    const forecast = await runTcnDecompositionForecast(ctx.ticker, decomposition, DEFAULT_TCN_CONFIG);
    return { forecast, fromCache: true };
  }
  const forecast = await runTcnDecompositionForecast(ctx.ticker, decomposition, DEFAULT_TCN_CONFIG, {
    yieldControl: ctx.yieldControl,
    isCancelled: ctx.isCancelled,
    useCache: true,
  });
  return { forecast: ctx.isCancelled() ? null : forecast, fromCache: false };
}

// ---------------------------------------------------------------------------
// Bootstrap-completion notifications — mounted hooks (useTcnVolatility,
// useTcnDiagnosticsBundle) subscribe so the boot trainer's cache writes
// surface immediately instead of waiting for the next bar tick to re-run
// their hydration effects.
// ---------------------------------------------------------------------------

/** Everything the app-level bootstrap fits: the decomposition forecast plus the multi-domain heads. */
export type TcnBootHead = TcnTrainableHead | 'decomposition';

type TcnBootListener = (ticker: string, head: TcnBootHead) => void;

const bootListeners = new Set<TcnBootListener>();

export function subscribeTcnBootUpdates(listener: TcnBootListener): () => void {
  bootListeners.add(listener);
  return () => {
    bootListeners.delete(listener);
  };
}

export function notifyTcnBootUpdate(ticker: string, head: TcnBootHead): void {
  for (const listener of [...bootListeners]) {
    try {
      listener(ticker, head);
    } catch {
      // a broken listener never blocks the remaining notifications
    }
  }
}
