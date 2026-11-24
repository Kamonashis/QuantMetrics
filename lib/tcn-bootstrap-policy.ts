// Pure decision logic for the app-level TCN bootstrap trainer
// (hooks/use-tcn-bootstrap.ts). The hooks stay thin: readiness conditions and
// "once" semantics are decided here so they are unit-testable and identical
// for every caller. No React, no I/O.

/** Heads the bootstrap trainer can fit (mirrors the bundle's retrain set). */
export type TcnTrainableHead = 'volatility' | 'risk_var' | 'regression';

/** Same engine minimum as the multi-domain run functions. */
export const TCN_BOOT_MIN_BARS = 70;

export type TcnBootstrapAction =
  | { kind: 'train'; ticker: string; heads: TcnTrainableHead[] }
  | { kind: 'skip'; ticker: string; reason: 'already-bootstrapped' | 'training-in-flight' }
  | {
      kind: 'defer';
      ticker: string;
      reason: 'no-ticker' | 'insufficient-bars' | 'awaiting-live-bars' | 'synthetic-data';
    };

export interface TcnBootstrapInputs {
  ticker: string;
  /** Tickers whose one-time bootstrap already completed this app session. */
  bootstrapped: ReadonlySet<string>;
  barsCount: number;
  /** Bars provenance: 'live_market' trains; anything else defers. */
  barsSource: string | null | undefined;
  /** selectDefaultRegressionFactors(ticker, bars, allStockBars) !== null. */
  regressionInputsReady: boolean;
  /** A fit for this ticker is already running (bootstrap or a retrain button). */
  trainingInFlight: boolean;
}

const BASE_HEADS: TcnTrainableHead[] = ['volatility', 'risk_var'];

/**
 * One fit per ticker per app session, then manual-only. A miss defers (the
 * watcher re-arms when data arrives); completion lands in the caller's
 * session registry and never re-fires — only retrain buttons refresh after
 * that. Live-tick signature drift re-arms the watcher but cannot re-train a
 * registered ticker.
 */
export function nextTcnBootstrapAction(input: TcnBootstrapInputs): TcnBootstrapAction {
  const ticker = input.ticker.trim().toUpperCase();
  if (!ticker) return { kind: 'defer', ticker, reason: 'no-ticker' };
  // In-flight first: an aborted run is not in the registry yet, and starting
  // a second fit for the same ticker while one is winding down must be a
  // no-op (the run's completion re-arms the watcher).
  if (input.trainingInFlight) return { kind: 'skip', ticker, reason: 'training-in-flight' };
  if (input.bootstrapped.has(ticker)) return { kind: 'skip', ticker, reason: 'already-bootstrapped' };
  if (!Number.isFinite(input.barsCount) || input.barsCount < TCN_BOOT_MIN_BARS) {
    return { kind: 'defer', ticker, reason: 'insufficient-bars' };
  }
  // The bootstrap trains real history only; synthetic fallback waits for
  // live bars (manual training buttons remain available meanwhile).
  if (input.barsSource !== 'live_market') {
    return {
      kind: 'defer',
      ticker,
      reason: input.barsSource === 'synthetic_gbm' ? 'synthetic-data' : 'awaiting-live-bars',
    };
  }
  const heads: TcnTrainableHead[] = input.regressionInputsReady
    ? [...BASE_HEADS, 'regression']
    : [...BASE_HEADS];
  return { kind: 'train', ticker, heads };
}
