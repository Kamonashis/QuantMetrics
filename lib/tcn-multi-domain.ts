// Pure TypeScript Multi-Channel, Multi-Horizon (1d / 5d / 22d) Dilated Causal TCN
// with Causal Multi-Head Self-Attention & Target-Specific Loss Heads.
//
// Zero React/DOM dependencies (guarded sessionStorage persistence only).
// Powers:
//   1. Volatility Modeling & Volatility T4 Forecast Horse Race (QLIKE log-variance head)
//   2. Analysis Risk VaR & Expected Shortfall (Asymmetric Pinball quantile head)
//   3. Regression Multi-Asset Nonlinear Sequence Forecasting (Huber multi-channel head)
//   4. Edge & Volatility T8 Signal Confluence (Zero-retrain cached fusion)
//   5. F&O Expected Move vs TCN Forecast Dislocation (Variance-time interpolation)
//   6. Correlation Dynamics & Spillover Forecasting (Fisher-z tanh bounded head)

import type { GarchForecastPoint, PriceBar } from './quant-engine';
import type { VaRBacktestModelResult } from './analysis-engine';
import { qlike, mseLogVariance } from './vol-analytics';
import type { TcnDecompositionForecast } from './tcn-attention';
import { computeHistoricalBacktest } from './tcn-attention';
import {
  buildCausalReturnVolFeatures,
  fitAndFilterCausalHmm,
  type CausalHmmResult,
  type CausalHmmSummary,
} from './hmm';
import {
  extractDominantHmmSeries,
  HMM_REGIME_TOKENS,
  type CorrRegimePerfRow,
  type HmmStateIdx,
  type RegRegimePerfRow,
  type RiskRegimePerfRow,
  type VolRegimePerfRow,
} from './hmm-regime-overlay';

export type ForecastHorizon = 1 | 5 | 22;
export const MULTI_HORIZONS: readonly ForecastHorizon[] = [1, 5, 22] as const;

export type TcnDomainKey = 'volatility' | 'risk_var' | 'regression' | 'correlation';

export interface TcnMultiDomainConfig {
  lookback: number;
  dilations: number[];
  kernelSize: 2 | 3;
  numHeads: number;
  hiddenChannels: number;
  epochs: number;
  learningRate: number;
  trainSplit: number;
  seed: number;
}

export const DEFAULT_MULTIDOMAIN_CONFIG: TcnMultiDomainConfig = {
  lookback: 22,
  dilations: [1, 2, 4],
  kernelSize: 3,
  numHeads: 2,
  hiddenChannels: 8,
  epochs: 36,
  learningRate: 0.012,
  trainSplit: 0.76,
  seed: 42,
};

export interface AttentionShockDate {
  date: string;
  lag: number;
  weight: number;
  metricLabel: string;
  metricValue: number;
}

export interface MultiDomainRunOptions {
  config?: Partial<TcnMultiDomainConfig>;
  useCache?: boolean;
  forceRetrain?: boolean;
  yieldControl?: () => Promise<void>;
  isCancelled?: () => boolean;
}

// ---------------------------------------------------------------------------
// 1. DOMAIN OUTPUT INTERFACES
// ---------------------------------------------------------------------------

export interface VolHorizonForecast {
  horizon: ForecastHorizon;
  /** Forecast average daily variance over next h trading sessions */
  dailyVariance: number;
  /** Annualized realized volatility sqrt(252 * dailyVariance) (decimal, e.g. 0.24) */
  annualizedVol: number;
  /** Lower 99% annualized volatility bound */
  lower99Vol: number;
  /** Upper 99% annualized volatility bound */
  upper99Vol: number;
  /** Strictly out-of-sample holdout QLIKE loss */
  holdoutQlike: number;
  /** Strictly out-of-sample holdout MSE on log-variance */
  holdoutMseLogVar: number;
  /** Number of holdout sessions scored */
  holdoutCount: number;
}

export interface TcnVolatilityForecast {
  ticker: string;
  cacheKey: string;
  dataSignature: string;
  asOfDate: string;
  horizons: Record<ForecastHorizon, VolHorizonForecast>;
  /** Strictly out-of-sample daily variance aligned to returns[] (NaN during train/warmup) */
  oosDailyVarianceByHorizon: Record<ForecastHorizon, number[]>;
  /** Conditional annualized vol series aligned to returns[] for ModelingView chart */
  fittedAnnualizedVol: number[];
  /** Index into returns[] where out-of-sample holdout begins */
  holdoutStartIdx: number;
  /** Forward daily vol cone interpolated smoothly across 1d / 5d / 22d anchors */
  forecastCone: GarchForecastPoint[];
  /** Causal self-attention lag weights (length = lookback, sums to 1) */
  attentionWeights: number[];
  topShockDates: AttentionShockDate[];
  parameterCount: number;
  receptiveField: number;
  trainingDurationMs: number;
  cacheHit: boolean;
  /** Strictly causal 3-state HMM volatility regime feature state fed into the TCN */
  hmmRegime: CausalHmmSummary | null;
  /** Strictly causal dominant HMM state (0=Calm, 1=Watchful, 2=Crisis) aligned to returns[] */
  hmmStates: HmmStateIdx[];
  /** Posterior probability of the dominant causal HMM state aligned to returns[] */
  hmmProbs: number[];
  /** Per-regime 1d OOS performance breakdown stratified by causal HMM state at prediction origin */
  regimePerformance: VolRegimePerfRow[];
}

export interface RiskHorizonForecast {
  horizon: ForecastHorizon;
  /** Positive loss magnitude (decimal, e.g. 0.025 = 2.5% loss) */
  var95: number;
  var99: number;
  es95: number;
  es99: number;
  /** ES99 / VaR99 tail-heaviness ratio */
  tailRatio: number;
  /** Strictly out-of-sample empirical exceedance rates on holdout */
  oosExceedanceRate95: number;
  oosExceedanceRate99: number;
  holdoutCount: number;
}

export interface TcnRiskVarForecast {
  ticker: string;
  cacheKey: string;
  dataSignature: string;
  asOfDate: string;
  horizons: Record<ForecastHorizon, RiskHorizonForecast>;
  /** Full VaRBacktestModelResult for direct insertion into VaRExplorer & HMM Attribution */
  backtestRow: VaRBacktestModelResult;
  /** Strictly OOS 1d VaR95 / VaR99 thresholds aligned to returns[] (NaN outside holdout) */
  oosVar95Series: number[];
  oosVar99Series: number[];
  /** Realized 1d loss (-r_t) aligned to returns[] on OOS holdout (NaN outside holdout) */
  oosDailyLossSeries: number[];
  /** Session dates aligned to returns[] */
  dates: string[];
  holdoutStartIdx: number;
  attentionWeights: number[];
  topShockDates: AttentionShockDate[];
  parameterCount: number;
  receptiveField: number;
  trainingDurationMs: number;
  cacheHit: boolean;
  /** Strictly causal 3-state HMM volatility regime feature state fed into the TCN */
  hmmRegime: CausalHmmSummary | null;
  /** Strictly causal dominant HMM state aligned to returns[] */
  hmmStates: HmmStateIdx[];
  /** Posterior probability of the dominant causal HMM state aligned to returns[] */
  hmmProbs: number[];
  /** Per-regime 1d OOS VaR exceedance breakdown stratified by causal HMM state at prediction origin (t-1) */
  regimePerformance: RiskRegimePerfRow[];
}

export interface RegressionHorizonForecast {
  horizon: ForecastHorizon;
  /** Predicted cumulative return of dependent ticker over next h sessions (decimal) */
  predictedReturn: number;
  lower99: number;
  upper99: number;
  oosR2: number;
  olsOosR2: number;
  /** OOS R² improvement over lag-matched OLS baseline */
  nonlinearLiftR2: number;
  oosRmse: number;
  olsOosRmse: number;
  directionalAccuracyPct: number;
  holdoutCount: number;
}

export interface FactorSequenceImportance {
  ticker: string;
  isDependentLag: boolean;
  /** True when this entry measures the joint permutation impact of the 3 HMM regime channels */
  isHmmRegime?: boolean;
  /** Normalized share of holdout permutation RMSE degradation (0..100) */
  importancePct: number;
  /** Mean RMSE increase in basis points when factor channel is permuted */
  rmseIncreaseBps: number;
}

export interface RegressionOosPoint {
  date: string;
  actualReturn: number;
  predictedReturn: number;
  olsPredictedReturn: number;
  hmmState: HmmStateIdx;
  hmmProb: number;
}

export interface TcnMultiAssetRegressionResult {
  dependentTicker: string;
  independentTickers: string[];
  cacheKey: string;
  dataSignature: string;
  asOfDate: string;
  horizons: Record<ForecastHorizon, RegressionHorizonForecast>;
  factorImportance: FactorSequenceImportance[];
  attentionWeights: number[];
  topTransmissionDates: AttentionShockDate[];
  holdoutCount: number;
  parameterCount: number;
  receptiveField: number;
  trainingDurationMs: number;
  cacheHit: boolean;
  /** Strictly causal 3-state HMM volatility regime feature state fed into the TCN */
  hmmRegime: CausalHmmSummary | null;
  /** Strictly causal dominant HMM state aligned to dependent return series */
  hmmStates: HmmStateIdx[];
  /** Posterior probability of the dominant causal HMM state aligned to dependent return series */
  hmmProbs: number[];
  /** Strictly OOS 1d holdout trajectory annotated with prediction-origin causal HMM state */
  oosHistory: RegressionOosPoint[];
  /** Per-regime 1d OOS regression accuracy breakdown */
  regimePerformance: RegRegimePerfRow[];
}

export interface CorrelationHorizonForecast {
  horizon: ForecastHorizon;
  /** Forecast pairwise correlation strictly bounded in (-1, 1) */
  predictedCorr: number;
  lower99: number;
  upper99: number;
  deltaVsCurrent: number;
  oosMae: number;
  oosRmse: number;
  holdoutCount: number;
}

export interface CorrelationSpilloverAlert {
  level: 'ELEVATED' | 'WATCH' | 'CALM';
  maxDelta: number;
  peakHorizon: ForecastHorizon;
  peakCorr: number;
  summary: string;
}

export interface TcnCorrelationForecast {
  tickerA: string;
  tickerB: string;
  cacheKey: string;
  dataSignature: string;
  asOfDate: string;
  currentCorr: number;
  horizons: Record<ForecastHorizon, CorrelationHorizonForecast>;
  /** Aligned dynamic correlation path for charting */
  dynamicCorrelations: {
    date: string;
    correlation: number;
    isOos: boolean;
    hmmState?: HmmStateIdx;
    hmmProb?: number;
  }[];
  /** Forward cone over forecastDays (strictly bounded in (-1, 1)) */
  forecastCone: { day: number; correlation: number; upper99: number; lower99: number }[];
  spilloverAlert: CorrelationSpilloverAlert;
  attentionWeights: number[];
  topShockDates: AttentionShockDate[];
  holdoutStartIdx: number;
  parameterCount: number;
  receptiveField: number;
  trainingDurationMs: number;
  cacheHit: boolean;
  /** Strictly causal 3-state HMM joint volatility regime feature state fed into the TCN */
  hmmRegime: CausalHmmSummary | null;
  /** Strictly causal dominant joint HMM state aligned to dynamicCorrelations[] */
  hmmStates: HmmStateIdx[];
  /** Posterior probability of the dominant joint HMM state */
  hmmProbs: number[];
  /** Per-regime 1d OOS correlation accuracy breakdown */
  regimePerformance: CorrRegimePerfRow[];
}

export interface TcnExpectedMoveEstimate {
  dte: number;
  /** Interpolated annualized TCN realized volatility (decimal) */
  annualizedVol: number;
  /** 1-sigma expected move in percent: sigma * sqrt(dte/252) * 100 */
  sigmaMovePct: number;
  /** Straddle-equivalent expected move in percent: 0.8 * sigmaMovePct */
  straddleEqMovePct: number;
  /** 99% move bound in percent: 2.576 * sigmaMovePct */
  move99Pct: number;
  /** True when dte > 22 and flat variance extrapolation from 22d anchor is used */
  extrapolated: boolean;
}

export type TcnMoveDislocationVerdict = 'RICH vs TCN' | 'CHEAP vs TCN' | 'ALIGNED';

export interface TcnConfluenceSummary {
  alignment: 'ALIGNED' | 'MIXED' | 'DIVERGENT' | 'PARTIAL';
  tcnRv1d: number | null;
  tcnRv5d: number | null;
  tcnRv22d: number | null;
  /** Direct comparison IV30 - TCN RV22 (decimal; RV √252 vs IV √365 never rescaled) */
  forwardVrp22: number | null;
  /** Term slope of TCN vol: TCN RV5 - TCN RV22 (positive = front spike / backwardation) */
  tcnVolSlope: number | null;
  volRegimeRead: 'SHORT-PREMIUM FAVORED' | 'LONG-GAMMA FAVORED' | 'NEUTRAL VOL' | 'UNAVAILABLE';
  priceTiltPct: number | null;
  pricePiAsymmetry: number | null;
  cyclicalSnrDb: number | null;
  cyclicalSnrRegime: string | null;
  backtestSharpe30d: number | null;
  rationale: string[];
  hasVolArtifact: boolean;
  hasDecompArtifact: boolean;
}

// ---------------------------------------------------------------------------
// 2. MATH UTILITIES, PRNG, STATISTICS
// ---------------------------------------------------------------------------

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function studentT99(df: number): number {
  if (df <= 2) return 9.925;
  if (df <= 5) return 4.032;
  if (df <= 10) return 3.169;
  if (df <= 20) return 2.845;
  if (df <= 30) return 2.75;
  if (df <= 60) return 2.66;
  return 2.576;
}

function cdfNormal(z: number): number {
  const b1 = 0.31938153;
  const b2 = -0.356563782;
  const b3 = 1.781477937;
  const b4 = -1.821255978;
  const b5 = 1.330274429;
  const p = 0.2316419;
  const c = 0.3989422804014337;
  if (z >= 0) {
    const t = 1.0 / (1.0 + p * z);
    return 1.0 - c * Math.exp(-0.5 * z * z) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1);
  }
  const t = 1.0 / (1.0 - p * z);
  return c * Math.exp(-0.5 * z * z) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1);
}

function chiSquarePValue(x: number, df: number): number {
  if (x <= 0 || !Number.isFinite(x)) return 1.0;
  const s = Math.sqrt(2 / (9 * df));
  const z = (Math.pow(x / df, 1 / 3) - (1 - 2 / (9 * df))) / s;
  return clamp(1.0 - cdfNormal(z), 1e-10, 1.0);
}

function safeAtanh(rho: number): number {
  const r = clamp(rho, -0.985, 0.985);
  return 0.5 * Math.log((1 + r) / (1 - r));
}

function nextWeekdayIso(isoDate: string, tradingDaysAhead: number): string {
  const parts = isoDate.split('-').map((p) => parseInt(p, 10));
  if (parts.length !== 3 || parts.some((p) => !Number.isFinite(p))) return `t+${tradingDaysAhead}`;
  const dt = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2]));
  let added = 0;
  while (added < tradingDaysAhead) {
    dt.setUTCDate(dt.getUTCDate() + 1);
    const day = dt.getUTCDay();
    if (day !== 0 && day !== 6) added++;
  }
  const y = dt.getUTCFullYear();
  const m = String(dt.getUTCMonth() + 1).padStart(2, '0');
  const d = String(dt.getUTCDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function hashSeriesFingerprint(values: number[], dates: string[], prefix: string): string {
  const n = values.length;
  const firstDate = dates[0] ?? 'none';
  const lastDate = dates[dates.length - 1] ?? 'none';
  let acc = 2166136261 >>> 0;
  const stride = Math.max(1, Math.floor(n / 64));
  for (let i = 0; i < n; i += stride) {
    const q = Math.round((Number.isFinite(values[i]) ? values[i] : 0) * 1e5);
    acc ^= q & 0xffffffff;
    acc = Math.imul(acc, 16777619) >>> 0;
  }
  if (n > 0) {
    const lastQ = Math.round((Number.isFinite(values[n - 1]) ? values[n - 1] : 0) * 1e5);
    acc ^= lastQ & 0xffffffff;
    acc = Math.imul(acc, 16777619) >>> 0;
  }
  return `${prefix}|n${n}|${firstDate}|${lastDate}|${acc.toString(16)}`;
}

function mergeConfig(partial?: Partial<TcnMultiDomainConfig>, seriesLen = 200): TcnMultiDomainConfig {
  const raw: TcnMultiDomainConfig = {
    ...DEFAULT_MULTIDOMAIN_CONFIG,
    ...partial,
  };
  const numHeads = clamp(Math.floor(raw.numHeads), 1, 4);
  let hiddenChannels = clamp(Math.floor(raw.hiddenChannels), 4, 24);
  if (hiddenChannels % numHeads !== 0) {
    hiddenChannels = Math.max(numHeads, Math.round(hiddenChannels / numHeads) * numHeads);
  }
  const maxL = Math.max(8, Math.min(36, Math.floor(seriesLen * 0.25)));
  const lookback = clamp(Math.floor(raw.lookback), 8, maxL);
  const kernelSize: 2 | 3 = raw.kernelSize === 2 ? 2 : 3;
  const dilations = raw.dilations.length > 0 ? raw.dilations.map((d) => clamp(Math.floor(d), 1, 8)) : [1, 2, 4];
  const epochs = clamp(Math.floor(raw.epochs), 12, 100);
  const learningRate = clamp(raw.learningRate, 0.001, 0.04);
  const trainSplit = clamp(raw.trainSplit, 0.65, 0.85);
  return {
    lookback,
    dilations,
    kernelSize,
    numHeads,
    hiddenChannels,
    epochs,
    learningRate,
    trainSplit,
    seed: raw.seed ?? 42,
  };
}

function configKey(cfg: TcnMultiDomainConfig): string {
  return `L${cfg.lookback}_D${cfg.dilations.join('-')}_K${cfg.kernelSize}_H${cfg.numHeads}_C${cfg.hiddenChannels}_E${cfg.epochs}_S${cfg.seed}_HMM3P`;
}

export function computeMultiDomainReceptiveField(dilations: number[], kernelSize: number): number {
  const sum = dilations.reduce((a, d) => a + Math.max(1, d), 0);
  return 1 + Math.max(0, kernelSize - 1) * sum;
}

// ---------------------------------------------------------------------------
// 3. SHARED MULTI-DOMAIN LRU + SESSION STORAGE CACHE
// ---------------------------------------------------------------------------

const SESSION_KEY = 'quantmetrics_tcn_multidomain_cache_v3_hmm_perf';
const MAX_ENTRIES_PER_DOMAIN = 16;

const domainCaches: Record<TcnDomainKey, Map<string, unknown>> = {
  volatility: new Map(),
  risk_var: new Map(),
  regression: new Map(),
  correlation: new Map(),
};

const latestVolatilityByTicker = new Map<string, string>();
let sessionHydrated = false;

function lruGet<T>(map: Map<string, unknown>, key: string): T | null {
  if (!map.has(key)) return null;
  const val = map.get(key) as T;
  map.delete(key);
  map.set(key, val);
  return val;
}

function lruSet(map: Map<string, unknown>, key: string, value: unknown): void {
  if (map.has(key)) map.delete(key);
  map.set(key, value);
  while (map.size > MAX_ENTRIES_PER_DOMAIN) {
    const oldest = map.keys().next().value;
    if (oldest === undefined) break;
    map.delete(oldest);
  }
}

function persistMultiDomainCache(): void {
  if (typeof window === 'undefined' || !window.sessionStorage) return;
  try {
    const payload: Record<string, [string, unknown][]> = {};
    for (const domain of Object.keys(domainCaches) as TcnDomainKey[]) {
      const entries = Array.from(domainCaches[domain].entries()).slice(-6);
      payload[domain] = entries;
    }
    const serialized = JSON.stringify({
      v: 3,
      caches: payload,
      latestVol: Array.from(latestVolatilityByTicker.entries()).slice(-12),
    });
    if (serialized.length < 2_500_000) {
      window.sessionStorage.setItem(SESSION_KEY, serialized);
    }
  } catch {
    // Ignore storage quota / private-mode errors gracefully
  }
}

export function hydrateTcnMultiDomainCache(): void {
  if (sessionHydrated) return;
  sessionHydrated = true;
  if (typeof window === 'undefined' || !window.sessionStorage) return;
  try {
    const raw = window.sessionStorage.getItem(SESSION_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw) as {
      v?: number;
      caches?: Record<string, [string, unknown][]>;
      latestVol?: [string, string][];
    };
    if (!parsed || parsed.v !== 3 || !parsed.caches) return;
    for (const domain of Object.keys(domainCaches) as TcnDomainKey[]) {
      const entries = parsed.caches[domain];
      if (Array.isArray(entries)) {
        for (const [k, v] of entries) {
          if (typeof k === 'string' && v && typeof v === 'object') {
            lruSet(domainCaches[domain], k, v);
          }
        }
      }
    }
    if (Array.isArray(parsed.latestVol)) {
      for (const [tk, ck] of parsed.latestVol) {
        if (typeof tk === 'string' && typeof ck === 'string') {
          latestVolatilityByTicker.set(tk, ck);
        }
      }
    }
  } catch {
    // Corrupted cache ignored
  }
}

export function clearTcnMultiDomainCache(domain?: TcnDomainKey): void {
  if (domain) {
    domainCaches[domain].clear();
    if (domain === 'volatility') latestVolatilityByTicker.clear();
  } else {
    for (const d of Object.keys(domainCaches) as TcnDomainKey[]) {
      domainCaches[d].clear();
    }
    latestVolatilityByTicker.clear();
  }
  if (typeof window !== 'undefined' && window.sessionStorage) {
    try {
      if (!domain) window.sessionStorage.removeItem(SESSION_KEY);
      else persistMultiDomainCache();
    } catch {
      // ignore
    }
  }
}

export function getTcnMultiDomainCacheStats(): Record<TcnDomainKey, number> {
  return {
    volatility: domainCaches.volatility.size,
    risk_var: domainCaches.risk_var.size,
    regression: domainCaches.regression.size,
    correlation: domainCaches.correlation.size,
  };
}

export function getCachedTcnVolatilityForecast(
  ticker: string,
  bars?: PriceBar[],
  config?: Partial<TcnMultiDomainConfig>
): TcnVolatilityForecast | null {
  hydrateTcnMultiDomainCache();
  if (bars && bars.length >= 40) {
    const sig = buildVolatilityDataSignature(ticker, bars);
    const cfg = mergeConfig(config, bars.length - 1);
    const key = `vol|${sig}|${configKey(cfg)}`;
    const exact = lruGet<TcnVolatilityForecast>(domainCaches.volatility, key);
    if (exact) return { ...exact, cacheHit: true };
  }
  const latestKey = latestVolatilityByTicker.get(ticker.toUpperCase());
  if (latestKey) {
    const found = lruGet<TcnVolatilityForecast>(domainCaches.volatility, latestKey);
    if (found) return { ...found, cacheHit: true };
  }
  return null;
}

export function getCachedTcnRiskVarForecast(
  ticker: string,
  returns: number[],
  dates: string[],
  config?: Partial<TcnMultiDomainConfig>
): TcnRiskVarForecast | null {
  hydrateTcnMultiDomainCache();
  if (returns.length < 60) return null;
  const sig = hashSeriesFingerprint(returns, dates, `risk_${ticker.toUpperCase()}`);
  const cfg = mergeConfig(config, returns.length);
  const key = `risk|${sig}|${configKey(cfg)}`;
  const cached = lruGet<TcnRiskVarForecast>(domainCaches.risk_var, key);
  return cached ? { ...cached, cacheHit: true } : null;
}

export function getCachedTcnRegressionForecast(
  dependentTicker: string,
  independentTickers: string[],
  returnsByTicker: Record<string, number[]>,
  dates: string[],
  config?: Partial<TcnMultiDomainConfig>
): TcnMultiAssetRegressionResult | null {
  hydrateTcnMultiDomainCache();
  const y = returnsByTicker[dependentTicker] ?? [];
  if (y.length < 60) return null;
  const factors = independentTickers.slice(0, 6);
  const sig = buildRegressionSignature(dependentTicker, factors, returnsByTicker, dates);
  const cfg = mergeConfig(config, y.length);
  const key = `reg|${sig}|${configKey(cfg)}`;
  const cached = lruGet<TcnMultiAssetRegressionResult>(domainCaches.regression, key);
  return cached ? { ...cached, cacheHit: true } : null;
}

export function getCachedTcnCorrelationForecast(
  tickerA: string,
  tickerB: string,
  returnsA: number[],
  returnsB: number[],
  dates: string[],
  forecastDays = 30,
  config?: Partial<TcnMultiDomainConfig>
): TcnCorrelationForecast | null {
  hydrateTcnMultiDomainCache();
  const n = Math.min(returnsA.length, returnsB.length);
  if (n < 60) return null;
  const sig = buildCorrelationSignature(tickerA, tickerB, returnsA, returnsB, dates, forecastDays);
  const cfg = mergeConfig(config, n);
  const key = `corr|${sig}|${configKey(cfg)}`;
  const cached = lruGet<TcnCorrelationForecast>(domainCaches.correlation, key);
  return cached ? { ...cached, cacheHit: true } : null;
}

// ---------------------------------------------------------------------------
// 4. GENERALIZED MULTI-CHANNEL, MULTI-OUTPUT TCN-ATTENTION BACKBONE
// ---------------------------------------------------------------------------

interface ParamBlock {
  w: Float64Array;
  g: Float64Array;
  m: Float64Array;
  v: Float64Array;
}

function makeParam(size: number, scale: number, rng: () => number): ParamBlock {
  const w = new Float64Array(size);
  if (scale > 0) {
    for (let i = 0; i < size; i++) w[i] = (rng() * 2 - 1) * scale;
  }
  return {
    w,
    g: new Float64Array(size),
    m: new Float64Array(size),
    v: new Float64Array(size),
  };
}

interface MultiHeadTcnNet {
  cIn: number;
  cOut: number;
  d: number;
  k: number;
  numHeads: number;
  headDim: number;
  inW: ParamBlock; // [cIn * d]
  inB: ParamBlock; // [d]
  blocks: { dilation: number; convW: ParamBlock; convB: ParamBlock }[];
  wq: ParamBlock;
  wk: ParamBlock;
  wv: ParamBlock;
  wo: ParamBlock;
  bo: ParamBlock;
  outW: ParamBlock; // [cOut * d]
  outB: ParamBlock; // [cOut]
  skipW: ParamBlock; // [cOut] skip from x[(L-1)*cIn + 0]
  allParams: ParamBlock[];
}

function createMultiHeadNet(
  cIn: number,
  cOut: number,
  cfg: TcnMultiDomainConfig,
  seed: number,
  initSkip = 0.2
): MultiHeadTcnNet {
  const rng = createRng(seed);
  const d = cfg.hiddenChannels;
  const k = cfg.kernelSize;
  const numHeads = cfg.numHeads;
  const headDim = d / numHeads;

  const inW = makeParam(cIn * d, Math.sqrt(2 / (cIn + d)), rng);
  const inB = makeParam(d, 0, rng);

  const blocks = cfg.dilations.map((dilation) => ({
    dilation,
    convW: makeParam(k * d * d, Math.sqrt(1.5 / (k * d + d)), rng),
    convB: makeParam(d, 0, rng),
  }));

  const attnScale = Math.sqrt(1.5 / (d + d));
  const wq = makeParam(d * d, attnScale, rng);
  const wk = makeParam(d * d, attnScale, rng);
  const wv = makeParam(d * d, attnScale, rng);
  const wo = makeParam(d * d, attnScale * 0.5, rng);
  const bo = makeParam(d, 0, rng);

  const outW = makeParam(cOut * d, Math.sqrt(1 / d) * 0.4, rng);
  const outB = makeParam(cOut, 0, rng);
  const skipW = makeParam(cOut, 0, rng);
  for (let o = 0; o < cOut; o++) skipW.w[o] = initSkip;

  const allParams: ParamBlock[] = [inW, inB];
  for (const b of blocks) allParams.push(b.convW, b.convB);
  allParams.push(wq, wk, wv, wo, bo, outW, outB, skipW);

  return { cIn, cOut, d, k, numHeads, headDim, inW, inB, blocks, wq, wk, wv, wo, bo, outW, outB, skipW, allParams };
}

function countMultiHeadParams(net: MultiHeadTcnNet): number {
  let total = 0;
  for (const p of net.allParams) total += p.w.length;
  return total;
}

function cloneNetWeights(net: MultiHeadTcnNet): Float64Array[] {
  return net.allParams.map((p) => new Float64Array(p.w));
}

function restoreNetWeights(net: MultiHeadTcnNet, snap: Float64Array[]): void {
  for (let i = 0; i < net.allParams.length; i++) {
    net.allParams[i].w.set(snap[i]);
  }
}

interface MultiForwardWorkspace {
  x: Float64Array;
  L: number;
  h0: Float64Array;
  blockInputs: Float64Array[];
  convPreAct: Float64Array[];
  convRes: Float64Array[];
  lnNorms: Float64Array[];
  lnInvStd: Float64Array[];
  hFinal: Float64Array;
  qLast: Float64Array;
  K: Float64Array;
  V: Float64Array;
  attnWeights: Float64Array;
  context: Float64Array;
  attnRes: Float64Array;
  finalLnNorm: Float64Array;
  finalLnInvStd: number;
  preds: Float64Array;
  gPreds: Float64Array;
  gFinalLnOut: Float64Array;
  gAttnRes: Float64Array;
  gContext: Float64Array;
  gQLast: Float64Array;
  gH: Float64Array;
  gK: Float64Array;
  gV: Float64Array;
  gAlpha: Float64Array;
  gPrevA: Float64Array;
  gPrevB: Float64Array;
  gRes: Float64Array;
}

function createMultiWorkspace(net: MultiHeadTcnNet, L: number): MultiForwardWorkspace {
  const { d, numHeads, cOut } = net;
  const numBlocks = net.blocks.length;
  const h0 = new Float64Array(L * d);
  const blockInputs: Float64Array[] = new Array(numBlocks);
  const convPreAct: Float64Array[] = new Array(numBlocks);
  const convRes: Float64Array[] = new Array(numBlocks);
  const lnNorms: Float64Array[] = new Array(numBlocks);
  const lnInvStd: Float64Array[] = new Array(numBlocks);

  for (let b = 0; b < numBlocks; b++) {
    blockInputs[b] = h0;
    convPreAct[b] = new Float64Array(L * d);
    convRes[b] = new Float64Array(L * d);
    lnNorms[b] = new Float64Array(L * d);
    lnInvStd[b] = new Float64Array(L);
  }

  return {
    x: new Float64Array(L * net.cIn),
    L,
    h0,
    blockInputs,
    convPreAct,
    convRes,
    lnNorms,
    lnInvStd,
    hFinal: numBlocks > 0 ? lnNorms[numBlocks - 1] : h0,
    qLast: new Float64Array(d),
    K: new Float64Array(L * d),
    V: new Float64Array(L * d),
    attnWeights: new Float64Array(numHeads * L),
    context: new Float64Array(d),
    attnRes: new Float64Array(d),
    finalLnNorm: new Float64Array(d),
    finalLnInvStd: 1,
    preds: new Float64Array(cOut),
    gPreds: new Float64Array(cOut),
    gFinalLnOut: new Float64Array(d),
    gAttnRes: new Float64Array(d),
    gContext: new Float64Array(d),
    gQLast: new Float64Array(d),
    gH: new Float64Array(L * d),
    gK: new Float64Array(L * d),
    gV: new Float64Array(L * d),
    gAlpha: new Float64Array(L),
    gPrevA: new Float64Array(L * d),
    gPrevB: new Float64Array(L * d),
    gRes: new Float64Array(L * d),
  };
}

const EPS_LN = 1e-5;

function silu(x: number): number {
  const xc = clamp(x, -20, 20);
  return xc / (1 + Math.exp(-xc));
}

function siluGrad(x: number): number {
  const xc = clamp(x, -20, 20);
  const sig = 1 / (1 + Math.exp(-xc));
  return sig * (1 + xc * (1 - sig));
}

function lnForward(src: Float64Array, srcOff: number, dst: Float64Array, dstOff: number, d: number): number {
  let mean = 0;
  for (let c = 0; c < d; c++) mean += src[srcOff + c];
  mean /= d;
  let varSum = 0;
  for (let c = 0; c < d; c++) {
    const diff = src[srcOff + c] - mean;
    varSum += diff * diff;
  }
  const invStd = 1 / Math.sqrt(varSum / d + EPS_LN);
  for (let c = 0; c < d; c++) dst[dstOff + c] = (src[srcOff + c] - mean) * invStd;
  return invStd;
}

function lnBackward(
  gOut: Float64Array,
  gOutOff: number,
  norm: Float64Array,
  normOff: number,
  invStd: number,
  d: number,
  gIn: Float64Array,
  gInOff: number
): void {
  let sumG = 0;
  let sumGN = 0;
  for (let c = 0; c < d; c++) {
    const g = gOut[gOutOff + c];
    sumG += g;
    sumGN += g * norm[normOff + c];
  }
  const invD = 1 / d;
  for (let c = 0; c < d; c++) {
    const g = gOut[gOutOff + c];
    const nVal = norm[normOff + c];
    gIn[gInOff + c] += invStd * (g - sumG * invD - nVal * sumGN * invD);
  }
}

function forwardMulti(net: MultiHeadTcnNet, x: Float64Array, L: number, ws: MultiForwardWorkspace): Float64Array {
  const { cIn, cOut, d, k, numHeads, headDim } = net;
  ws.x = x;
  const h0 = ws.h0;
  const inW = net.inW.w;
  const inB = net.inB.w;

  // 1. Generalized multi-channel input projection
  for (let t = 0; t < L; t++) {
    const xBase = t * cIn;
    const hBase = t * d;
    for (let c = 0; c < d; c++) {
      let sum = inB[c];
      for (let ch = 0; ch < cIn; ch++) {
        sum += x[xBase + ch] * inW[ch * d + c];
      }
      h0[hBase + c] = sum;
    }
  }

  // 2. Dilated causal TCN residual blocks
  let curr = h0;
  for (let b = 0; b < net.blocks.length; b++) {
    const blk = net.blocks[b];
    const dil = blk.dilation;
    const convW = blk.convW.w;
    const convB = blk.convB.w;
    ws.blockInputs[b] = curr;
    const pre = ws.convPreAct[b];
    const res = ws.convRes[b];
    const norm = ws.lnNorms[b];
    const invStd = ws.lnInvStd[b];

    for (let t = 0; t < L; t++) {
      const outBase = t * d;
      for (let co = 0; co < d; co++) {
        let sum = convB[co];
        for (let j = 0; j < k; j++) {
          const srcT = t - j * dil;
          if (srcT >= 0) {
            const srcBase = srcT * d;
            const wBase = j * d * d + co;
            for (let ci = 0; ci < d; ci++) {
              sum += curr[srcBase + ci] * convW[wBase + ci * d];
            }
          }
        }
        pre[outBase + co] = sum;
        res[outBase + co] = curr[outBase + co] + silu(sum);
      }
      invStd[t] = lnForward(res, outBase, norm, outBase, d);
    }
    curr = norm;
  }

  ws.hFinal = curr;
  const lastBase = (L - 1) * d;

  // 3. Causal Multi-Head Self-Attention (query at latest step L-1 attending causally to 0..L-1)
  const qLast = ws.qLast;
  const wq = net.wq.w;
  const wk = net.wk.w;
  const wv = net.wv.w;
  for (let co = 0; co < d; co++) {
    let sum = 0;
    for (let ci = 0; ci < d; ci++) sum += curr[lastBase + ci] * wq[ci * d + co];
    qLast[co] = sum;
  }

  const K = ws.K;
  const V = ws.V;
  for (let t = 0; t < L; t++) {
    const hBase = t * d;
    for (let co = 0; co < d; co++) {
      let sK = 0;
      let sV = 0;
      for (let ci = 0; ci < d; ci++) {
        const hv = curr[hBase + ci];
        const wi = ci * d + co;
        sK += hv * wk[wi];
        sV += hv * wv[wi];
      }
      K[hBase + co] = sK;
      V[hBase + co] = sV;
    }
  }

  const attnWeights = ws.attnWeights;
  const context = ws.context;
  const invSqrtDk = 1 / Math.sqrt(headDim);

  for (let h = 0; h < numHeads; h++) {
    const hOff = h * headDim;
    const wOff = h * L;
    let maxScore = -Infinity;
    for (let t = 0; t < L; t++) {
      let dot = 0;
      const kBase = t * d + hOff;
      for (let m = 0; m < headDim; m++) dot += qLast[hOff + m] * K[kBase + m];
      const sc = dot * invSqrtDk;
      attnWeights[wOff + t] = sc;
      if (sc > maxScore) maxScore = sc;
    }
    let expSum = 0;
    for (let t = 0; t < L; t++) {
      const e = Math.exp(attnWeights[wOff + t] - maxScore);
      attnWeights[wOff + t] = e;
      expSum += e;
    }
    const invSum = expSum > 0 ? 1 / expSum : 1 / L;
    for (let t = 0; t < L; t++) attnWeights[wOff + t] *= invSum;
    for (let m = 0; m < headDim; m++) {
      let ctxVal = 0;
      for (let t = 0; t < L; t++) ctxVal += attnWeights[wOff + t] * V[t * d + hOff + m];
      context[hOff + m] = ctxVal;
    }
  }

  const attnRes = ws.attnRes;
  const wo = net.wo.w;
  const bo = net.bo.w;
  for (let co = 0; co < d; co++) {
    let sum = bo[co];
    for (let ci = 0; ci < d; ci++) sum += context[ci] * wo[ci * d + co];
    attnRes[co] = curr[lastBase + co] + sum;
  }

  ws.finalLnInvStd = lnForward(attnRes, 0, ws.finalLnNorm, 0, d);

  // 4. Multi-Output Head Readout + Persistence Skip on channel 0
  const lastX0 = x[(L - 1) * cIn];
  const outW = net.outW.w;
  const outB = net.outB.w;
  const skipW = net.skipW.w;
  for (let o = 0; o < cOut; o++) {
    let pred = outB[o] + skipW[o] * lastX0;
    const oOff = o * d;
    for (let c = 0; c < d; c++) pred += ws.finalLnNorm[c] * outW[oOff + c];
    ws.preds[o] = pred;
  }
  return ws.preds;
}

function backwardMulti(net: MultiHeadTcnNet, ws: MultiForwardWorkspace, gPreds: Float64Array): void {
  const { cIn, cOut, d, k, numHeads, headDim } = net;
  const { L, x, hFinal, qLast, K, V, attnWeights, context, finalLnNorm, finalLnInvStd } = ws;
  const lastBase = (L - 1) * d;
  const lastX0 = x[(L - 1) * cIn];

  const gFinalLnOut = ws.gFinalLnOut;
  gFinalLnOut.fill(0);
  const outW = net.outW.w;
  const outWG = net.outW.g;
  const outBG = net.outB.g;
  const skipWG = net.skipW.g;

  for (let o = 0; o < cOut; o++) {
    const dp = gPreds[o];
    if (dp === 0) continue;
    outBG[o] += dp;
    skipWG[o] += dp * lastX0;
    const oOff = o * d;
    for (let c = 0; c < d; c++) {
      outWG[oOff + c] += dp * finalLnNorm[c];
      gFinalLnOut[c] += dp * outW[oOff + c];
    }
  }

  const gAttnRes = ws.gAttnRes;
  gAttnRes.fill(0);
  lnBackward(gFinalLnOut, 0, finalLnNorm, 0, finalLnInvStd, d, gAttnRes, 0);

  const gH = ws.gH;
  gH.fill(0);
  for (let c = 0; c < d; c++) {
    gH[lastBase + c] += gAttnRes[c];
    net.bo.g[c] += gAttnRes[c];
  }

  const gContext = ws.gContext;
  const woW = net.wo.w;
  const woG = net.wo.g;
  for (let ci = 0; ci < d; ci++) {
    let sum = 0;
    const cVal = context[ci];
    const rowOff = ci * d;
    for (let co = 0; co < d; co++) {
      const go = gAttnRes[co];
      woG[rowOff + co] += cVal * go;
      sum += go * woW[rowOff + co];
    }
    gContext[ci] = sum;
  }

  const gQLast = ws.gQLast;
  const gK = ws.gK;
  const gV = ws.gV;
  const gAlpha = ws.gAlpha;
  gQLast.fill(0);
  gK.fill(0);
  gV.fill(0);
  const invSqrtDk = 1 / Math.sqrt(headDim);

  for (let h = 0; h < numHeads; h++) {
    const hOff = h * headDim;
    const wOff = h * L;
    for (let t = 0; t < L; t++) {
      const a = attnWeights[wOff + t];
      const vBase = t * d + hOff;
      let dot = 0;
      for (let m = 0; m < headDim; m++) {
        const gc = gContext[hOff + m];
        dot += gc * V[vBase + m];
        gV[vBase + m] += a * gc;
      }
      gAlpha[t] = dot;
    }
    let sumAG = 0;
    for (let t = 0; t < L; t++) sumAG += attnWeights[wOff + t] * gAlpha[t];
    for (let t = 0; t < L; t++) {
      const gScore = attnWeights[wOff + t] * (gAlpha[t] - sumAG) * invSqrtDk;
      const kBase = t * d + hOff;
      for (let m = 0; m < headDim; m++) {
        gQLast[hOff + m] += gScore * K[kBase + m];
        gK[kBase + m] += gScore * qLast[hOff + m];
      }
    }
  }

  const wqW = net.wq.w;
  const wqG = net.wq.g;
  for (let ci = 0; ci < d; ci++) {
    const hLast = hFinal[lastBase + ci];
    const rowOff = ci * d;
    let sumQ = 0;
    for (let co = 0; co < d; co++) {
      const gq = gQLast[co];
      wqG[rowOff + co] += hLast * gq;
      sumQ += gq * wqW[rowOff + co];
    }
    gH[lastBase + ci] += sumQ;
  }

  const wkW = net.wk.w;
  const wkG = net.wk.g;
  const wvW = net.wv.w;
  const wvG = net.wv.g;
  for (let t = 0; t < L; t++) {
    const hBase = t * d;
    for (let ci = 0; ci < d; ci++) {
      const hv = hFinal[hBase + ci];
      const rowOff = ci * d;
      let sumKV = 0;
      for (let co = 0; co < d; co++) {
        const gk = gK[hBase + co];
        const gv = gV[hBase + co];
        wkG[rowOff + co] += hv * gk;
        wvG[rowOff + co] += hv * gv;
        sumKV += gk * wkW[rowOff + co] + gv * wvW[rowOff + co];
      }
      gH[hBase + ci] += sumKV;
    }
  }

  // Dilated TCN stack backward
  let gCurr = gH;
  const gRes = ws.gRes;
  let useA = true;
  for (let b = net.blocks.length - 1; b >= 0; b--) {
    const blk = net.blocks[b];
    const dil = blk.dilation;
    const inp = ws.blockInputs[b];
    const pre = ws.convPreAct[b];
    const norm = ws.lnNorms[b];
    const invStd = ws.lnInvStd[b];
    const convW = blk.convW.w;
    const convG = blk.convW.g;
    const convBG = blk.convB.g;

    const gPrev = useA ? ws.gPrevA : ws.gPrevB;
    useA = !useA;
    gPrev.fill(0);
    gRes.fill(0);

    for (let t = 0; t < L; t++) {
      const base = t * d;
      lnBackward(gCurr, base, norm, base, invStd[t], d, gRes, base);
    }

    for (let t = 0; t < L; t++) {
      const outBase = t * d;
      for (let co = 0; co < d; co++) {
        const gr = gRes[outBase + co];
        gPrev[outBase + co] += gr;
        const gPre = gr * siluGrad(pre[outBase + co]);
        convBG[co] += gPre;
        for (let j = 0; j < k; j++) {
          const srcT = t - j * dil;
          if (srcT >= 0) {
            const srcBase = srcT * d;
            const wBase = j * d * d + co;
            for (let ci = 0; ci < d; ci++) {
              const wIdx = wBase + ci * d;
              convG[wIdx] += inp[srcBase + ci] * gPre;
              gPrev[srcBase + ci] += convW[wIdx] * gPre;
            }
          }
        }
      }
    }
    gCurr = gPrev;
  }

  // Generalized input projection backward
  const inWG = net.inW.g;
  const inBG = net.inB.g;
  for (let t = 0; t < L; t++) {
    const xBase = t * cIn;
    const hBase = t * d;
    for (let c = 0; c < d; c++) {
      const gh = gCurr[hBase + c];
      inBG[c] += gh;
      for (let ch = 0; ch < cIn; ch++) {
        inWG[ch * d + c] += x[xBase + ch] * gh;
      }
    }
  }
}

function adamStepMulti(net: MultiHeadTcnNet, batchSize: number, lr: number, step: number, weightDecay = 1e-4): void {
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const invB = 1 / Math.max(1, batchSize);

  let gradSq = 0;
  for (const p of net.allParams) {
    for (let i = 0; i < p.g.length; i++) {
      const g = p.g[i] * invB;
      gradSq += g * g;
    }
  }
  const gradNorm = Math.sqrt(gradSq);
  const clipScale = gradNorm > 2.0 ? 2.0 / gradNorm : 1.0;

  const bc1 = 1 - Math.pow(beta1, step);
  const bc2 = 1 - Math.pow(beta2, step);

  for (const p of net.allParams) {
    const isBias = p === net.inB || p === net.bo || p === net.outB || p === net.skipW || net.blocks.some((b) => b.convB === p);
    const w = p.w;
    const gArr = p.g;
    const mArr = p.m;
    const vArr = p.v;
    for (let i = 0; i < w.length; i++) {
      const g = gArr[i] * invB * clipScale + (isBias ? 0 : weightDecay * w[i]);
      const m = beta1 * mArr[i] + (1 - beta1) * g;
      const v = beta2 * vArr[i] + (1 - beta2) * g * g;
      mArr[i] = m;
      vArr[i] = v;
      w[i] -= (lr * (m / bc1)) / (Math.sqrt(v / bc2) + eps);
    }
  }
}

type HeadLossMode =
  | { type: 'huber'; delta: number }
  | { type: 'qlike_logvar'; targetStds: number[] }
  | { type: 'pinball'; quantiles: number[] }; // quantile tau in (0,1), e.g. 0.95 or 0.99 for positive loss VaR

function computeHeadLossAndGrad(
  preds: Float64Array,
  targets: Float64Array,
  gOut: Float64Array,
  mode: HeadLossMode
): number {
  const cOut = preds.length;
  let totalLoss = 0;
  let validCount = 0;

  for (let o = 0; o < cOut; o++) {
    const y = targets[o];
    if (!Number.isFinite(y)) {
      gOut[o] = 0;
      continue;
    }
    validCount++;
    const p = preds[o];
    const diff = p - y;

    if (mode.type === 'huber') {
      const d = mode.delta;
      const absD = Math.abs(diff);
      if (absD <= d) {
        totalLoss += 0.5 * diff * diff;
        gOut[o] = diff;
      } else {
        totalLoss += d * (absD - 0.5 * d);
        gOut[o] = diff >= 0 ? d : -d;
      }
    } else if (mode.type === 'qlike_logvar') {
      // Standardized log-variance QLIKE-aligned asymmetric loss:
      // In raw log-var space u_hat - u = std * (p - y).
      // QLIKE(v, v_hat) = u_hat + exp(u - u_hat) -> d/dp = std * (1 - exp(std*(y - p)))
      // Blended with light Huber anchor for numerical stability on extreme outlier shocks.
      const scale = Math.max(0.15, Math.min(2.0, mode.targetStds[o] ?? 1.0));
      const rawErr = clamp((y - p) * scale, -2.5, 2.5);
      const expTerm = Math.exp(rawErr);
      const qlikeLoss = expTerm - rawErr - 1;
      const huberLoss = Math.abs(diff) <= 1.5 ? 0.5 * diff * diff : 1.5 * (Math.abs(diff) - 0.75);
      totalLoss += 0.7 * qlikeLoss + 0.3 * huberLoss;
      const qlikeGrad = scale * (1 - expTerm);
      const huberGrad = clamp(diff, -1.5, 1.5);
      gOut[o] = clamp(0.7 * qlikeGrad + 0.3 * huberGrad, -3.0, 3.0);
    } else {
      // Smoothed Pinball loss for upper-tail loss quantile tau (e.g. 0.95 or 0.99 for positive Loss = -r)
      // err = y - p: if y > p (under-predicted VaR / exceedance), penalty is tau * (y - p); else (1-tau)*(p - y)
      const tau = mode.quantiles[o] ?? 0.95;
      const err = y - p;
      const smoothEps = 0.08;
      if (err > smoothEps) {
        totalLoss += tau * err;
        gOut[o] = -tau;
      } else if (err < -smoothEps) {
        totalLoss += (1 - tau) * -err;
        gOut[o] = 1 - tau;
      } else {
        // Quadratic smoothing near 0
        const s = err / smoothEps; // [-1, 1]
        const slope = -tau * 0.5 * (1 + s) + (1 - tau) * 0.5 * (1 - s);
        totalLoss += 0.5 * smoothEps * (tau * (1 + s) * (1 + s) * 0.25 + (1 - tau) * (1 - s) * (1 - s) * 0.25);
        gOut[o] = slope;
      }
    }
  }

  const inv = validCount > 0 ? 1 / validCount : 0;
  for (let o = 0; o < cOut; o++) gOut[o] *= inv;
  return totalLoss * inv;
}

function averageAttentionWeights(ws: MultiForwardWorkspace, numHeads: number, L: number): number[] {
  const out = new Array<number>(L).fill(0);
  for (let h = 0; h < numHeads; h++) {
    const off = h * L;
    for (let t = 0; t < L; t++) out[t] += ws.attnWeights[off + t] / numHeads;
  }
  return out;
}

async function trainMultiHeadModel(params: {
  inputs: Float64Array[];
  targets: Float64Array[];
  cIn: number;
  cOut: number;
  config: TcnMultiDomainConfig;
  lossMode: HeadLossMode;
  initSkip?: number;
  yieldControl?: () => Promise<void>;
  isCancelled?: () => boolean;
}): Promise<{
  net: MultiHeadTcnNet;
  ws: MultiForwardWorkspace;
  splitIdx: number;
  bestWeights: Float64Array[];
} | null> {
  const { inputs, targets, cIn, cOut, config, lossMode, initSkip = 0.2, yieldControl, isCancelled } = params;
  const total = inputs.length;
  if (total < 16) return null;

  const splitIdx = clamp(Math.floor(total * config.trainSplit), 8, total - 6);
  const net = createMultiHeadNet(cIn, cOut, config, config.seed, initSkip);
  const ws = createMultiWorkspace(net, config.lookback);
  const shuffleRng = createRng(config.seed * 7 + 19);

  const order = new Int32Array(splitIdx);
  for (let i = 0; i < splitIdx; i++) order[i] = i;
  const batchSize = clamp(Math.floor(splitIdx / 8), 4, 16);

  let bestHoldout = Infinity;
  let bestWeights = cloneNetWeights(net);
  let step = 0;

  for (let epoch = 1; epoch <= config.epochs; epoch++) {
    if (isCancelled?.()) return null;

    // Fisher-Yates shuffle on training indices only
    for (let i = splitIdx - 1; i > 0; i--) {
      const j = Math.floor(shuffleRng() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }

    const cosine = 0.5 * (1 + Math.cos((Math.PI * (epoch - 1)) / Math.max(1, config.epochs - 1)));
    const lr = config.learningRate * (0.15 + 0.85 * cosine);

    for (let bStart = 0; bStart < splitIdx; bStart += batchSize) {
      const bEnd = Math.min(splitIdx, bStart + batchSize);
      for (const p of net.allParams) p.g.fill(0);
      for (let bi = bStart; bi < bEnd; bi++) {
        const idx = order[bi];
        const preds = forwardMulti(net, inputs[idx], config.lookback, ws);
        computeHeadLossAndGrad(preds, targets[idx], ws.gPreds, lossMode);
        backwardMulti(net, ws, ws.gPreds);
      }
      step++;
      adamStepMulti(net, bEnd - bStart, lr, step);
    }

    // Evaluate strictly out-of-sample holdout loss
    let holdLoss = 0;
    let holdN = 0;
    for (let idx = splitIdx; idx < total; idx++) {
      const preds = forwardMulti(net, inputs[idx], config.lookback, ws);
      const l = computeHeadLossAndGrad(preds, targets[idx], ws.gPreds, lossMode);
      if (Number.isFinite(l)) {
        holdLoss += l;
        holdN++;
      }
    }
    const meanHold = holdN > 0 ? holdLoss / holdN : Infinity;
    if (meanHold < bestHoldout) {
      bestHoldout = meanHold;
      bestWeights = cloneNetWeights(net);
    }

    if (yieldControl && epoch % 8 === 0) {
      await yieldControl();
    }
  }

  restoreNetWeights(net, bestWeights);
  return { net, ws, splitIdx, bestWeights };
}

function standardizeChannels(
  rawFeatures: number[][],
  trainEndRow: number
): { normed: number[][]; means: number[]; stds: number[] } {
  const n = rawFeatures.length;
  const cIn = rawFeatures[0]?.length ?? 0;
  const means = new Array<number>(cIn).fill(0);
  const stds = new Array<number>(cIn).fill(1);
  const statN = Math.max(2, Math.min(n, trainEndRow));

  for (let ch = 0; ch < cIn; ch++) {
    let s = 0;
    for (let t = 0; t < statN; t++) s += rawFeatures[t][ch];
    const m = s / statN;
    let sq = 0;
    for (let t = 0; t < statN; t++) {
      const d = rawFeatures[t][ch] - m;
      sq += d * d;
    }
    means[ch] = m;
    stds[ch] = Math.sqrt(sq / Math.max(1, statN - 1)) || 1e-4;
  }

  const normed: number[][] = new Array(n);
  for (let t = 0; t < n; t++) {
    const row = new Array<number>(cIn);
    for (let ch = 0; ch < cIn; ch++) {
      row[ch] = clamp((rawFeatures[t][ch] - means[ch]) / stds[ch], -6, 6);
    }
    normed[t] = row;
  }
  return { normed, means, stds };
}

/**
 * Appends 3 strictly causal forward-filtered HMM volatility regime probability channels
 * [2*P(Calm)-1, 2*P(Watchful)-1, 2*P(Crisis)-1] in bounded [-1, 1] scale AFTER z-score
 * standardization so low-occupancy training states never explode during out-of-sample stress.
 */
function appendHmmRegimeChannels(
  normed: number[][],
  hmm: CausalHmmResult | null
): number[][] {
  const n = normed.length;
  const out: number[][] = new Array(n);
  const posteriors = hmm?.filteredPosteriors;
  for (let t = 0; t < n; t++) {
    const base = normed[t];
    const p = posteriors?.[t];
    if (p && p.length >= 3) {
      out[t] = [...base, 2 * p[0] - 1, 2 * p[1] - 1, 2 * p[2] - 1];
    } else {
      out[t] = [...base, 0, 0, 0];
    }
  }
  return out;
}

function packWindow(normed: number[][], startIdx: number, lookback: number, cIn: number): Float64Array {
  const buf = new Float64Array(lookback * cIn);
  for (let t = 0; t < lookback; t++) {
    const row = normed[startIdx + t];
    const base = t * cIn;
    for (let ch = 0; ch < cIn; ch++) buf[base + ch] = row[ch];
  }
  return buf;
}

function extractTopShockDates(
  attnWeights: number[],
  dates: string[],
  metricValues: number[],
  metricLabel: string,
  topN = 3
): AttentionShockDate[] {
  const L = attnWeights.length;
  const n = dates.length;
  const start = n - L;
  const items: AttentionShockDate[] = [];
  for (let t = 0; t < L; t++) {
    const idx = start + t;
    if (idx < 0 || idx >= n) continue;
    items.push({
      date: dates[idx],
      lag: L - 1 - t,
      weight: attnWeights[t],
      metricLabel,
      metricValue: metricValues[idx] ?? 0,
    });
  }
  items.sort((a, b) => b.weight - a.weight);
  return items.slice(0, topN);
}

// ---------------------------------------------------------------------------
// 5. DOMAIN A: MULTI-HORIZON VOLATILITY MODEL (1d / 5d / 22d)
// ---------------------------------------------------------------------------

function buildVolatilityDataSignature(ticker: string, bars: PriceBar[]): string {
  const closes = bars.map((b) => b.close);
  const dates = bars.map((b) => b.date);
  return hashSeriesFingerprint(closes, dates, `vol_${ticker.toUpperCase()}`);
}

export async function runTcnVolatilityForecast(
  ticker: string,
  bars: PriceBar[],
  options?: MultiDomainRunOptions & { forecastDays?: number }
): Promise<TcnVolatilityForecast | null> {
  if (!bars || bars.length < 70) return null;
  hydrateTcnMultiDomainCache();

  const nRet = bars.length - 1;
  const cfg = mergeConfig(options?.config, nRet);
  const dataSignature = buildVolatilityDataSignature(ticker, bars);
  const cacheKey = `vol|${dataSignature}|${configKey(cfg)}`;
  const useCache = options?.useCache !== false;
  const forceRetrain = options?.forceRetrain === true;

  if (useCache && !forceRetrain) {
    const cached = lruGet<TcnVolatilityForecast>(domainCaches.volatility, cacheKey);
    if (cached) {
      latestVolatilityByTicker.set(ticker.toUpperCase(), cacheKey);
      return { ...cached, cacheHit: true };
    }
  }

  const t0 = typeof performance !== 'undefined' ? performance.now() : 0;

  // Build log returns & Garman-Klass daily variance proxy
  const returns: number[] = new Array(nRet);
  const retDates: string[] = new Array(nRet);
  const dailyVarProxy: number[] = new Array(nRet);

  for (let i = 1; i < bars.length; i++) {
    const prev = bars[i - 1];
    const curr = bars[i];
    const r = prev.close > 0 && curr.close > 0 ? Math.log(curr.close / prev.close) : 0;
    returns[i - 1] = r;
    retDates[i - 1] = curr.date;

    let v = r * r;
    if (curr.open > 0 && curr.high > 0 && curr.low > 0 && curr.close > 0 && curr.high >= curr.low) {
      const hl = Math.log(curr.high / curr.low);
      const co = Math.log(curr.close / curr.open);
      const gk = 0.5 * hl * hl - (2 * Math.LN2 - 1) * co * co;
      if (Number.isFinite(gk) && gk > 1e-9) {
        // Blend GK intraday variance with overnight squared return
        v = 0.65 * gk + 0.35 * (r * r);
      }
    }
    dailyVarProxy[i - 1] = Math.max(1e-8, v);
  }

  // Construct 6-channel feature matrix [logVar, r, |r|, negLev, logRv5, logRv22]
  const rawFeatures: number[][] = new Array(nRet);
  for (let t = 0; t < nRet; t++) {
    const r = returns[t];
    const absR = Math.abs(r);
    const negLev = r < 0 ? absR : 0;
    let s5 = 0;
    const c5 = Math.min(5, t + 1);
    for (let j = 0; j < c5; j++) s5 += dailyVarProxy[t - j];
    let s22 = 0;
    const c22 = Math.min(22, t + 1);
    for (let j = 0; j < c22; j++) s22 += dailyVarProxy[t - j];
    rawFeatures[t] = [
      Math.log(dailyVarProxy[t]),
      r,
      absR,
      negLev,
      Math.log(Math.max(1e-8, s5 / c5)),
      Math.log(Math.max(1e-8, s22 / c22)),
    ];
  }

  const L = cfg.lookback;
  const maxSampleEnd = nRet - 2; // need at least 1d forward target
  const numSamples = maxSampleEnd - L + 1;
  if (numSamples < 24) return null;

  const splitIdx = clamp(Math.floor(numSamples * cfg.trainSplit), 10, numSamples - 8);
  const trainEndRow = splitIdx + L;

  const { normed: baseNormed } = standardizeChannels(rawFeatures, trainEndRow);
  const hmm = fitAndFilterCausalHmm(
    buildCausalReturnVolFeatures(returns, dailyVarProxy),
    trainEndRow
  );
  const normed = appendHmmRegimeChannels(baseNormed, hmm);
  const cIn = 9; // 6 standardized vol features + 3 causal HMM regime probabilities

  // Target log forward average daily realized variance for h = 1, 5, 22
  const rawTargets: number[][] = new Array(numSamples);
  const sampleEndIdx: number[] = new Array(numSamples);
  for (let s = 0; s < numSamples; s++) {
    const endT = s + L - 1;
    sampleEndIdx[s] = endT;
    const row = [NaN, NaN, NaN];
    for (let hi = 0; hi < 3; hi++) {
      const h = MULTI_HORIZONS[hi];
      if (endT + h < nRet) {
        let sumSq = 0;
        for (let k = 1; k <= h; k++) {
          const rk = returns[endT + k];
          sumSq += rk * rk;
        }
        row[hi] = Math.log(Math.max(1e-8, sumSq / h));
      }
    }
    rawTargets[s] = row;
  }

  // Target standardization strictly on training slice
  const tMeans = [0, 0, 0];
  const tStds = [1, 1, 1];
  for (let hi = 0; hi < 3; hi++) {
    let sum = 0;
    let cnt = 0;
    for (let s = 0; s < splitIdx; s++) {
      const v = rawTargets[s][hi];
      if (Number.isFinite(v)) {
        sum += v;
        cnt++;
      }
    }
    const m = cnt > 0 ? sum / cnt : Math.log(1e-4);
    let sq = 0;
    for (let s = 0; s < splitIdx; s++) {
      const v = rawTargets[s][hi];
      if (Number.isFinite(v)) {
        const d = v - m;
        sq += d * d;
      }
    }
    tMeans[hi] = m;
    tStds[hi] = Math.sqrt(sq / Math.max(1, cnt - 1)) || 0.5;
  }

  const inputs: Float64Array[] = new Array(numSamples);
  const targets: Float64Array[] = new Array(numSamples);
  for (let s = 0; s < numSamples; s++) {
    inputs[s] = packWindow(normed, s, L, cIn);
    const tgt = new Float64Array(3);
    for (let hi = 0; hi < 3; hi++) {
      const v = rawTargets[s][hi];
      tgt[hi] = Number.isFinite(v) ? (v - tMeans[hi]) / tStds[hi] : NaN;
    }
    targets[s] = tgt;
  }

  const trained = await trainMultiHeadModel({
    inputs,
    targets,
    cIn,
    cOut: 3,
    config: cfg,
    lossMode: { type: 'qlike_logvar', targetStds: tStds },
    initSkip: 0.35,
    yieldControl: options?.yieldControl,
    isCancelled: options?.isCancelled,
  });

  if (!trained) return null;
  const { net, ws } = trained;

  // Evaluate strictly out-of-sample predictions across all horizons + fitted series
  const oosDailyVarianceByHorizon: Record<ForecastHorizon, number[]> = {
    1: new Array<number>(nRet).fill(NaN),
    5: new Array<number>(nRet).fill(NaN),
    22: new Array<number>(nRet).fill(NaN),
  };

  const fittedAnnualizedVol = new Array<number>(nRet);
  // Fill warmup period with sqrt(252 * rolling 20d var proxy)
  for (let t = 0; t < nRet; t++) {
    let s20 = 0;
    const c20 = Math.min(20, t + 1);
    for (let j = 0; j < c20; j++) s20 += dailyVarProxy[t - j];
    fittedAnnualizedVol[t] = Math.sqrt(252 * (s20 / c20));
  }

  const { states: hmmStates, probs: hmmProbs } = extractDominantHmmSeries(hmm?.filteredPosteriors);
  const volRegAcc = [0, 1, 2].map(() => ({
    count: 0,
    qlikeSum: 0,
    mseSum: 0,
    predVolSum: 0,
    realVolSum: 0,
  }));

  const holdResidLogVar: number[][] = [[], [], []];
  const holdQlikeList: number[][] = [[], [], []];
  const holdMseLogList: number[][] = [[], [], []];

  for (let s = 0; s < numSamples; s++) {
    const endT = sampleEndIdx[s];
    const preds = forwardMulti(net, inputs[s], L, ws);
    const logV1 = clamp(preds[0] * tStds[0] + tMeans[0], -14, -1.5);
    const dailyV1 = Math.exp(logV1);
    fittedAnnualizedVol[endT] = Math.sqrt(252 * dailyV1);

    if (s >= splitIdx) {
      for (let hi = 0; hi < 3; hi++) {
        const h = MULTI_HORIZONS[hi];
        const logVh = clamp(preds[hi] * tStds[hi] + tMeans[hi], -14, -1.5);
        const dailyVh = Math.exp(logVh);
        oosDailyVarianceByHorizon[h][endT] = dailyVh;

        const actualLogV = rawTargets[s][hi];
        if (Number.isFinite(actualLogV)) {
          const actualDailyV = Math.exp(actualLogV);
          const qVal = qlike(actualDailyV * h, dailyVh * h);
          const mVal = mseLogVariance(actualDailyV * h, dailyVh * h);
          holdResidLogVar[hi].push(actualLogV - logVh);
          holdQlikeList[hi].push(qVal);
          holdMseLogList[hi].push(mVal);

          if (hi === 0 && Number.isFinite(qVal)) {
            const st = hmmStates[endT] ?? 0;
            const bucket = volRegAcc[st];
            bucket.count++;
            bucket.qlikeSum += qVal;
            bucket.mseSum += Number.isFinite(mVal) ? mVal : 0;
            bucket.predVolSum += Math.sqrt(252 * dailyVh);
            bucket.realVolSum += Math.sqrt(252 * actualDailyV);
          }
        }
      }
    }
  }

  const regimePerformance: VolRegimePerfRow[] = ([0, 1, 2] as const).map((st) => {
    const b = volRegAcc[st];
    return {
      state: st,
      label: HMM_REGIME_TOKENS[st].label,
      count: b.count,
      meanQlike: b.count > 0 ? b.qlikeSum / b.count : null,
      meanMseLogVar: b.count > 0 ? b.mseSum / b.count : null,
      meanPredVol: b.count > 0 ? b.predVolSum / b.count : null,
      meanRealizedVol: b.count > 0 ? b.realVolSum / b.count : null,
    };
  });

  // Latest forward 1d / 5d / 22d inference
  const latestInput = packWindow(normed, nRet - L, L, cIn);
  const latestPreds = forwardMulti(net, latestInput, L, ws);
  const attentionWeights = averageAttentionWeights(ws, cfg.numHeads, L);

  const horizons = {} as Record<ForecastHorizon, VolHorizonForecast>;
  for (let hi = 0; hi < 3; hi++) {
    const h = MULTI_HORIZONS[hi];
    const logV = clamp(latestPreds[hi] * tStds[hi] + tMeans[hi], -14, -1.5);
    const dailyVariance = Math.exp(logV);
    const annualizedVol = Math.sqrt(252 * dailyVariance);
    const resids = holdResidLogVar[hi];
    const residRmse =
      resids.length > 1
        ? Math.sqrt(resids.reduce((acc, e) => acc + e * e, 0) / Math.max(1, resids.length - 1))
        : 0.45;
    const tCrit = studentT99(Math.max(2, resids.length - 1));
    const halfLog = clamp(tCrit * residRmse * 0.42, 0.08, 1.15);
    const lower99Vol = Math.max(0.02, Math.sqrt(252 * Math.exp(logV - halfLog)));
    const upper99Vol = Math.min(3.5, Math.sqrt(252 * Math.exp(logV + halfLog)));

    const qlArr = holdQlikeList[hi].filter(Number.isFinite);
    const mseArr = holdMseLogList[hi].filter(Number.isFinite);
    horizons[h] = {
      horizon: h,
      dailyVariance,
      annualizedVol,
      lower99Vol,
      upper99Vol,
      holdoutQlike: qlArr.length > 0 ? qlArr.reduce((a, b) => a + b, 0) / qlArr.length : NaN,
      holdoutMseLogVar: mseArr.length > 0 ? mseArr.reduce((a, b) => a + b, 0) / mseArr.length : NaN,
      holdoutCount: qlArr.length,
    };
  }

  // Update final bar conditional vol smoothly
  fittedAnnualizedVol[nRet - 1] = horizons[1].annualizedVol;

  // Build daily forward cone up to forecastDays (default 30) interpolating across (1d, 5d, 22d)
  const fcDays = clamp(options?.forecastDays ?? 30, 5, 90);
  const asOfDate = retDates[nRet - 1] ?? bars[bars.length - 1].date;
  const forecastCone: GarchForecastPoint[] = [];
  for (let day = 1; day <= fcDays; day++) {
    const est = interpolateTcnAnnualizedVol(horizons, day);
    const halfBand95 = Math.max(0.015, (est.upper99 - est.lower99) * 0.38);
    const halfBand68 = halfBand95 * 0.52;
    forecastCone.push({
      day,
      date: nextWeekdayIso(asOfDate, day),
      annualizedVol: est.vol,
      upper68: est.vol + halfBand68,
      lower68: Math.max(0.01, est.vol - halfBand68),
      upper95: est.vol + halfBand95,
      lower95: Math.max(0.01, est.vol - halfBand95),
    });
  }

  const topShockDates = extractTopShockDates(
    attentionWeights,
    retDates,
    returns.map((r) => r * 100),
    'Return %'
  );

  const t1 = typeof performance !== 'undefined' ? performance.now() : 0;

  const result: TcnVolatilityForecast = {
    ticker: ticker.toUpperCase(),
    cacheKey,
    dataSignature,
    asOfDate,
    horizons,
    oosDailyVarianceByHorizon,
    fittedAnnualizedVol,
    holdoutStartIdx: sampleEndIdx[splitIdx] ?? Math.floor(nRet * 0.76),
    forecastCone,
    attentionWeights,
    topShockDates,
    parameterCount: countMultiHeadParams(net),
    receptiveField: computeMultiDomainReceptiveField(cfg.dilations, cfg.kernelSize),
    trainingDurationMs: Math.max(1, Math.round(t1 - t0)),
    cacheHit: false,
    hmmRegime: hmm?.summary ?? null,
    hmmStates,
    hmmProbs,
    regimePerformance,
  };

  if (useCache) {
    lruSet(domainCaches.volatility, cacheKey, result);
    latestVolatilityByTicker.set(ticker.toUpperCase(), cacheKey);
    persistMultiDomainCache();
  }
  return result;
}

function interpolateTcnAnnualizedVol(
  horizons: Record<ForecastHorizon, VolHorizonForecast>,
  day: number
): { vol: number; lower99: number; upper99: number; extrapolated: boolean } {
  const v1 = horizons[1].annualizedVol ** 2;
  const v5 = horizons[5].annualizedVol ** 2;
  const v22 = horizons[22].annualizedVol ** 2;

  const lo1 = horizons[1].lower99Vol;
  const hi1 = horizons[1].upper99Vol;
  const lo5 = horizons[5].lower99Vol;
  const hi5 = horizons[5].upper99Vol;
  const lo22 = horizons[22].lower99Vol;
  const hi22 = horizons[22].upper99Vol;

  if (day <= 1) {
    return { vol: horizons[1].annualizedVol, lower99: lo1, upper99: hi1, extrapolated: false };
  }
  if (day <= 5) {
    const w = (day - 1) / 4;
    return {
      vol: Math.sqrt((1 - w) * v1 + w * v5),
      lower99: (1 - w) * lo1 + w * lo5,
      upper99: (1 - w) * hi1 + w * hi5,
      extrapolated: false,
    };
  }
  if (day <= 22) {
    const w = (day - 5) / 17;
    return {
      vol: Math.sqrt((1 - w) * v5 + w * v22),
      lower99: (1 - w) * lo5 + w * lo22,
      upper99: (1 - w) * hi5 + w * hi22,
      extrapolated: false,
    };
  }
  // Beyond 22d: flat variance at 22d anchor with gentle sqrt-time uncertainty expansion
  const extraScale = 1 + 0.04 * Math.sqrt(day - 22);
  const spread = (hi22 - lo22) * 0.5 * extraScale;
  return {
    vol: horizons[22].annualizedVol,
    lower99: Math.max(0.01, horizons[22].annualizedVol - spread),
    upper99: horizons[22].annualizedVol + spread,
    extrapolated: true,
  };
}

// ---------------------------------------------------------------------------
// 6. DOMAIN B: ANALYSIS RISK QUANTILE VaR & ES MODEL (1d / 5d / 22d)
// ---------------------------------------------------------------------------

export async function runTcnRiskVarForecast(
  ticker: string,
  returns: number[],
  dates: string[],
  options?: MultiDomainRunOptions
): Promise<TcnRiskVarForecast | null> {
  const n = returns.length;
  if (n < 70) return null;
  hydrateTcnMultiDomainCache();

  const cfg = mergeConfig(options?.config, n);
  const dataSignature = hashSeriesFingerprint(returns, dates, `risk_${ticker.toUpperCase()}`);
  const cacheKey = `risk|${dataSignature}|${configKey(cfg)}`;
  const useCache = options?.useCache !== false;
  const forceRetrain = options?.forceRetrain === true;

  if (useCache && !forceRetrain) {
    const cached = lruGet<TcnRiskVarForecast>(domainCaches.risk_var, cacheKey);
    if (cached) return { ...cached, cacheHit: true };
  }

  const t0 = typeof performance !== 'undefined' ? performance.now() : 0;

  // Build 6-channel tail-risk feature matrix: [loss_t, |r_t|, negShock_t, vol5, vol22, skew22]
  const rawFeatures: number[][] = new Array(n);
  for (let t = 0; t < n; t++) {
    const r = returns[t];
    const absR = Math.abs(r);
    const negShock = r < 0 ? absR : 0;
    const c5 = Math.min(5, t + 1);
    let sq5 = 0;
    for (let j = 0; j < c5; j++) sq5 += returns[t - j] ** 2;
    const vol5 = Math.sqrt(sq5 / c5);

    const c22 = Math.min(22, t + 1);
    let m22 = 0;
    for (let j = 0; j < c22; j++) m22 += returns[t - j];
    m22 /= c22;
    let s2 = 0;
    let s3 = 0;
    for (let j = 0; j < c22; j++) {
      const d = returns[t - j] - m22;
      s2 += d * d;
      s3 += d * d * d;
    }
    const v22 = s2 / Math.max(1, c22 - 1);
    const vol22 = Math.sqrt(Math.max(1e-8, v22));
    const skew22 = v22 > 1e-10 ? clamp((s3 / c22) / Math.pow(v22, 1.5), -4, 4) : 0;

    rawFeatures[t] = [-r, absR, negShock, vol5, vol22, skew22];
  }

  const L = cfg.lookback;
  const maxEnd = n - 2;
  const numSamples = maxEnd - L + 1;
  if (numSamples < 24) return null;

  const splitIdx = clamp(Math.floor(numSamples * cfg.trainSplit), 10, numSamples - 8);
  const trainEndRow = splitIdx + L;
  const { normed: baseNormed } = standardizeChannels(rawFeatures, trainEndRow);
  const hmm = fitAndFilterCausalHmm(buildCausalReturnVolFeatures(returns), trainEndRow);
  const normed = appendHmmRegimeChannels(baseNormed, hmm);
  const cIn = 9; // 6 standardized tail-risk features + 3 causal HMM regime probabilities

  // 6 outputs: [VaR95_1d, VaR99_1d, VaR95_5d, VaR99_5d, VaR95_22d, VaR99_22d]
  // Target for each pair (h, alpha) is positive forward cumulative loss L_{t,h} = -sum_{k=1}^h r_{t+k}
  const rawLosses: number[][] = new Array(numSamples);
  const sampleEndIdx: number[] = new Array(numSamples);
  for (let s = 0; s < numSamples; s++) {
    const endT = s + L - 1;
    sampleEndIdx[s] = endT;
    const row = [NaN, NaN, NaN];
    for (let hi = 0; hi < 3; hi++) {
      const h = MULTI_HORIZONS[hi];
      if (endT + h < n) {
        let cumR = 0;
        for (let k = 1; k <= h; k++) cumR += returns[endT + k];
        row[hi] = -cumR;
      }
    }
    rawLosses[s] = row;
  }

  // Horizon loss scale strictly from training slice + empirical tail ES/VaR multipliers
  const hScale = [0.015, 0.035, 0.075];
  const empiricalVar95 = [0.02, 0.045, 0.09];
  const empiricalVar99 = [0.03, 0.065, 0.13];
  const esMult95 = [1.26, 1.26, 1.26];
  const esMult99 = [1.22, 1.22, 1.22];

  for (let hi = 0; hi < 3; hi++) {
    const tr: number[] = [];
    for (let s = 0; s < splitIdx; s++) {
      const v = rawLosses[s][hi];
      if (Number.isFinite(v)) tr.push(v);
    }
    if (tr.length >= 10) {
      const sorted = [...tr].sort((a, b) => a - b);
      const mean = tr.reduce((a, b) => a + b, 0) / tr.length;
      const sd = Math.sqrt(tr.reduce((a, v) => a + (v - mean) ** 2, 0) / Math.max(1, tr.length - 1)) || 0.01;
      hScale[hi] = sd;
      const q95 = Math.max(0.004, sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))]);
      const q99 = Math.max(q95 * 1.15, sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.99))]);
      empiricalVar95[hi] = q95;
      empiricalVar99[hi] = q99;

      const tail95 = sorted.filter((v) => v >= q95);
      const tail99 = sorted.filter((v) => v >= q99);
      const mean95 = tail95.length > 0 ? tail95.reduce((a, b) => a + b, 0) / tail95.length : q95 * 1.26;
      const mean99 = tail99.length > 0 ? tail99.reduce((a, b) => a + b, 0) / tail99.length : q99 * 1.22;
      esMult95[hi] = clamp(mean95 / Math.max(1e-4, q95), 1.12, 1.65);
      esMult99[hi] = clamp(mean99 / Math.max(1e-4, q99), 1.08, 1.55);
    }
  }

  const inputs: Float64Array[] = new Array(numSamples);
  const targets: Float64Array[] = new Array(numSamples);
  for (let s = 0; s < numSamples; s++) {
    inputs[s] = packWindow(normed, s, L, cIn);
    const tgt = new Float64Array(6);
    for (let hi = 0; hi < 3; hi++) {
      const lv = rawLosses[s][hi];
      const zL = Number.isFinite(lv) ? lv / hScale[hi] : NaN;
      tgt[hi * 2] = zL;
      tgt[hi * 2 + 1] = zL;
    }
    targets[s] = tgt;
  }

  const trained = await trainMultiHeadModel({
    inputs,
    targets,
    cIn,
    cOut: 6,
    config: { ...cfg, seed: cfg.seed + 101 },
    lossMode: { type: 'pinball', quantiles: [0.95, 0.99, 0.95, 0.99, 0.95, 0.99] },
    initSkip: 0.05,
    yieldControl: options?.yieldControl,
    isCancelled: options?.isCancelled,
  });

  if (!trained) return null;
  const { net, ws } = trained;

  const decodePair = (raw95: number, raw99: number, hi: number) => {
    const floor95 = empiricalVar95[hi] * 0.45;
    const cap95 = empiricalVar95[hi] * 3.2;
    const v95 = clamp(raw95 * hScale[hi], floor95, cap95);
    const v99 = clamp(Math.max(v95 * 1.12, raw99 * hScale[hi]), v95 * 1.12, empiricalVar99[hi] * 3.5);
    const e95 = Math.max(v95 * 1.08, v95 * esMult95[hi]);
    const e99 = Math.max(v99 * 1.06, e95, v99 * esMult99[hi]);
    return { var95: v95, var99: v99, es95: e95, es99: e99 };
  };

  // Strictly OOS holdout backtest evaluation
  const { states: hmmStates, probs: hmmProbs } = extractDominantHmmSeries(hmm?.filteredPosteriors);
  const oosVar95Series = new Array<number>(n).fill(NaN);
  const oosVar99Series = new Array<number>(n).fill(NaN);
  const oosDailyLossSeries = new Array<number>(n).fill(NaN);
  const exc95Count = [0, 0, 0];
  const exc99Count = [0, 0, 0];
  const evalCount = [0, 0, 0];

  const riskRegAcc = [0, 1, 2].map(() => ({
    count: 0,
    exc95: 0,
    exc99: 0,
    var99Sum: 0,
    lossSum: 0,
  }));

  const excIdx95: number[] = [];
  const excIdx99: number[] = [];
  const hitSeq99: number[] = [];

  for (let s = splitIdx; s < numSamples; s++) {
    const endT = sampleEndIdx[s];
    const preds = forwardMulti(net, inputs[s], L, ws);
    for (let hi = 0; hi < 3; hi++) {
      const actualLoss = rawLosses[s][hi];
      if (!Number.isFinite(actualLoss)) continue;
      const dec = decodePair(preds[hi * 2], preds[hi * 2 + 1], hi);
      evalCount[hi]++;
      const hit95 = actualLoss > dec.var95;
      const hit99 = actualLoss > dec.var99;
      if (hit95) exc95Count[hi]++;
      if (hit99) exc99Count[hi]++;

      if (hi === 0) {
        const retIdx = endT + 1;
        oosVar95Series[retIdx] = dec.var95;
        oosVar99Series[retIdx] = dec.var99;
        oosDailyLossSeries[retIdx] = actualLoss;
        if (hit95) excIdx95.push(retIdx);
        if (hit99) excIdx99.push(retIdx);
        hitSeq99.push(hit99 ? 1 : 0);

        // Strictly causal regime attribution at prediction origin endT (= retIdx - 1)
        const st = hmmStates[endT] ?? 0;
        const bucket = riskRegAcc[st];
        bucket.count++;
        if (hit95) bucket.exc95++;
        if (hit99) bucket.exc99++;
        bucket.var99Sum += dec.var99;
        bucket.lossSum += actualLoss;
      }
    }
  }

  const regimePerformance: RiskRegimePerfRow[] = ([0, 1, 2] as const).map((st) => {
    const b = riskRegAcc[st];
    return {
      state: st,
      label: HMM_REGIME_TOKENS[st].label,
      count: b.count,
      exceedance95Count: b.exc95,
      exceedance95RatePct: b.count > 0 ? (b.exc95 / b.count) * 100 : null,
      exceedance99Count: b.exc99,
      exceedance99RatePct: b.count > 0 ? (b.exc99 / b.count) * 100 : null,
      meanVar99: b.count > 0 ? b.var99Sum / b.count : null,
      meanRealizedLoss: b.count > 0 ? b.lossSum / b.count : null,
    };
  });

  // Forward latest multi-horizon VaR / ES predictions
  const latestInput = packWindow(normed, n - L, L, cIn);
  const latestPreds = forwardMulti(net, latestInput, L, ws);
  const attentionWeights = averageAttentionWeights(ws, cfg.numHeads, L);

  const horizons = {} as Record<ForecastHorizon, RiskHorizonForecast>;
  for (let hi = 0; hi < 3; hi++) {
    const h = MULTI_HORIZONS[hi];
    const dec = decodePair(latestPreds[hi * 2], latestPreds[hi * 2 + 1], hi);
    const cnt = evalCount[hi];
    horizons[h] = {
      horizon: h,
      var95: dec.var95,
      var99: dec.var99,
      es95: dec.es95,
      es99: dec.es99,
      tailRatio: dec.es99 / Math.max(1e-6, dec.var99),
      oosExceedanceRate95: cnt > 0 ? (exc95Count[hi] / cnt) * 100 : 0,
      oosExceedanceRate99: cnt > 0 ? (exc99Count[hi] / cnt) * 100 : 0,
      holdoutCount: cnt,
    };
  }

  // Kupiec POF & Christoffersen Independence on strictly OOS 1d holdout
  const nOos = Math.max(1, evalCount[0]);
  const calcKupiec = (x: number, p: number) => {
    if (x === 0) return { lr: Math.max(0, -2 * nOos * Math.log(1 - p)), pVal: chiSquarePValue(-2 * nOos * Math.log(1 - p), 1) };
    if (x >= nOos) return { lr: 25, pVal: 1e-6 };
    const pObs = x / nOos;
    const termNull = (nOos - x) * Math.log(1 - p) + x * Math.log(p);
    const termAlt = (nOos - x) * Math.log(1 - pObs) + x * Math.log(pObs);
    const lr = Math.max(0, -2 * (termNull - termAlt));
    return { lr, pVal: chiSquarePValue(lr, 1) };
  };
  const kup95 = calcKupiec(exc95Count[0], 0.05);
  const kup99 = calcKupiec(exc99Count[0], 0.01);

  let n00 = 0, n01 = 0, n10 = 0, n11 = 0;
  for (let i = 1; i < hitSeq99.length; i++) {
    const p = hitSeq99[i - 1];
    const c = hitSeq99[i];
    if (p === 0 && c === 0) n00++;
    else if (p === 0 && c === 1) n01++;
    else if (p === 1 && c === 0) n10++;
    else n11++;
  }
  const pi01 = n00 + n01 > 0 ? n01 / (n00 + n01) : 0.01;
  const pi11 = n10 + n11 > 0 ? n11 / (n10 + n11) : 0.01;
  const pi = exc99Count[0] / nOos;
  let lrInd = 0;
  if (n01 > 0 && n11 > 0 && pi > 0 && pi < 1 && pi01 > 0 && pi01 < 1 && pi11 > 0 && pi11 < 1) {
    const lnNull = (n00 + n10) * Math.log(1 - pi) + (n01 + n11) * Math.log(pi);
    const lnAlt = n00 * Math.log(1 - pi01) + n01 * Math.log(pi01) + n10 * Math.log(1 - pi11) + n11 * Math.log(pi11);
    lrInd = Math.max(0, -2 * (lnNull - lnAlt));
  }
  const pValInd = chiSquarePValue(lrInd, 1);

  // Scale OOS 99% hits to a 250-day Basel equivalent for fair traffic-light grading
  const scaledHits250 = (exc99Count[0] / nOos) * 250;
  const baselZone: 'Green' | 'Yellow' | 'Red' =
    scaledHits250 >= 10 ? 'Red' : scaledHits250 >= 5 ? 'Yellow' : 'Green';

  const backtestRow: VaRBacktestModelResult = {
    modelName: 'TCN-Attention (Quantile OOS)',
    key: 'tcn-attention',
    var95Pct: horizons[1].var95 * 100,
    var99Pct: horizons[1].var99 * 100,
    es99Pct: horizons[1].es99 * 100,
    exceedances95: exc95Count[0],
    exceedances99: exc99Count[0],
    expectedExceedances95: Number((nOos * 0.05).toFixed(1)),
    expectedExceedances99: Number((nOos * 0.01).toFixed(1)),
    kupiecLr95: kup95.lr,
    kupiecPVal95: kup95.pVal,
    kupiecVerdict95: kup95.pVal >= 0.05 ? 'Pass' : 'Fail',
    kupiecLr99: kup99.lr,
    kupiecPVal99: kup99.pVal,
    kupiecVerdict99: kup99.pVal >= 0.05 ? 'Pass' : 'Fail',
    christoffersenLr99: lrInd,
    christoffersenPVal99: pValInd,
    christoffersenVerdict99: pValInd >= 0.05 ? 'Pass (Independent)' : 'Fail (Clustered)',
    baselZone,
    exceedanceIdx95: excIdx95,
    exceedanceIdx99: excIdx99,
  };

  const retDates = dates.length === n + 1 ? dates.slice(1) : dates.slice(-n);
  const topShockDates = extractTopShockDates(
    attentionWeights,
    retDates,
    returns.map((r) => r * 100),
    'Return %'
  );

  const t1 = typeof performance !== 'undefined' ? performance.now() : 0;

  const result: TcnRiskVarForecast = {
    ticker: ticker.toUpperCase(),
    cacheKey,
    dataSignature,
    asOfDate: retDates[retDates.length - 1] ?? 'latest',
    horizons,
    backtestRow,
    oosVar95Series,
    oosVar99Series,
    oosDailyLossSeries,
    dates: retDates,
    holdoutStartIdx: (sampleEndIdx[splitIdx] ?? Math.floor(n * 0.76)) + 1,
    attentionWeights,
    topShockDates,
    parameterCount: countMultiHeadParams(net),
    receptiveField: computeMultiDomainReceptiveField(cfg.dilations, cfg.kernelSize),
    trainingDurationMs: Math.max(1, Math.round(t1 - t0)),
    cacheHit: false,
    hmmRegime: hmm?.summary ?? null,
    hmmStates,
    hmmProbs,
    regimePerformance,
  };

  if (useCache) {
    lruSet(domainCaches.risk_var, cacheKey, result);
    persistMultiDomainCache();
  }
  return result;
}

// ---------------------------------------------------------------------------
// 7. DOMAIN C: MULTI-ASSET NONLINEAR SEQUENCE REGRESSION (1d / 5d / 22d)
// ---------------------------------------------------------------------------

function buildRegressionSignature(
  dependentTicker: string,
  independentTickers: string[],
  returnsByTicker: Record<string, number[]>,
  dates: string[]
): string {
  const y = returnsByTicker[dependentTicker] ?? [];
  const x0 = returnsByTicker[independentTickers[0] ?? ''] ?? [];
  const combined = y.map((v, i) => v + (x0[i] ?? 0) * 0.37);
  return hashSeriesFingerprint(
    combined,
    dates,
    `reg_${dependentTicker.toUpperCase()}_${independentTickers.map((t) => t.toUpperCase()).join('+')}`
  );
}

function solveSmallOls(X: number[][], y: number[]): number[] {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  if (n === 0 || p === 0) return [];
  const A: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const b = new Array<number>(p).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      b[j] += X[i][j] * y[i];
      for (let k = 0; k < p; k++) A[j][k] += X[i][j] * X[i][k];
    }
  }
  // Tiny ridge for numerical stability
  for (let j = 0; j < p; j++) A[j][j] += 1e-6;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < p; col++) {
    let piv = col;
    for (let r = col + 1; r < p; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    const diag = M[col][col];
    if (Math.abs(diag) < 1e-12) continue;
    for (let r = 0; r < p; r++) {
      if (r === col) continue;
      const f = M[r][col] / diag;
      for (let j = col; j <= p; j++) M[r][j] -= f * M[col][j];
    }
  }
  return M.map((row, i) => (Math.abs(row[i]) > 1e-12 ? row[p] / row[i] : 0));
}

export async function runTcnMultiAssetRegression(
  dependentTicker: string,
  independentTickers: string[],
  returnsByTicker: Record<string, number[]>,
  dates: string[],
  options?: MultiDomainRunOptions
): Promise<TcnMultiAssetRegressionResult | null> {
  const factors = independentTickers.slice(0, 6);
  const yRaw = returnsByTicker[dependentTicker] ?? [];
  const factorSeries = factors.map((f) => returnsByTicker[f] ?? []);
  const n = Math.min(yRaw.length, ...factorSeries.map((s) => s.length));
  if (n < 70 || factors.length === 0) return null;

  hydrateTcnMultiDomainCache();
  const cfg = mergeConfig(options?.config, n);
  const dataSignature = buildRegressionSignature(dependentTicker, factors, returnsByTicker, dates);
  const cacheKey = `reg|${dataSignature}|${configKey(cfg)}`;
  const useCache = options?.useCache !== false;
  const forceRetrain = options?.forceRetrain === true;

  if (useCache && !forceRetrain) {
    const cached = lruGet<TcnMultiAssetRegressionResult>(domainCaches.regression, cacheKey);
    if (cached) return { ...cached, cacheHit: true };
  }

  const t0 = typeof performance !== 'undefined' ? performance.now() : 0;

  // Align tail to common length n
  const y = yRaw.slice(yRaw.length - n);
  const X = factorSeries.map((s) => s.slice(s.length - n));
  const baseCIn = 2 + factors.length; // [Y_t, X1_t..Xk_t, crossDispersion_t]

  const rawFeatures: number[][] = new Array(n);
  for (let t = 0; t < n; t++) {
    const row = new Array<number>(baseCIn);
    row[0] = y[t];
    let meanX = 0;
    for (let j = 0; j < factors.length; j++) {
      row[1 + j] = X[j][t];
      meanX += X[j][t];
    }
    meanX /= factors.length;
    row[baseCIn - 1] = y[t] - meanX;
    rawFeatures[t] = row;
  }

  const L = cfg.lookback;
  const maxEnd = n - 2;
  const numSamples = maxEnd - L + 1;
  if (numSamples < 24) return null;

  const splitIdx = clamp(Math.floor(numSamples * cfg.trainSplit), 10, numSamples - 8);
  const trainEndRow = splitIdx + L;
  const { normed: baseNormed } = standardizeChannels(rawFeatures, trainEndRow);
  const hmm = fitAndFilterCausalHmm(buildCausalReturnVolFeatures(y), trainEndRow);
  const normed = appendHmmRegimeChannels(baseNormed, hmm);
  const cIn = baseCIn + 3; // + 3 causal HMM volatility regime probabilities

  // Forward cumulative returns of Y over 1d, 5d, 22d
  const rawTargets: number[][] = new Array(numSamples);
  const sampleEndIdx: number[] = new Array(numSamples);
  for (let s = 0; s < numSamples; s++) {
    const endT = s + L - 1;
    sampleEndIdx[s] = endT;
    const row = [NaN, NaN, NaN];
    for (let hi = 0; hi < 3; hi++) {
      const h = MULTI_HORIZONS[hi];
      if (endT + h < n) {
        let sumR = 0;
        for (let k = 1; k <= h; k++) sumR += y[endT + k];
        row[hi] = sumR;
      }
    }
    rawTargets[s] = row;
  }

  const tMeans = [0, 0, 0];
  const tStds = [1, 1, 1];
  for (let hi = 0; hi < 3; hi++) {
    let sum = 0;
    let cnt = 0;
    for (let s = 0; s < splitIdx; s++) {
      const v = rawTargets[s][hi];
      if (Number.isFinite(v)) {
        sum += v;
        cnt++;
      }
    }
    const m = cnt > 0 ? sum / cnt : 0;
    let sq = 0;
    for (let s = 0; s < splitIdx; s++) {
      const v = rawTargets[s][hi];
      if (Number.isFinite(v)) sq += (v - m) ** 2;
    }
    tMeans[hi] = m;
    tStds[hi] = Math.sqrt(sq / Math.max(1, cnt - 1)) || 0.01;
  }

  const inputs: Float64Array[] = new Array(numSamples);
  const targets: Float64Array[] = new Array(numSamples);
  for (let s = 0; s < numSamples; s++) {
    inputs[s] = packWindow(normed, s, L, cIn);
    const tgt = new Float64Array(3);
    for (let hi = 0; hi < 3; hi++) {
      const v = rawTargets[s][hi];
      tgt[hi] = Number.isFinite(v) ? (v - tMeans[hi]) / tStds[hi] : NaN;
    }
    targets[s] = tgt;
  }

  const trained = await trainMultiHeadModel({
    inputs,
    targets,
    cIn,
    cOut: 3,
    config: { ...cfg, seed: cfg.seed + 203 },
    lossMode: { type: 'huber', delta: 1.5 },
    initSkip: 0.05,
    yieldControl: options?.yieldControl,
    isCancelled: options?.isCancelled,
  });

  if (!trained) return null;
  const { net, ws } = trained;

  // Lag-matched OLS baseline on exact same train/holdout split (features: [1, Y_t, X1_t..Xk_t, 5d_avg_Y, 5d_avg_X1..Xk])
  const buildOlsRow = (endT: number) => {
    const row: number[] = [1, y[endT]];
    for (let j = 0; j < factors.length; j++) row.push(X[j][endT]);
    let y5 = 0;
    const c5 = Math.min(5, endT + 1);
    for (let k = 0; k < c5; k++) y5 += y[endT - k];
    row.push(y5 / c5);
    return row;
  };

  const olsBetas: number[][] = [];
  for (let hi = 0; hi < 3; hi++) {
    const Xtr: number[][] = [];
    const yTr: number[] = [];
    for (let s = 0; s < splitIdx; s++) {
      const v = rawTargets[s][hi];
      if (Number.isFinite(v)) {
        Xtr.push(buildOlsRow(sampleEndIdx[s]));
        yTr.push(v);
      }
    }
    olsBetas.push(solveSmallOls(Xtr, yTr));
  }

  // Evaluate OOS TCN vs OOS OLS on holdout
  const retDates = dates.length >= n ? dates.slice(dates.length - n) : dates;
  const { states: hmmStates, probs: hmmProbs } = extractDominantHmmSeries(hmm?.filteredPosteriors);
  const oosHistory: RegressionOosPoint[] = [];
  const regRegAcc = [0, 1, 2].map(() => ({
    count: 0,
    sqErrTcn: 0,
    sqErrOls: 0,
    dirHits: 0,
    predSum: 0,
    actSum: 0,
  }));

  const holdActual: number[][] = [[], [], []];
  const holdTcnPred: number[][] = [[], [], []];
  const holdOlsPred: number[][] = [[], [], []];
  const holdDirHits = [0, 0, 0];

  for (let s = splitIdx; s < numSamples; s++) {
    const endT = sampleEndIdx[s];
    const preds = forwardMulti(net, inputs[s], L, ws);
    const olsRow = buildOlsRow(endT);
    for (let hi = 0; hi < 3; hi++) {
      const act = rawTargets[s][hi];
      if (!Number.isFinite(act)) continue;
      const tcnP = preds[hi] * tStds[hi] + tMeans[hi];
      const beta = olsBetas[hi];
      let olsP = 0;
      for (let j = 0; j < beta.length; j++) olsP += beta[j] * (olsRow[j] ?? 0);
      holdActual[hi].push(act);
      holdTcnPred[hi].push(tcnP);
      holdOlsPred[hi].push(olsP);
      const isDirHit = (tcnP >= 0 && act >= 0) || (tcnP < 0 && act < 0);
      if (isDirHit) holdDirHits[hi]++;

      if (hi === 0) {
        const st = hmmStates[endT] ?? 0;
        const prob = hmmProbs[endT] ?? 1 / 3;
        const targetIdx = Math.min(retDates.length - 1, endT + 1);
        oosHistory.push({
          date: retDates[targetIdx] ?? retDates[endT] ?? `t-${numSamples - s}`,
          actualReturn: act,
          predictedReturn: tcnP,
          olsPredictedReturn: olsP,
          hmmState: st,
          hmmProb: prob,
        });
        const bucket = regRegAcc[st];
        bucket.count++;
        bucket.sqErrTcn += (act - tcnP) ** 2;
        bucket.sqErrOls += (act - olsP) ** 2;
        if (isDirHit) bucket.dirHits++;
        bucket.predSum += tcnP;
        bucket.actSum += act;
      }
    }
  }

  const regimePerformance: RegRegimePerfRow[] = ([0, 1, 2] as const).map((st) => {
    const b = regRegAcc[st];
    return {
      state: st,
      label: HMM_REGIME_TOKENS[st].label,
      count: b.count,
      oosRmse: b.count > 0 ? Math.sqrt(b.sqErrTcn / b.count) : null,
      olsOosRmse: b.count > 0 ? Math.sqrt(b.sqErrOls / b.count) : null,
      hitRatePct: b.count > 0 ? (b.dirHits / b.count) * 100 : null,
      meanPredReturn: b.count > 0 ? b.predSum / b.count : null,
      meanActualReturn: b.count > 0 ? b.actSum / b.count : null,
    };
  });

  // Forward latest multi-horizon inference
  const latestInput = packWindow(normed, n - L, L, cIn);
  const latestPreds = forwardMulti(net, latestInput, L, ws);
  const attentionWeights = averageAttentionWeights(ws, cfg.numHeads, L);

  const horizons = {} as Record<ForecastHorizon, RegressionHorizonForecast>;
  for (let hi = 0; hi < 3; hi++) {
    const h = MULTI_HORIZONS[hi];
    const act = holdActual[hi];
    const tP = holdTcnPred[hi];
    const oP = holdOlsPred[hi];
    const cnt = act.length;
    const meanAct = cnt > 0 ? act.reduce((a, b) => a + b, 0) / cnt : 0;
    let ssTot = 0;
    let ssTcn = 0;
    let ssOls = 0;
    for (let i = 0; i < cnt; i++) {
      ssTot += (act[i] - meanAct) ** 2;
      ssTcn += (act[i] - tP[i]) ** 2;
      ssOls += (act[i] - oP[i]) ** 2;
    }
    const oosR2 = ssTot > 1e-12 ? clamp(1 - ssTcn / ssTot, -1, 0.99) : 0;
    const olsOosR2 = ssTot > 1e-12 ? clamp(1 - ssOls / ssTot, -1, 0.99) : 0;
    const oosRmse = cnt > 0 ? Math.sqrt(ssTcn / cnt) : tStds[hi];
    const olsOosRmse = cnt > 0 ? Math.sqrt(ssOls / cnt) : tStds[hi];
    const predRet = clamp(latestPreds[hi] * tStds[hi] + tMeans[hi], -0.45, 0.45);
    const tCrit = studentT99(Math.max(2, cnt - 1));
    const halfPi = Math.max(0.005, tCrit * oosRmse);

    horizons[h] = {
      horizon: h,
      predictedReturn: predRet,
      lower99: predRet - halfPi,
      upper99: predRet + halfPi,
      oosR2,
      olsOosR2,
      nonlinearLiftR2: oosR2 - olsOosR2,
      oosRmse,
      olsOosRmse,
      directionalAccuracyPct: cnt > 0 ? (holdDirHits[hi] / cnt) * 100 : 50,
      holdoutCount: cnt,
    };
  }

  // Seeded permutation feature importance across holdout windows (using 1d RMSE degradation)
  // Permutes each asset channel individually and the 3 HMM regime channels jointly as one group
  const permRng = createRng(cfg.seed + 777);
  const baseRmse1 = horizons[1].oosRmse;
  const featureGroups: { label: string; channels: number[]; isDependentLag: boolean; isHmmRegime?: boolean }[] = [
    { label: dependentTicker, channels: [0], isDependentLag: true },
    ...factors.map((f, idx) => ({ label: f, channels: [1 + idx], isDependentLag: false })),
    {
      label: 'HMM Regimes',
      channels: [baseCIn, baseCIn + 1, baseCIn + 2],
      isDependentLag: false,
      isHmmRegime: true,
    },
  ];
  const rawDeltas: number[] = [];
  const holdLen = numSamples - splitIdx;

  for (let g = 0; g < featureGroups.length; g++) {
    const grp = featureGroups[g];
    // Build permutation indices over holdout samples
    const perm = Array.from({ length: holdLen }, (_, i) => splitIdx + i);
    for (let i = perm.length - 1; i > 0; i--) {
      const j = Math.floor(permRng() * (i + 1));
      const tmp = perm[i];
      perm[i] = perm[j];
      perm[j] = tmp;
    }
    let sqErr = 0;
    let cnt = 0;
    const scratch = new Float64Array(L * cIn);
    for (let i = 0; i < holdLen; i++) {
      const s = splitIdx + i;
      const donor = perm[i];
      const act = rawTargets[s][0];
      if (!Number.isFinite(act)) continue;
      scratch.set(inputs[s]);
      const donorBuf = inputs[donor];
      for (let t = 0; t < L; t++) {
        for (const ch of grp.channels) {
          scratch[t * cIn + ch] = donorBuf[t * cIn + ch];
        }
      }
      const preds = forwardMulti(net, scratch, L, ws);
      const p1 = preds[0] * tStds[0] + tMeans[0];
      sqErr += (act - p1) ** 2;
      cnt++;
    }
    const permRmse = cnt > 0 ? Math.sqrt(sqErr / cnt) : baseRmse1;
    rawDeltas.push(Math.max(0, permRmse - baseRmse1));
  }

  const deltaSum = rawDeltas.reduce((a, b) => a + b, 0);
  const factorImportance: FactorSequenceImportance[] = featureGroups
    .map((grp, idx) => ({
      ticker: grp.label,
      isDependentLag: grp.isDependentLag,
      isHmmRegime: grp.isHmmRegime,
      importancePct: deltaSum > 1e-10 ? (rawDeltas[idx] / deltaSum) * 100 : 100 / featureGroups.length,
      rmseIncreaseBps: rawDeltas[idx] * 10000,
    }))
    .sort((a, b) => b.importancePct - a.importancePct);

  const topTransmissionDates = extractTopShockDates(
    attentionWeights,
    retDates,
    y.map((r) => r * 100),
    `${dependentTicker} %`
  );

  const t1 = typeof performance !== 'undefined' ? performance.now() : 0;

  const result: TcnMultiAssetRegressionResult = {
    dependentTicker: dependentTicker.toUpperCase(),
    independentTickers: factors.map((f) => f.toUpperCase()),
    cacheKey,
    dataSignature,
    asOfDate: retDates[retDates.length - 1] ?? 'latest',
    horizons,
    factorImportance,
    attentionWeights,
    topTransmissionDates,
    holdoutCount: holdLen,
    parameterCount: countMultiHeadParams(net),
    receptiveField: computeMultiDomainReceptiveField(cfg.dilations, cfg.kernelSize),
    trainingDurationMs: Math.max(1, Math.round(t1 - t0)),
    cacheHit: false,
    hmmRegime: hmm?.summary ?? null,
    hmmStates,
    hmmProbs,
    oosHistory,
    regimePerformance,
  };

  if (useCache) {
    lruSet(domainCaches.regression, cacheKey, result);
    persistMultiDomainCache();
  }
  return result;
}

// ---------------------------------------------------------------------------
// 8. DOMAIN D: DYNAMIC CORRELATION & SPILLOVER FORECASTING (1d / 5d / 22d)
// ---------------------------------------------------------------------------

function buildCorrelationSignature(
  tickerA: string,
  tickerB: string,
  returnsA: number[],
  returnsB: number[],
  dates: string[],
  forecastDays: number
): string {
  const n = Math.min(returnsA.length, returnsB.length);
  const prod = new Array<number>(n);
  for (let i = 0; i < n; i++) prod[i] = returnsA[i] * 0.6 + returnsB[i] * 0.4;
  return hashSeriesFingerprint(
    prod,
    dates.slice(-n),
    `corr_${tickerA.toUpperCase()}_${tickerB.toUpperCase()}_F${forecastDays}`
  );
}

export async function runTcnCorrelationForecast(
  tickerA: string,
  tickerB: string,
  returnsA: number[],
  returnsB: number[],
  dates: string[],
  forecastDays = 30,
  options?: MultiDomainRunOptions
): Promise<TcnCorrelationForecast | null> {
  const n = Math.min(returnsA.length, returnsB.length);
  if (n < 70) return null;
  hydrateTcnMultiDomainCache();

  const cfg = mergeConfig(options?.config, n);
  const dataSignature = buildCorrelationSignature(tickerA, tickerB, returnsA, returnsB, dates, forecastDays);
  const cacheKey = `corr|${dataSignature}|${configKey(cfg)}`;
  const useCache = options?.useCache !== false;
  const forceRetrain = options?.forceRetrain === true;

  if (useCache && !forceRetrain) {
    const cached = lruGet<TcnCorrelationForecast>(domainCaches.correlation, cacheKey);
    if (cached) return { ...cached, cacheHit: true };
  }

  const t0 = typeof performance !== 'undefined' ? performance.now() : 0;

  const rA = returnsA.slice(returnsA.length - n);
  const rB = returnsB.slice(returnsB.length - n);
  const alignedDates = dates.length >= n ? dates.slice(dates.length - n) : Array.from({ length: n }, (_, i) => `T-${n - i}`);

  // Compute causal EWMA dynamic correlation proxy (lambda = 0.92) + Fisher-z series
  const lambda = 0.92;
  const ewmaRho = new Array<number>(n);
  const fisherZ = new Array<number>(n);
  let varA = Math.max(1e-6, rA[0] * rA[0] + 1e-4);
  let varB = Math.max(1e-6, rB[0] * rB[0] + 1e-4);
  let covAB = rA[0] * rB[0];

  for (let t = 0; t < n; t++) {
    varA = lambda * varA + (1 - lambda) * rA[t] * rA[t];
    varB = lambda * varB + (1 - lambda) * rB[t] * rB[t];
    covAB = lambda * covAB + (1 - lambda) * rA[t] * rB[t];
    const denom = Math.sqrt(Math.max(1e-12, varA * varB));
    const rho = clamp(covAB / denom, -0.97, 0.97);
    ewmaRho[t] = rho;
    fisherZ[t] = safeAtanh(rho);
  }

  // 5-channel features: [fisherZ_t, zA*zB, |rA|+|rB|, coCrash, rolling20Z]
  const rawFeatures: number[][] = new Array(n);
  const jointShockMagnitude = new Array<number>(n);
  for (let t = 0; t < n; t++) {
    const sA = Math.sqrt(Math.max(1e-8, varA));
    const sB = Math.sqrt(Math.max(1e-8, varB));
    const crossShock = clamp((rA[t] / sA) * (rB[t] / sB), -6, 6);
    const jointVol = Math.abs(rA[t]) + Math.abs(rB[t]);
    const coCrash = rA[t] < 0 && rB[t] < 0 ? jointVol : 0;
    const c20 = Math.min(20, t + 1);
    let mA = 0, mB = 0;
    for (let j = 0; j < c20; j++) { mA += rA[t - j]; mB += rB[t - j]; }
    mA /= c20; mB /= c20;
    let cAB = 0, cAA = 0, cBB = 0;
    for (let j = 0; j < c20; j++) {
      const da = rA[t - j] - mA;
      const db = rB[t - j] - mB;
      cAB += da * db; cAA += da * da; cBB += db * db;
    }
    const r20 = cAA > 1e-12 && cBB > 1e-12 ? clamp(cAB / Math.sqrt(cAA * cBB), -0.96, 0.96) : ewmaRho[t];
    rawFeatures[t] = [fisherZ[t], crossShock, jointVol, coCrash, safeAtanh(r20)];
    jointShockMagnitude[t] = jointVol * 100;
  }

  const L = cfg.lookback;
  const maxEnd = n - 2;
  const numSamples = maxEnd - L + 1;
  if (numSamples < 24) return null;

  const splitIdx = clamp(Math.floor(numSamples * cfg.trainSplit), 10, numSamples - 8);
  const trainEndRow = splitIdx + L;
  const { normed: baseNormed } = standardizeChannels(rawFeatures, trainEndRow);
  const jointReturns = new Array<number>(n);
  const jointVarProxy = new Array<number>(n);
  for (let t = 0; t < n; t++) {
    jointReturns[t] = 0.5 * (rA[t] + rB[t]);
    jointVarProxy[t] = Math.max(1e-8, 0.5 * (rA[t] * rA[t] + rB[t] * rB[t]));
  }
  const hmm = fitAndFilterCausalHmm(
    buildCausalReturnVolFeatures(jointReturns, jointVarProxy),
    trainEndRow
  );
  const normed = appendHmmRegimeChannels(baseNormed, hmm);
  const cIn = 8; // 5 standardized correlation features + 3 causal joint HMM regime probabilities

  // Forward average Fisher-z correlation targets over 1d, 5d, 22d
  const rawTargets: number[][] = new Array(numSamples);
  const sampleEndIdx: number[] = new Array(numSamples);
  for (let s = 0; s < numSamples; s++) {
    const endT = s + L - 1;
    sampleEndIdx[s] = endT;
    const row = [NaN, NaN, NaN];
    for (let hi = 0; hi < 3; hi++) {
      const h = MULTI_HORIZONS[hi];
      if (endT + h < n) {
        let sumZ = 0;
        for (let k = 1; k <= h; k++) sumZ += fisherZ[endT + k];
        row[hi] = sumZ / h;
      }
    }
    rawTargets[s] = row;
  }

  const tMeans = [0, 0, 0];
  const tStds = [1, 1, 1];
  for (let hi = 0; hi < 3; hi++) {
    let sum = 0;
    let cnt = 0;
    for (let s = 0; s < splitIdx; s++) {
      const v = rawTargets[s][hi];
      if (Number.isFinite(v)) { sum += v; cnt++; }
    }
    const m = cnt > 0 ? sum / cnt : 0;
    let sq = 0;
    for (let s = 0; s < splitIdx; s++) {
      const v = rawTargets[s][hi];
      if (Number.isFinite(v)) sq += (v - m) ** 2;
    }
    tMeans[hi] = m;
    tStds[hi] = Math.sqrt(sq / Math.max(1, cnt - 1)) || 0.2;
  }

  const inputs: Float64Array[] = new Array(numSamples);
  const targets: Float64Array[] = new Array(numSamples);
  for (let s = 0; s < numSamples; s++) {
    inputs[s] = packWindow(normed, s, L, cIn);
    const tgt = new Float64Array(3);
    for (let hi = 0; hi < 3; hi++) {
      const v = rawTargets[s][hi];
      tgt[hi] = Number.isFinite(v) ? (v - tMeans[hi]) / tStds[hi] : NaN;
    }
    targets[s] = tgt;
  }

  const trained = await trainMultiHeadModel({
    inputs,
    targets,
    cIn,
    cOut: 3,
    config: { ...cfg, seed: cfg.seed + 307 },
    lossMode: { type: 'huber', delta: 1.5 },
    initSkip: 0.45,
    yieldControl: options?.yieldControl,
    isCancelled: options?.isCancelled,
  });

  if (!trained) return null;
  const { net, ws } = trained;

  const { states: hmmStates, probs: hmmProbs } = extractDominantHmmSeries(hmm?.filteredPosteriors);
  const holdoutStartIdx = sampleEndIdx[splitIdx] ?? Math.floor(n * 0.76);
  const dynamicCorrelations: {
    date: string;
    correlation: number;
    isOos: boolean;
    hmmState?: HmmStateIdx;
    hmmProb?: number;
  }[] = alignedDates.map((d, i) => ({
    date: d,
    correlation: ewmaRho[i],
    isOos: false,
    hmmState: hmmStates[i],
    hmmProb: hmmProbs[i],
  }));

  const corrRegAcc = [0, 1, 2].map(() => ({
    count: 0,
    absErrSum: 0,
    sqErrSum: 0,
    predSum: 0,
    actSum: 0,
  }));

  const holdErrZ: number[][] = [[], [], []];
  const holdErrRho: number[][] = [[], [], []];

  for (let s = 0; s < numSamples; s++) {
    const endT = sampleEndIdx[s];
    const preds = forwardMulti(net, inputs[s], L, ws);
    const z1 = clamp(preds[0] * tStds[0] + tMeans[0], -2.6, 2.6);
    const rho1 = Math.tanh(z1);
    dynamicCorrelations[endT] = {
      date: alignedDates[endT],
      correlation: rho1,
      isOos: s >= splitIdx,
      hmmState: hmmStates[endT],
      hmmProb: hmmProbs[endT],
    };

    if (s >= splitIdx) {
      for (let hi = 0; hi < 3; hi++) {
        const actZ = rawTargets[s][hi];
        if (!Number.isFinite(actZ)) continue;
        const predZ = clamp(preds[hi] * tStds[hi] + tMeans[hi], -2.6, 2.6);
        const actRho = Math.tanh(actZ);
        const predRho = Math.tanh(predZ);
        const errRho = actRho - predRho;
        holdErrZ[hi].push(actZ - predZ);
        holdErrRho[hi].push(errRho);

        if (hi === 0) {
          const st = hmmStates[endT] ?? 0;
          const bucket = corrRegAcc[st];
          bucket.count++;
          bucket.absErrSum += Math.abs(errRho);
          bucket.sqErrSum += errRho * errRho;
          bucket.predSum += predRho;
          bucket.actSum += actRho;
        }
      }
    }
  }

  const regimePerformance: CorrRegimePerfRow[] = ([0, 1, 2] as const).map((st) => {
    const b = corrRegAcc[st];
    return {
      state: st,
      label: HMM_REGIME_TOKENS[st].label,
      count: b.count,
      oosMae: b.count > 0 ? b.absErrSum / b.count : null,
      oosRmse: b.count > 0 ? Math.sqrt(b.sqErrSum / b.count) : null,
      meanPredCorr: b.count > 0 ? b.predSum / b.count : null,
      meanActualCorr: b.count > 0 ? b.actSum / b.count : null,
    };
  });

  // Forward latest multi-horizon inference
  const latestInput = packWindow(normed, n - L, L, cIn);
  const latestPreds = forwardMulti(net, latestInput, L, ws);
  const attentionWeights = averageAttentionWeights(ws, cfg.numHeads, L);
  const currentCorr = ewmaRho[n - 1];

  const horizons = {} as Record<ForecastHorizon, CorrelationHorizonForecast>;
  const zAnchors = [0, 0, 0];
  const zHalfBand = [0.2, 0.25, 0.35];

  for (let hi = 0; hi < 3; hi++) {
    const h = MULTI_HORIZONS[hi];
    const predZ = clamp(latestPreds[hi] * tStds[hi] + tMeans[hi], -2.6, 2.6);
    const predRho = Math.tanh(predZ);
    const errsZ = holdErrZ[hi];
    const errsRho = holdErrRho[hi];
    const cnt = errsRho.length;
    const rmseZ =
      cnt > 1 ? Math.sqrt(errsZ.reduce((a, e) => a + e * e, 0) / Math.max(1, cnt - 1)) : 0.22;
    const maeRho = cnt > 0 ? errsRho.reduce((a, e) => a + Math.abs(e), 0) / cnt : 0.08;
    const rmseRho = cnt > 0 ? Math.sqrt(errsRho.reduce((a, e) => a + e * e, 0) / cnt) : 0.1;
    const tCrit = studentT99(Math.max(2, cnt - 1));
    const hz = clamp(tCrit * rmseZ * 0.5, 0.06, 1.05);

    zAnchors[hi] = predZ;
    zHalfBand[hi] = hz;

    horizons[h] = {
      horizon: h,
      predictedCorr: predRho,
      lower99: Math.tanh(predZ - hz),
      upper99: Math.tanh(predZ + hz),
      deltaVsCurrent: predRho - currentCorr,
      oosMae: maeRho,
      oosRmse: rmseRho,
      holdoutCount: cnt,
    };
  }

  // Build strictly (-1, 1) forward cone interpolated in Fisher-z space
  const fcDays = clamp(forecastDays, 5, 60);
  const forecastCone: { day: number; correlation: number; upper99: number; lower99: number }[] = [];
  for (let day = 1; day <= fcDays; day++) {
    let zVal: number;
    let hz: number;
    if (day <= 1) {
      zVal = zAnchors[0];
      hz = zHalfBand[0];
    } else if (day <= 5) {
      const w = (day - 1) / 4;
      zVal = (1 - w) * zAnchors[0] + w * zAnchors[1];
      hz = (1 - w) * zHalfBand[0] + w * zHalfBand[1];
    } else if (day <= 22) {
      const w = (day - 5) / 17;
      zVal = (1 - w) * zAnchors[1] + w * zAnchors[2];
      hz = (1 - w) * zHalfBand[1] + w * zHalfBand[2];
    } else {
      zVal = zAnchors[2];
      hz = Math.min(1.25, zHalfBand[2] * (1 + 0.03 * Math.sqrt(day - 22)));
    }
    forecastCone.push({
      day,
      correlation: Math.tanh(zVal),
      upper99: Math.tanh(zVal + hz),
      lower99: Math.tanh(zVal - hz),
    });
  }

  // Spillover / correlation-jump detection
  let peakHorizon: ForecastHorizon = 5;
  let maxDelta = horizons[5].deltaVsCurrent;
  for (const h of MULTI_HORIZONS) {
    if (horizons[h].deltaVsCurrent > maxDelta) {
      maxDelta = horizons[h].deltaVsCurrent;
      peakHorizon = h;
    }
  }
  const peakCorr = horizons[peakHorizon].predictedCorr;
  let level: CorrelationSpilloverAlert['level'] = 'CALM';
  let summary = `Pairwise co-movement projected near current baseline (${currentCorr.toFixed(2)} → ${peakCorr.toFixed(2)} at ${peakHorizon}d).`;
  if (maxDelta >= 0.1 || (peakCorr >= 0.78 && maxDelta >= 0.04)) {
    level = 'ELEVATED';
    summary = `Spillover jump flagged: ${tickerA}/${tickerB} correlation projected +${maxDelta.toFixed(2)} to ${peakCorr.toFixed(2)} over ${peakHorizon}d, compressing diversification.`;
  } else if (maxDelta >= 0.045) {
    level = 'WATCH';
    summary = `Correlation tightening: projected +${maxDelta.toFixed(2)} rise to ${peakCorr.toFixed(2)} at ${peakHorizon}d horizon.`;
  }

  const topShockDates = extractTopShockDates(attentionWeights, alignedDates, jointShockMagnitude, 'Joint |r| %');
  const t1 = typeof performance !== 'undefined' ? performance.now() : 0;

  const result: TcnCorrelationForecast = {
    tickerA: tickerA.toUpperCase(),
    tickerB: tickerB.toUpperCase(),
    cacheKey,
    dataSignature,
    asOfDate: alignedDates[n - 1] ?? 'latest',
    currentCorr,
    horizons,
    dynamicCorrelations,
    forecastCone,
    spilloverAlert: {
      level,
      maxDelta,
      peakHorizon,
      peakCorr,
      summary,
    },
    attentionWeights,
    topShockDates,
    holdoutStartIdx,
    parameterCount: countMultiHeadParams(net),
    receptiveField: computeMultiDomainReceptiveField(cfg.dilations, cfg.kernelSize),
    trainingDurationMs: Math.max(1, Math.round(t1 - t0)),
    cacheHit: false,
    hmmRegime: hmm?.summary ?? null,
    hmmStates,
    hmmProbs,
    regimePerformance,
  };

  if (useCache) {
    lruSet(domainCaches.correlation, cacheKey, result);
    persistMultiDomainCache();
  }
  return result;
}

// ---------------------------------------------------------------------------
// 9. ZERO-RETRAIN HELPERS: F&O EXPECTED MOVE & EDGE / T8 SIGNAL CONFLUENCE
// ---------------------------------------------------------------------------

/**
 * Variance-time interpolation across cached (1d, 5d, 22d) TCN-Attention annualized
 * volatility anchors to compute an apples-to-apples expected move for any option DTE.
 */
export function interpolateTcnExpectedMove(
  volForecast: TcnVolatilityForecast,
  dte: number
): TcnExpectedMoveEstimate {
  const safeDte = Math.max(0.5, dte);
  const { vol, extrapolated } = interpolateTcnAnnualizedVol(volForecast.horizons, safeDte);
  const sigmaMovePct = vol * Math.sqrt(safeDte / 252) * 100;
  return {
    dte: safeDte,
    annualizedVol: vol,
    sigmaMovePct,
    straddleEqMovePct: 0.8 * sigmaMovePct,
    move99Pct: 2.576 * sigmaMovePct,
    extrapolated,
  };
}

export function classifyTcnMoveDislocation(
  impliedPct: number | null,
  tcnMovePct: number | null
): { verdict: TcnMoveDislocationVerdict | null; diffPct: number | null; ratio: number | null } {
  if (impliedPct === null || tcnMovePct === null || !Number.isFinite(impliedPct) || !Number.isFinite(tcnMovePct) || tcnMovePct <= 0) {
    return { verdict: null, diffPct: null, ratio: null };
  }
  const diffPct = impliedPct - tcnMovePct;
  const ratio = impliedPct / tcnMovePct;
  let verdict: TcnMoveDislocationVerdict = 'ALIGNED';
  if (ratio >= 1.12) verdict = 'RICH vs TCN';
  else if (ratio <= 0.88) verdict = 'CHEAP vs TCN';
  return { verdict, diffPct, ratio };
}

/**
 * Combines cached TCN-Attention Volatility (1d/5d/22d) and Price Decomposition
 * artifacts with live IV30 / RV20 and terminal signals without retraining.
 * Preserves QuantMetrics' direct RV (√252) vs IV (√365) basis convention.
 */
export function evaluateTcnConfluence(params: {
  iv30: number | null;
  rv20: number | null;
  volForecast: TcnVolatilityForecast | null;
  decompForecast: TcnDecompositionForecast | null;
  /** Optional terminal signal orientation: e.g. 'premium-harvest' | 'long-gamma' | 'directional' */
  primaryOrientation?: 'short-premium' | 'long-gamma' | 'neutral';
}): TcnConfluenceSummary {
  const { iv30, rv20, volForecast, decompForecast, primaryOrientation = 'neutral' } = params;

  const tcnRv1d = volForecast?.horizons[1].annualizedVol ?? null;
  const tcnRv5d = volForecast?.horizons[5].annualizedVol ?? null;
  const tcnRv22d = volForecast?.horizons[22].annualizedVol ?? null;
  const forwardVrp22 = iv30 !== null && tcnRv22d !== null ? iv30 - tcnRv22d : null;
  const tcnVolSlope = tcnRv5d !== null && tcnRv22d !== null ? tcnRv5d - tcnRv22d : null;

  let volRegimeRead: TcnConfluenceSummary['volRegimeRead'] = 'UNAVAILABLE';
  if (forwardVrp22 !== null) {
    if (forwardVrp22 >= 0.025) volRegimeRead = 'SHORT-PREMIUM FAVORED';
    else if (forwardVrp22 <= -0.02) volRegimeRead = 'LONG-GAMMA FAVORED';
    else volRegimeRead = 'NEUTRAL VOL';
  }

  let priceTiltPct: number | null = null;
  let pricePiAsymmetry: number | null = null;
  let cyclicalSnrDb: number | null = null;
  let cyclicalSnrRegime: string | null = null;
  let backtestSharpe30d: number | null = null;

  if (decompForecast) {
    priceTiltPct = decompForecast.deltaPct;
    const upSpan = Math.max(0, decompForecast.upper99 - decompForecast.lastPrice);
    const dnSpan = Math.max(0, decompForecast.lastPrice - decompForecast.lower99);
    const totalSpan = upSpan + dnSpan;
    pricePiAsymmetry = totalSpan > 1e-9 ? (upSpan - dnSpan) / totalSpan : 0;
    cyclicalSnrDb = decompForecast.signalToNoise?.db ?? null;
    cyclicalSnrRegime = decompForecast.signalToNoise?.regime ?? null;
    if (decompForecast.backtestSignals && decompForecast.backtestSignals.length >= 3) {
      backtestSharpe30d = computeHistoricalBacktest(decompForecast.backtestSignals, 0).summary.annualizedSharpe;
    }
  }

  const rationale: string[] = [];
  if (forwardVrp22 !== null && tcnRv22d !== null && iv30 !== null) {
    const pts = (forwardVrp22 * 100).toFixed(1);
    rationale.push(
      `IV30 (${(iv30 * 100).toFixed(1)}%) vs TCN 22d RV (${(tcnRv22d * 100).toFixed(1)}%) implies ${forwardVrp22 >= 0 ? '+' : ''}${pts} vol pts forward VRP (direct √365 vs √252 basis).`
    );
  }
  if (tcnVolSlope !== null && tcnRv5d !== null && tcnRv22d !== null) {
    rationale.push(
      `TCN term curve is ${tcnVolSlope > 0.01 ? 'in backwardation' : tcnVolSlope < -0.01 ? 'in contango' : 'flat'} (5d ${(tcnRv5d * 100).toFixed(1)}% vs 22d ${(tcnRv22d * 100).toFixed(1)}%).`
    );
  }
  if (rv20 !== null && tcnRv22d !== null) {
    const shift = (tcnRv22d - rv20) * 100;
    rationale.push(
      `TCN 22d forecast sits ${shift >= 0 ? '+' : ''}${shift.toFixed(1)} vol pts relative to trailing 20d realized (${(rv20 * 100).toFixed(1)}%).`
    );
  }
  if (priceTiltPct !== null) {
    rationale.push(
      `Cached TCN Decomposition t+1 tilt: ${priceTiltPct >= 0 ? '+' : ''}${priceTiltPct.toFixed(2)}%${cyclicalSnrRegime ? ` · Cyclical SNR ${cyclicalSnrRegime}` : ''}${backtestSharpe30d !== null ? ` · 30d Sharpe ${backtestSharpe30d.toFixed(2)}` : ''}.`
    );
  }

  let alignment: TcnConfluenceSummary['alignment'] = 'PARTIAL';
  if (volForecast) {
    if (primaryOrientation === 'short-premium') {
      alignment =
        volRegimeRead === 'SHORT-PREMIUM FAVORED'
          ? 'ALIGNED'
          : volRegimeRead === 'LONG-GAMMA FAVORED'
            ? 'DIVERGENT'
            : 'MIXED';
    } else if (primaryOrientation === 'long-gamma') {
      alignment =
        volRegimeRead === 'LONG-GAMMA FAVORED'
          ? 'ALIGNED'
          : volRegimeRead === 'SHORT-PREMIUM FAVORED'
            ? 'DIVERGENT'
            : 'MIXED';
    } else {
      alignment = volRegimeRead === 'UNAVAILABLE' ? 'PARTIAL' : 'ALIGNED';
    }
  }

  return {
    alignment,
    tcnRv1d,
    tcnRv5d,
    tcnRv22d,
    forwardVrp22,
    tcnVolSlope,
    volRegimeRead,
    priceTiltPct,
    pricePiAsymmetry,
    cyclicalSnrDb,
    cyclicalSnrRegime,
    backtestSharpe30d,
    rationale,
    hasVolArtifact: volForecast !== null,
    hasDecompArtifact: decompForecast !== null,
  };
}
