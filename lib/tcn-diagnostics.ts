// Pure TypeScript aggregation & comparative diagnostic builders for the
// Unified TCN-Attention Diagnostics Dashboard (Volatility, Risk VaR,
// Multi-Asset Regression, and Signal Confluence).
// Zero React/DOM dependencies — relative imports only for node --test compatibility.

import type { PriceBar } from './quant-engine';
import type { HistoricalBacktestResult, TcnDecompositionForecast } from './tcn-attention';
import {
  MULTI_HORIZONS,
  type AttentionShockDate,
  type ForecastHorizon,
  type TcnConfluenceSummary,
  type TcnDomainKey,
  type TcnMultiAssetRegressionResult,
  type TcnRiskVarForecast,
  type TcnVolatilityForecast,
} from './tcn-multi-domain';
import { HMM_REGIME_TOKENS, type HmmStateIdx } from './hmm-regime-overlay';

export type TcnModuleKey = 'volatility' | 'risk_var' | 'regression' | 'confluence';
export type TcnModuleStatus = 'idle' | 'cached' | 'training' | 'ready' | 'insufficient';

export interface TcnModuleStatuses {
  volatility: TcnModuleStatus;
  risk_var: TcnModuleStatus;
  regression: TcnModuleStatus;
  confluence: TcnModuleStatus;
}

export interface AlignedRegressionInputs {
  dependentTicker: string;
  independentTickers: string[];
  returnsByTicker: Record<string, number[]>;
  dates: string[];
}

export interface OosScorecardRow {
  module: TcnModuleKey;
  label: string;
  headSpec: string;
  status: TcnModuleStatus;
  primaryMetricLabel: string;
  primaryMetricValue: string;
  secondaryMetricLabel: string;
  secondaryMetricValue: string;
  verdict: string;
  verdictTone: 'positive' | 'gold' | 'negative' | 'neutral';
  oosSampleCount: number | null;
  targetView: 'modeling' | 'analysis' | 'regression' | 'volatility';
  targetParams?: Record<string, string>;
  targetLabel: string;
}

export interface HorizonMatrixRow {
  horizon: ForecastHorizon;
  label: string;
  volAnnualized: number | null;
  volLower99: number | null;
  volUpper99: number | null;
  volOosQlike: number | null;
  var95: number | null;
  var99: number | null;
  es99: number | null;
  tailRatio: number | null;
  regReturn: number | null;
  regLower99: number | null;
  regUpper99: number | null;
  regHitRatePct: number | null;
  regLiftR2: number | null;
  vrpNote: string;
}

export interface UnifiedLagPoint {
  /** Lag in sessions before forecast origin (1 = most recent session t-1, L = oldest) */
  lag: number;
  volWeight: number | null;
  riskWeight: number | null;
  regWeight: number | null;
  consensusWeight: number;
  /** Strictly causal dominant HMM regime at this lookback lag session */
  hmmState: HmmStateIdx | null;
  /** Posterior confidence of the dominant HMM regime at this lag session */
  hmmProb: number | null;
}

export interface CrossModuleRegimeRow {
  state: HmmStateIdx;
  label: 'Calm' | 'Watchful' | 'Crisis';
  volCount: number;
  volMeanQlike: number | null;
  volMeanPredVol: number | null;
  riskCount: number;
  riskExc95RatePct: number | null;
  riskExc99RatePct: number | null;
  regCount: number;
  regHitRatePct: number | null;
  regOosRmse: number | null;
  decompCount: number;
  decompHitRatePct: number | null;
  decompCumNetPct: number | null;
}

export interface JointShockDateRow {
  date: string;
  lag: number;
  moduleCount: number;
  modules: ('volatility' | 'risk_var' | 'regression')[];
  combinedWeight: number;
  volWeight: number | null;
  riskWeight: number | null;
  regWeight: number | null;
  metricSummary: string;
}

export interface UnifiedAttentionTimeline {
  maxLookback: number;
  lagCurves: UnifiedLagPoint[];
  jointShockDates: JointShockDateRow[];
  activeModules: ('volatility' | 'risk_var' | 'regression')[];
}

export interface ModuleTelemetryRow {
  module: 'volatility' | 'risk_var' | 'regression' | 'decomposition';
  label: string;
  status: TcnModuleStatus;
  cacheState: 'Cache Hit' | 'Fresh Fit' | 'Training' | 'Not Cached' | 'Insufficient';
  latencyMs: number | null;
  parameterCount: number | null;
  receptiveField: number | null;
  asOfDate: string | null;
  canRetrain: boolean;
}

export interface UnifiedCacheTelemetry {
  rows: ModuleTelemetryRow[];
  totalParameters: number;
  totalLatencyMs: number;
  cacheHitsCount: number;
  readyTrainableCount: number;
  multiDomainCounts: Record<TcnDomainKey, number>;
  multiDomainTotal: number;
  decompCachedCount: number;
  decompWeightTensors: number;
}

const PREFERRED_FACTORS = ['SPY', 'QQQ', 'MSFT', 'NVDA', 'AAPL', 'GOOGL', 'AMZN', 'JPM'];

/**
 * Selects up to 2 liquid benchmark factors from the already-loaded workspace bar cache
 * and aligns log-return series by exact session date intersection so the Multi-Asset
 * Regression head runs automatically inside Edge with zero extra API fetches.
 */
export function selectDefaultRegressionFactors(
  ticker: string,
  bars: PriceBar[],
  allStockBars: Record<string, PriceBar[]>
): AlignedRegressionInputs | null {
  const dep = ticker.trim().toUpperCase();
  if (!dep || !bars || bars.length < 70) return null;

  const candidates: string[] = [];
  for (const sym of PREFERRED_FACTORS) {
    if (sym === dep) continue;
    if ((allStockBars[sym]?.length ?? 0) >= 70) {
      candidates.push(sym);
      if (candidates.length === 2) break;
    }
  }
  if (candidates.length < 2) {
    for (const [sym, series] of Object.entries(allStockBars)) {
      const upper = sym.toUpperCase();
      if (upper === dep || candidates.includes(upper)) continue;
      if ((series?.length ?? 0) >= 70) {
        candidates.push(upper);
        if (candidates.length >= 2) break;
      }
    }
  }
  if (candidates.length === 0) return null;

  // Build date -> close maps for exact date intersection
  const depMap = new Map<string, number>();
  for (const b of bars) {
    if (b.close > 0 && Number.isFinite(b.close)) depMap.set(b.date, b.close);
  }

  const factorMaps = candidates.map((sym) => {
    const m = new Map<string, number>();
    const raw = allStockBars[sym] ?? [];
    for (const b of raw) {
      if (b.close > 0 && Number.isFinite(b.close)) m.set(b.date, b.close);
    }
    return { sym, map: m };
  });

  const commonDates: string[] = [];
  for (const b of bars) {
    if (!depMap.has(b.date)) continue;
    let allHave = true;
    for (const fm of factorMaps) {
      if (!fm.map.has(b.date)) {
        allHave = false;
        break;
      }
    }
    if (allHave) commonDates.push(b.date);
  }

  if (commonDates.length < 70) return null;

  const retDates: string[] = [];
  const returnsByTicker: Record<string, number[]> = { [dep]: [] };
  for (const sym of candidates) returnsByTicker[sym] = [];

  for (let i = 1; i < commonDates.length; i++) {
    const prevD = commonDates[i - 1];
    const currD = commonDates[i];
    const p0 = depMap.get(prevD)!;
    const p1 = depMap.get(currD)!;
    retDates.push(currD);
    returnsByTicker[dep].push(Math.log(p1 / p0));
    for (const fm of factorMaps) {
      const f0 = fm.map.get(prevD)!;
      const f1 = fm.map.get(currD)!;
      returnsByTicker[fm.sym].push(Math.log(f1 / f0));
    }
  }

  if (retDates.length < 65) return null;

  return {
    dependentTicker: dep,
    independentTickers: candidates,
    returnsByTicker,
    dates: retDates,
  };
}

function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  return `${(v * 100).toFixed(digits)}%`;
}

function fmtSignedPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`;
}

export function buildUnifiedOosScorecard(params: {
  volatility: TcnVolatilityForecast | null;
  riskVar: TcnRiskVarForecast | null;
  regression: TcnMultiAssetRegressionResult | null;
  confluence: TcnConfluenceSummary;
  statuses: TcnModuleStatuses;
}): OosScorecardRow[] {
  const { volatility, riskVar, regression, confluence, statuses } = params;

  // 1. Volatility QLIKE row
  let volVerdict = statuses.volatility === 'training' ? 'Training' : statuses.volatility === 'insufficient' ? 'Insufficient' : 'Pending';
  let volTone: OosScorecardRow['verdictTone'] = 'neutral';
  if (volatility) {
    const q22 = volatility.horizons[22].holdoutQlike;
    if (q22 <= 0.35) {
      volVerdict = 'Sharp OOS Fit';
      volTone = 'positive';
    } else if (q22 <= 0.75) {
      volVerdict = 'Moderate OOS';
      volTone = 'gold';
    } else {
      volVerdict = 'Noisy Regime';
      volTone = 'negative';
    }
  }

  const volRow: OosScorecardRow = {
    module: 'volatility',
    label: 'Volatility (1d · 5d · 22d)',
    headSpec: 'QLIKE Log-Variance',
    status: statuses.volatility,
    primaryMetricLabel: '22d OOS QLIKE',
    primaryMetricValue: volatility ? volatility.horizons[22].holdoutQlike.toFixed(3) : '—',
    secondaryMetricLabel: '1d / 5d MSE log-RV',
    secondaryMetricValue: volatility
      ? `${volatility.horizons[1].holdoutMseLogVar.toFixed(2)} / ${volatility.horizons[5].holdoutMseLogVar.toFixed(2)}`
      : '—',
    verdict: volVerdict,
    verdictTone: volTone,
    oosSampleCount: volatility ? volatility.horizons[1].holdoutCount : null,
    targetView: 'modeling',
    targetLabel: 'Modeling',
  };

  // 2. Risk VaR / ES Pinball row
  let riskVerdict = statuses.risk_var === 'training' ? 'Training' : statuses.risk_var === 'insufficient' ? 'Insufficient' : 'Pending';
  let riskTone: OosScorecardRow['verdictTone'] = 'neutral';
  if (riskVar) {
    const bz = riskVar.backtestRow.baselZone;
    const kupPass = riskVar.backtestRow.kupiecVerdict95 === 'Pass';
    if (bz === 'Green' && kupPass) {
      riskVerdict = 'Basel Green · POF Pass';
      riskTone = 'positive';
    } else if (bz === 'Red') {
      riskVerdict = 'Basel Red · Breach Alert';
      riskTone = 'negative';
    } else {
      riskVerdict = `Basel ${bz} · POF ${riskVar.backtestRow.kupiecVerdict95}`;
      riskTone = 'gold';
    }
  }

  const riskRow: OosScorecardRow = {
    module: 'risk_var',
    label: 'Risk VaR & ES (1d · 5d · 22d)',
    headSpec: 'Pinball Quantile 95/99',
    status: statuses.risk_var,
    primaryMetricLabel: 'Kupiec 95 / 99 p-val',
    primaryMetricValue: riskVar
      ? `${riskVar.backtestRow.kupiecPVal95.toFixed(2)} / ${riskVar.backtestRow.kupiecPVal99.toFixed(2)}`
      : '—',
    secondaryMetricLabel: '1d OOS 95% / 99% Hits',
    secondaryMetricValue: riskVar
      ? `${fmtPct(riskVar.horizons[1].oosExceedanceRate95, 1)} / ${fmtPct(riskVar.horizons[1].oosExceedanceRate99, 1)}`
      : '—',
    verdict: riskVerdict,
    verdictTone: riskTone,
    oosSampleCount: riskVar ? riskVar.horizons[1].holdoutCount : null,
    targetView: 'analysis',
    targetParams: { tab: 'risk' },
    targetLabel: 'Risk',
  };

  // 3. Multi-Asset Regression row
  let regVerdict = statuses.regression === 'training' ? 'Training' : statuses.regression === 'insufficient' ? 'Insufficient' : 'Pending';
  let regTone: OosScorecardRow['verdictTone'] = 'neutral';
  if (regression) {
    const lift5 = regression.horizons[5].nonlinearLiftR2;
    const hit5 = regression.horizons[5].directionalAccuracyPct;
    if (lift5 > 0.005 && hit5 >= 52) {
      regVerdict = 'Beats OLS · Positive Hit';
      regTone = 'positive';
    } else if (lift5 >= -0.02) {
      regVerdict = 'Near OLS Baseline';
      regTone = 'gold';
    } else {
      regVerdict = 'Trails Linear OLS';
      regTone = 'negative';
    }
  }

  const regRow: OosScorecardRow = {
    module: 'regression',
    label: regression
      ? `Multi-Asset Reg (${regression.independentTickers.join(', ')})`
      : 'Multi-Asset Regression',
    headSpec: 'Huber Multi-Channel',
    status: statuses.regression,
    primaryMetricLabel: '5d OOS R² (Lift vs OLS)',
    primaryMetricValue: regression
      ? `${regression.horizons[5].oosR2.toFixed(3)} (${regression.horizons[5].nonlinearLiftR2 >= 0 ? '+' : ''}${regression.horizons[5].nonlinearLiftR2.toFixed(3)})`
      : '—',
    secondaryMetricLabel: '5d / 22d Hit Rate',
    secondaryMetricValue: regression
      ? `${regression.horizons[5].directionalAccuracyPct.toFixed(0)}% / ${regression.horizons[22].directionalAccuracyPct.toFixed(0)}%`
      : '—',
    verdict: regVerdict,
    verdictTone: regTone,
    oosSampleCount: regression ? regression.holdoutCount : null,
    targetView: 'regression',
    targetLabel: 'Regression',
  };

  // 4. Signal Confluence row
  let confTone: OosScorecardRow['verdictTone'] = 'neutral';
  if (confluence.alignment === 'ALIGNED') confTone = 'positive';
  else if (confluence.alignment === 'MIXED') confTone = 'gold';
  else if (confluence.alignment === 'DIVERGENT') confTone = 'negative';

  const confRow: OosScorecardRow = {
    module: 'confluence',
    label: 'Signal Confluence (VRP + Decomp)',
    headSpec: 'Zero-Retrain Fusion',
    status: statuses.confluence,
    primaryMetricLabel: 'IV30 − TCN RV22',
    primaryMetricValue: fmtSignedPct(confluence.forwardVrp22, 1),
    secondaryMetricLabel: 'Regime / Decomp Tilt',
    secondaryMetricValue: `${confluence.volRegimeRead === 'UNAVAILABLE' ? 'No IV' : confluence.volRegimeRead.replace(' FAVORED', '')}${
      confluence.priceTiltPct !== null
        ? ` · ${confluence.priceTiltPct >= 0 ? '+' : ''}${confluence.priceTiltPct.toFixed(2)}%`
        : ''
    }`,
    verdict: confluence.alignment,
    verdictTone: confTone,
    oosSampleCount: volatility ? volatility.horizons[22].holdoutCount : null,
    targetView: 'volatility',
    targetParams: { tab: 'decision' },
    targetLabel: 'T8 Decision',
  };

  return [volRow, riskRow, regRow, confRow];
}

export function buildHorizonComparisonMatrix(params: {
  volatility: TcnVolatilityForecast | null;
  riskVar: TcnRiskVarForecast | null;
  regression: TcnMultiAssetRegressionResult | null;
  iv30: number | null;
}): HorizonMatrixRow[] {
  const { volatility, riskVar, regression, iv30 } = params;
  const labels: Record<ForecastHorizon, string> = {
    1: '1d (Next Session)',
    5: '5d (1 Trading Week)',
    22: '22d (1 Trading Month)',
  };

  return MULTI_HORIZONS.map((h) => {
    const vh = volatility?.horizons[h] ?? null;
    const rh = riskVar?.horizons[h] ?? null;
    const regh = regression?.horizons[h] ?? null;

    let vrpNote = '—';
    if (vh) {
      if (h === 22 && iv30 !== null) {
        const diff = iv30 - vh.annualizedVol;
        vrpNote = `IV30 ${diff >= 0 ? '+' : ''}${(diff * 100).toFixed(1)} pts vs TCN`;
      } else if (volatility) {
        const slope = vh.annualizedVol - volatility.horizons[22].annualizedVol;
        vrpNote =
          h === 22
            ? 'Monthly vol anchor'
            : `${slope >= 0 ? '+' : ''}${(slope * 100).toFixed(1)} pts vs 22d`;
      }
    }

    return {
      horizon: h,
      label: labels[h],
      volAnnualized: vh?.annualizedVol ?? null,
      volLower99: vh?.lower99Vol ?? null,
      volUpper99: vh?.upper99Vol ?? null,
      volOosQlike: vh?.holdoutQlike ?? null,
      var95: rh?.var95 ?? null,
      var99: rh?.var99 ?? null,
      es99: rh?.es99 ?? null,
      tailRatio: rh?.tailRatio ?? null,
      regReturn: regh?.predictedReturn ?? null,
      regLower99: regh?.lower99 ?? null,
      regUpper99: regh?.upper99 ?? null,
      regHitRatePct: regh?.directionalAccuracyPct ?? null,
      regLiftR2: regh?.nonlinearLiftR2 ?? null,
      vrpNote,
    };
  });
}

/**
 * Aligns causal self-attention lag profiles (t-L ... t-1) and merges top attended
 * historical shock dates across Volatility, Risk VaR, and Multi-Asset Regression.
 */
export function aggregateCausalAttentionTimeline(params: {
  volatility: TcnVolatilityForecast | null;
  riskVar: TcnRiskVarForecast | null;
  regression: TcnMultiAssetRegressionResult | null;
}): UnifiedAttentionTimeline {
  const { volatility, riskVar, regression } = params;
  const volW = volatility?.attentionWeights ?? null;
  const riskW = riskVar?.attentionWeights ?? null;
  const regW = regression?.attentionWeights ?? null;

  const activeModules: ('volatility' | 'risk_var' | 'regression')[] = [];
  if (volW && volW.length > 0) activeModules.push('volatility');
  if (riskW && riskW.length > 0) activeModules.push('risk_var');
  if (regW && regW.length > 0) activeModules.push('regression');

  const maxLookback = Math.max(volW?.length ?? 0, riskW?.length ?? 0, regW?.length ?? 0);

  const hmmSource =
    volatility && volatility.hmmStates && volatility.hmmStates.length > 0
      ? { states: volatility.hmmStates, probs: volatility.hmmProbs }
      : riskVar && riskVar.hmmStates && riskVar.hmmStates.length > 0
        ? { states: riskVar.hmmStates, probs: riskVar.hmmProbs }
        : regression && regression.hmmStates && regression.hmmStates.length > 0
          ? { states: regression.hmmStates, probs: regression.hmmProbs }
          : null;

  // attentionWeights[0] is oldest lag (L), attentionWeights[L-1] is most recent lag (1)
  const lagCurves: UnifiedLagPoint[] = [];
  for (let lag = maxLookback; lag >= 1; lag--) {
    const vIdx = volW ? volW.length - lag : -1;
    const rIdx = riskW ? riskW.length - lag : -1;
    const gIdx = regW ? regW.length - lag : -1;

    const vw = volW && vIdx >= 0 && vIdx < volW.length ? volW[vIdx] : null;
    const rw = riskW && rIdx >= 0 && rIdx < riskW.length ? riskW[rIdx] : null;
    const gw = regW && gIdx >= 0 && gIdx < regW.length ? regW[gIdx] : null;

    let sum = 0;
    let cnt = 0;
    if (vw !== null) {
      sum += vw;
      cnt++;
    }
    if (rw !== null) {
      sum += rw;
      cnt++;
    }
    if (gw !== null) {
      sum += gw;
      cnt++;
    }

    const hIdx = hmmSource ? hmmSource.states.length - lag : -1;
    const hmmState =
      hmmSource && hIdx >= 0 && hIdx < hmmSource.states.length ? hmmSource.states[hIdx] : null;
    const hmmProb =
      hmmSource && hIdx >= 0 && hIdx < hmmSource.probs.length ? hmmSource.probs[hIdx] : null;

    lagCurves.push({
      lag,
      volWeight: vw,
      riskWeight: rw,
      regWeight: gw,
      consensusWeight: cnt > 0 ? sum / cnt : 0,
      hmmState,
      hmmProb,
    });
  }

  // Merge top shock dates across heads by ISO date
  const byDate = new Map<
    string,
    {
      date: string;
      lag: number;
      volWeight: number | null;
      riskWeight: number | null;
      regWeight: number | null;
      notes: string[];
    }
  >();

  const ingest = (
    mod: 'volatility' | 'risk_var' | 'regression',
    items: AttentionShockDate[] | undefined
  ) => {
    if (!items) return;
    for (const item of items) {
      let entry = byDate.get(item.date);
      if (!entry) {
        entry = {
          date: item.date,
          lag: item.lag,
          volWeight: null,
          riskWeight: null,
          regWeight: null,
          notes: [],
        };
        byDate.set(item.date, entry);
      }
      if (mod === 'volatility') entry.volWeight = item.weight;
      else if (mod === 'risk_var') entry.riskWeight = item.weight;
      else entry.regWeight = item.weight;
      const valStr = Number.isFinite(item.metricValue)
        ? `${item.metricValue >= 0 && item.metricLabel.includes('Return') ? '+' : ''}${item.metricValue.toFixed(2)}`
        : '—';
      entry.notes.push(`${item.metricLabel} ${valStr}`);
    }
  };

  ingest('volatility', volatility?.topShockDates);
  ingest('risk_var', riskVar?.topShockDates);
  ingest('regression', regression?.topTransmissionDates);

  const jointShockDates: JointShockDateRow[] = Array.from(byDate.values())
    .map((e) => {
      const modules: ('volatility' | 'risk_var' | 'regression')[] = [];
      let combinedWeight = 0;
      if (e.volWeight !== null) {
        modules.push('volatility');
        combinedWeight += e.volWeight;
      }
      if (e.riskWeight !== null) {
        modules.push('risk_var');
        combinedWeight += e.riskWeight;
      }
      if (e.regWeight !== null) {
        modules.push('regression');
        combinedWeight += e.regWeight;
      }
      return {
        date: e.date,
        lag: e.lag,
        moduleCount: modules.length,
        modules,
        combinedWeight,
        volWeight: e.volWeight,
        riskWeight: e.riskWeight,
        regWeight: e.regWeight,
        metricSummary: e.notes.slice(0, 2).join(' · '),
      };
    })
    .sort((a, b) => {
      if (b.moduleCount !== a.moduleCount) return b.moduleCount - a.moduleCount;
      return b.combinedWeight - a.combinedWeight;
    })
    .slice(0, 8);

  return {
    maxLookback,
    lagCurves,
    jointShockDates,
    activeModules,
  };
}

export function summarizeCacheAndLatency(params: {
  volatility: TcnVolatilityForecast | null;
  riskVar: TcnRiskVarForecast | null;
  regression: TcnMultiAssetRegressionResult | null;
  decompForecast: TcnDecompositionForecast | null;
  statuses: TcnModuleStatuses;
  multiDomainCounts: Record<TcnDomainKey, number>;
  decompStats: { forecastCount: number; sensitivityCount: number; totalWeightTensors: number };
}): UnifiedCacheTelemetry {
  const { volatility, riskVar, regression, decompForecast, statuses, multiDomainCounts, decompStats } = params;

  const toCacheState = (
    status: TcnModuleStatus,
    cacheHit: boolean | null
  ): ModuleTelemetryRow['cacheState'] => {
    if (status === 'training') return 'Training';
    if (status === 'insufficient') return 'Insufficient';
    if (cacheHit === true) return 'Cache Hit';
    if (cacheHit === false) return 'Fresh Fit';
    return 'Not Cached';
  };

  const rows: ModuleTelemetryRow[] = [
    {
      module: 'volatility',
      label: 'Volatility QLIKE (1d/5d/22d)',
      status: statuses.volatility,
      cacheState: toCacheState(statuses.volatility, volatility ? volatility.cacheHit : null),
      latencyMs: volatility ? volatility.trainingDurationMs : null,
      parameterCount: volatility ? volatility.parameterCount : null,
      receptiveField: volatility ? volatility.receptiveField : null,
      asOfDate: volatility ? volatility.asOfDate : null,
      canRetrain: true,
    },
    {
      module: 'risk_var',
      label: 'Risk VaR / ES Pinball (1d/5d/22d)',
      status: statuses.risk_var,
      cacheState: toCacheState(statuses.risk_var, riskVar ? riskVar.cacheHit : null),
      latencyMs: riskVar ? riskVar.trainingDurationMs : null,
      parameterCount: riskVar ? riskVar.parameterCount : null,
      receptiveField: riskVar ? riskVar.receptiveField : null,
      asOfDate: riskVar ? riskVar.asOfDate : null,
      canRetrain: true,
    },
    {
      module: 'regression',
      label: 'Multi-Asset Regression (1d/5d/22d)',
      status: statuses.regression,
      cacheState: toCacheState(statuses.regression, regression ? regression.cacheHit : null),
      latencyMs: regression ? regression.trainingDurationMs : null,
      parameterCount: regression ? regression.parameterCount : null,
      receptiveField: regression ? regression.receptiveField : null,
      asOfDate: regression ? regression.asOfDate : null,
      canRetrain: true,
    },
    {
      module: 'decomposition',
      label: 'Price Decomposition (t+1 4-Comp)',
      status: decompForecast ? 'cached' : 'idle',
      cacheState: decompForecast ? 'Cache Hit' : 'Not Cached',
      latencyMs: decompForecast ? decompForecast.trainingDurationMs : null,
      parameterCount: decompForecast ? decompForecast.parameterCount * 4 : null,
      receptiveField: decompForecast ? decompForecast.receptiveField : null,
      asOfDate: decompForecast ? decompForecast.lastDate : null,
      canRetrain: false,
    },
  ];

  let totalParameters = 0;
  let totalLatencyMs = 0;
  let cacheHitsCount = 0;
  let readyTrainableCount = 0;

  for (const r of rows) {
    if (r.parameterCount !== null) totalParameters += r.parameterCount;
    if (r.latencyMs !== null) totalLatencyMs += r.latencyMs;
    if (r.cacheState === 'Cache Hit') cacheHitsCount++;
    if (r.canRetrain && (r.status === 'ready' || r.status === 'cached')) readyTrainableCount++;
  }

  const multiDomainTotal =
    multiDomainCounts.volatility +
    multiDomainCounts.risk_var +
    multiDomainCounts.regression +
    multiDomainCounts.correlation;

  return {
    rows,
    totalParameters,
    totalLatencyMs,
    cacheHitsCount,
    readyTrainableCount,
    multiDomainCounts,
    multiDomainTotal,
    decompCachedCount: decompStats.forecastCount,
    decompWeightTensors: decompStats.totalWeightTensors,
  };
}

/**
 * Aggregates Calm / Watchful / Crisis strictly out-of-sample forecast accuracy
 * across all ready TCN-Attention modules for the Unified Diagnostics Dashboard.
 */
export function buildCrossModuleRegimeBreakdown(params: {
  volatility: TcnVolatilityForecast | null;
  riskVar: TcnRiskVarForecast | null;
  regression: TcnMultiAssetRegressionResult | null;
  decompBacktest?: HistoricalBacktestResult | null;
}): CrossModuleRegimeRow[] {
  const { volatility, riskVar, regression, decompBacktest = null } = params;
  return ([0, 1, 2] as const).map((st) => {
    const vr = volatility?.regimePerformance?.find((r) => r.state === st);
    const rr = riskVar?.regimePerformance?.find((r) => r.state === st);
    const gr = regression?.regimePerformance?.find((r) => r.state === st);
    const dr = decompBacktest?.regimeBreakdown?.find((r) => r.state === st);
    return {
      state: st,
      label: HMM_REGIME_TOKENS[st].label,
      volCount: vr?.count ?? 0,
      volMeanQlike: vr?.meanQlike ?? null,
      volMeanPredVol: vr?.meanPredVol ?? null,
      riskCount: rr?.count ?? 0,
      riskExc95RatePct: rr?.exceedance95RatePct ?? null,
      riskExc99RatePct: rr?.exceedance99RatePct ?? null,
      regCount: gr?.count ?? 0,
      regHitRatePct: gr?.hitRatePct ?? null,
      regOosRmse: gr?.oosRmse ?? null,
      decompCount: dr?.count ?? 0,
      decompHitRatePct: dr?.directionalHitRatePct ?? null,
      decompCumNetPct: dr?.cumNetReturnPct ?? null,
    };
  });
}
