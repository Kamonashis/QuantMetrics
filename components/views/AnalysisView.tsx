'use client';

import React, { useState, useEffect, useMemo } from 'react';
import {
  PriceBar,
  calculateReturns,
  computeReturnStats,
  computeRollingAnnualizedVol,
  computeAcfPacfSquaredReturns,
  decomposeTimeSeries,
  computeAdfTest,
  IntradayBar,
  computeRealizedVolatilityAndJumps,
  computeDiurnalVolatilityProfile,
  fitGjrGarch,
} from '@/lib/quant-engine';
import {
  computeStationarityBattery,
  computeArchLmLjungBoxBattery,
  computeEvtTailEstimator,
  computeRollingMomentsWithBands,
  computeStructuralBreaks,
  computeHurstAndVarianceRatio,
  computeNewsImpactCurve,
  computeVolatilitySignaturePlot,
  computeOvernightIntradaySplit,
  computeVaRBacktestSuite,
  computeDrawdownAnalytics,
  computeDownsideAndBenchmark,
  computeMicrostructureDiagnostics,
} from '@/lib/analysis-engine';
import { fitGaussianHmm, type HmmFit } from '@/lib/hmm';
import { AnalysisOverviewTab } from './analysis/AnalysisOverviewTab';
import { AnalysisStationarityTab } from './analysis/AnalysisStationarityTab';
import { AnalysisDynamicsTab } from './analysis/AnalysisDynamicsTab';
import { AnalysisRiskTab } from './analysis/AnalysisRiskTab';
import { AnalysisMicrostructureTab } from './analysis/AnalysisMicrostructureTab';
import { ShimmerButton } from '@/components/ui/ShimmerButton';
import { TickerSuggest } from '@/components/ui/TickerSuggest';
import { AnimatedTabs } from '@/components/ui/animated-tabs';
import { Iv30Chip } from '@/components/views/Iv30Chip';
import { TrapStateChip } from '@/components/views/TrapStateChip';
import {
  Play,
  RotateCcw,
  Zap,
  Calendar,
  Layers,
  Scale,
  Activity,
  ShieldAlert,
  Waves,
  Download,
  Settings2,
  ChevronDown,
} from 'lucide-react';

interface AnalysisViewProps {
  ticker: string;
  onTickerChange: (ticker: string) => void;
  startDate: string;
  onStartDateChange: (date: string) => void;
  endDate: string;
  onEndDateChange: (date: string) => void;
  returnType: 'simple' | 'log';
  onReturnTypeChange: (type: 'simple' | 'log') => void;
  rollingWindows: string;
  onRollingWindowsChange: (win: string) => void;
  decompPeriod: number;
  onDecompPeriodChange: (p: number) => void;
  decompModel: 'additive' | 'multiplicative';
  onDecompModelChange: (m: 'additive' | 'multiplicative') => void;
  bars: PriceBar[];
  intradayBars: IntradayBar[];
  resolution: 'daily' | 'intraday';
  onResolutionChange: (res: 'daily' | 'intraday') => void;
  isLoading: boolean;
  onRunAnalysis: () => void;
  companyName: string;
}

export type AnalysisSubTabId =
  | 'overview'
  | 'stationarity'
  | 'dynamics'
  | 'risk'
  | 'microstructure';

export function AnalysisView({
  ticker,
  onTickerChange,
  startDate,
  onStartDateChange,
  endDate,
  onEndDateChange,
  returnType,
  onReturnTypeChange,
  rollingWindows,
  onRollingWindowsChange,
  decompPeriod,
  onDecompPeriodChange,
  decompModel,
  onDecompModelChange,
  bars,
  intradayBars,
  resolution,
  onResolutionChange,
  isLoading,
  onRunAnalysis,
  companyName,
}: AnalysisViewProps) {
  const [localTicker, setLocalTicker] = useState(ticker);
  const [activeSubTab, setActiveSubTab] = useState<AnalysisSubTabId>('overview');
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Deep-link: ?tab=dynamics restores the T1–T5 selection (replaceState keeps history clean).
  // Deferred with a 0ms timeout so the setState doesn't cascade synchronously inside the effect.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const timer = setTimeout(() => {
      const t = new URLSearchParams(window.location.search).get('tab');
      const valid: AnalysisSubTabId[] = ['overview', 'stationarity', 'dynamics', 'risk', 'microstructure'];
      if (t && (valid as string[]).includes(t)) {
        setActiveSubTab(t as AnalysisSubTabId);
      }
    }, 0);
    return () => clearTimeout(timer);
  }, []);

  const handleSubTabChange = (id: AnalysisSubTabId) => {
    setActiveSubTab(id);
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href);
      url.searchParams.set('tab', id);
      window.history.replaceState(null, '', url.toString());
    }
  };

  // Filter daily bars by start and end date
  const filteredBars = useMemo(() => {
    if (!bars || bars.length === 0) return [];
    return bars.filter((b) => {
      if (startDate && b.date < startDate) return false;
      if (endDate && b.date > endDate) return false;
      return true;
    });
  }, [bars, startDate, endDate]);

  const dates = useMemo(() => filteredBars.map((b) => b.date), [filteredBars]);
  const prices = useMemo(() => filteredBars.map((b) => b.close), [filteredBars]);

  // Compute returns based on selected returnType
  const returns = useMemo(() => {
    return calculateReturns(prices, returnType);
  }, [prices, returnType]);

  // Compute distribution statistics
  const returnStats = useMemo(() => {
    return computeReturnStats(returns);
  }, [returns]);

  // Parse rolling windows
  const parsedWindows = useMemo(() => {
    const list = rollingWindows
      .split(',')
      .map((w) => parseInt(w.trim(), 10))
      .filter((w) => !isNaN(w) && w > 1 && w < 300);
    return list.length > 0 ? list : [10, 20, 30];
  }, [rollingWindows]);

  // Compute rolling annualized volatility
  const rollingVolSeries = useMemo(() => {
    if (returns.length < 5) return [];
    return computeRollingAnnualizedVol(dates, returns, parsedWindows);
  }, [dates, returns, parsedWindows]);

  // Compute Seasonal Decomposition
  const decompResult = useMemo(() => {
    return decomposeTimeSeries(dates, prices, decompPeriod, decompModel);
  }, [dates, prices, decompPeriod, decompModel]);

  // Compute ADF test on decomposition residuals
  const adfResult = useMemo(() => {
    const validResiduals = decompResult.residual.filter(
      (v): v is number => v !== null && !isNaN(v)
    );
    return computeAdfTest(validResiduals);
  }, [decompResult]);

  // High-Frequency 1-Minute Realized Volatility & Jump Decomposition
  const dailyRealizedVol = useMemo(() => {
    return computeRealizedVolatilityAndJumps(intradayBars);
  }, [intradayBars]);

  // Intraday 390-min Diurnal Volatility Profile
  const diurnalProfile = useMemo(() => {
    return computeDiurnalVolatilityProfile(intradayBars);
  }, [intradayBars]);

  // =============================================================
  // ADVANCED ANALYSIS SUITE ENGINES (TIERS 1-5)
  // =============================================================

  // Tier 1: Stationarity Battery (ADF, PP, KPSS on Price, Return, Squared Return)
  const stationarityBattery = useMemo(() => {
    return computeStationarityBattery(prices, returns);
  }, [prices, returns]);

  // Tier 1: ARCH-LM & Ljung-Box Battery
  const archLmLjungBox = useMemo(() => {
    return computeArchLmLjungBoxBattery(returns);
  }, [returns]);

  // Tier 2: EVT Hill Estimator & POT-GPD Tail Risk
  const evtTail = useMemo(() => {
    return computeEvtTailEstimator(returns);
  }, [returns]);

  // Tier 1: Rolling Moments with Lo (2002) Confidence Intervals
  const rollingMoments = useMemo(() => {
    return computeRollingMomentsWithBands(dates, returns, 60);
  }, [dates, returns]);

  // Tier 1: Structural Breaks (CUSUM-of-Squares)
  const structuralBreaks = useMemo(() => {
    return computeStructuralBreaks(dates, returns, prices);
  }, [dates, returns, prices]);

  // Tier 1: Hurst Exponent & Lo-MacKinlay Variance Ratio
  const hurstAndVr = useMemo(() => {
    return computeHurstAndVarianceRatio(returns);
  }, [returns]);

  // Tier 3: News Impact Curve (Symmetric vs GJR-GARCH) — from fitted GJR-GARCH params
  const newsImpactPoints = useMemo(() => {
    const gjr = fitGjrGarch(returns);
    return computeNewsImpactCurve(returns, gjr.summary.omega, gjr.summary.alpha, gjr.summary.gamma, gjr.summary.beta);
  }, [returns]);

  // Tier 3: Volatility Signature Plot
  const volSignaturePoints = useMemo(() => {
    return computeVolatilitySignaturePlot(intradayBars);
  }, [intradayBars]);

  // Tier 3: Overnight vs Intraday Split
  const overnightIntradaySplit = useMemo(() => {
    return computeOvernightIntradaySplit(filteredBars, intradayBars);
  }, [filteredBars, intradayBars]);

  // Tier 4: 5-Model VaR & ES Backtesting Suite
  const varBacktestModels = useMemo(() => {
    return computeVaRBacktestSuite(returns);
  }, [returns]);

  // HMM volatility-regime fit — shared by Overview & Risk tabs. Features: [r, |r|, log GK-RV].
  const [hmmK, setHmmK] = useState<2 | 3 | 4>(3);
  const hmmFit = useMemo<HmmFit | null>(() => {
    if (!filteredBars || filteredBars.length < 60) return null;
    const feats: number[][] = [];
    for (let i = 1; i < filteredBars.length; i++) {
      const b = filteredBars[i];
      const prev = filteredBars[i - 1];
      const o = b.open, h = b.high, l = b.low, c = b.close;
      const po = prev.close;
      if (!(o > 0 && h > 0 && l > 0 && c > 0 && po > 0)) continue;
      const r =
        returnType === 'log'
          ? Math.log(c / po)
          : (c - po) / po;
      const gk = 0.5 * Math.log(h / l) ** 2 - (2 * Math.LN2 - 1) * Math.log(c / o) ** 2;
      feats.push([r, Math.abs(r), Math.log(Math.max(gk, 1e-8))]);
    }
    return fitGaussianHmm(feats, { k: hmmK, rankFeature: 2 });
  }, [filteredBars, returnType, hmmK]);

  // Tier 4: Drawdown Analytics & Underwater Curve
  const drawdownResult = useMemo(() => {
    return computeDrawdownAnalytics(dates, prices);
  }, [dates, prices]);

  // Tier 4: Downside Risk, Asymmetric Beta & HAC Probabilistic Sharpe Ratio
  const downsideResult = useMemo(() => {
    return computeDownsideAndBenchmark(returns, dates);
  }, [returns, dates]);

  // Tier 5: Microstructure Diagnostics (Roll & Corwin-Schultz spreads, Amihud, Realized Semivariance)
  const microstructureResult = useMemo(() => {
    return computeMicrostructureDiagnostics(intradayBars);
  }, [intradayBars]);

  // Handle ticker submission
  const handleTickerSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (localTicker.trim()) {
      onTickerChange(localTicker.trim().toUpperCase());
      onRunAnalysis();
    }
  };

  // CSV Export Handler
  const handleExportCsv = () => {
    if (filteredBars.length === 0) return;
    const header = 'Date,Open,High,Low,Close,Volume,Return\n';
    const rows = filteredBars.map((b, idx) => {
      const ret = idx > 0 && returns[idx - 1] !== undefined ? returns[idx - 1].toFixed(6) : '';
      return `${b.date},${b.open},${b.high},${b.low},${b.close},${b.volume},${ret}`;
    });
    const csvContent = 'data:text/csv;charset=utf-8,' + encodeURIComponent(header + rows.join('\n'));
    const link = document.createElement('a');
    link.setAttribute('href', csvContent);
    link.setAttribute('download', `${ticker}_quantitative_analysis.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Parameters Header Card */}
      <div className="panel p-5">
        <form onSubmit={handleTickerSubmit} className="space-y-4">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-zinc-100 dark:border-zinc-800/80">
            <div>
              <div className="flex items-center gap-2">
                <span className="kicker">
                  Single-Asset Econometric Workstation
                </span>
                <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
                <span className="text-xs text-zinc-500 dark:text-zinc-400">
                  {companyName} ({ticker})
                </span>
              </div>
              <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100 mt-0.5 text-balance">
                Econometric Distributions, Volatility Dynamics &amp; Tail Risk
              </h2>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              {/* Resolution Segmented Control */}
              <div className="scrollbar-none flex max-w-full items-center gap-1 overflow-x-auto p-1 bg-zinc-100 dark:bg-zinc-800 rounded-lg">
                <button
                  type="button"
                  onClick={() => onResolutionChange('daily')}
                  className={`min-h-[44px] shrink-0 px-4 text-xs font-semibold rounded-md transition-colors cursor-pointer inline-flex items-center ${
                    resolution === 'daily'
                      ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-xs'
                      : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
                  }`}
                >
                  Daily (1Y)
                </button>
                <button
                  type="button"
                  onClick={() => onResolutionChange('intraday')}
                  className={`min-h-[44px] shrink-0 px-4 text-xs font-semibold rounded-md transition-colors flex items-center gap-1.5 cursor-pointer ${
                    resolution === 'intraday'
                      ? 'bg-gold-500 text-zinc-950 font-bold shadow-xs'
                      : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
                  }`}
                >
                  <Zap className="w-3.5 h-3.5" />
                  Intraday 1-Min (7D)
                </button>
              </div>

              <ShimmerButton
                type="submit"
                disabled={isLoading}
                className="hidden min-h-[44px] px-4 md:inline-flex"
              >
                {isLoading ? (
                  <>
                    <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                    Calculating...
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current text-gold-500" />
                    <span>Run Econometrics</span>
                  </>
                )}
              </ShimmerButton>
            </div>
          </div>

          {/* Primary controls — always visible: ticker + date range */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            {/* Ticker Input */}
            <div className="col-span-2 md:col-span-2 lg:col-span-2">
              <label htmlFor="analysis-ticker-input" className="block text-zinc-600 dark:text-zinc-400 font-medium mb-1">
                Ticker Symbol
              </label>
              <TickerSuggest
                id="analysis-ticker-input"
                ariaLabel="Ticker Symbol"
                value={localTicker}
                onChange={setLocalTicker}
                placeholder="Search U.S. tickers — e.g. NVDA"
                inputClassName="h-11 w-full px-3 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono font-bold text-base text-zinc-900 dark:text-zinc-100 focus:outline-hidden focus:ring-1 focus:ring-gold-500"
              />
            </div>

            {/* Start Date */}
            <div>
              <label htmlFor="analysis-start-date" className="block text-zinc-600 dark:text-zinc-400 font-medium mb-1">
                Start Date
              </label>
              <input
                id="analysis-start-date"
                type="date"
                value={startDate}
                onChange={(e) => onStartDateChange(e.target.value)}
                className="h-11 w-full px-2.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono text-base text-zinc-900 dark:text-zinc-100 focus:outline-hidden focus:ring-1 focus:ring-gold-500"
              />
            </div>

            {/* End Date */}
            <div>
              <label htmlFor="analysis-end-date" className="block text-zinc-600 dark:text-zinc-400 font-medium mb-1">
                End Date
              </label>
              <input
                id="analysis-end-date"
                type="date"
                value={endDate}
                onChange={(e) => onEndDateChange(e.target.value)}
                className="h-11 w-full px-2.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono text-base text-zinc-900 dark:text-zinc-100 focus:outline-hidden focus:ring-1 focus:ring-gold-500"
              />
            </div>
          </div>

          {/* Advanced parameters — collapsed behind a single disclosure, summary line always visible */}
          <div className="rounded-xl border border-zinc-100 dark:border-zinc-800/60">
            <button
              type="button"
              onClick={() => setAdvancedOpen((v) => !v)}
              aria-expanded={advancedOpen}
              aria-controls="analysis-advanced-fields"
              className="flex min-h-[44px] w-full cursor-pointer items-center justify-between gap-2 px-3 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-gold-400/60"
            >
              <span className="flex min-w-0 items-center gap-2 text-[11px] text-zinc-500 dark:text-zinc-400">
                <Settings2 className="h-3.5 w-3.5 shrink-0 text-zinc-400" aria-hidden="true" />
                <span className="truncate font-mono">
                  Advanced · {returnType === 'log' ? 'Log returns' : 'Simple returns'} · Windows{' '}
                  {rollingWindows} · {decompModel === 'additive' ? 'Additive' : 'Multiplicative'} p={decompPeriod}
                </span>
              </span>
              <ChevronDown
                className={`h-4 w-4 shrink-0 text-zinc-400 transition-transform duration-200 ${advancedOpen ? 'rotate-180' : ''}`}
                aria-hidden="true"
              />
            </button>
            {advancedOpen && (
              <div
                id="analysis-advanced-fields"
                className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 border-t border-zinc-100 dark:border-zinc-800/60 p-3 text-xs"
              >
                {/* Return Type */}
                <div>
                  <label htmlFor="analysis-return-type" className="block text-zinc-600 dark:text-zinc-400 font-medium mb-1">
                    Return Type
                  </label>
                  <select
                    id="analysis-return-type"
                    value={returnType}
                    onChange={(e) => onReturnTypeChange(e.target.value as 'simple' | 'log')}
                    className="h-11 w-full px-3 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 text-base text-zinc-900 dark:text-zinc-100 font-medium focus:outline-hidden focus:ring-1 focus:ring-gold-500"
                  >
                    <option value="log">Log (Continuously Compounded)</option>
                    <option value="simple">Simple Arithmetic (ΔP/P)</option>
                  </select>
                </div>

                {/* Rolling Volatility Window(s) */}
                <div>
                  <label htmlFor="analysis-rolling-windows" className="block text-zinc-600 dark:text-zinc-400 font-medium mb-1">
                    Rolling Windows (days)
                  </label>
                  <input
                    id="analysis-rolling-windows"
                    type="text"
                    inputMode="decimal"
                    value={rollingWindows}
                    onChange={(e) => onRollingWindowsChange(e.target.value)}
                    placeholder="10,20,30"
                    className="h-11 w-full px-3 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono text-base text-zinc-900 dark:text-zinc-100 focus:outline-hidden focus:ring-1 focus:ring-gold-500"
                  />
                </div>

                {/* Decomposition Settings */}
                <div>
                  <label htmlFor="analysis-decomp-period" className="block text-zinc-600 dark:text-zinc-400 font-medium mb-1">
                    Decomp Period &amp; Model
                  </label>
                  <div className="flex items-center gap-1.5">
                    <input
                      id="analysis-decomp-period"
                      type="number"
                      inputMode="numeric"
                      min="5"
                      max="120"
                      value={decompPeriod}
                      onChange={(e) => onDecompPeriodChange(Math.max(5, parseInt(e.target.value, 10) || 30))}
                      className="h-11 w-16 px-2 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono text-center text-base text-zinc-900 dark:text-zinc-100 focus:outline-hidden focus:ring-1 focus:ring-gold-500"
                      title="Period (e.g. 21 days)"
                    />
                    <select
                      aria-label="Decomposition model"
                      value={decompModel}
                      onChange={(e) => onDecompModelChange(e.target.value as 'additive' | 'multiplicative')}
                      className="h-11 flex-1 px-2 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 text-base text-zinc-900 dark:text-zinc-100 focus:outline-hidden focus:ring-1 focus:ring-gold-500"
                    >
                      <option value="additive">Additive</option>
                      <option value="multiplicative">Multiplicative</option>
                    </select>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Sticky full-width Run button — mobile, thumb reach, above the bottom tab bar */}
          <ShimmerButton
            type="submit"
            disabled={isLoading}
            className="sticky bottom-[calc(3.5rem+env(safe-area-inset-bottom,0px)+0.5rem)] z-30 flex h-12 w-full text-sm md:hidden"
          >
            {isLoading ? (
              <>
                <RotateCcw className="w-4 h-4 animate-spin" />
                Calculating…
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current text-gold-500" />
                <span>Run Econometrics</span>
              </>
            )}
          </ShimmerButton>
          {/* Clearance so the last Advanced field can scroll clear of the sticky Run button */}
          <div className="h-14 md:hidden" aria-hidden="true" />
        </form>

        {/* Popular Ticker Quick Selector */}
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 mt-3 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 text-xs">
          <span className="text-zinc-400">Popular Tickers:</span>
          {['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'JPM'].map((t) => (
            <button
              key={t}
              onClick={() => {
                setLocalTicker(t);
                onTickerChange(t);
                onRunAnalysis();
              }}
              className={`inline-flex h-11 min-w-[44px] cursor-pointer items-center justify-center px-3 font-mono transition-colors hover:text-gold-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-gold-400/60 ${
                ticker === t ? 'text-gold-500 font-bold' : 'text-zinc-500 dark:text-zinc-400'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Common Sample & Data Honesty Banner + KPI strip (2×3 on xs, 3×2 → 6×1 on lg) */}
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between rounded-xl border border-zinc-200 dark:border-zinc-800/90 bg-zinc-50 dark:bg-[#101014] px-4 py-3 text-xs">
        <div className="flex min-w-0 items-start gap-2.5 sm:items-center">
          <Calendar className="mt-0.5 h-4 w-4 shrink-0 text-gold-500 sm:mt-0" />
          <div className="min-w-0 text-pretty text-zinc-700 dark:text-zinc-300">
            <span className="font-semibold text-zinc-900 dark:text-zinc-100">
              Sample Bounds:
            </span>{' '}
            Analyzing{' '}
            <strong className="font-mono text-gold-600 dark:text-gold-400">
              {filteredBars.length}
            </strong>{' '}
            daily observations ({dates[0] || startDate} → {dates[dates.length - 1] || endDate}) ·{' '}
            <strong className="font-mono text-zinc-800 dark:text-zinc-200">
              {intradayBars.length}
            </strong>{' '}
            1-minute intraday bars.
          </div>
        </div>

        <div className="grid shrink-0 grid-cols-3 gap-px overflow-hidden rounded-lg border border-[var(--hairline)] bg-[var(--hairline)] sm:grid-cols-6">
          {[
            { label: 'IV30', node: <Iv30Chip ticker={ticker} /> },
            { label: 'TRAP', node: <TrapStateChip ticker={ticker} /> },
            { label: 'Ann Vol', node: <span className="font-mono text-sm font-semibold tabular-nums text-zinc-800 dark:text-zinc-100">{(returnStats.annualizedVol * 100).toFixed(2)}%</span> },
            { label: 'Skew', node: <span className="font-mono text-sm font-semibold tabular-nums text-zinc-800 dark:text-zinc-100">{returnStats.skewness.toFixed(2)}</span> },
            { label: 'Kurtosis', node: <span className="font-mono text-sm font-semibold tabular-nums text-zinc-800 dark:text-zinc-100">{returnStats.excessKurtosis.toFixed(2)}</span> },
            { label: 'Hurst H', node: <span className="font-mono text-sm font-semibold tabular-nums text-zinc-800 dark:text-zinc-100">{hurstAndVr.hurstExponent.toFixed(3)}</span> },
          ].map(({ label, node }) => (
            <div
              key={label}
              className="flex flex-col items-center justify-center gap-0.5 bg-[var(--bg-panel)] px-2 py-2"
            >
              <span className="font-mono text-[8px] font-medium uppercase leading-none tracking-[0.16em] text-zinc-400 dark:text-zinc-500">
                {label}
              </span>
              <span className="flex min-h-[20px] items-center">{node}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Segmented Sub-Tab Navigation Strip — horizontally scrollable chips, edge fade, deep-linkable via ?tab= */}
      <AnimatedTabs
        value={activeSubTab}
        onChange={(id) => handleSubTabChange(id as AnalysisSubTabId)}
        tabs={[
          { id: 'overview', label: 'Overview & Time Series', icon: Layers, meta: 'T1' },
          { id: 'stationarity', label: 'Stationarity & Distributions', icon: Scale, meta: 'T2' },
          { id: 'dynamics', label: 'Volatility Dynamics', icon: Activity, meta: 'T3' },
          { id: 'risk', label: 'Risk & Drawdowns', icon: ShieldAlert, meta: 'T4' },
          { id: 'microstructure', label: 'Microstructure & Liquidity', icon: Waves, meta: 'T5' },
        ]}
      />

      {/* Sub-Tab Content Rendering */}
      {activeSubTab === 'overview' && (
        <AnalysisOverviewTab
          ticker={ticker}
          companyName={companyName}
          filteredBars={filteredBars}
          dates={dates}
          prices={prices}
          returns={returns}
          returnType={returnType}
          onReturnTypeChange={onReturnTypeChange}
          returnStats={returnStats}
          rollingVolSeries={rollingVolSeries}
          rollingMoments={rollingMoments}
          structuralBreaks={structuralBreaks}
          hurstAndVr={hurstAndVr}
          decompResult={decompResult}
          adfResult={adfResult}
          decompPeriod={decompPeriod}
          onDecompPeriodChange={onDecompPeriodChange}
          decompModel={decompModel}
          onDecompModelChange={onDecompModelChange}
          onExportCsv={handleExportCsv}
          hmmFit={hmmFit}
          hmmK={hmmK}
          onHmmKChange={setHmmK}
        />
      )}

      {activeSubTab === 'stationarity' && (
        <AnalysisStationarityTab
          stationarityBattery={stationarityBattery}
          archLmLjungBox={archLmLjungBox}
          evtTail={evtTail}
          returns={returns}
          stats={returnStats}
          ticker={ticker}
        />
      )}

      {activeSubTab === 'dynamics' && (
        <AnalysisDynamicsTab
          newsImpactPoints={newsImpactPoints}
          volSignaturePoints={volSignaturePoints}
          overnightIntradaySplit={overnightIntradaySplit}
          dailyRealizedVol={dailyRealizedVol}
          diurnalProfile={diurnalProfile}
          ticker={ticker}
        />
      )}

      {activeSubTab === 'risk' && (
        <AnalysisRiskTab
          varBacktestModels={varBacktestModels}
          drawdownResult={drawdownResult}
          downsideResult={downsideResult}
          ticker={ticker}
          hmmFit={hmmFit}
        />
      )}

      {activeSubTab === 'microstructure' && (
        <AnalysisMicrostructureTab
          microstructure={microstructureResult}
          ticker={ticker}
        />
      )}
    </div>
  );
}
