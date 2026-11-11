'use client';

import React, { useState, useMemo } from 'react';
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
import { AnalysisOverviewTab } from './analysis/AnalysisOverviewTab';
import { AnalysisStationarityTab } from './analysis/AnalysisStationarityTab';
import { AnalysisDynamicsTab } from './analysis/AnalysisDynamicsTab';
import { AnalysisRiskTab } from './analysis/AnalysisRiskTab';
import { AnalysisMicrostructureTab } from './analysis/AnalysisMicrostructureTab';
import { ShimmerButton } from '@/components/ui/ShimmerButton';
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
    return list.length > 0 ? list : [30, 60, 90];
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

  // Tier 3: News Impact Curve (Symmetric vs GJR-GARCH)
  const newsImpactPoints = useMemo(() => {
    return computeNewsImpactCurve(returns);
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
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Parameters Header Card */}
      <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
        <form onSubmit={handleTickerSubmit} className="space-y-4">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-100 dark:border-slate-800/80">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono font-semibold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
                  Single-Asset Econometric Workstation
                </span>
                <span className="text-slate-300 dark:text-slate-700" aria-hidden="true">·</span>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {companyName} ({ticker})
                </span>
              </div>
              <h2 className="text-xl font-bold tracking-tight text-slate-900 dark:text-slate-100 mt-0.5">
                Econometric Distributions, Volatility Dynamics &amp; Tail Risk
              </h2>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              {/* Resolution Segmented Control */}
              <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg">
                <button
                  type="button"
                  onClick={() => onResolutionChange('daily')}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors cursor-pointer ${
                    resolution === 'daily'
                      ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-xs'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                  }`}
                >
                  Daily (1Y)
                </button>
                <button
                  type="button"
                  onClick={() => onResolutionChange('intraday')}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors flex items-center gap-1.5 cursor-pointer ${
                    resolution === 'intraday'
                      ? 'bg-emerald-500 text-slate-950 font-bold shadow-xs'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                  }`}
                >
                  <Zap className="w-3.5 h-3.5" />
                  Intraday 1-Min (7D)
                </button>
              </div>

              <ShimmerButton
                type="submit"
                disabled={isLoading}
                className="px-4 py-2"
              >
                {isLoading ? (
                  <>
                    <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                    Calculating...
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current text-amber-500" />
                    <span>Run Econometrics</span>
                  </>
                )}
              </ShimmerButton>
            </div>
          </div>

          {/* Input Controls Row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 text-xs">
            {/* Ticker Input */}
            <div>
              <label className="block text-slate-600 dark:text-slate-400 font-medium mb-1">
                Ticker Symbol
              </label>
              <input
                type="text"
                value={localTicker}
                onChange={(e) => setLocalTicker(e.target.value.toUpperCase())}
                placeholder="AAPL, MSFT..."
                className="w-full px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 font-mono font-bold text-slate-900 dark:text-slate-100 focus:outline-hidden focus:ring-1 focus:ring-emerald-500"
              />
            </div>

            {/* Return Type */}
            <div>
              <label className="block text-slate-600 dark:text-slate-400 font-medium mb-1">
                Return Type
              </label>
              <select
                value={returnType}
                onChange={(e) => onReturnTypeChange(e.target.value as 'simple' | 'log')}
                className="w-full px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 font-medium focus:outline-hidden focus:ring-1 focus:ring-emerald-500"
              >
                <option value="log">Log (Continuously Compounded)</option>
                <option value="simple">Simple Arithmetic (ΔP/P)</option>
              </select>
            </div>

            {/* Start Date */}
            <div>
              <label className="block text-slate-600 dark:text-slate-400 font-medium mb-1">
                Start Date
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => onStartDateChange(e.target.value)}
                className="w-full px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100 focus:outline-hidden focus:ring-1 focus:ring-emerald-500"
              />
            </div>

            {/* End Date */}
            <div>
              <label className="block text-slate-600 dark:text-slate-400 font-medium mb-1">
                End Date
              </label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => onEndDateChange(e.target.value)}
                className="w-full px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100 focus:outline-hidden focus:ring-1 focus:ring-emerald-500"
              />
            </div>

            {/* Rolling Volatility Window(s) */}
            <div>
              <label className="block text-slate-600 dark:text-slate-400 font-medium mb-1">
                Rolling Windows (days)
              </label>
              <input
                type="text"
                value={rollingWindows}
                onChange={(e) => onRollingWindowsChange(e.target.value)}
                placeholder="30,60,90"
                className="w-full px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100 focus:outline-hidden focus:ring-1 focus:ring-emerald-500"
              />
            </div>

            {/* Decomposition Settings */}
            <div>
              <label className="block text-slate-600 dark:text-slate-400 font-medium mb-1">
                Decomp Period &amp; Model
              </label>
              <div className="flex items-center gap-1.5">
                <input
                  type="number"
                  min="5"
                  max="120"
                  value={decompPeriod}
                  onChange={(e) => onDecompPeriodChange(Math.max(5, parseInt(e.target.value, 10) || 30))}
                  className="w-16 px-2 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 font-mono text-center text-slate-900 dark:text-slate-100 focus:outline-hidden focus:ring-1 focus:ring-emerald-500"
                  title="Period (e.g. 21 days)"
                />
                <select
                  value={decompModel}
                  onChange={(e) => onDecompModelChange(e.target.value as 'additive' | 'multiplicative')}
                  className="flex-1 px-2 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:outline-hidden focus:ring-1 focus:ring-emerald-500"
                >
                  <option value="additive">Additive</option>
                  <option value="multiplicative">Multiplicative</option>
                </select>
              </div>
            </div>
          </div>
        </form>

        {/* Popular Ticker Quick Selector */}
        <div className="flex items-center gap-2 mt-3 pt-3 border-t border-slate-100 dark:border-slate-800/60 text-xs">
          <span className="text-slate-400">Popular Tickers:</span>
          {['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'JPM'].map((t) => (
            <button
              key={t}
              onClick={() => {
                setLocalTicker(t);
                onTickerChange(t);
                onRunAnalysis();
              }}
              className={`hover:text-emerald-500 font-mono transition-colors cursor-pointer ${
                ticker === t ? 'text-emerald-500 font-bold' : 'text-slate-500 dark:text-slate-400'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Common Sample & Data Honesty Banner */}
      <div className="bg-slate-50 dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/90 rounded-xl px-4 py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2.5">
          <Calendar className="w-4 h-4 text-emerald-500 shrink-0" />
          <div className="text-slate-700 dark:text-slate-300">
            <span className="font-semibold text-slate-900 dark:text-slate-100">
              Sample Bounds:
            </span>{' '}
            Analyzing{' '}
            <strong className="font-mono text-emerald-600 dark:text-emerald-400">
              {filteredBars.length}
            </strong>{' '}
            daily observations ({dates[0] || startDate} → {dates[dates.length - 1] || endDate}) ·{' '}
            <strong className="font-mono text-slate-800 dark:text-slate-200">
              {intradayBars.length}
            </strong>{' '}
            1-minute intraday bars.
          </div>
        </div>

        <div className="flex items-center gap-3 font-mono text-[11px] text-slate-500 dark:text-slate-400">
          <span>
            Ann Vol: <strong>{(returnStats.annualizedVol * 100).toFixed(2)}%</strong>
          </span>
          <span className="text-slate-300 dark:text-slate-700" aria-hidden="true">·</span>
          <span>
            Kurtosis: <strong>{returnStats.excessKurtosis.toFixed(2)}</strong>
          </span>
          <span className="text-slate-300 dark:text-slate-700" aria-hidden="true">·</span>
          <span>
            Hurst H: <strong>{hurstAndVr.hurstExponent.toFixed(3)}</strong>
          </span>
        </div>
      </div>

      {/* Segmented Sub-Tab Navigation Strip */}
      <div className="flex items-center gap-1 border-b border-slate-200 dark:border-slate-800 pb-0 overflow-x-auto">
        <button
          type="button"
          onClick={() => setActiveSubTab('overview')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all whitespace-nowrap cursor-pointer ${
            activeSubTab === 'overview'
              ? 'border-emerald-500 text-emerald-600 dark:text-emerald-400 bg-emerald-50/40 dark:bg-emerald-950/20'
              : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:border-slate-300'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>Overview &amp; Time Series</span>
          <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
            Tier 1
          </span>
        </button>

        <button
          type="button"
          onClick={() => setActiveSubTab('stationarity')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all whitespace-nowrap cursor-pointer ${
            activeSubTab === 'stationarity'
              ? 'border-emerald-500 text-emerald-600 dark:text-emerald-400 bg-emerald-50/40 dark:bg-emerald-950/20'
              : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:border-slate-300'
          }`}
        >
          <Scale className="w-4 h-4" />
          <span>Stationarity &amp; Distributions</span>
          <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
            Tier 2
          </span>
        </button>

        <button
          type="button"
          onClick={() => setActiveSubTab('dynamics')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all whitespace-nowrap cursor-pointer ${
            activeSubTab === 'dynamics'
              ? 'border-emerald-500 text-emerald-600 dark:text-emerald-400 bg-emerald-50/40 dark:bg-emerald-950/20'
              : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:border-slate-300'
          }`}
        >
          <Activity className="w-4 h-4" />
          <span>Volatility Dynamics</span>
          <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
            Tier 3
          </span>
        </button>

        <button
          type="button"
          onClick={() => setActiveSubTab('risk')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all whitespace-nowrap cursor-pointer ${
            activeSubTab === 'risk'
              ? 'border-emerald-500 text-emerald-600 dark:text-emerald-400 bg-emerald-50/40 dark:bg-emerald-950/20'
              : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:border-slate-300'
          }`}
        >
          <ShieldAlert className="w-4 h-4" />
          <span>Risk &amp; Drawdowns</span>
          <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
            Tier 4
          </span>
        </button>

        <button
          type="button"
          onClick={() => setActiveSubTab('microstructure')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all whitespace-nowrap cursor-pointer ${
            activeSubTab === 'microstructure'
              ? 'border-emerald-500 text-emerald-600 dark:text-emerald-400 bg-emerald-50/40 dark:bg-emerald-950/20'
              : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:border-slate-300'
          }`}
        >
          <Waves className="w-4 h-4" />
          <span>Microstructure &amp; Liquidity</span>
          <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
            Tier 5
          </span>
        </button>
      </div>

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
