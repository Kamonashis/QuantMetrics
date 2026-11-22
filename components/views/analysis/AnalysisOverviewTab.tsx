'use client';

import React, { useState } from 'react';
import {
  PriceBar,
  ReturnStats,
  DecompositionResult,
  RollingVolSeries,
  AdfTestResult,
} from '@/lib/quant-engine';
import {
  RollingMomentsResult,
  StructuralBreakResult,
  HurstVarianceRatioResult,
  InnovationDiagnostics,
} from '@/lib/analysis-engine';
import { TimeSeriesChart } from '@/components/charts/TimeSeriesChart';
import { ReturnsChart } from '@/components/charts/ReturnsChart';
import { RollingVolatilityChart } from '@/components/charts/RollingVolatilityChart';
import { DecompositionChart } from '@/components/charts/DecompositionChart';
import { RegimeTimeline, regimeNames } from '@/components/charts/RegimeTimeline';
import type { Data, Layout } from 'plotly.js';
import { RegimeChip } from '@/components/ui/RegimeChip';
import { RegimeTransitionMatrix } from '@/components/ui/RegimeTransitionMatrix';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens } from '@/lib/plotly-theme';
import { AlgoSelect } from '@/components/ui/AlgoSelect';
import type { HmmFit } from '@/lib/hmm';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import {
  VarianceRatioEquation,
} from '@/components/math/MathEquations';
import {
  TrendingUp,
  Activity,
  Layers,
  Download,
  AlertTriangle,
  Info,
} from 'lucide-react';

interface AnalysisOverviewTabProps {
  ticker: string;
  companyName: string;
  filteredBars: PriceBar[];
  dates: string[];
  prices: number[];
  returns: number[];
  returnType: 'simple' | 'log';
  onReturnTypeChange: (type: 'simple' | 'log') => void;
  returnStats: ReturnStats;
  rollingVolSeries: RollingVolSeries[];
  rollingMoments: RollingMomentsResult;
  structuralBreaks: StructuralBreakResult;
  hurstAndVr: HurstVarianceRatioResult;
  /** σ̂-standardized battery (GARCH conditional variance filter) — optional */
  innovation?: InnovationDiagnostics | null;
  decompResult: DecompositionResult;
  adfResult: AdfTestResult;
  decompPeriod: number;
  onDecompPeriodChange: (p: number) => void;
  decompModel: 'additive' | 'multiplicative';
  onDecompModelChange: (m: 'additive' | 'multiplicative') => void;
  cycleBand: { min: number; max: number };
  onCycleBandChange: (band: { min: number; max: number }) => void;
  onExportCsv: () => void;
  hmmFit?: HmmFit | null;
  hmmK?: 2 | 3 | 4;
  onHmmKChange?: (k: 2 | 3 | 4) => void;
  /** session date per posterior — dates[i] aligns with hmmFit.posteriors[i] */
  hmmDates?: string[];
}

export function AnalysisOverviewTab({
  ticker,
  companyName,
  filteredBars,
  dates,
  prices,
  returns,
  returnType,
  onReturnTypeChange,
  returnStats,
  rollingVolSeries,
  rollingMoments,
  structuralBreaks,
  hurstAndVr,
  innovation = null,
  decompResult,
  adfResult,
  decompPeriod,
  onDecompPeriodChange,
  decompModel,
  onDecompModelChange,
  cycleBand,
  onCycleBandChange,
  onExportCsv,
  hmmFit,
  hmmK,
  onHmmKChange,
  hmmDates,
}: AnalysisOverviewTabProps) {
  const [skewHoverIdx, setSkewHoverIdx] = useState<number | null>(null);
  // Memory battery: raw returns vs σ̂-standardized innovation diagnostic (GARCH filter)
  const [memMode, setMemMode] = useState<'raw' | 'std'>('raw');
  const memStdActive = memMode === 'std' && innovation !== null && innovation !== undefined;
  const mem = memStdActive
    ? { h: innovation.hurstExponent, regime: innovation.hurstRegime, vrs: innovation.varianceRatios, caption: 'σ̂-standardized (r / √(GARCH σ̂²ₜ)) — genuine memory vs vol-clustering artifact' }
    : { h: hurstAndVr.hurstExponent, regime: hurstAndVr.hurstRegime, vrs: hurstAndVr.varianceRatios, caption: 'raw returns — vol clustering contaminates H' };

  // Format price bars for TimeSeriesChart
  const priceChartBars = React.useMemo(() => {
    return filteredBars.map((b) => ({
      date: b.date,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
      volume: b.volume,
    }));
  }, [filteredBars]);

  const isDark = usePlotlyDarkTheme();
  const tokens = React.useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  // Rolling skew data for Plotly
  const skewData = rollingMoments.rollingSkew;
  const nSkew = skewData.length;
  const skewActiveIdx = skewHoverIdx !== null && skewHoverIdx < nSkew ? skewHoverIdx : nSkew - 1;
  const activeSkewPoint = skewData[skewActiveIdx] || { date: '', skew: 0, lowerCi: 0, upperCi: 0 };

  const { skewTraces, skewLayout } = React.useMemo<{
    skewTraces: Data[];
    skewLayout: Partial<Layout>;
  }>(() => {
    if (skewData.length === 0) return { skewTraces: [], skewLayout: {} };
    const xs = skewData.map((d) => d.date);
    return {
      skewTraces: [
        {
          type: 'scatter',
          mode: 'lines',
          name: 'Upper 95% CI',
          x: xs,
          y: skewData.map((d) => d.upperCi),
          line: { color: '#64748b', width: 1, dash: 'dot' },
          hovertemplate: 'Upper CI: %{y:.3f}<extra></extra>',
          showlegend: false,
        },
        {
          type: 'scatter',
          mode: 'lines',
          name: '95% Confidence Band',
          x: xs,
          y: skewData.map((d) => d.lowerCi),
          fill: 'tonexty',
          fillcolor: 'rgba(16,185,129,0.12)',
          line: { color: '#64748b', width: 1, dash: 'dot' },
          hovertemplate: 'Lower CI: %{y:.3f}<extra></extra>',
        },
        {
          type: 'scatter',
          mode: 'lines',
          name: '60d Skew',
          x: xs,
          y: skewData.map((d) => d.skew),
          line: { color: '#10b981', width: 1.8 },
          hovertemplate: '60d Skew: %{y:.3f}<extra></extra>',
        },
      ],
      skewLayout: {
        margin: { l: 48, r: 16, t: 10, b: 36 },
        shapes: [
          {
            type: 'line',
            xref: 'paper',
            x0: 0,
            x1: 1,
            yref: 'y',
            y0: 0,
            y1: 0,
            line: { color: tokens.zeroLine, width: 1, dash: 'dot' },
          },
        ],
        yaxis: {
          title: { text: 'Skewness', font: { size: 10, color: tokens.textMuted } },
        },
      },
    };
  }, [skewData, tokens]);

  return (
    <div className="space-y-6">
      {/* Structural Break Alert Banner if detected */}
      {structuralBreaks.hasSignificantBreak && (
        <div className="bg-gold-500/10 border border-gold-500/30 rounded-xl p-4 text-xs text-gold-900 dark:text-gold-200 flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 text-gold-500 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="font-semibold text-gold-800 dark:text-gold-100 flex items-center gap-2">
              <span>CUSUM-of-Squares Volatility Regime Shift Detected</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-gold-500/20 text-gold-700 dark:text-gold-300 font-mono">
                p &lt; 0.05
              </span>
            </div>
            <p className="text-gold-700 dark:text-gold-300 leading-relaxed">
              Brown-Durbin-Evans recursive CUSUM-sq identified{' '}
              <strong>{structuralBreaks.breakDates.length}</strong> variance change point(s). Most prominent on{' '}
              <strong className="font-mono">{structuralBreaks.breakDates[0]?.date}</strong>. Historical stationarity
              assumptions prior to this boundary may be biased.
            </p>
          </div>
        </div>
      )}

      {/* Primary Price Trajectory & Volume */}
      <div className="panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-zinc-100 dark:border-zinc-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-gold-500" />
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Asset Price Trajectory &amp; CUSUM Variance Regime Shifts
              </h3>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              {ticker} · {companyName} · {dates.length} trading days with recursive cumulative sum bounds
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onExportCsv}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-zinc-700 dark:text-zinc-200 text-xs font-medium hover:border-gold-500 transition-colors cursor-pointer"
            >
              <Download className="w-3.5 h-3.5 text-zinc-400" />
              <span>Export CSV</span>
            </button>
          </div>
        </div>

        <div className="mt-4">
          <ChartFullscreen title="Price Trajectory & Volatility Regime Shifts" subtitle={`${ticker} · ${companyName}`}>
            <TimeSeriesChart
              bars={priceChartBars}
              breakPoints={structuralBreaks.breakDates}
              cusumSqPoints={structuralBreaks.cusumSqPoints}
              title="Price Trajectory & Volatility Regime Shifts"
              subtitle={`${ticker} · ${companyName}`}
            />
          </ChartFullscreen>
        </div>

        {/* HMM Volatility Regimes — timeline below price chart */}
        <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800/60">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-2">
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[10px] font-mono uppercase tracking-[0.16em] text-gold-600 dark:text-gold-400">Volatility Regimes (HMM)</span>
                <RegimeChip fit={hmmFit ?? null} />
              </div>
              <p className="text-[10px] text-zinc-500 dark:text-zinc-400 mt-0.5">
                Unsupervised latent states, discovered not labelled — calm→crisis ranked by mean GK-RV · K = {hmmK ?? 3} · BIC {hmmFit ? hmmFit.bic.toFixed(0) : '—'} · {hmmFit?.converged ? `converged in ${hmmFit.iterations} iters` : 'did not converge'} · HMM ranks vs CUSUM heuristic
              </p>
            </div>
            {onHmmKChange && (
              <AlgoSelect
                ariaLabel="HMM state count"
                value={String(hmmK ?? 3)}
                onChange={(v) => onHmmKChange(Number(v) as 2 | 3 | 4)}
                options={[
                  { value: '2', label: '2 states', hint: 'Calm | Stress' },
                  { value: '3', label: '3 states', hint: 'Calm | Normal | Crisis' },
                  { value: '4', label: '4 states', hint: 'Calm | Watchful | Normal | Crisis' },
                ]}
                className="w-52"
              />
            )}
          </div>
          {hmmFit ? (
            <>
              <RegimeTimeline posteriors={hmmFit.posteriors} labels={regimeNames(hmmFit.k)} height={44} dates={hmmDates} />
              <RegimeTransitionMatrix fit={hmmFit} className="mt-2 border-t border-zinc-100 dark:border-zinc-800/60 pt-2" labels={regimeNames(hmmFit.k)} />
            </>
          ) : (
            <div className="rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] p-4 text-center font-mono text-[10px] text-zinc-500">
              HMM did not converge on this window — regimes suppressed rather than fabricated.
            </div>
          )}
        </div>

        {/* Structural Break Dates List */}
        {structuralBreaks.breakDates.length > 0 && (
          <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 flex flex-wrap items-center gap-3 text-xs">
            <span className="text-zinc-400 font-mono text-[11px]">Break Points:</span>
            {structuralBreaks.breakDates.map((b) => (
              <span
                key={b.date}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-gold-50 dark:bg-gold-950/40 border border-gold-200 dark:border-gold-800/60 font-mono text-gold-800 dark:text-gold-300 text-[11px]"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-gold-500" />
                {b.date} ({b.reason})
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Return Series & Return Type Switcher */}
      <div className="panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-zinc-100 dark:border-zinc-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-gold-500" />
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Empirical Return Innovations (r_t)
              </h3>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Comparing arithmetic vs. continuously compounded logarithmic returns
            </p>
          </div>

          <div className="flex items-center gap-2">
            <div className="scrollbar-none inline-flex max-w-full overflow-x-auto p-0.5 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 text-xs">
              <button
                type="button"
                onClick={() => onReturnTypeChange('log')}
                className={`shrink-0 px-3 py-1 rounded-md font-medium transition-colors cursor-pointer ${
                  returnType === 'log'
                    ? 'bg-white dark:bg-zinc-800 text-gold-600 dark:text-gold-400 shadow-xs'
                    : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900'
                }`}
              >
                Log Return (ln P_t/P_t-1)
              </button>
              <button
                type="button"
                onClick={() => onReturnTypeChange('simple')}
                className={`shrink-0 px-3 py-1 rounded-md font-medium transition-colors cursor-pointer ${
                  returnType === 'simple'
                    ? 'bg-white dark:bg-zinc-800 text-gold-600 dark:text-gold-400 shadow-xs'
                    : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900'
                }`}
              >
                Simple Return (ΔP/P)
              </button>
            </div>
          </div>
        </div>

        <div className="mt-4">
          <ChartFullscreen title="Empirical Return Innovations">
            <ReturnsChart dates={dates.slice(1)} returns={returns} returnType={returnType} />
          </ChartFullscreen>
        </div>
      </div>

      {/* Distribution Moments & Confidence Intervals Panel */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Mean */}
        <div className="bg-white dark:bg-[#101014] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
            <span>Daily Mean (μ)</span>
            <span className="font-mono text-[10px]">Ann. {(returnStats.mean * 252 * 100).toFixed(2)}%</span>
          </div>
          <div className="mt-2 text-2xl font-semibold font-mono text-zinc-900 dark:text-zinc-100">
            {(returnStats.mean * 100).toFixed(3)}%
          </div>
          <div className="mt-2 text-[11px] text-zinc-400 flex items-center justify-between">
            <span>Range: [{(returnStats.min * 100).toFixed(2)}%, {(returnStats.max * 100).toFixed(2)}%]</span>
            <span>n = {returns.length}</span>
          </div>
        </div>

        {/* Volatility */}
        <div className="bg-white dark:bg-[#101014] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
            <span>Annualized Volatility (σ_ann)</span>
            <span className="font-mono text-[10px]">Daily {(returnStats.stdDev * 100).toFixed(2)}%</span>
          </div>
          <div className="mt-2 text-2xl font-semibold font-mono text-gold-600 dark:text-gold-400">
            {(returnStats.annualizedVol * 100).toFixed(2)}%
          </div>
          <div className="mt-2 text-[11px] text-zinc-400 flex items-center justify-between">
            <span>Variance: {(returnStats.variance * 10000).toFixed(2)} bps²</span>
            <span>√252 Scale</span>
          </div>
        </div>

        {/* Skewness with Confidence Interval */}
        <div className="bg-white dark:bg-[#101014] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
            <span>Skewness (S)</span>
            <span className="font-mono text-[10px]">Lo (2002) 95% CI</span>
          </div>
          <div className="mt-2 text-2xl font-semibold font-mono text-zinc-900 dark:text-zinc-100">
            {returnStats.skewness.toFixed(3)}
          </div>
          <div className="mt-2 text-[11px] font-mono text-zinc-400">
            CI: [{(returnStats.skewness - 1.96 * Math.sqrt(6 / Math.max(1, returns.length))).toFixed(2)},{' '}
            {(returnStats.skewness + 1.96 * Math.sqrt(6 / Math.max(1, returns.length))).toFixed(2)}]
          </div>
        </div>

        {/* Excess Kurtosis with Confidence Interval */}
        <div className="bg-white dark:bg-[#101014] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
            <span>Excess Kurtosis (K)</span>
            <span className="font-mono text-[10px]">Normal = 0.0</span>
          </div>
          <div className="mt-2 text-2xl font-semibold font-mono text-gold-600 dark:text-gold-400">
            {returnStats.excessKurtosis.toFixed(3)}
          </div>
          <div className="mt-2 text-[11px] font-mono text-zinc-400">
            CI: [{(returnStats.excessKurtosis - 1.96 * Math.sqrt(24 / Math.max(1, returns.length))).toFixed(2)},{' '}
            {(returnStats.excessKurtosis + 1.96 * Math.sqrt(24 / Math.max(1, returns.length))).toFixed(2)}]
          </div>
        </div>
      </div>

      {/* Rolling Moments with Lo (2002) Confidence Bands */}
      <div className="grid-cards">
        {/* Rolling Volatility */}
        <div>
          <ChartFullscreen title="Rolling Annualized Volatility">
            <RollingVolatilityChart series={rollingVolSeries} />
          </ChartFullscreen>
        </div>

        {/* Rolling Skewness with Confidence Interval */}
        <div className="panel p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Rolling Skewness (60-Day) &amp; Lo (2002) Confidence Bands
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Negative skew breach indicates emergent left-tail crash hazard
              </p>
            </div>
            <div className="font-mono text-xs text-zinc-400">
              {activeSkewPoint.date}: <strong className="text-gold-500">{activeSkewPoint.skew.toFixed(3)}</strong>
            </div>
          </div>

          <div className="mt-4">
            <ChartFullscreen title="Rolling Skewness (60-Day) & Lo (2002) Confidence Bands">
              <PlotlyChart
                data={skewTraces}
                layout={skewLayout}
                plotType="2d"
                rangeSlider
                hovermode="x unified"
                showLegend={false}
                height={240}
                showWheelHint
                exportFilename={`${ticker}-rolling-skewness`}
                uirevisionKey={`skew-${ticker}-${nSkew}`}
                onHover={(ev) => {
                  const pt = (ev as { points?: Array<{ pointIndex?: number }> })?.points?.[0];
                  if (pt && typeof pt.pointIndex === 'number') setSkewHoverIdx(pt.pointIndex);
                }}
                onUnhover={() => setSkewHoverIdx(null)}
              />
            </ChartFullscreen>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-[11px] text-zinc-400 font-mono mt-2">
            <span className="tabular-nums">{skewData[0]?.date}</span>
            <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-0.5 order-3 sm:order-none w-full sm:w-auto">
              <span className="inline-flex items-center gap-1">
                <span className="w-2.5 h-0.5 bg-gold-500 inline-block" /> 60d Skew
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="w-2.5 h-0.5 bg-zinc-400 border-b border-dashed inline-block" /> 95% Confidence Band
              </span>
            </div>
            <span className="tabular-nums">{skewData[nSkew - 1]?.date}</span>
          </div>
        </div>
      </div>

      {/* Hurst Exponent & Lo-MacKinlay Variance Ratio Battery */}
      <div className="panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-zinc-100 dark:border-zinc-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-gold-500" />
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Memory Structure &amp; Random Walk Tests: Hurst &amp; Lo-MacKinlay Variance Ratio
              </h3>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Evaluating martingale difference hypothesis: VR(q) = 1 under pure Brownian motion
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
            <div className="inline-flex items-center rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] p-0.5 text-[9px] font-bold uppercase tracking-[0.16em]">
              {(['raw', 'std'] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMemMode(m)}
                  aria-pressed={memMode === m}
                  className={`cursor-pointer rounded-md px-2.5 py-1 transition-colors ${
                    memMode === m
                      ? 'bg-gold-500/15 text-gold-800 dark:text-gold-200'
                      : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-100'
                  }`}
                >
                  {m === 'raw' ? 'Raw returns' : 'σ̂-std'}
                </button>
              ))}
            </div>
            <span>
              Hurst H:{' '}
              <strong className="text-gold-600 dark:text-gold-400 text-sm">
                {mem.h.toFixed(3)}
              </strong>
            </span>
            <span className="text-zinc-300 dark:text-zinc-700">·</span>
            <span className="text-zinc-600 dark:text-zinc-300 font-medium">
              {mem.regime}
            </span>
          </div>
        </div>
        <p className="mt-1 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400 dark:text-zinc-500">
          {mem.caption}
          {memStdActive && innovation ? ` · ADF p ${innovation.adfPValue < 0.001 ? '< 0.001' : innovation.adfPValue.toFixed(4)} (residuals) · min VR z ${innovation.vrMinZ.toFixed(2)}` : ''}
        </p>

        <div className="mt-4 overflow-x-auto">
          <table className="table-sticky-first w-full text-xs text-left">
            <thead>
              <tr className="border-b border-zinc-200 dark:border-zinc-800 text-zinc-400 font-mono">
                <th className="py-2.5 px-3">Horizon Lag (q)</th>
                <th className="py-2.5 px-3">Variance Ratio VR(q)</th>
                <th className="py-2.5 px-3">Lo-MacKinlay z-Statistic</th>
                <th className="py-2.5 px-3">Asymptotic p-Value</th>
                <th className="py-2.5 px-3">Statistical Regime Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60 font-mono">
              {mem.vrs.map((vr) => (
                <tr key={vr.q} className="hover:bg-zinc-50/50 dark:hover:bg-zinc-900/30">
                  <td className="py-2.5 px-3 font-semibold text-zinc-900 dark:text-zinc-100">
                    q = {vr.q} days
                  </td>
                  <td className="py-2.5 px-3 font-semibold text-gold-600 dark:text-gold-400">
                    {vr.vr.toFixed(3)}
                  </td>
                  <td className="py-2.5 px-3 text-zinc-600 dark:text-zinc-300">
                    {vr.zStat > 0 ? `+${vr.zStat.toFixed(2)}` : vr.zStat.toFixed(2)}
                  </td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[11px] ${
                        vr.pValue < 0.05
                          ? 'bg-gold-100 dark:bg-gold-950/40 text-gold-800 dark:text-gold-300'
                          : 'bg-gold-100 dark:bg-gold-950/40 text-gold-800 dark:text-gold-300'
                      }`}
                    >
                      {vr.pValue < 0.001 ? '< 0.001' : vr.pValue.toFixed(3)}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-zinc-700 dark:text-zinc-300 font-sans">
                    {vr.vr > 1.05
                      ? 'Positive serial correlation (Momentum persistence)'
                      : vr.vr < 0.95
                      ? 'Negative serial correlation (Mean-reverting tendencies)'
                      : 'Consistent with Random Walk (Martingale difference)'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 text-xs text-zinc-500 dark:text-zinc-400 flex items-center justify-between gap-3">
          <div className="flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-zinc-400" />
            <span>
              {memMode === 'raw'
                ? hurstAndVr.randomWalkVerdict
                : 'Innovation view — Ĥ ≈ 0.5 ⇒ raw H was a vol-clustering artifact (no momentum tilt); Ĥ ≳ 0.6 ⇒ serial dependence survives vol normalization.'}
            </span>
          </div>
          <VarianceRatioEquation className="hidden md:inline" />
        </div>
      </div>

      {/* Structural Time Series Decomposition & TCN-Attention Forecast */}
      <div className="panel p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between pb-4 border-b border-zinc-100 dark:border-zinc-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-gold-500" />
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Structural Decomposition (Trend · Cyclical · Seasonal · Residual) &amp; TCN-Attention Forecast
              </h3>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Fourier bandpass cyclical extraction with component-wise Dilated Causal TCN + Multi-Head Self-Attention next-day price reconstruction
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-1.5">
              <label htmlFor="overview-decomp-period" className="text-xs text-zinc-500 font-medium">
                Seasonal:
              </label>
              <select
                id="overview-decomp-period"
                value={decompPeriod}
                onChange={(e) => onDecompPeriodChange(parseInt(e.target.value, 10))}
                className="px-2 py-1 rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-xs font-mono"
              >
                <option value={5}>5d (Weekly)</option>
                <option value={21}>21d (Monthly)</option>
                <option value={30}>30d (6-Week)</option>
                <option value={63}>63d (Quarterly)</option>
              </select>
            </div>

            <div className="flex items-center gap-1.5">
              <label htmlFor="overview-cycle-band" className="text-xs text-zinc-500 font-medium">
                Cycle Band:
              </label>
              <select
                id="overview-cycle-band"
                value={`${cycleBand.min}-${cycleBand.max}`}
                onChange={(e) => {
                  const [minStr, maxStr] = e.target.value.split('-');
                  onCycleBandChange({
                    min: parseInt(minStr, 10),
                    max: parseInt(maxStr, 10),
                  });
                }}
                className="px-2 py-1 rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-xs font-mono"
              >
                <option value="15-63">15–63d (Short)</option>
                <option value="25-126">25–126d (Business)</option>
                <option value="40-180">40–180d (Macro)</option>
              </select>
            </div>

            <div className="scrollbar-none inline-flex max-w-full overflow-x-auto p-0.5 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 text-xs">
              <button
                type="button"
                onClick={() => onDecompModelChange('multiplicative')}
                className={`shrink-0 px-2.5 py-1 rounded-md font-medium transition-colors cursor-pointer ${
                  decompModel === 'multiplicative'
                    ? 'bg-white dark:bg-zinc-800 text-gold-600 dark:text-gold-400 shadow-xs'
                    : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
                }`}
              >
                Multiplicative
              </button>
              <button
                type="button"
                onClick={() => onDecompModelChange('additive')}
                className={`shrink-0 px-2.5 py-1 rounded-md font-medium transition-colors cursor-pointer ${
                  decompModel === 'additive'
                    ? 'bg-white dark:bg-zinc-800 text-gold-600 dark:text-gold-400 shadow-xs'
                    : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
                }`}
              >
                Additive
              </button>
            </div>
          </div>
        </div>

        <div className="mt-4">
          <ChartFullscreen title="Structural Decomposition & TCN-Attention Forecast">
            <DecompositionChart ticker={ticker} decomposition={decompResult} adfTest={adfResult} />
          </ChartFullscreen>
        </div>
      </div>
    </div>
  );
}
