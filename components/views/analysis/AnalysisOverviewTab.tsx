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
} from '@/lib/analysis-engine';
import { TimeSeriesChart } from '@/components/charts/TimeSeriesChart';
import { ReturnsChart } from '@/components/charts/ReturnsChart';
import { RollingVolatilityChart } from '@/components/charts/RollingVolatilityChart';
import { DecompositionChart } from '@/components/charts/DecompositionChart';
import {
  SkewnessEquation,
  KurtosisEquation,
  AnnualizedVolEquation,
  VarianceRatioEquation,
  DecompositionAdditiveEquation,
  DecompositionMultiplicativeEquation,
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
  decompResult: DecompositionResult;
  adfResult: AdfTestResult;
  decompPeriod: number;
  onDecompPeriodChange: (p: number) => void;
  decompModel: 'additive' | 'multiplicative';
  onDecompModelChange: (m: 'additive' | 'multiplicative') => void;
  onExportCsv: () => void;
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
  decompResult,
  adfResult,
  decompPeriod,
  onDecompPeriodChange,
  decompModel,
  onDecompModelChange,
  onExportCsv,
}: AnalysisOverviewTabProps) {
  const [skewHoverIdx, setSkewHoverIdx] = useState<number | null>(null);

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

  // Rolling skew data scaling for pure SVG
  const skewData = rollingMoments.rollingSkew;
  const nSkew = skewData.length;
  let minSkew = -2.5;
  let maxSkew = 2.5;
  if (nSkew > 0) {
    for (const d of skewData) {
      if (d.lowerCi < minSkew) minSkew = d.lowerCi;
      if (d.upperCi > maxSkew) maxSkew = d.upperCi;
    }
  }
  const skewRange = maxSkew - minSkew || 1;
  const svgW = 600;
  const svgH = 220;
  const padL = 40;
  const padR = 20;
  const padT = 20;
  const padB = 30;
  const cW = svgW - padL - padR;
  const cH = svgH - padT - padB;

  const getSkewX = (i: number) => padL + (i / Math.max(1, nSkew - 1)) * cW;
  const getSkewY = (val: number) => padT + cH - ((val - minSkew) / skewRange) * cH;

  const skewActiveIdx = skewHoverIdx !== null ? skewHoverIdx : nSkew - 1;
  const activeSkewPoint = skewData[skewActiveIdx] || { date: '', skew: 0, lowerCi: 0, upperCi: 0 };

  return (
    <div className="space-y-6">
      {/* Structural Break Alert Banner if detected */}
      {structuralBreaks.hasSignificantBreak && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 text-xs text-amber-900 dark:text-amber-200 flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="font-semibold text-amber-800 dark:text-amber-100 flex items-center gap-2">
              <span>CUSUM-of-Squares Volatility Regime Shift Detected</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-700 dark:text-amber-300 font-mono">
                p &lt; 0.05
              </span>
            </div>
            <p className="text-amber-700 dark:text-amber-300 leading-relaxed">
              Brown-Durbin-Evans recursive CUSUM-sq identified{' '}
              <strong>{structuralBreaks.breakDates.length}</strong> variance change point(s). Most prominent on{' '}
              <strong className="font-mono">{structuralBreaks.breakDates[0]?.date}</strong>. Historical stationarity
              assumptions prior to this boundary may be biased.
            </p>
          </div>
        </div>
      )}

      {/* Primary Price Trajectory & Volume */}
      <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/90 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 dark:border-slate-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-emerald-500" />
              <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                Asset Price Trajectory &amp; CUSUM Variance Regime Shifts
              </h3>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              {ticker} · {companyName} · {dates.length} trading days with recursive cumulative sum bounds
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onExportCsv}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 text-xs font-medium hover:border-emerald-500 transition-colors cursor-pointer"
            >
              <Download className="w-3.5 h-3.5 text-slate-400" />
              <span>Export CSV</span>
            </button>
          </div>
        </div>

        <div className="mt-4">
          <TimeSeriesChart bars={priceChartBars} />
        </div>

        {/* Structural Break Dates List */}
        {structuralBreaks.breakDates.length > 0 && (
          <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800/60 flex flex-wrap items-center gap-3 text-xs">
            <span className="text-slate-400 font-mono text-[11px]">Break Points:</span>
            {structuralBreaks.breakDates.map((b) => (
              <span
                key={b.date}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800/60 font-mono text-amber-800 dark:text-amber-300 text-[11px]"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                {b.date} ({b.reason})
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Return Series & Return Type Switcher */}
      <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/90 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 dark:border-slate-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-500" />
              <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                Empirical Return Innovations (r_t)
              </h3>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Comparing arithmetic vs. continuously compounded logarithmic returns
            </p>
          </div>

          <div className="flex items-center gap-2">
            <div className="inline-flex p-0.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-xs">
              <button
                type="button"
                onClick={() => onReturnTypeChange('log')}
                className={`px-3 py-1 rounded-md font-medium transition-colors cursor-pointer ${
                  returnType === 'log'
                    ? 'bg-white dark:bg-slate-800 text-emerald-600 dark:text-emerald-400 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                }`}
              >
                Log Return (ln P_t/P_t-1)
              </button>
              <button
                type="button"
                onClick={() => onReturnTypeChange('simple')}
                className={`px-3 py-1 rounded-md font-medium transition-colors cursor-pointer ${
                  returnType === 'simple'
                    ? 'bg-white dark:bg-slate-800 text-emerald-600 dark:text-emerald-400 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                }`}
              >
                Simple Return (ΔP/P)
              </button>
            </div>
          </div>
        </div>

        <div className="mt-4">
          <ReturnsChart dates={dates.slice(1)} returns={returns} returnType={returnType} />
        </div>
      </div>

      {/* Distribution Moments & Confidence Intervals Panel */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Mean */}
        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/90 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
            <span>Daily Mean (μ)</span>
            <span className="font-mono text-[10px]">Ann. {(returnStats.mean * 252 * 100).toFixed(2)}%</span>
          </div>
          <div className="mt-2 text-2xl font-semibold font-mono text-slate-900 dark:text-slate-100">
            {(returnStats.mean * 100).toFixed(3)}%
          </div>
          <div className="mt-2 text-[11px] text-slate-400 flex items-center justify-between">
            <span>Range: [{(returnStats.min * 100).toFixed(2)}%, {(returnStats.max * 100).toFixed(2)}%]</span>
            <span>n = {returns.length}</span>
          </div>
        </div>

        {/* Volatility */}
        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/90 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
            <span>Annualized Volatility (σ_ann)</span>
            <span className="font-mono text-[10px]">Daily {(returnStats.stdDev * 100).toFixed(2)}%</span>
          </div>
          <div className="mt-2 text-2xl font-semibold font-mono text-emerald-600 dark:text-emerald-400">
            {(returnStats.annualizedVol * 100).toFixed(2)}%
          </div>
          <div className="mt-2 text-[11px] text-slate-400 flex items-center justify-between">
            <span>Variance: {(returnStats.variance * 10000).toFixed(2)} bps²</span>
            <span>√252 Scale</span>
          </div>
        </div>

        {/* Skewness with Confidence Interval */}
        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/90 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
            <span>Skewness (S)</span>
            <span className="font-mono text-[10px]">Lo (2002) 95% CI</span>
          </div>
          <div className="mt-2 text-2xl font-semibold font-mono text-slate-900 dark:text-slate-100">
            {returnStats.skewness.toFixed(3)}
          </div>
          <div className="mt-2 text-[11px] font-mono text-slate-400">
            CI: [{(returnStats.skewness - 1.96 * Math.sqrt(6 / Math.max(1, returns.length))).toFixed(2)},{' '}
            {(returnStats.skewness + 1.96 * Math.sqrt(6 / Math.max(1, returns.length))).toFixed(2)}]
          </div>
        </div>

        {/* Excess Kurtosis with Confidence Interval */}
        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/90 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
            <span>Excess Kurtosis (K)</span>
            <span className="font-mono text-[10px]">Normal = 0.0</span>
          </div>
          <div className="mt-2 text-2xl font-semibold font-mono text-amber-600 dark:text-amber-400">
            {returnStats.excessKurtosis.toFixed(3)}
          </div>
          <div className="mt-2 text-[11px] font-mono text-slate-400">
            CI: [{(returnStats.excessKurtosis - 1.96 * Math.sqrt(24 / Math.max(1, returns.length))).toFixed(2)},{' '}
            {(returnStats.excessKurtosis + 1.96 * Math.sqrt(24 / Math.max(1, returns.length))).toFixed(2)}]
          </div>
        </div>
      </div>

      {/* Rolling Moments with Lo (2002) Confidence Bands */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Rolling Volatility */}
        <div>
          <RollingVolatilityChart series={rollingVolSeries} />
        </div>

        {/* Rolling Skewness with Confidence Interval */}
        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/90 rounded-xl p-5 shadow-xs">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800/60">
            <div>
              <h4 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                Rolling Skewness (60-Day) &amp; Lo (2002) Confidence Bands
              </h4>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Negative skew breach indicates emergent left-tail crash hazard
              </p>
            </div>
            <div className="font-mono text-xs text-slate-400">
              {activeSkewPoint.date}: <strong className="text-emerald-500">{activeSkewPoint.skew.toFixed(3)}</strong>
            </div>
          </div>

          <div className="mt-4">
            <svg
              viewBox={`0 0 ${svgW} ${svgH}`}
              className="w-full h-56 select-none"
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const ratio = Math.max(0, Math.min(1, (mouseX - padL) / cW));
                const idx = Math.round(ratio * (nSkew - 1));
                setSkewHoverIdx(idx);
              }}
              onMouseLeave={() => setSkewHoverIdx(null)}
            >
              {/* Zero line */}
              <line
                x1={padL}
                y1={getSkewY(0)}
                x2={svgW - padR}
                y2={getSkewY(0)}
                stroke="#64748b"
                strokeDasharray="2 2"
                strokeWidth={1}
                opacity={0.4}
              />

              {/* Confidence Band Polygon */}
              {nSkew > 1 && (
                <path
                  d={`
                    M ${getSkewX(0)} ${getSkewY(skewData[0].upperCi)}
                    ${skewData.map((d, i) => `L ${getSkewX(i)} ${getSkewY(d.upperCi)}`).join(' ')}
                    ${skewData.slice().reverse().map((d, i) => `L ${getSkewX(nSkew - 1 - i)} ${getSkewY(d.lowerCi)}`).join(' ')}
                    Z
                  `}
                  fill="#10b981"
                  fillOpacity={0.12}
                />
              )}

              {/* Lower CI line */}
              {nSkew > 1 && (
                <path
                  d={skewData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getSkewX(i)} ${getSkewY(d.lowerCi)}`).join(' ')}
                  fill="none"
                  stroke="#64748b"
                  strokeDasharray="3 3"
                  strokeWidth={1}
                />
              )}

              {/* Upper CI line */}
              {nSkew > 1 && (
                <path
                  d={skewData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getSkewX(i)} ${getSkewY(d.upperCi)}`).join(' ')}
                  fill="none"
                  stroke="#64748b"
                  strokeDasharray="3 3"
                  strokeWidth={1}
                />
              )}

              {/* Center Skew line */}
              {nSkew > 1 && (
                <path
                  d={skewData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getSkewX(i)} ${getSkewY(d.skew)}`).join(' ')}
                  fill="none"
                  stroke="#10b981"
                  strokeWidth={2}
                />
              )}

              {/* Active hover vertical line */}
              {skewActiveIdx >= 0 && skewActiveIdx < nSkew && (
                <line
                  x1={getSkewX(skewActiveIdx)}
                  y1={padT}
                  x2={getSkewX(skewActiveIdx)}
                  y2={svgH - padB}
                  stroke="#94a3b8"
                  strokeWidth={1}
                  strokeDasharray="2 2"
                />
              )}
            </svg>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono mt-2">
            <span>{skewData[0]?.date}</span>
            <div className="flex items-center gap-3">
              <span className="inline-flex items-center gap-1">
                <span className="w-2.5 h-0.5 bg-emerald-500 inline-block" /> 60d Skew
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="w-2.5 h-0.5 bg-slate-400 border-b border-dashed inline-block" /> 95% Confidence Band
              </span>
            </div>
            <span>{skewData[nSkew - 1]?.date}</span>
          </div>
        </div>
      </div>

      {/* Hurst Exponent & Lo-MacKinlay Variance Ratio Battery */}
      <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/90 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 dark:border-slate-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-500" />
              <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                Memory Structure &amp; Random Walk Tests: Hurst &amp; Lo-MacKinlay Variance Ratio
              </h3>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Evaluating martingale difference hypothesis: VR(q) = 1 under pure Brownian motion
            </p>
          </div>

          <div className="flex items-center gap-3 font-mono text-xs">
            <span>
              Hurst H:{' '}
              <strong className="text-emerald-600 dark:text-emerald-400 text-sm">
                {hurstAndVr.hurstExponent.toFixed(3)}
              </strong>
            </span>
            <span className="text-slate-300 dark:text-slate-700">·</span>
            <span className="text-slate-600 dark:text-slate-300 font-medium">
              {hurstAndVr.hurstRegime}
            </span>
          </div>
        </div>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-400 font-mono">
                <th className="py-2.5 px-3">Horizon Lag (q)</th>
                <th className="py-2.5 px-3">Variance Ratio VR(q)</th>
                <th className="py-2.5 px-3">Lo-MacKinlay z-Statistic</th>
                <th className="py-2.5 px-3">Asymptotic p-Value</th>
                <th className="py-2.5 px-3">Statistical Regime Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono">
              {hurstAndVr.varianceRatios.map((vr) => (
                <tr key={vr.q} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/30">
                  <td className="py-2.5 px-3 font-semibold text-slate-900 dark:text-slate-100">
                    q = {vr.q} days
                  </td>
                  <td className="py-2.5 px-3 font-semibold text-emerald-600 dark:text-emerald-400">
                    {vr.vr.toFixed(3)}
                  </td>
                  <td className="py-2.5 px-3 text-slate-600 dark:text-slate-300">
                    {vr.zStat > 0 ? `+${vr.zStat.toFixed(2)}` : vr.zStat.toFixed(2)}
                  </td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[11px] ${
                        vr.pValue < 0.05
                          ? 'bg-amber-100 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300'
                          : 'bg-emerald-100 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-300'
                      }`}
                    >
                      {vr.pValue < 0.001 ? '< 0.001' : vr.pValue.toFixed(3)}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-slate-700 dark:text-slate-300 font-sans">
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

        <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800/60 text-xs text-slate-500 dark:text-slate-400 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-slate-400" />
            <span>{hurstAndVr.randomWalkVerdict}</span>
          </div>
          <VarianceRatioEquation className="hidden md:inline" />
        </div>
      </div>

      {/* Classical Time Series Decomposition */}
      <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/90 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 dark:border-slate-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-500" />
              <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                Classical Structural Decomposition (Trend · Seasonal · Residual)
              </h3>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Separating macro trend, periodic calendar components, and irregular noise
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <label className="text-xs text-slate-500 font-medium">Period:</label>
              <select
                value={decompPeriod}
                onChange={(e) => onDecompPeriodChange(parseInt(e.target.value, 10))}
                className="px-2 py-1 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-mono"
              >
                <option value={5}>5 Days (Weekly)</option>
                <option value={21}>21 Days (Monthly)</option>
                <option value={63}>63 Days (Quarterly)</option>
              </select>
            </div>

            <div className="inline-flex p-0.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-xs">
              <button
                type="button"
                onClick={() => onDecompModelChange('additive')}
                className={`px-2.5 py-1 rounded-md font-medium transition-colors cursor-pointer ${
                  decompModel === 'additive'
                    ? 'bg-white dark:bg-slate-800 text-emerald-600 dark:text-emerald-400 shadow-xs'
                    : 'text-slate-500 hover:text-slate-900'
                }`}
              >
                Additive
              </button>
              <button
                type="button"
                onClick={() => onDecompModelChange('multiplicative')}
                className={`px-2.5 py-1 rounded-md font-medium transition-colors cursor-pointer ${
                  decompModel === 'multiplicative'
                    ? 'bg-white dark:bg-slate-800 text-emerald-600 dark:text-emerald-400 shadow-xs'
                    : 'text-slate-500 hover:text-slate-900'
                }`}
              >
                Multiplicative
              </button>
            </div>
          </div>
        </div>

        <div className="mt-4">
          <DecompositionChart decomposition={decompResult} adfTest={adfResult} />
        </div>
      </div>
    </div>
  );
}
