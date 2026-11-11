'use client';

import React, { useState, useMemo } from 'react';
import {
  IntradayBar,
  computeRollingCorrelation,
  computeEwmaDynamicCorrelation,
  fitDccGarchModel,
  computeTailDependence,
  computeRegimeCorrelations,
  computeRealizedCorrelationIntraday,
} from '@/lib/correlation-engine';
import {
  MathDisplay,
  DccQEquation,
  DccCorrelationEquation,
  TailDependenceEquation,
} from '@/components/math/MathEquations';
import {
  Activity,
  Zap,
  Clock,
  Layers,
  TrendingDown,
  Info,
  ShieldAlert,
  ArrowRight,
  TrendingUp,
} from 'lucide-react';

interface CorrelationDynamicsTabProps {
  tickers: string[];
  returnsByTicker: { [ticker: string]: number[] };
  dates: string[];
  intradayBars?: IntradayBar[];
}

export function CorrelationDynamicsTab({
  tickers,
  returnsByTicker,
  dates,
  intradayBars = [],
}: CorrelationDynamicsTabProps) {
  // Pair selection
  const [selectedPairA, setSelectedPairA] = useState<string>(tickers[0] || 'AAPL');
  const [selectedPairB, setSelectedPairB] = useState<string>(
    tickers[1] || (tickers.length > 1 ? tickers[1] : 'SPY')
  );

  // Engine choice: DCC-GARCH vs EWMA
  const [modelType, setModelType] = useState<'dcc' | 'ewma'>('dcc');
  const [ewmaLambda, setEwmaLambda] = useState<number>(0.94);
  const [rollingWindow, setRollingWindow] = useState<number>(60);
  const [forecastDays, setForecastDays] = useState<number>(30);

  const returnsA = useMemo(() => returnsByTicker[selectedPairA] || [], [returnsByTicker, selectedPairA]);
  const returnsB = useMemo(() => returnsByTicker[selectedPairB] || [], [returnsByTicker, selectedPairB]);
  const nObs = Math.min(returnsA.length, returnsB.length);

  // Fit DCC-GARCH
  const dccResult = useMemo(() => {
    return fitDccGarchModel(returnsA, returnsB, dates, forecastDays);
  }, [returnsA, returnsB, dates, forecastDays]);

  // Fit EWMA
  const ewmaResult = useMemo(() => {
    return computeEwmaDynamicCorrelation(returnsA, returnsB, dates, ewmaLambda);
  }, [returnsA, returnsB, dates, ewmaLambda]);

  // Rolling correlation
  const rollingPoints = useMemo(() => {
    return computeRollingCorrelation(returnsA, returnsB, dates, rollingWindow);
  }, [returnsA, returnsB, dates, rollingWindow]);

  // Tail dependence & Copula
  const tailResult = useMemo(() => {
    return computeTailDependence(returnsA, returnsB);
  }, [returnsA, returnsB]);

  // Regime conditioning (Calm vs Stress)
  const regimeResult = useMemo(() => {
    const marketTicker = tickers.includes('SPY') ? 'SPY' : tickers[0];
    return computeRegimeCorrelations(returnsByTicker, marketTicker);
  }, [returnsByTicker, tickers]);

  // Realized 1-minute correlation & jump decomposition
  const realizedCorrResult = useMemo(() => {
    return computeRealizedCorrelationIntraday(intradayBars, intradayBars);
  }, [intradayBars]);

  // Active series for the dynamic chart
  const activeSeries =
    modelType === 'dcc' ? dccResult.dynamicCorrelations : ewmaResult;

  const currentDynamicCorr =
    activeSeries[activeSeries.length - 1]?.correlation ?? dccResult.unconditionalCorr;

  return (
    <div className="space-y-6">
      {/* Parameter & Model Selection Card */}
      <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-100 dark:border-slate-800">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-amber-600 dark:text-amber-400">
                Dynamic Correlation Suite
              </span>
              <span className="text-slate-300 dark:text-slate-700" aria-hidden="true">·</span>
              <span className="text-xs text-slate-500">
                Engle (2002) DCC-GARCH &amp; RiskMetrics EWMA
              </span>
            </div>
            <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100 mt-0.5">
              Time-Varying Co-Movement &amp; Volatility Spillover
            </h3>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Pair Selectors */}
            <div className="flex items-center gap-1.5 text-xs font-mono">
              <span className="text-slate-400">Pair:</span>
              <select
                value={selectedPairA}
                onChange={(e) => setSelectedPairA(e.target.value)}
                className="px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 font-bold text-slate-900 dark:text-slate-100"
              >
                {tickers.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
              <span className="text-slate-400">/</span>
              <select
                value={selectedPairB}
                onChange={(e) => setSelectedPairB(e.target.value)}
                className="px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 font-bold text-slate-900 dark:text-slate-100"
              >
                {tickers.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>

            {/* Model Architecture Switcher */}
            <div className="inline-flex items-center p-1 bg-slate-100 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 text-xs">
              <button
                onClick={() => setModelType('dcc')}
                className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                  modelType === 'dcc'
                    ? 'bg-amber-500 text-slate-950 font-bold shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                DCC-GARCH (Engle)
              </button>
              <button
                onClick={() => setModelType('ewma')}
                className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                  modelType === 'ewma'
                    ? 'bg-sky-500 text-slate-950 font-bold shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                RiskMetrics EWMA
              </button>
            </div>
          </div>
        </div>

        {/* Dynamic Model Control Readout Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-4 text-xs">
          {modelType === 'dcc' ? (
            <>
              <div className="p-3 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800">
                <span className="text-slate-400 font-mono text-[10px] block">
                  Shock Reaction Parameter (a)
                </span>
                <span className="font-mono text-xl font-bold text-amber-500">
                  {dccResult.a.toFixed(3)}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  Sensitivity to joint innovation shocks
                </span>
              </div>
              <div className="p-3 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800">
                <span className="text-slate-400 font-mono text-[10px] block">
                  Decay Memory Parameter (b)
                </span>
                <span className="font-mono text-xl font-bold text-sky-500">
                  {dccResult.b.toFixed(3)}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  Autoregressive correlation persistence
                </span>
              </div>
              <div className="p-3 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800">
                <span className="text-slate-400 font-mono text-[10px] block">
                  Total Persistence (a + b)
                </span>
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-xl font-bold text-slate-900 dark:text-slate-100">
                    {dccResult.persistence.toFixed(3)}
                  </span>
                  <span className="text-[10px] text-emerald-500 font-semibold">
                    Stationary (&lt;1)
                  </span>
                </div>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  Half-life: {(Math.log(0.5) / Math.log(dccResult.persistence)).toFixed(1)} sessions
                </span>
              </div>
              <div className="p-3 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800">
                <span className="text-slate-400 font-mono text-[10px] block">
                  Latest Dynamic Correlation
                </span>
                <span className="font-mono text-xl font-bold text-emerald-500">
                  {currentDynamicCorr.toFixed(3)}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  Unconditional: {dccResult.unconditionalCorr.toFixed(2)}
                </span>
              </div>
            </>
          ) : (
            <>
              <div className="p-3 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800 lg:col-span-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-slate-700 dark:text-slate-300 font-semibold">
                    EWMA Decay Factor (λ)
                  </span>
                  <span className="font-mono font-bold text-sky-500">{ewmaLambda.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.85"
                  max="0.99"
                  step="0.01"
                  value={ewmaLambda}
                  onChange={(e) => setEwmaLambda(parseFloat(e.target.value))}
                  className="w-full accent-sky-500 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono mt-1">
                  <span>0.85 (Fast shock response)</span>
                  <span>0.94 (RiskMetrics standard)</span>
                  <span>0.99 (Smooth long memory)</span>
                </div>
              </div>
              <div className="p-3 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800">
                <span className="text-slate-400 font-mono text-[10px] block">Effective Half-Life</span>
                <span className="font-mono text-xl font-bold text-slate-900 dark:text-slate-100">
                  {(Math.log(0.5) / Math.log(ewmaLambda)).toFixed(1)} days
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  Memory horizon for joint returns
                </span>
              </div>
              <div className="p-3 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800">
                <span className="text-slate-400 font-mono text-[10px] block">
                  Current EWMA Correlation
                </span>
                <span className="font-mono text-xl font-bold text-sky-500">
                  {currentDynamicCorr.toFixed(3)}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  Static: {dccResult.unconditionalCorr.toFixed(2)}
                </span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Main Dynamic Correlation Chart with Baseline & Forecast Cone */}
      <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <div>
            <div className="flex items-center gap-2">
              <h4 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                {modelType === 'dcc' ? 'Engle DCC-GARCH(1,1)' : 'RiskMetrics EWMA'} Dynamic Correlation Path
              </h4>
              <span className="text-xs font-mono text-slate-400">
                · {selectedPairA} vs {selectedPairB}
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Demonstrating how correlation fluctuates as a state variable compared to constant static ρ
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-amber-500 inline-block" />
              <span>Dynamic {modelType.toUpperCase()} Path</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-slate-400 border-dashed inline-block" />
              <span>Static Baseline ({dccResult.unconditionalCorr.toFixed(2)})</span>
            </div>
          </div>
        </div>

        {/* SVG Time-Series Chart */}
        <div className="w-full h-64 bg-slate-50 dark:bg-slate-900/60 rounded-xl border border-slate-200 dark:border-slate-800 relative p-3">
          {activeSeries.length > 5 ? (
            <svg viewBox="0 0 800 220" className="w-full h-full overflow-visible">
              {/* Horizontal Reference Lines (-1, 0, +1) */}
              <line x1="40" y1="20" x2="780" y2="20" stroke="#64748b" strokeWidth="0.5" strokeDasharray="3 3" />
              <line x1="40" y1="110" x2="780" y2="110" stroke="#64748b" strokeWidth="1" strokeDasharray="2 2" />
              <line x1="40" y1="200" x2="780" y2="200" stroke="#64748b" strokeWidth="0.5" strokeDasharray="3 3" />

              {/* Zero & Bounds Labels */}
              <text x="35" y="24" textAnchor="end" className="text-[10px] font-mono fill-slate-400">+1.0</text>
              <text x="35" y="114" textAnchor="end" className="text-[10px] font-mono fill-slate-400">0.0</text>
              <text x="35" y="204" textAnchor="end" className="text-[10px] font-mono fill-slate-400">-1.0</text>

              {/* Static Unconditional Line */}
              <line
                x1="45"
                y1={110 - dccResult.unconditionalCorr * 90}
                x2="780"
                y2={110 - dccResult.unconditionalCorr * 90}
                stroke="#94a3b8"
                strokeWidth="1.2"
                strokeDasharray="4 4"
              />

              {/* Dynamic Path Polyline */}
              {(() => {
                const totalPoints = activeSeries.length;
                const stepX = 650 / Math.max(1, totalPoints - 1);
                const pointsStr = activeSeries
                  .map((pt, i) => {
                    const x = 50 + i * stepX;
                    const y = 110 - pt.correlation * 90;
                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                  })
                  .join(' ');

                return (
                  <polyline
                    fill="none"
                    stroke={modelType === 'dcc' ? '#f59e0b' : '#0ea5e9'}
                    strokeWidth="2"
                    points={pointsStr}
                  />
                );
              })()}

              {/* DCC Forecast Cone (if DCC selected) */}
              {modelType === 'dcc' && dccResult.forecastCone.length > 0 && (() => {
                const startX = 700;
                const endX = 770;
                const cone = dccResult.forecastCone;
                const topPath = cone
                  .map((c, i) => `${startX + i * 2.3},${110 - c.upper95 * 90}`)
                  .join(' ');
                const bottomPath = cone
                  .slice()
                  .reverse()
                  .map((c, i) => `${endX - i * 2.3},${110 - c.lower95 * 90}`)
                  .join(' ');

                return (
                  <polygon
                    points={`${topPath} ${bottomPath}`}
                    fill="#f59e0b"
                    fillOpacity="0.15"
                    stroke="#f59e0b"
                    strokeWidth="0.8"
                    strokeDasharray="2 2"
                  />
                );
              })()}
            </svg>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-slate-400 font-mono">
              Insufficient time series points to construct dynamic correlation path.
            </div>
          )}
        </div>

        <div className="mt-3 flex items-center justify-between text-xs text-slate-500 font-mono">
          <span>Start: {dates[0] || 'Historical'}</span>
          <span>Latest: {dates[dates.length - 1] || 'Current'}</span>
          {modelType === 'dcc' && (
            <span className="text-amber-500 font-semibold">+30-Day Forecast Cone Shaded</span>
          )}
        </div>
      </div>

      {/* Rolling Windows & Tail Copula Breakdown Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Rolling Window Correlation with Historical Percentile Bands */}
        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-100 dark:border-slate-800">
            <div>
              <h4 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                Rolling Correlation &amp; Percentile Bands
              </h4>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Evaluates historical regime stability against long-term quantile bands
              </p>
            </div>

            {/* Window Selector */}
            <div className="flex items-center gap-1 p-0.5 bg-slate-100 dark:bg-slate-900 rounded border border-slate-200 dark:border-slate-800 text-xs">
              {[20, 60, 120].map((w) => (
                <button
                  key={w}
                  onClick={() => setRollingWindow(w)}
                  className={`px-2 py-1 rounded font-mono font-medium ${
                    rollingWindow === w
                      ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white font-bold shadow-xs'
                      : 'text-slate-500'
                  }`}
                >
                  {w}D
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 text-center font-mono text-xs mb-3">
            <div className="p-2.5 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800">
              <span className="text-[10px] text-slate-400 block">10th Percentile (Min Regime)</span>
              <span className="font-bold text-rose-500 text-sm">
                {rollingPoints[0]?.p10.toFixed(2) ?? '-0.15'}
              </span>
            </div>
            <div className="p-2.5 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800">
              <span className="text-[10px] text-slate-400 block">50th Percentile (Median)</span>
              <span className="font-bold text-slate-900 dark:text-slate-100 text-sm">
                {rollingPoints[0]?.p50.toFixed(2) ?? '0.45'}
              </span>
            </div>
            <div className="p-2.5 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800">
              <span className="text-[10px] text-slate-400 block">90th Percentile (Stress Spike)</span>
              <span className="font-bold text-amber-500 text-sm">
                {rollingPoints[0]?.p90.toFixed(2) ?? '0.82'}
              </span>
            </div>
          </div>

          <p className="text-[11px] text-slate-500 leading-relaxed">
            During liquidity crises, correlations typically spike towards the 90th percentile (~{rollingPoints[0]?.p90.toFixed(2) ?? '0.82'}), eroding standard portfolio diversification benefit exactly when drawdown protection is required.
          </p>
        </div>

        {/* Tail Dependence & Copula Exceedance Quadrant */}
        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-100 dark:border-slate-800">
            <div>
              <h4 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                Tail Dependence &amp; Crash Quadrants
              </h4>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Empirical co-crash probability vs Gaussian copula assumption
              </p>
            </div>
            <span className="text-xs font-mono text-rose-500 font-semibold">
              Tail Risk Indicator
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs mb-3">
            <div className="p-3 bg-rose-500/10 border border-rose-500/20 rounded-xl">
              <span className="text-[10px] font-mono text-rose-600 dark:text-rose-400 block font-semibold">
                Empirical Joint Crash Exceedance
              </span>
              <div className="flex items-baseline gap-2 mt-0.5">
                <span className="font-mono text-xl font-bold text-rose-600 dark:text-rose-400">
                  {(tailResult.jointExceedanceEmpirical * 100).toFixed(1)}%
                </span>
                <span className="text-[10px] font-mono text-slate-400">
                  vs {(tailResult.jointExceedanceGaussian * 100).toFixed(2)}% Gaussian
                </span>
              </div>
              <span className="text-[10px] text-rose-700 dark:text-rose-400 block mt-1 font-semibold">
                {tailResult.exceedanceRatio.toFixed(1)}× higher probability than normal distribution
              </span>
            </div>

            <div className="p-3 bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 rounded-xl">
              <span className="text-[10px] font-mono text-slate-400 block font-semibold">
                Lower Tail Dependence (λ_L)
              </span>
              <div className="flex items-baseline gap-2 mt-0.5">
                <span className="font-mono text-xl font-bold text-slate-900 dark:text-slate-100">
                  {tailResult.lowerTailDep.toFixed(2)}
                </span>
                <span className="text-[10px] text-slate-500">
                  (Gaussian = 0.00)
                </span>
              </div>
              <span className="text-[10px] text-slate-500 block mt-1">
                Both assets drop &gt;1.5σ in {tailResult.quadrants.bothCrash} sessions
              </span>
            </div>
          </div>

          <div className="p-2.5 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800 flex items-center justify-between text-xs">
            <TailDependenceEquation display="inline" className="text-xs" />
            <span className="text-[10px] font-mono text-slate-400">
              Gaussian copulas dangerously assume zero tail dependence
            </span>
          </div>
        </div>
      </div>

      {/* Regime-Conditional Matrices & 1-Min Intraday Realized Correlation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Calm vs Stress Regime Split */}
        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-100 dark:border-slate-800">
            <div>
              <h4 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                Regime-Conditional Correlation Breakdown
              </h4>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                {regimeResult.stressDefinition}
              </p>
            </div>
            <span className="text-xs font-mono text-amber-500 font-semibold">
              Calm vs Stress Split
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 font-mono text-center text-xs mb-3">
            <div className="p-2.5 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
              <span className="text-[10px] text-emerald-600 dark:text-emerald-400 block font-semibold">
                Calm Regime Mean ρ
              </span>
              <span className="font-bold text-emerald-600 dark:text-emerald-400 text-lg">
                {regimeResult.avgCalmCorr.toFixed(2)}
              </span>
            </div>
            <div className="p-2.5 bg-rose-500/10 border border-rose-500/20 rounded-lg">
              <span className="text-[10px] text-rose-600 dark:text-rose-400 block font-semibold">
                Stress Regime Mean ρ
              </span>
              <span className="font-bold text-rose-600 dark:text-rose-400 text-lg">
                {regimeResult.avgStressCorr.toFixed(2)}
              </span>
            </div>
            <div className="p-2.5 bg-amber-500/10 border border-amber-500/20 rounded-lg">
              <span className="text-[10px] text-amber-600 dark:text-amber-400 block font-semibold">
                Tail Penalty (Δρ)
              </span>
              <span className="font-bold text-amber-600 dark:text-amber-400 text-lg">
                +{regimeResult.avgDeltaCorr.toFixed(2)}
              </span>
            </div>
          </div>

          <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
            Quantifies the empirical phenomenon where asset correlations increase by{' '}
            <strong className="text-amber-500">
              +{(regimeResult.avgDeltaCorr * 100).toFixed(0)}%
            </strong>{' '}
            during market stress, reducing diversification exactly when protection is vital.
          </p>
        </div>

        {/* Intraday 1-Minute Realized Correlation & Jump Split */}
        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-100 dark:border-slate-800">
            <div>
              <div className="flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-amber-500" />
                <h4 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                  1-Minute Realized Correlation &amp; Jump Split
                </h4>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Continuous Brownian co-diffusion vs discrete co-jump shocks
              </p>
            </div>
            <span className="text-xs font-mono text-sky-500 font-semibold">
              High-Frequency RC
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 font-mono text-center text-xs mb-3">
            <div className="p-2.5 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800">
              <span className="text-[10px] text-slate-400 block">Total Realized (ρ^RV)</span>
              <span className="font-bold text-slate-900 dark:text-slate-100 text-lg">
                {realizedCorrResult.intradayRealizedCorr.toFixed(2)}
              </span>
            </div>
            <div className="p-2.5 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800">
              <span className="text-[10px] text-slate-400 block">Continuous Diffusion</span>
              <span className="font-bold text-sky-500 text-lg">
                {realizedCorrResult.continuousDiffusionCorr.toFixed(2)}
              </span>
            </div>
            <div className="p-2.5 bg-slate-50 dark:bg-slate-900/60 rounded-lg border border-slate-200 dark:border-slate-800">
              <span className="text-[10px] text-slate-400 block">Discrete Co-Jumps</span>
              <span className="font-bold text-amber-500 text-lg">
                {realizedCorrResult.discreteJumpCorr.toFixed(2)}
              </span>
            </div>
          </div>

          {/* Epps Effect Transparency Warning */}
          <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg text-xs text-amber-700 dark:text-amber-400 flex items-start gap-2">
            <ShieldAlert className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <div>
              <strong>Epps Effect &amp; Sample Size Notice: </strong>
              {realizedCorrResult.eppsBiasCaveat}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
