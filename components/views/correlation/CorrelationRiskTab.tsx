'use client';

import React, { useState, useMemo } from 'react';
import {
  decomposePortfolioRisk,
  computeCorrelationVaR,
  computeBenchmarkWeights,
} from '@/lib/correlation-engine';
import {
  PortfolioVarianceEquation,
  RiskContributionEquation,
  DiversificationRatioEquation,
} from '@/components/math/MathEquations';
import {
  PieChart,
  ShieldAlert,
  Percent,
  Sliders,
  Check,
  TrendingDown,
  Layers,
  Sparkles,
  Info,
  Scale,
  Award,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

interface CorrelationRiskTabProps {
  tickers: string[];
  returnsByTicker: { [ticker: string]: number[] };
  shrunkCovariance: number[][];
  rawCovariance: number[][];
}

export function CorrelationRiskTab({
  tickers,
  returnsByTicker,
  shrunkCovariance,
  rawCovariance,
}: CorrelationRiskTabProps) {
  const n = tickers.length;

  // Active covariance matrix choice: Shrunk vs Raw
  const [useShrunkCov, setUseShrunkCov] = useState<boolean>(true);
  const activeCov = useShrunkCov && shrunkCovariance.length === n ? shrunkCovariance : rawCovariance;

  // Portfolio weights state (in percentage terms, e.g. 20 = 20%)
  const [weightsPct, setWeightsPct] = useState<number[]>(() => {
    if (n === 0) return [];
    const eq = Number((100 / n).toFixed(2));
    const init = new Array(n).fill(eq);
    return init;
  });

  const [showMathReference, setShowMathReference] = useState<boolean>(false);
  const [activeConfidence, setActiveConfidence] = useState<95 | 99>(95);

  // Normalize weights for calculations (sum = 1)
  const normalizedWeights = useMemo(() => {
    let sum = 0;
    for (const w of weightsPct) sum += Math.max(0, w);
    if (sum <= 0) return new Array(n).fill(1 / Math.max(1, n));
    return weightsPct.map((w) => Math.max(0, w) / sum);
  }, [weightsPct, n]);

  const currentWeightSum = useMemo(() => {
    return weightsPct.reduce((acc, curr) => acc + curr, 0);
  }, [weightsPct]);

  // Risk decomposition computation
  const riskDecomp = useMemo(() => {
    if (n === 0 || activeCov.length !== n) {
      return {
        weights: [],
        portfolioVol: 0,
        mctr: [],
        cctr: [],
        pctCtr: [],
        diversificationRatio: 1,
        diversificationBenefit: 0,
        weightedAvgVol: 0,
      };
    }
    return decomposePortfolioRisk(normalizedWeights, activeCov, tickers);
  }, [normalizedWeights, activeCov, tickers, n]);

  // Correlation VaR and Incremental VaR computation
  const varResult = useMemo(() => {
    if (n === 0) {
      return {
        parametricVaR95: 0,
        parametricVaR99: 0,
        historicalVaR95: 0,
        historicalVaR99: 0,
        expectedShortfall95: 0,
        expectedShortfall99: 0,
        tailPenalty: 0,
        incrementalVaR: [],
      };
    }
    return computeCorrelationVaR(returnsByTicker, normalizedWeights, activeConfidence / 100);
  }, [returnsByTicker, normalizedWeights, activeConfidence, n]);

  // Benchmark allocations
  const benchmarks = useMemo(() => {
    if (n === 0 || activeCov.length !== n) {
      return {
        equalWeight: [],
        minVariance: [],
        riskParity: [],
        equalWeightVol: 0,
        minVarVol: 0,
        riskParityVol: 0,
      };
    }
    return computeBenchmarkWeights(activeCov, tickers);
  }, [activeCov, tickers, n]);

  // Handler for weight slider
  const handleWeightChange = (index: number, val: number) => {
    setWeightsPct((prev) => {
      const copy = [...prev];
      copy[index] = val;
      return copy;
    });
  };

  // Preset loaders
  const loadEqualWeights = () => {
    const eq = Number((100 / n).toFixed(2));
    setWeightsPct(new Array(n).fill(eq));
  };

  const loadMinVariance = () => {
    if (benchmarks.minVariance.length === n) {
      setWeightsPct(benchmarks.minVariance.map((w) => Number((w * 100).toFixed(2))));
    }
  };

  const loadRiskParity = () => {
    if (benchmarks.riskParity.length === n) {
      setWeightsPct(benchmarks.riskParity.map((w) => Number((w * 100).toFixed(2))));
    }
  };

  const normalizeWeightsTo100 = () => {
    if (currentWeightSum <= 0) {
      loadEqualWeights();
      return;
    }
    setWeightsPct((prev) =>
      prev.map((w) => Number(((Math.max(0, w) / currentWeightSum) * 100).toFixed(2)))
    );
  };

  // Effective number of bets (concentration metrics)
  const capitalEnc = useMemo(() => {
    const sumSq = normalizedWeights.reduce((acc, w) => acc + w * w, 0);
    return sumSq > 0 ? 1 / sumSq : 1;
  }, [normalizedWeights]);

  const riskEnc = useMemo(() => {
    const sumSq = riskDecomp.pctCtr.reduce((acc, pct) => acc + (pct / 100) * (pct / 100), 0);
    return sumSq > 0 ? 1 / sumSq : 1;
  }, [riskDecomp.pctCtr]);

  // Annualized volatility (assuming 252 trading days)
  const annualFactor = Math.sqrt(252);
  const annPortfolioVol = riskDecomp.portfolioVol * annualFactor * 100;
  const annWeightedAvgVol = riskDecomp.weightedAvgVol * annualFactor * 100;

  return (
    <div className="space-y-6">
      {/* Top Banner & Control Toolbar */}
      <div className="panel p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-zinc-100 dark:border-zinc-800">
          <div>
            <div className="flex items-center gap-2">
              <span className="kicker">
                Tier 3 Risk Attribution
              </span>
              <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
              <span className="text-xs text-zinc-500 dark:text-zinc-400">
                Euler Marginal Decomposition &amp; Correlation VaR
              </span>
            </div>
            <h3 className="text-lg font-bold tracking-tight text-zinc-900 dark:text-zinc-100 mt-0.5">
              Portfolio Risk Budgeting &amp; Diversification Analysis
            </h3>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1 max-w-2xl">
              Turn the correlation matrix into executable risk metrics. Evaluate marginal risk contributions (MCTR), component risk (CCTR), parametric vs historical VaR tails, and compare benchmark allocations.
            </p>
          </div>

          {/* Quick preset action buttons */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-zinc-500 dark:text-zinc-400 mr-1">Presets:</span>
            <button
              type="button"
              onClick={loadEqualWeights}
              className="px-2.5 py-1 text-xs font-medium rounded-lg border border-zinc-200 dark:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-800 text-zinc-700 dark:text-zinc-300 transition-colors"
            >
              Equal Weight (1/N)
            </button>
            <button
              type="button"
              onClick={loadMinVariance}
              className="px-2.5 py-1 text-xs font-medium rounded-lg border border-gold-300 dark:border-gold-700/60 bg-gold-50 dark:bg-gold-950/30 text-gold-700 dark:text-gold-300 hover:bg-gold-100 dark:hover:bg-gold-900/50 transition-colors flex items-center gap-1"
            >
              <Award className="w-3.5 h-3.5 text-gold-500" />
              Min-Variance
            </button>
            <button
              type="button"
              onClick={loadRiskParity}
              className="px-2.5 py-1 text-xs font-medium rounded-lg border border-zinc-200 dark:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-800 text-zinc-700 dark:text-zinc-300 transition-colors"
            >
              Risk Parity (1/σ)
            </button>
            <button
              type="button"
              onClick={() => setUseShrunkCov((prev) => !prev)}
              className={`px-2.5 py-1 text-xs font-medium rounded-lg border transition-colors ${
                useShrunkCov
                  ? 'border-gold-500 bg-gold-500 dark:bg-gold-500/40 text-gold-700 dark:text-gold-300'
                  : 'border-zinc-200 dark:border-zinc-700 text-zinc-600 dark:text-zinc-400'
              }`}
              title="Toggle Ledoit-Wolf analytical shrinkage covariance vs sample covariance"
            >
              {useShrunkCov ? 'Shrunk Σ (Active)' : 'Sample Σ'}
            </button>
          </div>
        </div>

        {/* Key Metric Tiles */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3 mt-4">
          <div className="p-3 rounded-lg bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-100 dark:border-zinc-800/80">
            <div className="text-[11px] font-medium text-zinc-500 dark:text-zinc-400">
              Portfolio Volatility (Ann.)
            </div>
            <div className="text-xl font-bold font-mono text-gold-600 dark:text-gold-400 mt-0.5">
              {annPortfolioVol.toFixed(2)}%
            </div>
            <div className="text-[10px] text-zinc-400 dark:text-zinc-500 mt-0.5">
              Daily σ: {(riskDecomp.portfolioVol * 100).toFixed(3)}%
            </div>
          </div>

          <div className="p-3 rounded-lg bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-100 dark:border-zinc-800/80">
            <div className="text-[11px] font-medium text-zinc-500 dark:text-zinc-400">
              Diversification Ratio (DR)
            </div>
            <div className="text-xl font-bold font-mono text-gold-600 dark:text-gold-300 dark:text-gold-600 dark:text-gold-300 mt-0.5">
              {riskDecomp.diversificationRatio.toFixed(3)}x
            </div>
            <div className="text-[10px] text-zinc-400 dark:text-zinc-500 mt-0.5">
              Σ(w·σ) / σ_p
            </div>
          </div>

          <div className="p-3 rounded-lg bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-100 dark:border-zinc-800/80">
            <div className="text-[11px] font-medium text-zinc-500 dark:text-zinc-400">
              Diversification Benefit
            </div>
            <div className="text-xl font-bold font-mono text-gold-600 dark:text-gold-300 dark:text-gold-600 dark:text-gold-300 mt-0.5">
              {(riskDecomp.diversificationBenefit * 100).toFixed(1)}%
            </div>
            <div className="text-[10px] text-zinc-400 dark:text-zinc-500 mt-0.5">
              Vol reduced via decorrelation
            </div>
          </div>

          <div className="p-3 rounded-lg bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-100 dark:border-zinc-800/80">
            <div className="text-[11px] font-medium text-zinc-500 dark:text-zinc-400">
              Weighted Avg Vol (Σ w_i σ_i)
            </div>
            <div className="text-xl font-bold font-mono text-zinc-800 dark:text-zinc-200 mt-0.5">
              {annWeightedAvgVol.toFixed(2)}%
            </div>
            <div className="text-[10px] text-zinc-400 dark:text-zinc-500 mt-0.5">
              Zero-correlation benchmark
            </div>
          </div>

          <div className="p-3 rounded-lg bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-100 dark:border-zinc-800/80 col-span-2 md:col-span-4 lg:col-span-1">
            <div className="text-[11px] font-medium text-zinc-500 dark:text-zinc-400">
              Effective Bets (ENC)
            </div>
            <div className="text-xl font-bold font-mono text-gold-600 dark:text-gold-400 mt-0.5">
              {riskEnc.toFixed(1)} <span className="text-xs text-zinc-400">/ {n}</span>
            </div>
            <div className="text-[10px] text-zinc-400 dark:text-zinc-500 mt-0.5">
              Capital ENC: {capitalEnc.toFixed(1)}
            </div>
          </div>
        </div>

        {/* Normalization warning if sum != 100% */}
        {Math.abs(currentWeightSum - 100) > 0.5 && (
          <div className="mt-4 p-3 bg-gold-50 dark:bg-gold-950/30 border border-gold-200 dark:border-gold-800/50 rounded-lg flex items-center justify-between text-xs text-gold-800 dark:text-gold-300">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-gold-500 shrink-0" />
              <span>
                Current weights sum to <strong>{currentWeightSum.toFixed(1)}%</strong>. Calculations are automatically normalized to 100%, but you can re-scale weights explicitly.
              </span>
            </div>
            <button
              type="button"
              onClick={normalizeWeightsTo100}
              className="px-2.5 py-1 bg-gold-600 hover:bg-gold-700 text-white rounded font-medium transition-colors shrink-0 ml-3"
            >
              Normalize to 100%
            </button>
          </div>
        )}
      </div>

      {/* Main Grid: Interactive Weights & Risk Attribution Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (5 cols): Weight Sliders */}
        <div className="lg:col-span-5 panel p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-zinc-100 dark:border-zinc-800">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-gold-500" />
                <h4 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                  Portfolio Weight Sliders
                </h4>
              </div>
              <span className="text-xs font-mono text-zinc-500 dark:text-zinc-400">
                Sum: {currentWeightSum.toFixed(1)}%
              </span>
            </div>

            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 mb-4">
              Adjust capital allocation per asset. Notice how shifting weight into high-volatility or highly-correlated assets disproportionately inflates risk contribution.
            </p>

            <div className="space-y-3.5 max-h-[380px] overflow-y-auto pr-1">
              {tickers.map((ticker, idx) => {
                const wPct = weightsPct[idx] ?? 0;
                const normWPct = (normalizedWeights[idx] || 0) * 100;
                const riskPct = riskDecomp.pctCtr[idx] || 0;
                const isOverBudget = riskPct > normWPct * 1.35;
                const isUnderBudget = riskPct < normWPct * 0.65;

                return (
                  <div
                    key={ticker}
                    className="p-2.5 rounded-lg bg-zinc-50/80 dark:bg-zinc-900/40 border border-zinc-100 dark:border-zinc-800/60"
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-xs text-zinc-800 dark:text-zinc-200">
                          {ticker}
                        </span>
                        {isOverBudget && (
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-red-100 dark:bg-red-950/50 text-red-700 dark:text-red-400 font-semibold">
                            Risk Hog ({riskPct.toFixed(1)}%)
                          </span>
                        )}
                        {isUnderBudget && (
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-gold-100 dark:bg-gold-950/50 text-gold-700 dark:text-gold-400 font-semibold">
                            Diversifier ({riskPct.toFixed(1)}%)
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-2 font-mono text-xs">
                        <span className="text-zinc-500 dark:text-zinc-400">
                          Norm: {normWPct.toFixed(1)}%
                        </span>
                        <div className="flex items-center bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded px-1.5 py-0.5">
                          <input
                            type="number"
                            min="0"
                            max="100"
                            step="1"
                            value={wPct}
                            onChange={(e) => handleWeightChange(idx, Number(e.target.value))}
                            className="w-12 text-right bg-transparent focus:outline-hidden font-bold text-zinc-900 dark:text-zinc-100"
                          />
                          <span className="text-zinc-400 text-[10px] ml-0.5">%</span>
                        </div>
                      </div>
                    </div>

                    <input
                      type="range"
                      min="0"
                      max="100"
                      step="0.5"
                      value={wPct}
                      onChange={(e) => handleWeightChange(idx, Number(e.target.value))}
                      className="w-full h-1.5 bg-zinc-200 dark:bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-gold-600"
                    />

                    {/* Micro bar comparing Weight vs Risk Contribution */}
                    <div className="grid grid-cols-2 gap-2 mt-2 pt-1 border-t border-zinc-100 dark:border-zinc-800/40 text-[10px] font-mono">
                      <div>
                        <div className="flex justify-between text-zinc-500 dark:text-zinc-400 mb-0.5">
                          <span>Capital:</span>
                          <span>{normWPct.toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-1 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gold-400 rounded-full"
                            style={{ width: `${Math.min(100, normWPct)}%` }}
                          />
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-zinc-500 dark:text-zinc-400 mb-0.5">
                          <span>Risk (%CTR):</span>
                          <span
                            className={
                              riskPct > normWPct
                                ? 'text-red-500 font-bold'
                                : 'text-gold-500 font-bold'
                            }
                          >
                            {riskPct.toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full h-1 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              riskPct > normWPct ? 'bg-gold-500' : 'bg-gold-500'
                            }`}
                            style={{ width: `${Math.min(100, Math.max(0, riskPct))}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800 text-[11px] text-zinc-500 dark:text-zinc-400">
            <strong>Euler Identity:</strong> Total portfolio variance satisfies <code className="text-gold-600 dark:text-gold-400">σ_p = Σ w_i · MCTR_i</code>. When correlation is imperfect, risk contribution does not equal weight contribution.
          </div>
        </div>

        {/* Right Column (7 cols): Capital vs Risk Stacked Comparison & Table */}
        <div className="lg:col-span-7 panel p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-zinc-100 dark:border-zinc-800">
              <div className="flex items-center gap-2">
                <PieChart className="w-4 h-4 text-gold-500" />
                <h4 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                  Risk Budget vs Capital Budget (%CTR vs Weight)
                </h4>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <span className="flex items-center gap-1.5 text-gold-600 dark:text-gold-300 dark:text-gold-300 font-mono">
                  <span className="w-2.5 h-2.5 rounded-sm bg-gold-400 inline-block" /> Capital %
                </span>
                <span className="flex items-center gap-1.5 text-gold-600 dark:text-gold-400 font-mono">
                  <span className="w-2.5 h-2.5 rounded-sm bg-gold-500 inline-block" /> Risk %CTR
                </span>
              </div>
            </div>

            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 mb-3">
              The classic risk parity trap: a modest capital weight in a volatile, highly correlated asset often consumes the majority of portfolio volatility risk.
            </p>

            {/* Stacked Horizon Bar Visualization */}
            <div className="space-y-2.5 my-3 p-3.5 bg-zinc-50 dark:bg-zinc-900/50 rounded-lg border border-zinc-100 dark:border-zinc-800">
              <div className="text-[11px] font-mono text-zinc-500 dark:text-zinc-400 flex justify-between">
                <span>Capital Allocation Stack</span>
                <span>100%</span>
              </div>
              <div className="w-full h-5 rounded-md overflow-hidden flex bg-zinc-200 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-700">
                {tickers.map((ticker, idx) => {
                  const pct = (normalizedWeights[idx] || 0) * 100;
                  if (pct <= 0) return null;
                  const palette = ['#3b82f6', '#10b981', '#DEC084', '#ec4899', '#8b5cf6', '#06b6d4', '#6366f1', '#14b8a6'];
                  const color = palette[idx % palette.length];
                  return (
                    <div
                      key={ticker}
                      style={{ width: `${pct}%`, backgroundColor: color }}
                      className="h-full flex items-center justify-center text-[10px] font-bold text-white overflow-hidden text-ellipsis whitespace-nowrap px-0.5 border-r border-black/10 last:border-r-0"
                      title={`${ticker}: ${pct.toFixed(1)}%`}
                    >
                      {pct > 7 ? ticker : ''}
                    </div>
                  );
                })}
              </div>

              <div className="text-[11px] font-mono text-zinc-500 dark:text-zinc-400 flex justify-between pt-1">
                <span>Total Risk Contribution (%CTR Stack)</span>
                <span>100%</span>
              </div>
              <div className="w-full h-5 rounded-md overflow-hidden flex bg-zinc-200 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-700">
                {tickers.map((ticker, idx) => {
                  const pct = Math.max(0, riskDecomp.pctCtr[idx] || 0);
                  if (pct <= 0) return null;
                  const palette = ['#3b82f6', '#10b981', '#DEC084', '#ec4899', '#8b5cf6', '#06b6d4', '#6366f1', '#14b8a6'];
                  const color = palette[idx % palette.length];
                  return (
                    <div
                      key={ticker}
                      style={{ width: `${pct}%`, backgroundColor: color }}
                      className="h-full flex items-center justify-center text-[10px] font-bold text-white overflow-hidden text-ellipsis whitespace-nowrap px-0.5 border-r border-black/10 last:border-r-0"
                      title={`${ticker}: ${pct.toFixed(1)}% of total risk`}
                    >
                      {pct > 7 ? ticker : ''}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Detailed Euler Table */}
            <div className="overflow-x-auto mt-4">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-zinc-200 dark:border-zinc-700/80 text-zinc-500 dark:text-zinc-400">
                    <th className="py-2 px-2">Ticker</th>
                    <th className="py-2 px-2 text-right">Weight (w_i)</th>
                    <th className="py-2 px-2 text-right">Ann. Vol (σ_i)</th>
                    <th className="py-2 px-2 text-right">MCTR (bps)</th>
                    <th className="py-2 px-2 text-right">CCTR (bps)</th>
                    <th className="py-2 px-2 text-right">% Total Risk</th>
                    <th className="py-2 px-2 text-right">Ratio (%/w)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
                  {tickers.map((ticker, idx) => {
                    const normW = normalizedWeights[idx] || 0;
                    const assetVol = Math.sqrt(Math.max(1e-12, activeCov[idx]?.[idx] || 0)) * annualFactor * 100;
                    const mctrBps = (riskDecomp.mctr[idx] || 0) * 10000;
                    const cctrBps = (riskDecomp.cctr[idx] || 0) * 10000;
                    const pctCtr = riskDecomp.pctCtr[idx] || 0;
                    const ratio = normW > 0 ? (pctCtr / (normW * 100)) : 1;

                    return (
                      <tr key={ticker} className="hover:bg-zinc-50 dark:hover:bg-zinc-900/40">
                        <td className="py-2 px-2 font-bold text-zinc-800 dark:text-zinc-200">
                          {ticker}
                        </td>
                        <td className="py-2 px-2 text-right text-zinc-700 dark:text-zinc-300">
                          {(normW * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 px-2 text-right text-zinc-700 dark:text-zinc-300">
                          {assetVol.toFixed(1)}%
                        </td>
                        <td className="py-2 px-2 text-right text-zinc-600 dark:text-zinc-400">
                          {mctrBps.toFixed(1)}
                        </td>
                        <td className="py-2 px-2 text-right text-zinc-600 dark:text-zinc-400">
                          {cctrBps.toFixed(1)}
                        </td>
                        <td className="py-2 px-2 text-right font-bold text-gold-600 dark:text-gold-400">
                          {pctCtr.toFixed(1)}%
                        </td>
                        <td
                          className={`py-2 px-2 text-right font-bold ${
                            ratio > 1.25
                              ? 'text-red-600 dark:text-red-400'
                              : ratio < 0.75
                              ? 'text-gold-600 dark:text-gold-400'
                              : 'text-zinc-600 dark:text-zinc-400'
                          }`}
                        >
                          {ratio.toFixed(2)}x
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
                <tfoot>
                  <tr className="border-t-2 border-zinc-200 dark:border-zinc-700 font-bold text-zinc-900 dark:text-zinc-100 bg-zinc-50/50 dark:bg-zinc-900/30">
                    <td className="py-2 px-2">Portfolio Total</td>
                    <td className="py-2 px-2 text-right">100.0%</td>
                    <td className="py-2 px-2 text-right">{annPortfolioVol.toFixed(2)}%</td>
                    <td className="py-2 px-2 text-right">-</td>
                    <td className="py-2 px-2 text-right">
                      {((riskDecomp.portfolioVol) * 10000).toFixed(1)}
                    </td>
                    <td className="py-2 px-2 text-right text-gold-600 dark:text-gold-400">
                      100.0%
                    </td>
                    <td className="py-2 px-2 text-right">1.00x</td>
                  </tr>
                </tfoot>
              </table>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800 text-[11px] text-zinc-400">
            <strong>Risk/Weight Ratio:</strong> &gt;1.0 indicates an asset consuming more risk than capital; &lt;1.0 marks a natural hedge or volatility dampener.
          </div>
        </div>
      </div>

      {/* Row 2: Correlation-Aware VaR / ES and Benchmark Allocations */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (6 cols): Correlation VaR, Tail Penalty & Incremental VaR */}
        <div className="lg:col-span-6 panel p-5">
          <div className="flex items-center justify-between pb-3 border-b border-zinc-100 dark:border-zinc-800">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-gold-500" />
              <h4 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                Correlation-Aware VaR &amp; Fat-Tail Gap
              </h4>
            </div>

            <div className="flex items-center gap-1 bg-zinc-100 dark:bg-zinc-800/80 p-0.5 rounded-lg text-xs font-mono">
              <button
                type="button"
                onClick={() => setActiveConfidence(95)}
                className={`px-2 py-0.5 rounded-md font-medium transition-colors ${
                  activeConfidence === 95
                    ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-zinc-100 shadow-xs'
                    : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100'
                }`}
              >
                95% Conf
              </button>
              <button
                type="button"
                onClick={() => setActiveConfidence(99)}
                className={`px-2 py-0.5 rounded-md font-medium transition-colors ${
                  activeConfidence === 99
                    ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-zinc-100 shadow-xs'
                    : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100'
                }`}
              >
                99% Conf
              </button>
            </div>
          </div>

          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 mb-3">
            Parametric VaR assumes Gaussian joint normal distribution. The historical VaR gap reveals the correlation tail penalty (co-crash risk) in realistic market conditions.
          </p>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 my-3">
            <div className="p-3 bg-zinc-50 dark:bg-zinc-900/50 rounded-lg border border-zinc-100 dark:border-zinc-800">
              <div className="text-[11px] text-zinc-500 dark:text-zinc-400">
                Parametric VaR ({activeConfidence}%)
              </div>
              <div className="text-lg font-mono font-bold text-zinc-800 dark:text-zinc-200 mt-0.5">
                {(
                  (activeConfidence === 95
                    ? varResult.parametricVaR95
                    : varResult.parametricVaR99) * 100
                ).toFixed(2)}
                %
              </div>
              <div className="text-[10px] text-zinc-400">From wᵀ Σ w</div>
            </div>

            <div className="p-3 bg-zinc-50 dark:bg-zinc-900/50 rounded-lg border border-zinc-100 dark:border-zinc-800">
              <div className="text-[11px] text-zinc-500 dark:text-zinc-400">
                Historical VaR ({activeConfidence}%)
              </div>
              <div className="text-lg font-mono font-bold text-red-600 dark:text-red-400 mt-0.5">
                {(
                  (activeConfidence === 95
                    ? varResult.historicalVaR95
                    : varResult.historicalVaR99) * 100
                ).toFixed(2)}
                %
              </div>
              <div className="text-[10px] text-zinc-400">Empirical distribution</div>
            </div>

            <div className="p-3 bg-zinc-50 dark:bg-zinc-900/50 rounded-lg border border-zinc-100 dark:border-zinc-800 col-span-2 sm:col-span-1">
              <div className="text-[11px] text-zinc-500 dark:text-zinc-400">
                Tail Penalty Gap
              </div>
              <div className="text-lg font-mono font-bold text-gold-600 dark:text-gold-400 mt-0.5">
                +{(varResult.tailPenalty * 10000).toFixed(0)} bps
              </div>
              <div className="text-[10px] text-zinc-400">Hist VaR - Param VaR</div>
            </div>
          </div>

          <div className="p-3 bg-zinc-50 dark:bg-zinc-900/40 rounded-lg border border-zinc-100 dark:border-zinc-800 mb-4 flex items-center justify-between text-xs font-mono">
            <div>
              <span className="text-zinc-500 dark:text-zinc-400">
                Expected Shortfall / CVaR ({activeConfidence}%):{' '}
              </span>
              <strong className="text-red-600 dark:text-red-400">
                {(
                  (activeConfidence === 95
                    ? varResult.expectedShortfall95
                    : varResult.expectedShortfall99) * 100
                ).toFixed(2)}
                %
              </strong>
            </div>
            <span className="text-[11px] text-zinc-400">Average loss in worst 5% / 1%</span>
          </div>

          {/* Incremental VaR (IVaR) Ranking */}
          <div>
            <div className="flex items-center justify-between text-xs font-bold text-zinc-900 dark:text-zinc-100 mb-2">
              <span>Incremental VaR (IVaR) — Impact of +1% Capital Tilt</span>
              <span className="font-mono text-zinc-400 text-[11px]">Δ VaR (bps)</span>
            </div>

            <div className="space-y-1.5 max-h-[160px] overflow-y-auto pr-1">
              {varResult.incrementalVaR.map((item) => {
                const isIncreasing = item.ivarBps > 0;
                return (
                  <div
                    key={item.ticker}
                    className="flex items-center justify-between px-2.5 py-1.5 rounded bg-zinc-50/80 dark:bg-zinc-900/40 border border-zinc-100 dark:border-zinc-800/60 text-xs font-mono"
                  >
                    <span className="font-bold text-zinc-700 dark:text-zinc-300">
                      {item.ticker}
                    </span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-1.5 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${
                            isIncreasing ? 'bg-red-500' : 'bg-gold-500'
                          }`}
                          style={{
                            width: `${Math.min(100, Math.abs(item.ivarBps) * 8)}%`,
                          }}
                        />
                      </div>
                      <span
                        className={`font-semibold min-w-[55px] text-right ${
                          isIncreasing
                            ? 'text-red-600 dark:text-red-400'
                            : 'text-gold-600 dark:text-gold-400'
                        }`}
                      >
                        {isIncreasing ? '+' : ''}
                        {item.ivarBps.toFixed(1)} bps
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column (6 cols): Strategic Benchmark Allocations */}
        <div className="lg:col-span-6 panel p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-zinc-100 dark:border-zinc-800">
              <div className="flex items-center gap-2">
                <Scale className="w-4 h-4 text-gold-500" />
                <h4 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                  Strategic Benchmark Comparisons
                </h4>
              </div>
              <span className="text-xs font-mono text-zinc-400">Analytical Solutions</span>
            </div>

            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 mb-3">
              Compare your current allocation against three canonical portfolio models solved from the shrunk covariance matrix:
            </p>

            <div className="space-y-3">
              {/* Benchmark 1: Minimum Variance */}
              <div className="p-3 bg-gold-50/60 dark:bg-gold-950/20 border border-gold-200 dark:border-gold-800/40 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Award className="w-4 h-4 text-gold-600 dark:text-gold-400" />
                    <span className="text-xs font-bold text-zinc-900 dark:text-zinc-100">
                      Minimum-Variance Portfolio (w_mv)
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={loadMinVariance}
                    className="text-[11px] font-mono px-2 py-0.5 rounded bg-gold-600 text-white hover:bg-gold-700 transition-colors"
                  >
                    Apply Weights
                  </button>
                </div>
                <div className="flex items-center justify-between mt-2 text-xs font-mono">
                  <span className="text-zinc-500 dark:text-zinc-400">
                    Annualized Volatility:
                  </span>
                  <strong className="text-gold-700 dark:text-gold-300">
                    {(benchmarks.minVarVol * annualFactor * 100).toFixed(2)}%
                  </strong>
                </div>
                <div className="text-[10px] text-zinc-500 dark:text-zinc-400 mt-1">
                  Solved as <code className="text-gold-600 dark:text-gold-400">Σ̂⁻¹ 𝟏 / (𝟏ᵀ Σ̂⁻¹ 𝟏)</code> using Ledoit-Wolf shrinkage to eliminate estimation instability.
                </div>
              </div>

              {/* Benchmark 2: Risk Parity */}
              <div className="p-3 bg-zinc-50 dark:bg-zinc-900/40 border border-zinc-100 dark:border-zinc-800 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Scale className="w-4 h-4 text-gold-600 dark:text-gold-300" />
                    <span className="text-xs font-bold text-zinc-900 dark:text-zinc-100">
                      Risk Parity / Equal Risk Contribution (1/σ)
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={loadRiskParity}
                    className="text-[11px] font-mono px-2 py-0.5 rounded bg-zinc-200 dark:bg-zinc-700 hover:bg-zinc-300 dark:hover:bg-zinc-600 text-zinc-800 dark:text-zinc-200 transition-colors"
                  >
                    Apply Weights
                  </button>
                </div>
                <div className="flex items-center justify-between mt-2 text-xs font-mono">
                  <span className="text-zinc-500 dark:text-zinc-400">
                    Annualized Volatility:
                  </span>
                  <strong className="text-gold-600 dark:text-gold-300 dark:text-gold-600 dark:text-gold-300">
                    {(benchmarks.riskParityVol * annualFactor * 100).toFixed(2)}%
                  </strong>
                </div>
                <div className="text-[10px] text-zinc-500 dark:text-zinc-400 mt-1">
                  Allocates capital inversely proportional to individual asset standard deviation.
                </div>
              </div>

              {/* Benchmark 3: Equal-Weight 1/N */}
              <div className="p-3 bg-zinc-50 dark:bg-zinc-900/40 border border-zinc-100 dark:border-zinc-800 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-zinc-500" />
                    <span className="text-xs font-bold text-zinc-900 dark:text-zinc-100">
                      Equal Weight (1/N Naive Diversification)
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={loadEqualWeights}
                    className="text-[11px] font-mono px-2 py-0.5 rounded bg-zinc-200 dark:bg-zinc-700 hover:bg-zinc-300 dark:hover:bg-zinc-600 text-zinc-800 dark:text-zinc-200 transition-colors"
                  >
                    Apply Weights
                  </button>
                </div>
                <div className="flex items-center justify-between mt-2 text-xs font-mono">
                  <span className="text-zinc-500 dark:text-zinc-400">
                    Annualized Volatility:
                  </span>
                  <strong className="text-zinc-700 dark:text-zinc-300">
                    {(benchmarks.equalWeightVol * annualFactor * 100).toFixed(2)}%
                  </strong>
                </div>
                <div className="text-[10px] text-zinc-500 dark:text-zinc-400 mt-1">
                  DeMarzo &amp; Pflueger benchmark: robust to parameter uncertainty and out-of-sample estimation noise.
                </div>
              </div>
            </div>
          </div>

          {/* Estimation Caveat Banner */}
          <div className="mt-4 p-3 bg-zinc-100 dark:bg-zinc-900/80 rounded-lg border border-zinc-200 dark:border-zinc-800 text-[11px] text-zinc-600 dark:text-zinc-400">
            <strong>Estimation Risk Notice:</strong> Without shrinkage, unconstrained quadratic programming on sample covariance frequently outputs extreme leveraged corner weights (+350% / -250%). Ledoit-Wolf shrinkage dampens small noisy eigenvalues to keep allocations actionable.
          </div>
        </div>
      </div>

      {/* Expandable Mathematical Reference Panel */}
      <div className="panel p-5">
        <button
          type="button"
          onClick={() => setShowMathReference((prev) => !prev)}
          className="w-full flex items-center justify-between text-left text-sm font-bold text-zinc-900 dark:text-zinc-100"
        >
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-gold-500" />
            <span>Mathematical Formulations (Euler Risk Decomposition &amp; Diversification Ratio)</span>
          </div>
          {showMathReference ? (
            <ChevronUp className="w-4 h-4 text-zinc-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-zinc-400" />
          )}
        </button>

        {showMathReference && (
          <div className="mt-4 pt-4 border-t border-zinc-100 dark:border-zinc-800 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <div className="text-xs font-bold text-zinc-800 dark:text-zinc-200 mb-2">
                Portfolio Variance
              </div>
              <PortfolioVarianceEquation display="block" />
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-2">
                Quadratic form where w is the weight vector and Σ is the asset covariance matrix.
              </p>
            </div>

            <div>
              <div className="text-xs font-bold text-zinc-800 dark:text-zinc-200 mb-2">
                Euler Marginal Risk Contribution
              </div>
              <RiskContributionEquation display="block" />
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-2">
                Marginal risk ∂σ_p / ∂w_i multiplied by weight w_i decomposes total volatility exactly: Σ CCTR_i = σ_p.
              </p>
            </div>

            <div>
              <div className="text-xs font-bold text-zinc-800 dark:text-zinc-200 mb-2">
                Choueifaty Diversification Ratio (DR)
              </div>
              <DiversificationRatioEquation display="block" />
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-2">
                Ratio of weighted average individual volatility to diversified portfolio volatility. A higher ratio indicates superior diversification.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
