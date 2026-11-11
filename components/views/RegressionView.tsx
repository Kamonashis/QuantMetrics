'use client';

import React, { useState } from 'react';
import {
  PriceBar,
  calculateReturns,
  performRegressionAnalysis,
} from '@/lib/quant-engine';
import { ResidualDiagnosticsChart } from '@/components/charts/ResidualDiagnosticsChart';
import {
  RegressionModelEquation,
  OlsMatrixEquation,
  RobustIrlsEquation,
  VifEquation,
  ExpectedReturnEquation,
  MathSymbol,
} from '@/components/math/MathEquations';
import { ShimmerButton } from '@/components/ui/ShimmerButton';
import { Play, RotateCcw, HelpCircle, ArrowUpRight, TrendingUp, Info } from 'lucide-react';

interface RegressionViewProps {
  regressionTickers: string;
  onRegressionTickersChange: (val: string) => void;
  startDate: string;
  onStartDateChange: (val: string) => void;
  endDate: string;
  onEndDateChange: (val: string) => void;
  returnType: 'simple' | 'log';
  allStockBars: { [ticker: string]: PriceBar[] };
  isLoading: boolean;
  onRunRegression: (tickers: string[]) => void;
}

export function RegressionView({
  regressionTickers,
  onRegressionTickersChange,
  startDate,
  onStartDateChange,
  endDate,
  onEndDateChange,
  returnType,
  allStockBars,
  isLoading,
  onRunRegression,
}: RegressionViewProps) {
  const [localInput, setLocalInput] = useState(regressionTickers);
  const [regressionType, setRegressionType] = useState<
    'Standard OLS' | 'Robust Regression' | 'Rolling Regression'
  >('Standard OLS');
  const [rollingWindow, setRollingWindow] = useState<number>(60);

  const parsedTickers = React.useMemo(() => {
    return regressionTickers
      .split(',')
      .map((t) => t.trim().toUpperCase())
      .filter((t) => t.length > 0);
  }, [regressionTickers]);

  const dependentTicker = parsedTickers[0] || 'AAPL';
  const independentTickers = React.useMemo(() => {
    return parsedTickers.slice(1).length > 0 ? parsedTickers.slice(1) : ['SPY'];
  }, [parsedTickers]);

  // Align dates and returns
  const { returnsByTicker, dates } = React.useMemo(() => {
    const map: { [ticker: string]: number[] } = {};
    let commonDates: string[] = [];

    for (const t of [dependentTicker, ...independentTickers]) {
      const raw = allStockBars[t] || [];
      const filtered = raw.filter((b) => {
        if (startDate && b.date < startDate) return false;
        if (endDate && b.date > endDate) return false;
        return true;
      });
      const prices = filtered.map((b) => b.close);
      map[t] = calculateReturns(prices, returnType);
      if (filtered.length > commonDates.length) {
        commonDates = filtered.map((b) => b.date);
      }
    }
    return { returnsByTicker: map, dates: commonDates };
  }, [dependentTicker, independentTickers, allStockBars, startDate, endDate, returnType]);

  // Execute regression
  const regressionSummary = React.useMemo(() => {
    return performRegressionAnalysis(
      returnsByTicker,
      dates,
      dependentTicker,
      independentTickers,
      regressionType,
      rollingWindow
    );
  }, [returnsByTicker, dates, dependentTicker, independentTickers, regressionType, rollingWindow]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const list = localInput
      .split(',')
      .map((t) => t.trim().toUpperCase())
      .filter((t) => t.length > 0);
    if (list.length >= 2) {
      onRegressionTickersChange(list.join(', '));
      onRunRegression(list);
    }
  };

  const isMultiple = independentTickers.length > 1;

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Parameter Control Card */}
      <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-100 dark:border-slate-800/80">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono font-semibold uppercase tracking-wider text-purple-600 dark:text-purple-400">
                  Econometric Factor Analysis
                </span>
                <span className="text-slate-300 dark:text-slate-700" aria-hidden="true">·</span>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {isMultiple ? 'Multiple Linear Regression' : 'Simple Linear Regression (CAPM Beta)'}
                </span>
              </div>
              <h2 className="text-xl font-bold tracking-tight text-slate-900 dark:text-slate-100 mt-0.5">
                Factor Sensitivity &amp; Residual Econometrics
              </h2>
              <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-slate-600 dark:text-slate-300">
                <RegressionModelEquation display="inline" />
                <span className="text-slate-300 dark:text-slate-700" aria-hidden="true">·</span>
                {regressionType === 'Robust Regression' ? (
                  <RobustIrlsEquation display="inline" />
                ) : (
                  <OlsMatrixEquation display="inline" />
                )}
              </div>
            </div>


            <div className="flex items-center gap-2">
              <ShimmerButton
                type="submit"
                disabled={isLoading}
                className="px-4 py-2"
              >
                {isLoading ? (
                  <>
                    <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                    <span>Fitting Model...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current text-amber-500" />
                    <span>Run Regression</span>
                  </>
                )}
              </ShimmerButton>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
            {/* Tickers Input */}
            <div className="lg:col-span-2">
              <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1">
                Tickers: <strong className="text-purple-600 dark:text-purple-400">Y (Dependent)</strong>, followed by <strong className="text-sky-600 dark:text-sky-400">X₁, X₂, ... (Independent)</strong>
              </label>
              <input
                type="text"
                value={localInput}
                onChange={(e) => setLocalInput(e.target.value)}
                placeholder="AAPL, SPY, QQQ"
                className="w-full px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 font-mono font-bold text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-1 focus:ring-purple-500"
              />
              <span className="text-[11px] text-slate-400 block mt-1">
                Active: Dependent <strong>{dependentTicker}</strong> ~ Intercept + {independentTickers.join(' + ')}
              </span>
            </div>

            {/* Regression Type */}
            <div>
              <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1">
                Regression Engine
              </label>
              <select
                value={regressionType}
                onChange={(e) =>
                  setRegressionType(
                    e.target.value as 'Standard OLS' | 'Robust Regression' | 'Rolling Regression'
                  )
                }
                className="w-full px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 font-medium focus:outline-none focus:ring-1 focus:ring-purple-500"
              >
                <option value="Standard OLS">Standard OLS</option>
                <option value="Robust Regression">Robust Regression (Huber IRLS)</option>
                <option value="Rolling Regression">Rolling Window Regression</option>
              </select>
            </div>

            {/* Rolling Window / Settings */}
            <div>
              <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1">
                {regressionType === 'Rolling Regression' ? 'Rolling Window (days)' : 'Date Range Filter'}
              </label>
              {regressionType === 'Rolling Regression' ? (
                <input
                  type="number"
                  min="20"
                  max="180"
                  value={rollingWindow}
                  onChange={(e) => setRollingWindow(Math.max(20, parseInt(e.target.value, 10) || 60))}
                  className="w-full px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-1 focus:ring-purple-500"
                />
              ) : (
                <div className="grid grid-cols-2 gap-1.5">
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => onStartDateChange(e.target.value)}
                    className="px-2 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100 text-[11px]"
                  />
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => onEndDateChange(e.target.value)}
                    className="px-2 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 font-mono text-slate-900 dark:text-slate-100 text-[11px]"
                  />
                </div>
              )}
            </div>
          </div>
        </form>

        {/* Model Presets */}
        <div className="flex flex-wrap items-center gap-2 mt-4 pt-3 border-t border-slate-100 dark:border-slate-800/60 text-xs">
          <span className="text-slate-400">Benchmark Factor Presets:</span>
          {[
            { label: 'Single Factor (CAPM): AAPL ~ SPY', val: 'AAPL, SPY' },
            { label: 'Two-Factor Tech: NVDA ~ SPY, QQQ', val: 'NVDA, SPY, QQQ' },
            { label: 'Cloud Beta: MSFT ~ GOOGL, SPY', val: 'MSFT, GOOGL, SPY' },
          ].map((preset) => (
            <button
              key={preset.label}
              onClick={() => {
                setLocalInput(preset.val);
                onRegressionTickersChange(preset.val);
                onRunRegression(preset.val.split(',').map((s) => s.trim()));
              }}
              className="px-2.5 py-1 rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/60 hover:border-purple-500 text-slate-700 dark:text-slate-300 font-mono text-[11px] transition-colors"
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* Observation Availability Notice */}
      {regressionSummary.nObservations === 0 && (
        <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-xl text-xs text-amber-700 dark:text-amber-400 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 flex-shrink-0" />
            <span>
              {isLoading
                ? `Retrieving and aligning price series for ${dependentTicker} and ${independentTickers.join(', ')}...`
                : `Insufficient overlapping price observations for ${dependentTicker} and ${independentTickers.join(', ')} in selected date range. Click 'Run Regression' to fetch price bars.`}
            </span>
          </div>
        </div>
      )}

      {/* Model Fit Diagnostics Summary Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 p-4 rounded-xl">
          <span className="text-xs text-slate-500 dark:text-slate-400 block">
            R-Squared (<MathSymbol name="r_squared" />)
          </span>
          <span className="font-mono text-xl font-bold text-slate-900 dark:text-slate-100 tabular-nums">
            {(regressionSummary.rSquared * 100).toFixed(2)}%
          </span>
          <span className="text-[11px] text-slate-500 dark:text-slate-400 block mt-0.5">
            Variance explained
          </span>
        </div>

        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 p-4 rounded-xl">
          <span className="text-xs text-slate-500 dark:text-slate-400 block">
            Adjusted <MathSymbol name="adj_r_squared" />
          </span>
          <span className="font-mono text-xl font-bold text-sky-500 tabular-nums">
            {(regressionSummary.adjRSquared * 100).toFixed(2)}%
          </span>
          <span className="text-[11px] text-slate-500 dark:text-slate-400 block mt-0.5">
            Degrees of freedom adj.
          </span>
        </div>

        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 p-4 rounded-xl">
          <span className="text-xs text-slate-500 dark:text-slate-400 block">F-Statistic</span>
          <span className="font-mono text-xl font-bold text-slate-900 dark:text-slate-100 tabular-nums">
            {regressionSummary.fStatistic.toFixed(1)}
          </span>
          <span className="text-[11px] text-slate-500 dark:text-slate-400 block mt-0.5">
            p &lt; 0.001 (Joint sig.)
          </span>
        </div>

        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 p-4 rounded-xl">
          <span className="text-xs text-slate-500 dark:text-slate-400 block">Akaike / Bayesian</span>
          <span className="font-mono text-sm font-bold text-slate-900 dark:text-slate-100 tabular-nums">
            {regressionSummary.aic.toFixed(0)} / {regressionSummary.bic.toFixed(0)}
          </span>
          <span className="text-[11px] text-slate-500 dark:text-slate-400 block mt-0.5">
            Model selection AIC/BIC
          </span>
        </div>

        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 p-4 rounded-xl">
          <span className="text-xs text-slate-500 dark:text-slate-400 block">Durbin-Watson</span>
          <span className="font-mono text-xl font-bold text-emerald-500 tabular-nums">
            {regressionSummary.durbinWatson.toFixed(2)}
          </span>
          <span className="text-[11px] text-slate-500 dark:text-slate-400 block mt-0.5">
            ~2.0 (No serial residual corr)
          </span>
        </div>

        <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 p-4 rounded-xl bg-purple-50/20 dark:bg-purple-950/20 border-purple-200 dark:border-purple-800/40">
          <div className="flex items-center justify-between">
            <span className="text-xs text-purple-600 dark:text-purple-400 block font-semibold">Next-Day Forecast</span>
            <ExpectedReturnEquation display="inline" className="text-[10px]" />
          </div>
          <span
            className={`font-mono text-xl font-bold tabular-nums ${
              regressionSummary.forecastNextDay.expectedReturn >= 0
                ? 'text-emerald-500'
                : 'text-rose-500'
            }`}
          >
            {(regressionSummary.forecastNextDay.expectedReturn * 100).toFixed(2)}%
          </span>
          <span className="text-[11px] text-slate-500 dark:text-slate-400 block mt-0.5">
            95% CI: [{(regressionSummary.forecastNextDay.predictionInterval95[0] * 100).toFixed(1)}%,{' '}
            {(regressionSummary.forecastNextDay.predictionInterval95[1] * 100).toFixed(1)}%]
          </span>
        </div>
      </div>

      {/* Regression Coefficients & VIF Table */}
      <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs overflow-x-auto">
        <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-100 dark:border-slate-800">
          <div>
            <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
              Parameter Estimates, Significance &amp; Multicollinearity (VIF)
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Dependent Variable: {dependentTicker} · Observations: {regressionSummary.nObservations}
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
            Regression Error s: {(regressionSummary.standardErrorOfRegression * 100).toFixed(3)}%
          </span>
        </div>

        <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-800 text-xs font-mono">
          <thead>
            <tr>
              <th className="py-2.5 px-3 text-left font-semibold text-slate-500">Variable</th>
              <th className="py-2.5 px-3 text-right font-semibold text-slate-500">
                Coefficient (<MathSymbol name="beta" />)
              </th>
              <th className="py-2.5 px-3 text-right font-semibold text-slate-500">Std Error</th>
              <th className="py-2.5 px-3 text-right font-semibold text-slate-500">t-Statistic</th>
              <th className="py-2.5 px-3 text-right font-semibold text-slate-500">p-Value (P&gt;|t|)</th>
              <th className="py-2.5 px-3 text-right font-semibold text-slate-500">
                VIF (<VifEquation display="inline" className="text-[10px]" />)
              </th>
            </tr>
          </thead>

          <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
            {regressionSummary.coefficients.map((row) => {
              const isSignificant = row.pValue < 0.05;
              const hasHighVif = row.vif !== null && row.vif > 5;

              return (
                <tr key={row.variable} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/30">
                  <td className="py-2.5 px-3 font-bold text-slate-900 dark:text-slate-100">
                    {row.variable}
                  </td>
                  <td className="py-2.5 px-3 text-right font-bold tabular-nums text-slate-900 dark:text-slate-100">
                    {row.coefficient.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 text-right tabular-nums text-slate-600 dark:text-slate-400">
                    {row.stdError.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 text-right tabular-nums font-semibold text-slate-800 dark:text-slate-200">
                    {row.tStatistic.toFixed(3)}
                  </td>
                  <td className="py-2.5 px-3 text-right tabular-nums font-bold">
                    <span
                      className={
                        isSignificant
                          ? 'text-emerald-600 dark:text-emerald-400'
                          : 'text-slate-400'
                      }
                    >
                      {row.pValue < 0.0001 ? '< 0.0001 ***' : `${row.pValue.toFixed(4)} ${isSignificant ? '**' : ''}`}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-right tabular-nums font-semibold">
                    {row.vif !== null ? (
                      <span
                        className={
                          hasHighVif
                            ? 'text-rose-500 font-bold'
                            : 'text-slate-700 dark:text-slate-300'
                        }
                      >
                        {row.vif.toFixed(2)} {hasHighVif ? '(High Collinearity)' : ''}
                      </span>
                    ) : (
                      <span className="text-slate-400">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400 mt-3 pt-2 border-t border-slate-100 dark:border-slate-800">
          <span>* p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001</span>
          <span>VIF &lt; 5 indicates healthy absence of severe multicollinearity</span>
        </div>
      </div>

      {/* Residual Analysis Diagnostics Charts */}
      <ResidualDiagnosticsChart summary={regressionSummary} />
    </div>
  );
}
