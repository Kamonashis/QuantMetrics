'use client';

import React, { useState } from 'react';
import { DecompositionResult, AdfTestResult } from '@/lib/quant-engine';
import {
  DecompositionAdditiveEquation,
  DecompositionMultiplicativeEquation,
  AdfEquation,
} from '@/components/math/MathEquations';

interface DecompositionChartProps {
  decomposition: DecompositionResult;
  adfTest: AdfTestResult;
}

export function DecompositionChart({
  decomposition,
  adfTest,
}: DecompositionChartProps) {
  const [activeComponent, setActiveComponent] = useState<'all' | 'observed' | 'trend' | 'seasonal' | 'residual'>('all');

  const { dates, observed, trend, seasonal, residual, model, period } = decomposition;
  const n = dates.length;

  if (n === 0) return null;


  // Helper to render mini sparkline/path
  const renderSubplot = (
    title: string,
    data: (number | null)[],
    color: string,
    unit: string = ''
  ) => {
    // Filter non-nulls for min/max
    const valid = data.filter((v): v is number => v !== null && !isNaN(v));
    if (valid.length === 0) return null;

    const min = Math.min(...valid);
    const max = Math.max(...valid);
    const range = max - min || 1;

    const width = 760;
    const height = 110;
    const padLeft = 55;
    const padRight = 15;
    const padTop = 15;
    const padBottom = 20;

    const chartW = width - padLeft - padRight;
    const chartH = height - padTop - padBottom;

    const getX = (i: number) => padLeft + (i / Math.max(1, n - 1)) * chartW;
    const getY = (val: number) => padTop + chartH - ((val - min) / range) * chartH;

    const points: string[] = [];
    for (let i = 0; i < n; i++) {
      const v = data[i];
      if (v !== null && !isNaN(v)) {
        points.push(`${getX(i)},${getY(v)}`);
      }
    }

    return (
      <div className="bg-zinc-50/70 dark:bg-[#080d1a] border border-zinc-200 dark:border-zinc-800/60 rounded-lg p-3">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="font-semibold text-zinc-800 dark:text-zinc-200">
            {title}
          </span>
          <div className="flex items-center gap-2 font-mono text-[11px] text-zinc-500 dark:text-zinc-400 tabular-nums">
            <span>Min: {min.toFixed(2)}{unit}</span>
            <span>·</span>
            <span>Max: {max.toFixed(2)}{unit}</span>
          </div>
        </div>

        <div className="w-full aspect-[7.6/1.1]">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
            {/* Zero line if crosses zero */}
            {min < 0 && max > 0 && (
              <line
                x1={padLeft}
                y1={getY(0)}
                x2={width - padRight}
                y2={getY(0)}
                stroke="currentColor"
                className="text-zinc-300 dark:text-zinc-700"
                strokeDasharray="2 2"
              />
            )}

            <path
              d={`M ${points.join(' L ')}`}
              fill="none"
              stroke={color}
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
      </div>
    );
  };

  return (
    <div className="panel p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
              Seasonal Decomposition &amp; ADF Stationarity
            </h3>
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              · {model.toUpperCase()} (m = {period} days)
            </span>
          </div>
          <div className="mt-1 flex items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400">
            <span>Model:</span>
            {model === 'additive' ? (
              <DecompositionAdditiveEquation display="inline" className="text-zinc-700 dark:text-zinc-300" />
            ) : (
              <DecompositionMultiplicativeEquation display="inline" className="text-zinc-700 dark:text-zinc-300" />
            )}
          </div>
        </div>

        {/* Component selector */}
        <div className="flex items-center gap-1 p-1 bg-zinc-100 dark:bg-zinc-800/80 rounded-lg">
          {(['all', 'trend', 'seasonal', 'residual'] as const).map((comp) => (
            <button
              key={comp}
              onClick={() => setActiveComponent(comp)}
              className={`px-2.5 py-1 text-xs font-medium rounded-md capitalize transition-colors ${
                activeComponent === comp
                  ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-xs'
                  : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
              }`}
            >
              {comp}
            </button>
          ))}
        </div>
      </div>

      {/* Subplots */}
      <div className="space-y-3">
        {(activeComponent === 'all' || activeComponent === 'observed') &&
          renderSubplot('Observed Price (Y_t)', observed, '#06b6d4')}
        {(activeComponent === 'all' || activeComponent === 'trend') &&
          renderSubplot('Trend Component (T_t: Centered Moving Average)', trend, '#10b981')}
        {(activeComponent === 'all' || activeComponent === 'seasonal') &&
          renderSubplot('Seasonal Component (S_t: Repeating Periodic Factor)', seasonal, '#DEC084')}
        {(activeComponent === 'all' || activeComponent === 'residual') &&
          renderSubplot('Residual Noise Component (R_t: Isolated Remainder)', residual, '#f43f5e')}
      </div>

      {/* Augmented Dickey-Fuller (ADF) Test on Residuals Box */}
      <div className="mt-4 pt-4 border-t border-zinc-100 dark:border-zinc-800/80">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-zinc-700 dark:text-zinc-300 block">
              Augmented Dickey-Fuller (ADF) Test on Residuals
            </span>
            <div className="mt-0.5">
              <AdfEquation display="inline" className="text-xs text-zinc-600 dark:text-zinc-300" />
            </div>
          </div>
          <span
            className={`text-xs font-semibold ${
              adfTest.isStationary
                ? 'text-gold-600 dark:text-gold-400'
                : 'text-gold-600 dark:text-gold-400'
            }`}
          >
            {adfTest.isStationary ? 'Stationary (p < 0.05)' : 'Non-Stationary (Unit Root)'}
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-zinc-50 dark:bg-zinc-900/60 p-3 rounded-lg border border-zinc-200 dark:border-zinc-800/60 text-xs">
          <div>
            <div className="text-zinc-500 dark:text-zinc-400">Test Statistic (τ):</div>
            <div className="font-mono font-bold text-zinc-900 dark:text-zinc-100 text-sm tabular-nums">
              {adfTest.testStatistic.toFixed(3)}
            </div>
          </div>
          <div>
            <div className="text-zinc-500 dark:text-zinc-400">MacKinnon p-value:</div>
            <div className="font-mono font-bold text-zinc-900 dark:text-zinc-100 text-sm tabular-nums">
              {adfTest.pValue < 0.001 ? '< 0.001' : adfTest.pValue.toFixed(4)}
            </div>
          </div>
          <div>
            <div className="text-zinc-500 dark:text-zinc-400">1% Critical Value:</div>
            <div className="font-mono text-zinc-700 dark:text-zinc-300 tabular-nums">
              {adfTest.criticalValues['1%'].toFixed(3)}
            </div>
          </div>
          <div>
            <div className="text-zinc-500 dark:text-zinc-400">5% Critical Value:</div>
            <div className="font-mono text-zinc-700 dark:text-zinc-300 tabular-nums">
              {adfTest.criticalValues['5%'].toFixed(3)}
            </div>
          </div>
        </div>

        <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2">
          {adfTest.conclusion}
        </p>
      </div>
    </div>
  );
}
