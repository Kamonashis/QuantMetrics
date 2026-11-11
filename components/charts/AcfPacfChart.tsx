'use client';

import React, { useState } from 'react';
import { AcfPacfResult } from '@/lib/quant-engine';

interface AcfPacfChartProps {
  acfPacf: AcfPacfResult;
}

export function AcfPacfChart({ acfPacf }: AcfPacfResultProps & { acfPacf: AcfPacfResult }) {
  const [activeTab, setActiveTab] = useState<'acf' | 'pacf'>('acf');

  const { lags, acf, pacf, confidenceUpper, confidenceLower } = acfPacf;

  if (!lags || lags.length === 0) {
    return (
      <div className="h-56 flex items-center justify-center border border-slate-200 dark:border-slate-800 rounded-lg text-slate-500 text-xs">
        Insufficient return series for ACF/PACF computation
      </div>
    );
  }

  const values = activeTab === 'acf' ? acf : pacf;

  // Check how many lags exceed 95% Bartlett bounds
  const significantLags = values.filter(
    (v) => v > confidenceUpper || v < confidenceLower
  ).length;

  const width = 700;
  const height = 240;
  const padLeft = 45;
  const padRight = 20;
  const padTop = 20;
  const padBottom = 30;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;
  const zeroY = padTop + chartH / 2;

  // Clamped max limit to [-1, 1] or max value with margin
  const maxLimit = Math.max(0.5, Math.min(1.0, Math.max(...values.map(Math.abs), confidenceUpper * 1.5)));

  const getX = (lagIdx: number) =>
    padLeft + ((lagIdx + 0.5) / lags.length) * chartW;
  const getY = (val: number) =>
    padTop + chartH / 2 - (val / maxLimit) * (chartH / 2);

  const upperBandY = getY(confidenceUpper);
  const lowerBandY = getY(confidenceLower);

  return (
    <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/80 rounded-xl p-5 shadow-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm tracking-tight">
              {activeTab === 'acf' ? 'Autocorrelation (ACF)' : 'Partial Autocorrelation (PACF)'} of Squared Returns
            </h3>
            <span className="text-xs text-slate-500 dark:text-slate-400">
              · Volatility Clustering Diagnostics
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Dashed blue lines denote Bartlett&apos;s 95% white-noise confidence bounds (±1.96/√N)
          </p>
        </div>

        {/* Tab switch button control (allowed as interactive filter control) */}
        <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800/80 rounded-lg">
          <button
            onClick={() => setActiveTab('acf')}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
              activeTab === 'acf'
                ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            ACF (r_t²)
          </button>
          <button
            onClick={() => setActiveTab('pacf')}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
              activeTab === 'pacf'
                ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            PACF (r_t²)
          </button>
        </div>
      </div>

      <div className="relative w-full aspect-[7/2.4] min-h-[190px]">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
          {/* Zero baseline */}
          <line
            x1={padLeft}
            y1={zeroY}
            x2={width - padRight}
            y2={zeroY}
            stroke="currentColor"
            className="text-slate-400 dark:text-slate-600"
            strokeWidth="1.2"
          />

          {/* Upper confidence band */}
          <line
            x1={padLeft}
            y1={upperBandY}
            x2={width - padRight}
            y2={upperBandY}
            stroke="#38bdf8"
            strokeWidth="1.2"
            strokeDasharray="4 3"
          />
          <text
            x={padLeft - 6}
            y={upperBandY + 3}
            textAnchor="end"
            className="text-[9px] font-mono fill-sky-500 tabular-nums"
          >
            +{confidenceUpper.toFixed(2)}
          </text>

          {/* Lower confidence band */}
          <line
            x1={padLeft}
            y1={lowerBandY}
            x2={width - padRight}
            y2={lowerBandY}
            stroke="#38bdf8"
            strokeWidth="1.2"
            strokeDasharray="4 3"
          />
          <text
            x={padLeft - 6}
            y={lowerBandY + 3}
            textAnchor="end"
            className="text-[9px] font-mono fill-sky-500 tabular-nums"
          >
            {confidenceLower.toFixed(2)}
          </text>

          {/* Stems for each lag */}
          {lags.map((lag, i) => {
            const val = values[i];
            const x = getX(i);
            const y = getY(val);
            const isSignificant = val > confidenceUpper || val < confidenceLower;
            const stemColor = isSignificant ? '#f59e0b' : '#64748b';

            return (
              <g key={lag} className="group cursor-pointer">
                <line
                  x1={x}
                  y1={zeroY}
                  x2={x}
                  y2={y}
                  stroke={stemColor}
                  strokeWidth="2"
                  className="transition-colors group-hover:stroke-sky-400"
                />
                <circle
                  cx={x}
                  cy={y}
                  r="3.5"
                  fill={stemColor}
                  className="transition-all group-hover:r-5 group-hover:fill-sky-400"
                />
                {/* Lag number on X axis */}
                <text
                  x={x}
                  y={height - 8}
                  textAnchor="middle"
                  className="text-[9px] font-mono fill-slate-500 dark:fill-slate-400 tabular-nums"
                >
                  {lag}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs text-slate-500 dark:text-slate-400 mt-2 pt-2 border-t border-slate-100 dark:border-slate-800/60 gap-2">
        <div className="flex items-center gap-2">
          <span>Significant Lags (&gt;95% bound):</span>
          <span className="font-mono font-bold text-slate-900 dark:text-slate-100 tabular-nums">
            {significantLags} of {lags.length}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {significantLags > 2 ? (
            <span className="text-amber-600 dark:text-amber-400 font-medium">
              Persistent ARCH effects: Volatility clustering confirmed (GARCH recommended)
            </span>
          ) : (
            <span className="text-emerald-600 dark:text-emerald-400 font-medium">
              Weak autocorrelation: Return variance is approximately homoskedastic
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

interface AcfPacfResultProps {
  acfPacf: AcfPacfResult;
}
