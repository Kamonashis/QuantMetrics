'use client';

import React from 'react';
import { ReturnStats } from '@/lib/quant-engine';

interface ReturnDistributionChartProps {
  returns: number[];
  stats: ReturnStats;
}

export function ReturnDistributionChart({
  returns,
  stats,
}: ReturnDistributionChartProps) {
  if (!returns || returns.length === 0) {
    return null;
  }

  // 35 histogram bins
  const binCount = 35;
  const min = stats.min;
  const max = stats.max;
  const range = max - min || 0.01;
  const binWidth = range / binCount;

  const bins: number[] = new Array(binCount).fill(0);
  for (const r of returns) {
    const idx = Math.min(binCount - 1, Math.max(0, Math.floor((r - min) / binWidth)));
    bins[idx]++;
  }

  const maxCount = Math.max(...bins, 1);

  const width = 600;
  const height = 240;
  const padLeft = 45;
  const padRight = 20;
  const padTop = 20;
  const padBottom = 30;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const getX = (val: number) => padLeft + ((val - min) / range) * chartW;
  const getY = (count: number) => padTop + chartH - (count / maxCount) * chartH;

  // Theoretical Gaussian curve scaled to peak
  const normalPoints: string[] = [];
  const mean = stats.mean;
  const std = stats.stdDev || 0.01;
  const numSteps = 80;

  for (let s = 0; s <= numSteps; s++) {
    const xVal = min + (s / numSteps) * range;
    const z = (xVal - mean) / std;
    const phi = (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z);
    // Area scaling: total observations * binWidth * phi
    const expectedCount = returns.length * binWidth * phi;
    const yVal = getY(expectedCount);
    normalPoints.push(`${getX(xVal)},${Math.max(padTop, Math.min(padTop + chartH, yVal))}`);
  }

  return (
    <div className="panel p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
            Return Distribution & Moments
          </h3>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Empirical frequency vs. theoretical normal distribution
          </p>
        </div>

        {/* Clean unboxed statistics metrics without decorative pills */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
          <div>
            <span className="text-zinc-500 dark:text-zinc-400">Skewness:</span>{' '}
            <span className="font-mono font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
              {stats.skewness.toFixed(3)}
            </span>
          </div>
          <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
          <div>
            <span className="text-zinc-500 dark:text-zinc-400">Excess Kurtosis:</span>{' '}
            <span className="font-mono font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
              {stats.excessKurtosis.toFixed(3)}
            </span>
          </div>
          <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
          <div>
            <span className="text-zinc-500 dark:text-zinc-400">Jarque-Bera:</span>{' '}
            <span className="font-mono font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
              {stats.jarqueBera.toFixed(1)}
            </span>
          </div>
        </div>
      </div>

      <div className="relative w-full aspect-[6/2.4] min-h-[180px]">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
          {/* Grid lines */}
          {[0, 0.5, 1].map((pct) => {
            const count = Math.round(pct * maxCount);
            const yPos = getY(count);
            return (
              <g key={pct}>
                <line
                  x1={padLeft}
                  y1={yPos}
                  x2={width - padRight}
                  y2={yPos}
                  stroke="currentColor"
                  className="text-zinc-200 dark:text-zinc-800"
                  strokeDasharray="2 3"
                />
                <text
                  x={padLeft - 6}
                  y={yPos + 4}
                  textAnchor="end"
                  className="text-[10px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
                >
                  {count}
                </text>
              </g>
            );
          })}

          {/* Histogram Bars */}
          {bins.map((count, i) => {
            const binStart = min + i * binWidth;
            const x0 = getX(binStart);
            const x1 = getX(binStart + binWidth);
            const bWidth = Math.max(1, x1 - x0 - 1);
            const y = getY(count);
            const h = padTop + chartH - y;

            return (
              <rect
                key={i}
                x={x0}
                y={y}
                width={bWidth}
                height={h}
                fill="#38bdf8"
                className="opacity-70 hover:opacity-100 transition-opacity"
              />
            );
          })}

          {/* Normal Curve */}
          <path
            d={`M ${normalPoints.join(' ')}`}
            fill="none"
            stroke="#DEC084"
            strokeWidth="2"
            strokeDasharray="3 2"
          />

          {/* Mean vertical line */}
          <line
            x1={getX(mean)}
            y1={padTop}
            x2={getX(mean)}
            y2={padTop + chartH}
            stroke="#10b981"
            strokeWidth="1.5"
            strokeDasharray="4 2"
          />

          {/* X axis marks */}
          {[-0.04, -0.02, 0, 0.02, 0.04].map((tick) => {
            if (tick < min || tick > max) return null;
            return (
              <g key={tick}>
                <line
                  x1={getX(tick)}
                  y1={padTop + chartH}
                  x2={getX(tick)}
                  y2={padTop + chartH + 4}
                  stroke="currentColor"
                  className="text-zinc-400 dark:text-zinc-600"
                />
                <text
                  x={getX(tick)}
                  y={height - 8}
                  textAnchor="middle"
                  className="text-[10px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
                >
                  {(tick * 100).toFixed(0)}%
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="flex items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400 mt-2 pt-2 border-t border-zinc-100 dark:border-zinc-800/60">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 bg-gold-400 rounded-xs inline-block" />
            Empirical Returns
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-gold-500 inline-block border-b border-dashed" />
            Gaussian Reference
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-gold-500 inline-block" />
            Mean Return ({(stats.mean * 100).toFixed(2)}%)
          </span>
        </div>
        <div>
          {stats.excessKurtosis > 1 ? (
            <span className="text-gold-600 dark:text-gold-400 font-medium">
              Leptokurtic (Fat-Tailed Risk)
            </span>
          ) : (
            <span className="text-gold-600 dark:text-gold-400 font-medium">
              Mesokurtic / Normal
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
