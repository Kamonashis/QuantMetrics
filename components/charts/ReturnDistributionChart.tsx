'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { ReturnStats } from '@/lib/quant-engine';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

interface ReturnDistributionChartProps {
  returns: number[];
  stats: ReturnStats;
}

export function ReturnDistributionChart({ returns, stats }: ReturnDistributionChartProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const { traces, layout } = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (!returns || returns.length === 0) return { traces: [], layout: {} };

    const binCount = 35;
    const min = stats.min;
    const max = stats.max;
    const range = max - min || 0.01;
    const binWidth = range / binCount;

    const counts: number[] = new Array(binCount).fill(0);
    for (const r of returns) {
      const idx = Math.min(binCount - 1, Math.max(0, Math.floor((r - min) / binWidth)));
      counts[idx]++;
    }

    const binCentersPct = Array.from({ length: binCount }, (_, i) => (min + (i + 0.5) * binWidth) * 100);
    const mean = stats.mean;
    const std = stats.stdDev || 0.01;
    const numSteps = 100;
    const normX: number[] = [];
    const normY: number[] = [];
    for (let s = 0; s <= numSteps; s++) {
      const xVal = min + (s / numSteps) * range;
      const z = (xVal - mean) / std;
      const phi = (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z);
      normX.push(xVal * 100);
      normY.push(returns.length * binWidth * phi);
    }

    return {
      traces: [
        {
          type: 'bar',
          name: 'Empirical Returns',
          x: binCentersPct,
          y: counts,
          marker: {
            color: 'rgba(139, 139, 147, 0.58)',
            line: { color: QUANT_COLORS.goldLight, width: 0.6 },
          },
          hovertemplate: '%{x:+.2f}% · %{y} days<extra>Empirical</extra>',
        },
        {
          type: 'scatter',
          mode: 'lines',
          name: 'Gaussian Reference',
          x: normX,
          y: normY,
          line: { color: QUANT_COLORS.goldLight, width: 2, dash: 'dash' },
          hovertemplate: '%{x:+.2f}% · %{y:.1f} exp<extra>Gaussian</extra>',
        },
      ],
      layout: {
        bargap: 0.06,
        shapes: [
          {
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: mean * 100,
            x1: mean * 100,
            y0: 0,
            y1: 1,
            line: { color: QUANT_COLORS.positive, width: 1.4, dash: 'dot' },
          },
        ],
        annotations: [
          {
            x: mean * 100,
            y: 1.0,
            xref: 'x',
            yref: 'paper',
            text: `Mean ${(mean * 100).toFixed(2)}%`,
            showarrow: false,
            xanchor: 'left',
            yanchor: 'top',
            xshift: 5,
            font: { size: 9.5, color: QUANT_COLORS.positive },
          },
        ],
        xaxis: {
          ticksuffix: '%',
          gridcolor: tokens.grid,
        },
        yaxis: {
          title: { text: 'Frequency', font: { size: 10, color: tokens.textMuted } },
          rangemode: 'tozero',
          gridcolor: tokens.grid,
        },
      },
    };
  }, [returns, stats, tokens]);

  if (!returns || returns.length === 0) {
    return null;
  }

  return (
    <div className="panel p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
            Return Distribution &amp; Moments
          </h3>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Empirical frequency vs. theoretical normal distribution
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
          <div>
            <span className="text-zinc-500 dark:text-zinc-400">Skewness:</span>{' '}
            <span className="font-mono font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
              {stats.skewness.toFixed(3)}
            </span>
          </div>
          <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">
            ·
          </span>
          <div>
            <span className="text-zinc-500 dark:text-zinc-400">Excess Kurtosis:</span>{' '}
            <span className="font-mono font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
              {stats.excessKurtosis.toFixed(3)}
            </span>
          </div>
          <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">
            ·
          </span>
          <div>
            <span className="text-zinc-500 dark:text-zinc-400">Jarque-Bera:</span>{' '}
            <span className="font-mono font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
              {stats.jarqueBera.toFixed(1)}
            </span>
          </div>
        </div>
      </div>

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="x unified"
        showLegend
        height={265}
        exportFilename="quantmetrics-return-distribution"
        uirevisionKey={`rdist-${returns.length}-${stats.mean.toFixed(5)}`}
        showWheelHint
      />

      <div className="flex items-center justify-end text-[11px] text-zinc-500 dark:text-zinc-400 mt-2 pt-2 border-t border-[var(--hairline)]">
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
  );
}
