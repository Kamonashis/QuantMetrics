'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { DailyRealizedVol } from '@/lib/quant-engine';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

interface JumpDecompositionChartProps {
  dailyRV: DailyRealizedVol[];
  ticker: string;
}

export function JumpDecompositionChart({
  dailyRV,
  ticker,
}: JumpDecompositionChartProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const { traces, layout, avgJumpRatio, jumpDayCount } = useMemo<{
    traces: Data[];
    layout: Partial<Layout>;
    avgJumpRatio: number;
    jumpDayCount: number;
  }>(() => {
    if (!dailyRV || dailyRV.length === 0) {
      return { traces: [], layout: {}, avgJumpRatio: 0, jumpDayCount: 0 };
    }

    let totalRv = 0;
    let totalJump = 0;
    let sigCount = 0;
    for (const d of dailyRV) {
      totalRv += d.realizedVariance;
      totalJump += d.jumpComponent;
      if (d.hasSignificantJump) sigCount++;
    }
    const ratio = totalRv > 0 ? (totalJump / totalRv) * 100 : 0;

    const dates = dailyRV.map((d) => d.date);
    const contVolPct = dailyRV.map((d) => d.annualizedContinuousVol * 100);
    const jumpDeltaPct = dailyRV.map((d) =>
      Math.max(0, (d.annualizedRealizedVol - d.annualizedContinuousVol) * 100)
    );
    const normalJump = dailyRV.map((d, i) => (d.hasSignificantJump ? 0 : jumpDeltaPct[i]));
    const sigJump = dailyRV.map((d, i) => (d.hasSignificantJump ? jumpDeltaPct[i] : 0));

    const sigDates: string[] = [];
    const sigTopY: number[] = [];
    dailyRV.forEach((d) => {
      if (d.hasSignificantJump) {
        sigDates.push(d.date);
        sigTopY.push(d.annualizedRealizedVol * 100 + 1.2);
      }
    });

    const dataTraces: Data[] = [
      {
        type: 'bar',
        name: 'Continuous Diffusion (BV_t)',
        x: dates,
        y: contVolPct,
        marker: { color: 'rgba(201, 164, 92, 0.78)' },
        hovertemplate: '%{y:.2f}%<extra>Continuous BV_t</extra>',
      },
      {
        type: 'bar',
        name: 'Normal Jump Noise',
        x: dates,
        y: normalJump,
        marker: { color: 'rgba(222, 192, 132, 0.45)' },
        hovertemplate: '+%{y:.2f}%<extra>Jump Noise</extra>',
      },
      {
        type: 'bar',
        name: 'Significant Jump (Z > 1.96)',
        x: dates,
        y: sigJump,
        marker: { color: QUANT_COLORS.negative },
        hovertemplate: '+%{y:.2f}%<extra>Significant Jump</extra>',
      },
    ];

    if (sigDates.length > 0) {
      dataTraces.push({
        type: 'scatter',
        mode: 'markers',
        name: 'Jump Anomaly Flag',
        x: sigDates,
        y: sigTopY,
        marker: {
          symbol: 'triangle-up',
          size: 8,
          color: QUANT_COLORS.negative,
        },
        showlegend: false,
        hovertemplate: 'Significant Jump (p < 0.05)<extra></extra>',
      });
    }

    return {
      traces: dataTraces,
      layout: {
        barmode: 'stack',
        bargap: 0.25,
        yaxis: {
          ticksuffix: '%',
          rangemode: 'tozero',
          gridcolor: tokens.grid,
        },
      },
      avgJumpRatio: ratio,
      jumpDayCount: sigCount,
    };
  }, [dailyRV, tokens]);

  if (!dailyRV || dailyRV.length === 0) {
    return (
      <div className="h-60 flex items-center justify-center border border-[var(--hairline)] rounded-lg text-zinc-500 text-xs">
        No realized volatility days computed
      </div>
    );
  }

  return (
    <div className="panel p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
              Realized Volatility &amp; Jump Decomposition (Barndorff-Nielsen &amp; Shephard)
            </h3>
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              · {ticker} (Recent {dailyRV.length} Sessions)
            </span>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Disentangles continuous diffusive market variance (BV_t) from discrete jump shocks (J_t)
          </p>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <div>
            <span className="text-zinc-500 dark:text-zinc-400">Jump Share:</span>{' '}
            <span className="font-mono font-bold text-gold-500 tabular-nums">
              {avgJumpRatio.toFixed(1)}% of Total RV
            </span>
          </div>
          <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">
            ·
          </span>
          <div>
            <span className="text-zinc-500 dark:text-zinc-400">Significant Jumps (p &lt; 0.05):</span>{' '}
            <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100 tabular-nums">
              {jumpDayCount} of {dailyRV.length} days
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
        height={290}
        exportFilename={`quantmetrics-jump-decomp-${ticker.toLowerCase()}`}
        uirevisionKey={`jump-${ticker}-${dailyRV.length}`}
        showWheelHint
      />

      <div className="flex items-center justify-end text-xs text-zinc-500 dark:text-zinc-400 mt-2 pt-2 border-t border-[var(--hairline)] font-mono text-[11px]">
        <span>Bars/Day: ~390 mins</span>
      </div>
    </div>
  );
}
