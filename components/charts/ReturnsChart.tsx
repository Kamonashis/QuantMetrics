'use client';

import React, { useMemo } from 'react';
import type { Data, Layout, Shape } from 'plotly.js';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

interface ReturnsChartProps {
  dates: string[];
  returns: number[];
  returnType: 'simple' | 'log';
  title?: string;
}

export function ReturnsChart({
  dates,
  returns,
  returnType,
  title = 'Daily Returns Trajectory',
}: ReturnsChartProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const { traces, layout, latestDate, latestRet } = useMemo<{
    traces: Data[];
    layout: Partial<Layout>;
    latestDate: string | undefined;
    latestRet: number;
  }>(() => {
    if (!returns || returns.length === 0) {
      return { traces: [], layout: {}, latestDate: undefined, latestRet: 0 };
    }
    const alignedDates = dates.slice(1, returns.length + 1);
    const pctReturns = returns.map((r) => r * 100);
    const barColors = returns.map((r) =>
      r >= 0 ? 'rgba(63, 178, 127, 0.82)' : 'rgba(224, 101, 95, 0.82)'
    );

    // Compute empirical ±2σ reference envelope
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance =
      returns.reduce((acc, r) => acc + (r - mean) * (r - mean), 0) /
      Math.max(1, returns.length - 1);
    const sigmaPct = Math.sqrt(variance) * 100;
    const band2 = 2 * sigmaPct;

    const shapes: Partial<Shape>[] = [
      {
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: band2,
        y1: band2,
        line: { color: QUANT_COLORS.goldLight, width: 1, dash: 'dot' },
      },
      {
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: -band2,
        y1: -band2,
        line: { color: QUANT_COLORS.goldLight, width: 1, dash: 'dot' },
      },
    ];

    return {
      traces: [
        {
          type: 'bar',
          name: returnType === 'log' ? 'Log Return' : 'Simple Return',
          x: alignedDates,
          y: pctReturns,
          marker: { color: barColors },
          hovertemplate: '%{y:+.2f}%<extra>Return</extra>',
        },
      ],
      layout: {
        bargap: 0.08,
        shapes,
        yaxis: {
          ticksuffix: '%',
          zeroline: true,
          zerolinecolor: tokens.zeroLine,
          zerolinewidth: 1,
          gridcolor: tokens.grid,
        },
        annotations: [
          {
            xref: 'paper',
            yref: 'y',
            x: 1,
            y: band2,
            text: `+2σ (${band2.toFixed(1)}%)`,
            showarrow: false,
            xanchor: 'right',
            yanchor: 'bottom',
            font: { size: 9.5, color: QUANT_COLORS.goldLight },
          },
          {
            xref: 'paper',
            yref: 'y',
            x: 1,
            y: -band2,
            text: `−2σ (−${band2.toFixed(1)}%)`,
            showarrow: false,
            xanchor: 'right',
            yanchor: 'top',
            font: { size: 9.5, color: QUANT_COLORS.goldLight },
          },
        ],
      },
      latestDate: alignedDates[alignedDates.length - 1],
      latestRet: returns[returns.length - 1],
    };
  }, [dates, returns, returnType, tokens]);

  if (!returns || returns.length === 0) {
    return (
      <div className="h-60 flex items-center justify-center border border-[var(--hairline)] rounded-lg text-zinc-500">
        No return series calculated
      </div>
    );
  }

  return (
    <div className="panel p-5">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
            {title}
          </h3>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            {returnType === 'log'
              ? 'Continuously compounded log returns r_t = ln(P_t/P_{t-1})'
              : 'Simple percentage returns R_t = (P_t - P_{t-1})/P_{t-1}'}
          </p>
        </div>

        {latestDate && (
          <div className="text-right">
            <span className="text-xs text-zinc-500 dark:text-zinc-400 mr-1.5">{latestDate}:</span>
            <span
              className={`font-mono text-sm font-bold tabular-nums ${
                latestRet >= 0 ? 'text-emerald-500' : 'text-rose-500'
              }`}
            >
              {latestRet >= 0 ? '+' : ''}
              {(latestRet * 100).toFixed(2)}%
            </span>
          </div>
        )}
      </div>

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        rangeSlider
        hovermode="x unified"
        height={280}
        exportFilename="quantmetrics-daily-returns"
        uirevisionKey={`ret-${dates[1]}-${returns.length}-${returnType}`}
        showWheelHint
      />
    </div>
  );
}
