'use client';

import React, { useMemo } from 'react';
import type { Annotations, Data, Layout, Shape } from 'plotly.js';
import { RollingVolSeries } from '@/lib/quant-engine';
import { SERIES_COLORS as COLORS } from '@/lib/chart-palette';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

export interface ReferenceLine {
  /** Horizontal level in the same units as the series (annualized decimal). */
  value: number;
  /** Right-edge + legend label, e.g. '1.25×IV30 35.8%'. */
  label: string;
}

interface RollingVolatilityChartProps {
  series: RollingVolSeries[];
  title?: string;
  /** Optional horizontal reference levels (e.g. IV30, breakeven). Absent by default — call sites opt in. */
  referenceLines?: ReferenceLine[];
}

export function RollingVolatilityChart({
  series,
  title = 'Rolling Annualized Volatility (σ × √252)',
  referenceLines = [],
}: RollingVolatilityChartProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const { traces, layout, latestDate } = useMemo<{
    traces: Data[];
    layout: Partial<Layout>;
    latestDate: string | undefined;
  }>(() => {
    if (!series || series.length === 0 || !series[0].values.length) {
      return { traces: [], layout: {}, latestDate: undefined };
    }
    const dates = series[0].values.map((v) => v.date);

    const dataTraces: Data[] = series.map((s, idx) => ({
      type: 'scatter',
      mode: 'lines',
      name: `${s.window}d RV`,
      x: dates,
      y: s.values.map((v) => (v.vol !== null ? v.vol * 100 : null)),
      line: {
        color: COLORS[idx % COLORS.length],
        width: idx === 0 ? 1.9 : 1.5,
      },
      connectgaps: false,
      hovertemplate: `%{y:.2f}%<extra>${s.window}d RV</extra>`,
    }));

    const shapes: Partial<Shape>[] = referenceLines.map((rl) => ({
      type: 'line',
      xref: 'paper',
      yref: 'y',
      x0: 0,
      x1: 1,
      y0: rl.value * 100,
      y1: rl.value * 100,
      line: { color: QUANT_COLORS.goldLight, width: 1.3, dash: 'dash' },
    }));

    const annotations: Partial<Annotations>[] = referenceLines.map((rl) => ({
      xref: 'paper',
      yref: 'y',
      x: 1,
      y: rl.value * 100,
      text: rl.label,
      showarrow: false,
      xanchor: 'right',
      yanchor: 'bottom',
      font: { size: 9.5, color: QUANT_COLORS.goldLight },
    }));

    return {
      traces: dataTraces,
      layout: {
        shapes,
        annotations,
        yaxis: {
          ticksuffix: '%',
          rangemode: 'tozero',
          gridcolor: tokens.grid,
        },
      },
      latestDate: dates[dates.length - 1],
    };
  }, [series, referenceLines, tokens]);

  if (!series || series.length === 0 || !series[0].values.length) {
    return (
      <div className="h-60 flex items-center justify-center border border-[var(--hairline)] rounded-lg text-zinc-500">
        No rolling volatility computed
      </div>
    );
  }

  const lastIdx = series[0].values.length - 1;

  return (
    <div className="panel p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
        <div>
          <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
            {title}
          </h3>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Standard deviation annualized across user-selected rolling observation windows
          </p>
        </div>

        {/* Legend with latest values */}
        <div className="flex flex-wrap items-center gap-4 text-xs">
          {series.map((s, idx) => {
            const color = COLORS[idx % COLORS.length];
            const activeVal = s.values[lastIdx]?.vol;
            return (
              <div key={s.window} className="flex items-center gap-1.5">
                <span
                  className="w-2.5 h-2.5 rounded-full inline-block"
                  style={{ backgroundColor: color }}
                />
                <span className="text-zinc-500 dark:text-zinc-400 font-medium">{s.window}d:</span>
                <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100 tabular-nums">
                  {activeVal !== null && activeVal !== undefined
                    ? `${(activeVal * 100).toFixed(1)}%`
                    : '—'}
                </span>
              </div>
            );
          })}
          {referenceLines.map((rl) => (
            <div key={rl.label} className="flex items-center gap-1.5">
              <span className="inline-block h-0 w-4 border-t-2 border-dashed border-gold-500" />
              <span className="text-zinc-500 dark:text-zinc-400 font-medium">{rl.label}</span>
            </div>
          ))}
        </div>
      </div>

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        rangeSlider
        hovermode="x unified"
        height={295}
        exportFilename="quantmetrics-rolling-volatility"
        uirevisionKey={`rvol-${series[0].values[0]?.date}-${series[0].values.length}-${series
          .map((s) => s.window)
          .join('-')}`}
        showWheelHint
      />

      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-zinc-500 dark:text-zinc-400 mt-2">
        <span>
          Latest Date:{' '}
          <span className="font-mono text-zinc-800 dark:text-zinc-200 font-semibold">
            {latestDate}
          </span>
        </span>
        <span>
          Annualized Factor: <span className="font-mono">√252 ≈ 15.874</span>
        </span>
      </div>
    </div>
  );
}
