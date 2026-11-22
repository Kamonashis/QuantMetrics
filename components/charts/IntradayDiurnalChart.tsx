'use client';

import React, { useMemo } from 'react';
import type { Annotations, Data, Layout, Shape } from 'plotly.js';
import { EmptyState } from '@/components/ui/empty-state';
import { DiurnalPoint } from '@/lib/quant-engine';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

interface IntradayDiurnalChartProps {
  points: DiurnalPoint[];
  ticker: string;
}

const KEY_SESSION_TIMES = [
  { minute: 0, label: '09:30 Open' },
  { minute: 120, label: '11:30 EU Close' },
  { minute: 210, label: '13:00 Lunch Lull' },
  { minute: 360, label: '15:30 Power Hr' },
  { minute: 389, label: '16:00 MOC' },
];

export function IntradayDiurnalChart({ points, ticker }: IntradayDiurnalChartProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const { traces, layout, peakRatio } = useMemo<{
    traces: Data[];
    layout: Partial<Layout>;
    peakRatio: number;
  }>(() => {
    if (!points || points.length === 0) {
      return { traces: [], layout: {}, peakRatio: 1 };
    }
    const mins = points.map((p) => p.minuteOfDay);
    const labels = points.map((p) => p.timeLabel);
    const upper = points.map((p) => p.upperBand * 100);
    const lower = points.map((p) => p.lowerBand * 100);
    const mean = points.map((p) => p.avgAnnualizedVol * 100);

    const openVol = points[0]?.avgAnnualizedVol || 0.25;
    const middayVol = points[Math.min(210, points.length - 1)]?.avgAnnualizedVol || 0.12;
    const ratio = openVol / Math.max(middayVol, 1e-4);

    const shapes: Partial<Shape>[] = KEY_SESSION_TIMES.map((k) => ({
      type: 'line',
      xref: 'x',
      yref: 'paper',
      x0: k.minute,
      x1: k.minute,
      y0: 0,
      y1: 1,
      line: { color: tokens.grid, width: 1, dash: 'dot' },
    }));

    const annotations: Partial<Annotations>[] = KEY_SESSION_TIMES.map((k, idx) => ({
      x: k.minute,
      y: 1.0,
      xref: 'x',
      yref: 'paper',
      text: k.label,
      showarrow: false,
      xanchor: idx === 0 ? 'left' : idx === KEY_SESSION_TIMES.length - 1 ? 'right' : 'center',
      yanchor: 'top',
      font: { size: 9.5, color: tokens.textMuted },
    }));

    return {
      traces: [
        {
          type: 'scatter',
          mode: 'lines',
          name: '−1σ Band',
          x: mins,
          y: lower,
          customdata: labels,
          line: { width: 0 },
          showlegend: false,
          hovertemplate: '%{y:.2f}%<extra>−1σ Band</extra>',
        },
        {
          type: 'scatter',
          mode: 'lines',
          name: '±1σ Dispersion',
          x: mins,
          y: upper,
          customdata: labels,
          fill: 'tonexty',
          fillcolor: 'rgba(222, 192, 132, 0.16)',
          line: { width: 0 },
          hovertemplate: '%{y:.2f}%<extra>+1σ Band</extra>',
        },
        {
          type: 'scatter',
          mode: 'lines',
          name: 'Mean 1-Min Vol',
          x: mins,
          y: mean,
          customdata: labels,
          line: { color: QUANT_COLORS.goldLight, width: 2 },
          hovertemplate: '%{customdata} · %{y:.2f}%<extra>Mean Vol</extra>',
        },
      ],
      layout: {
        shapes,
        annotations,
        xaxis: {
          tickvals: [0, 60, 120, 195, 270, 330, 389],
          ticktext: ['09:30', '10:30', '11:30', '12:45', '14:00', '15:00', '16:00'],
          gridcolor: tokens.grid,
        },
        yaxis: {
          ticksuffix: '%',
          rangemode: 'tozero',
          gridcolor: tokens.grid,
        },
      },
      peakRatio: ratio,
    };
  }, [points, tokens]);

  if (!points || points.length === 0) {
    return (
      <EmptyState
        title="No intraday observations"
        detail="1-minute bars are needed for the diurnal profile"
        className="h-64"
      />
    );
  }

  return (
    <div className="panel p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
              Intraday Diurnal U-Shape Volatility Smile
            </h3>
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              · {ticker} (390 Trading Minutes)
            </span>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Empirical minute-by-minute volatility profile showing opening price discovery, midday
            lull, and market-on-close auction
          </p>
        </div>

        <div className="shrink-0 text-right font-mono text-xs text-zinc-500 dark:text-zinc-400 tabular-nums">
          <span>U-Shape Peak Ratio: </span>
          <strong className="text-gold-600 dark:text-gold-300">
            {peakRatio.toFixed(1)}× Open vs Midday
          </strong>
        </div>
      </div>

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="x unified"
        showLegend
        height={295}
        exportFilename={`quantmetrics-diurnal-${ticker.toLowerCase()}`}
        uirevisionKey={`diurnal-${ticker}-${points.length}`}
        showWheelHint
      />
    </div>
  );
}
