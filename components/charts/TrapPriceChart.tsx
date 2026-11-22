'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { EmptyState } from '@/components/ui/empty-state';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { QUANT_COLORS } from '@/lib/plotly-theme';

/**
 * Interactive Plotly daily close series with dealer levels:
 * Call Wall (gold dash), Put Wall (crimson dash), and Zero-Gamma Flip (slate dot).
 */
export function TrapPriceChart({
  closes,
  spot,
  callWall,
  putWall,
  flip,
}: {
  closes: { date: string; close: number }[];
  spot: number;
  callWall: number | null;
  putWall: number | null;
  flip: number | null;
}) {
  const isDark = usePlotlyDarkTheme();

  const dataSlice = useMemo(() => closes.slice(-180), [closes]);

  const { traces, layout } = useMemo(() => {
    if (dataSlice.length < 10) {
      return { traces: [] as Data[], layout: {} as Partial<Layout> };
    }

    const xs = dataSlice.map((d) => d.date);
    const ys = dataSlice.map((d) => d.close);

    const list: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Daily Close',
        x: xs,
        y: ys,
        line: {
          color: isDark ? '#ECECF1' : '#18181B',
          width: 1.8,
        },
        fill: 'tozeroy',
        fillcolor: 'rgba(222, 192, 132, 0.08)',
        hovertemplate: '<b>%{x}</b><br>Close: $%{y:.2f}<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'markers',
        name: 'Spot',
        x: [xs[xs.length - 1]],
        y: [spot],
        marker: {
          size: 7,
          color: QUANT_COLORS.goldLight,
          line: { color: isDark ? '#0B0B0D' : '#FFFFFF', width: 1.5 },
        },
        hovertemplate: '<b>Current Spot</b><br>$%{y:.2f}<extra></extra>',
      },
    ];

    const shapes: NonNullable<Layout['shapes']> = [];
    const annotations: NonNullable<Layout['annotations']> = [];

    const addLevel = (val: number | null, color: string, label: string, dash: 'dash' | 'dot') => {
      if (val === null || !Number.isFinite(val)) return;
      shapes.push({
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: val,
        y1: val,
        line: { color, width: 1.2, dash },
      });
      annotations.push({
        x: 0.99,
        y: val,
        xref: 'paper',
        yref: 'y',
        text: `${label} $${val.toFixed(1)}`,
        showarrow: false,
        xanchor: 'right',
        yanchor: 'bottom',
        font: { size: 9.5, color },
      });
    };

    addLevel(callWall, QUANT_COLORS.goldLight, 'CW', 'dash');
    addLevel(putWall, QUANT_COLORS.negative, 'PW', 'dash');
    addLevel(flip, QUANT_COLORS.slate, 'FLIP', 'dot');

    return {
      traces: list,
      layout: {
        shapes,
        annotations,
        margin: { l: 50, r: 16, t: 14, b: 30 },
      } as Partial<Layout>,
    };
  }, [dataSlice, spot, callWall, putWall, flip, isDark]);

  if (dataSlice.length < 10) {
    return (
      <EmptyState
        title="Not enough price history"
        detail="The trap overlay needs a daily close series"
        className="h-44"
      />
    );
  }

  return (
    <PlotlyChart
      data={traces}
      layout={layout}
      plotType="2d"
      hovermode="x unified"
      yTickPrefix="$"
      height={220}
      uirevisionKey={`trap-${dataSlice.length}-${spot.toFixed(1)}`}
      exportFilename="trap-price-levels"
      showWheelHint
    />
  );
}
