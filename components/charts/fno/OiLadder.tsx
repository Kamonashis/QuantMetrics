'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { EmptyState } from '@/components/ui/empty-state';
import { PlotlyChart } from '@/components/charts/PlotlyChart';
import { QUANT_COLORS } from '@/lib/plotly-theme';

/**
 * Interactive Plotly dual-sided OI butterfly ladder around spot:
 * Calls extending left in champagne gold, Puts extending right in crimson.
 */
export function OiLadder({
  rows,
  spot,
}: {
  rows: { strike: number; callOi: number; putOi: number; totalOi: number }[];
  spot: number;
}) {
  const { traces, layout } = useMemo(() => {
    if (rows.length === 0) {
      return { traces: [] as Data[], layout: {} as Partial<Layout> };
    }

    const sorted = [...rows].sort((a, b) => a.strike - b.strike);
    const strikes = sorted.map((r) => r.strike);

    const list: Data[] = [
      {
        type: 'bar',
        orientation: 'h',
        name: 'Call OI',
        y: strikes,
        x: sorted.map((r) => -r.callOi),
        customdata: sorted.map((r) => [r.callOi.toLocaleString(), r.totalOi.toLocaleString()]),
        marker: {
          color: sorted.map((r) =>
            Math.abs(r.strike - spot) / Math.max(spot, 1e-6) < 0.025
              ? 'rgba(222, 192, 132, 0.96)'
              : 'rgba(222, 192, 132, 0.62)'
          ),
        },
        hovertemplate:
          '<b>Strike $%{y} · Calls</b><br>Call OI: %{customdata[0]}<br>Total OI: %{customdata[1]}<extra></extra>',
      },
      {
        type: 'bar',
        orientation: 'h',
        name: 'Put OI',
        y: strikes,
        x: sorted.map((r) => r.putOi),
        customdata: sorted.map((r) => [r.putOi.toLocaleString(), r.totalOi.toLocaleString()]),
        marker: {
          color: sorted.map((r) =>
            Math.abs(r.strike - spot) / Math.max(spot, 1e-6) < 0.025
              ? 'rgba(224, 101, 95, 0.96)'
              : 'rgba(224, 101, 95, 0.62)'
          ),
        },
        hovertemplate:
          '<b>Strike $%{y} · Puts</b><br>Put OI: %{customdata[0]}<br>Total OI: %{customdata[1]}<extra></extra>',
      },
    ];

    const chartLayout: Partial<Layout> = {
      barmode: 'relative',
      margin: { l: 54, r: 18, t: 28, b: 40 },
      shapes: [
        {
          type: 'line',
          xref: 'x',
          yref: 'paper',
          x0: 0,
          x1: 0,
          y0: 0,
          y1: 1,
          line: { color: 'rgba(148,148,160,0.4)', width: 1, dash: 'dot' },
        },
        {
          type: 'line',
          xref: 'paper',
          yref: 'y',
          x0: 0,
          x1: 1,
          y0: spot,
          y1: spot,
          line: { color: QUANT_COLORS.goldLight, width: 1.3, dash: 'dash' },
        },
      ],
      annotations: [
        {
          x: 0.99,
          y: spot,
          xref: 'paper',
          yref: 'y',
          text: `Spot $${spot.toFixed(2)}`,
          showarrow: false,
          xanchor: 'right',
          yanchor: 'bottom',
          font: { size: 9.5, color: QUANT_COLORS.goldLight },
        },
      ],
    };

    return { traces: list, layout: chartLayout };
  }, [rows, spot]);

  if (rows.length === 0) {
    return (
      <EmptyState
        title="No strikes in ±15% band"
        detail="The chain has no usable rows in this window"
        className="h-40"
      />
    );
  }

  const height = Math.max(290, Math.min(520, rows.length * 16 + 76));

  return (
    <PlotlyChart
      data={traces}
      layout={layout}
      plotType="2d"
      hovermode="y unified"
      showLegend
      xAxisTitle="← Call OI · Put OI →"
      yAxisTitle="Strike"
      yTickPrefix="$"
      height={height}
      uirevisionKey={`oi-ladder-${spot.toFixed(1)}-${rows.length}`}
      exportFilename="oi-ladder"
      showWheelHint
    />
  );
}
