'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { PlotlyChart } from '@/components/charts/PlotlyChart';
import { QUANT_COLORS } from '@/lib/plotly-theme';
import { EmptyState } from '@/components/ui/empty-state';

export interface GexStrikePoint {
  strike: number;
  callOi: number;
  putOi: number;
  gex: number;
}

/**
 * Interactive Plotly GEX by strike: diverging horizontal bars (gold = long gamma,
 * crimson = short gamma) around zero, with Spot and Zero-Gamma Flip reference rules.
 */
export function GammaProfileChart({
  points,
  spot,
  flip,
  maxBars = 34,
}: {
  points: GexStrikePoint[];
  spot: number;
  flip: number | null;
  maxBars?: number;
}) {
  const trimmed = useMemo(() => {
    const sorted = [...points].sort((a, b) => a.strike - b.strike);
    const abs = points.map((p) => Math.abs(p.gex));
    const threshold = Math.max(...abs, 1) * 0.02;
    const meaningful = sorted.filter((p) => Math.abs(p.gex) >= threshold);
    if (meaningful.length <= maxBars) return meaningful;
    const start = Math.max(0, Math.floor((meaningful.length - maxBars) / 2));
    return meaningful.slice(start, start + maxBars);
  }, [points, maxBars]);

  const { traces, layout } = useMemo(() => {
    if (trimmed.length === 0) {
      return { traces: [] as Data[], layout: {} as Partial<Layout> };
    }

    const pos = trimmed.map((p) => (p.gex >= 0 ? p.gex / 1e6 : 0));
    const neg = trimmed.map((p) => (p.gex < 0 ? p.gex / 1e6 : 0));
    const strikes = trimmed.map((p) => p.strike);
    const custom = trimmed.map((p) => [
      p.callOi.toLocaleString(),
      p.putOi.toLocaleString(),
      (p.gex / 1e6).toFixed(2),
    ]);

    const list: Data[] = [
      {
        type: 'bar',
        orientation: 'h',
        name: 'Long Gamma',
        y: strikes,
        x: pos,
        customdata: custom,
        marker: { color: 'rgba(222, 192, 132, 0.80)' },
        hovertemplate:
          '<b>Strike $%{y}</b><br>GEX: %{customdata[2]}M $ / 1%<br>Call OI %{customdata[0]} · Put OI %{customdata[1]}<extra></extra>',
      },
      {
        type: 'bar',
        orientation: 'h',
        name: 'Short Gamma',
        y: strikes,
        x: neg,
        customdata: custom,
        marker: { color: 'rgba(224, 101, 95, 0.78)' },
        hovertemplate:
          '<b>Strike $%{y}</b><br>GEX: %{customdata[2]}M $ / 1%<br>Call OI %{customdata[0]} · Put OI %{customdata[1]}<extra></extra>',
      },
    ];

    const shapes: NonNullable<Layout['shapes']> = [
      {
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: 0,
        x1: 0,
        y0: 0,
        y1: 1,
        line: { color: 'rgba(148, 148, 160, 0.4)', width: 1, dash: 'dot' },
      },
      {
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: spot,
        y1: spot,
        line: { color: QUANT_COLORS.goldLight, width: 1.4, dash: 'dash' },
      },
    ];

    const annotations: NonNullable<Layout['annotations']> = [
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
    ];

    if (flip !== null && Number.isFinite(flip) && flip > 0) {
      shapes.push({
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: flip,
        y1: flip,
        line: { color: QUANT_COLORS.amber, width: 1.2, dash: 'dot' },
      });
      annotations.push({
        x: 0.01,
        y: flip,
        xref: 'paper',
        yref: 'y',
        text: `Flip $${flip.toFixed(2)}`,
        showarrow: false,
        xanchor: 'left',
        yanchor: 'bottom',
        font: { size: 9.5, color: QUANT_COLORS.amber },
      });
    }

    return {
      traces: list,
      layout: {
        barmode: 'relative',
        shapes,
        annotations,
        margin: { l: 54, r: 16, t: 28, b: 42 },
      } as Partial<Layout>,
    };
  }, [trimmed, spot, flip]);

  if (trimmed.length === 0) {
    return (
      <EmptyState
        title="No gamma exposure bars to plot"
        detail="Requires open interest across strikes"
        className="h-44"
      />
    );
  }

  const height = Math.max(300, Math.min(540, trimmed.length * 16 + 76));

  return (
    <div>
      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="y unified"
        showLegend
        xAxisTitle="Dealer GEX ($M per 1% move)"
        xTickSuffix="M"
        yAxisTitle="Strike"
        yTickPrefix="$"
        height={height}
        uirevisionKey={`gex-${spot.toFixed(1)}-${trimmed.length}`}
        exportFilename="gamma-profile"
        showWheelHint
      />
      <div className="mt-1 flex flex-wrap items-center gap-4 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
        <span>Long gamma: dealers re-hedge by selling rallies</span>
        <span>·</span>
        <span>Short gamma: dealers amplify moves</span>
        <span>·</span>
        <span>
          Spot ${spot.toFixed(2)} · {flip ? `Flip ≈ $${flip.toFixed(2)}` : 'No flip in grid'}
        </span>
      </div>
    </div>
  );
}
