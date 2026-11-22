'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { EmptyState } from '@/components/ui/empty-state';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

export interface IvRvPoint {
  date: string;
  iv: number | null; // annualized IV, decimal
  rv: number | null; // annualized RV, decimal
}

/**
 * Interactive dual-panel Plotly chart: IV vs RV on top, IV/RV ratio below
 * with shaded crimson regimes where IV < RV (underpriced risk).
 */
export function IvRvChart({ points, ivLabel }: { points: IvRvPoint[]; ivLabel: string }) {
  const isDark = usePlotlyDarkTheme();

  const data = useMemo(
    () =>
      points.filter(
        (p) => (p.iv !== null && Number.isFinite(p.iv)) || (p.rv !== null && Number.isFinite(p.rv))
      ),
    [points]
  );

  const { traces, layout } = useMemo(() => {
    if (data.length < 30) {
      return { traces: [] as Data[], layout: {} as Partial<Layout> };
    }
    const t = getPlotlyThemeTokens(isDark);
    const dates = data.map((p) => p.date);
    const ivs = data.map((p) => (p.iv !== null ? p.iv * 100 : null));
    const rvs = data.map((p) => (p.rv !== null ? p.rv * 100 : null));
    const ratios = data.map((p) =>
      p.iv !== null && p.rv !== null && p.rv > 0 ? Number((p.iv / p.rv).toFixed(3)) : null
    );

    const list: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: ivLabel,
        x: dates,
        y: ivs,
        yaxis: 'y',
        line: { color: QUANT_COLORS.goldLight, width: 2 },
        hovertemplate: `${ivLabel}: %{y:.1f}%<extra></extra>`,
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'RV 20d',
        x: dates,
        y: rvs,
        yaxis: 'y',
        line: { color: QUANT_COLORS.sky, width: 1.5, dash: 'dash' },
        hovertemplate: 'RV 20d: %{y:.1f}%<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'IV / RV Ratio',
        x: dates,
        y: ratios,
        yaxis: 'y2',
        line: { color: isDark ? 'rgba(236,236,241,0.72)' : 'rgba(24,24,27,0.68)', width: 1.3 },
        hovertemplate: 'IV/RV: %{y:.2f}×<extra></extra>',
      },
    ];

    // Shade IV < RV regimes on the bottom panel
    const shapes: NonNullable<Layout['shapes']> = [
      {
        type: 'line',
        xref: 'paper',
        yref: 'y2',
        x0: 0,
        x1: 1,
        y0: 1,
        y1: 1,
        line: { color: t.zeroLine, width: 1, dash: 'dash' },
      },
    ];

    let startIdx: number | null = null;
    for (let i = 0; i < data.length; i++) {
      const r = ratios[i];
      const below = r !== null && r < 1;
      if (below && startIdx === null) startIdx = i;
      if (!below && startIdx !== null) {
        shapes.push({
          type: 'rect',
          xref: 'x',
          yref: 'y2',
          x0: dates[startIdx],
          x1: dates[i],
          y0: 0.5,
          y1: 2.0,
          fillcolor: 'rgba(224, 101, 95, 0.14)',
          line: { width: 0 },
          layer: 'below',
        });
        startIdx = null;
      }
    }
    if (startIdx !== null) {
      shapes.push({
        type: 'rect',
        xref: 'x',
        yref: 'y2',
        x0: dates[startIdx],
        x1: dates[dates.length - 1],
        y0: 0.5,
        y1: 2.0,
        fillcolor: 'rgba(224, 101, 95, 0.14)',
        line: { width: 0 },
        layer: 'below',
      });
    }

    const chartLayout: Partial<Layout> = {
      shapes,
      margin: { l: 50, r: 16, t: 30, b: 22 },
      yaxis: {
        domain: [0.34, 1],
        ticksuffix: '%',
        title: { text: 'Vol %', font: { size: 10, color: t.textMuted } },
        gridcolor: t.grid,
      },
      yaxis2: {
        domain: [0, 0.26],
        range: [0.55, 1.85],
        title: { text: 'IV/RV', font: { size: 10, color: t.textMuted } },
        gridcolor: t.grid,
        tickfont: { size: 10, color: t.textMuted },
      },
    };

    return { traces: list, layout: chartLayout };
  }, [data, ivLabel, isDark]);

  if (data.length < 30) {
    return (
      <EmptyState
        title="Not enough overlapping IV/RV history"
        detail="Need at least 30 aligned sessions"
        className="h-44"
      />
    );
  }

  return (
    <div>
      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        rangeSlider
        hovermode="x unified"
        showLegend
        height={310}
        uirevisionKey={`ivrv-${ivLabel}-${data.length}`}
        exportFilename="iv-vs-rv"
        showWheelHint
      />
      <div className="mt-1 flex flex-wrap items-center gap-3 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
        <span>Crimson shading: IV &lt; RV (underpriced implied risk)</span>
      </div>
    </div>
  );
}
