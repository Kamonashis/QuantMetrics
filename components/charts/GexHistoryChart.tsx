'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import type { PanelRowOut } from '@/components/views/volatility/types';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { EmptyState } from '@/components/ui/empty-state';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

export interface GexHistoryChartProps {
  panel: PanelRowOut[];
  convention: 'squeeze' | 'inverse';
}

const MIN_BASELINE_SESSIONS = 20;

/**
 * Accrued daily Total GEX ($M) & Gamma Flip vs. Spot chart from `.cache/iv/{SYM}/panel.json`.
 * Never interpolates or backfills; surfaces count-aware sparse notices below 20 sessions.
 */
export function GexHistoryChart({ panel, convention }: GexHistoryChartProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const series = useMemo(() => {
    const convMult = convention === 'inverse' ? -1 : 1;
    const valid = [...panel]
      .filter(
        (r) =>
          r &&
          r.date &&
          ((r.totalGex !== null && Number.isFinite(r.totalGex)) ||
            (r.gammaFlip !== null && Number.isFinite(r.gammaFlip)))
      )
      .sort((a, b) => a.date.localeCompare(b.date));

    let maxAbsGex = 0;
    for (const r of valid) {
      if (r.totalGex !== null && Number.isFinite(r.totalGex)) {
        const v = Math.abs(r.totalGex * convMult);
        if (v > maxAbsGex) maxAbsGex = v;
      }
    }
    const useK = maxAbsGex > 0 && maxAbsGex < 1e6;
    const div = useK ? 1e3 : 1e6;
    const unit = useK ? 'K' : 'M';

    return {
      rows: valid,
      convMult,
      div,
      unit,
    };
  }, [panel, convention]);

  const n = series.rows.length;

  const { traces, layout } = useMemo(() => {
    if (n < 2) return { traces: [] as Data[], layout: {} as Partial<Layout> };

    const { rows, convMult, div, unit } = series;
    const dates = rows.map((r) => r.date);
    const spots = rows.map((r) => r.spot);
    const flips = rows.map((r) => (r.gammaFlip !== null && Number.isFinite(r.gammaFlip) ? r.gammaFlip : null));
    const gexVals = rows.map((r) =>
      r.totalGex !== null && Number.isFinite(r.totalGex) ? (r.totalGex * convMult) / div : null
    );
    const barColors = gexVals.map((v) =>
      v === null
        ? 'rgba(140, 140, 154, 0.25)'
        : v >= 0
          ? 'rgba(222, 192, 132, 0.82)'
          : 'rgba(224, 101, 95, 0.80)'
    );

    const list: Data[] = [
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Spot',
        x: dates,
        y: spots,
        yaxis: 'y',
        marker: { size: n <= 10 ? 5 : 3.5, color: isDark ? '#ECECF1' : '#18181B' },
        line: { color: isDark ? '#ECECF1' : '#18181B', width: 1.7 },
        hovertemplate: '<b>%{x}</b><br>Spot: $%{y:.2f}<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Gamma Flip',
        x: dates,
        y: flips,
        yaxis: 'y',
        connectgaps: false,
        marker: { size: n <= 10 ? 5 : 3.5, color: QUANT_COLORS.amber },
        line: { color: QUANT_COLORS.amber, width: 1.5, dash: 'dot' },
        hovertemplate: '<b>%{x}</b><br>Flip: $%{y:.2f}<extra></extra>',
      },
      {
        type: 'bar',
        name: `Net GEX ($${unit})`,
        x: dates,
        y: gexVals,
        yaxis: 'y2',
        marker: { color: barColors },
        hovertemplate: `<b>%{x}</b><br>Net GEX: %{y:+.2f}${unit}<extra></extra>`,
      },
    ];

    return {
      traces: list,
      layout: {
        margin: { l: 56, r: 16, t: 24, b: 32 },
        xaxis: {
          gridcolor: tokens.grid,
          linecolor: tokens.grid,
          tickfont: { size: 10, color: tokens.textMuted },
        },
        yaxis: {
          domain: [0.46, 1.0],
          title: { text: 'Spot / Flip', font: { size: 10, color: tokens.textMuted } },
          tickprefix: '$',
          gridcolor: tokens.grid,
          linecolor: tokens.grid,
          tickfont: { size: 10, color: tokens.textMuted },
          autorange: true,
        },
        yaxis2: {
          domain: [0.0, 0.36],
          anchor: 'x',
          title: { text: `GEX ($${unit})`, font: { size: 10, color: tokens.textMuted } },
          ticksuffix: unit,
          gridcolor: tokens.grid,
          zeroline: true,
          zerolinecolor: tokens.zeroLine,
          linecolor: tokens.grid,
          tickfont: { size: 10, color: tokens.textMuted },
          autorange: true,
        },
      } as Partial<Layout>,
    };
  }, [n, series, isDark, tokens]);

  if (n < 2) {
    return (
      <EmptyState
        title="Accrued GEX & Flip history building"
        detail={`Have ${n} of ${MIN_BASELINE_SESSIONS} sessions — daily EOD snapshots accrue automatically on chain fetches (min 2 to chart, never fabricated)`}
        className="h-56"
      />
    );
  }

  return (
    <div>
      {n < MIN_BASELINE_SESSIONS ? (
        <div className="mb-2.5 flex items-center justify-between gap-2 rounded-lg border border-gold-500/35 bg-gold-500/10 px-3 py-1.5 font-mono text-[10px] text-gold-800 dark:text-gold-200">
          <span>
            Sparse history: <strong>{n} / {MIN_BASELINE_SESSIONS}</strong> accrued sessions — plotting observed EOD snapshots only (no interpolation or backfill).
          </span>
        </div>
      ) : (
        <div className="mb-2 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
          {n} accrued EOD sessions · panel stores canonical squeeze baseline ({convention} sign applied)
        </div>
      )}

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        autoFitYOnLoad={false}
        hovermode="x unified"
        showLegend
        height={320}
        uirevisionKey={`gex-hist-${n}-${convention}`}
        exportFilename="gex-accrued-history"
        showWheelHint
      />
    </div>
  );
}
