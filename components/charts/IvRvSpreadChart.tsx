'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { rollingZ } from '@/lib/vol-signals';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

export interface IvRvSpreadPoint {
  date: string;
  iv30: number | null;
  rv: number | null;
}

/**
 * Interactive dual-panel Plotly IV30 (accrued) vs RV spread & 20-session rolling Z-score.
 * Preserves honest count-aware empty state below 2 overlapping sessions.
 */
export function IvRvSpreadChart({
  points,
  ivLabel = 'IV30 (accrued)',
  rvLabel = 'RV 20d',
}: {
  points: IvRvSpreadPoint[];
  ivLabel?: string;
  rvLabel?: string;
}) {
  const isDark = usePlotlyDarkTheme();

  const data = useMemo(
    () =>
      points.filter(
        (p) =>
          p.iv30 !== null &&
          Number.isFinite(p.iv30) &&
          p.rv !== null &&
          Number.isFinite(p.rv)
      ),
    [points]
  );

  const spread = useMemo(
    () => data.map((p) => (p.iv30 as number) - (p.rv as number)),
    [data]
  );
  const spreadZ = useMemo(() => rollingZ(spread, 20), [spread]);

  const { traces, layout } = useMemo(() => {
    if (data.length < 2) {
      return { traces: [] as Data[], layout: {} as Partial<Layout> };
    }
    const t = getPlotlyThemeTokens(isDark);
    const dates = data.map((p) => p.date);
    const ivs = data.map((p) => (p.iv30 as number) * 100);
    const rvs = data.map((p) => (p.rv as number) * 100);
    const zs = spreadZ.map((z) => (z === null ? null : Math.max(-2.5, Math.min(2.5, z))));
    const zReady = zs.some((z) => z !== null);

    const list: Data[] = [
      {
        type: 'scatter',
        mode: data.length < 15 ? 'lines+markers' : 'lines',
        name: ivLabel,
        x: dates,
        y: ivs,
        yaxis: 'y',
        line: { color: QUANT_COLORS.goldLight, width: 2 },
        hovertemplate: `${ivLabel}: %{y:.1f}%<extra></extra>`,
      },
      {
        type: 'scatter',
        mode: data.length < 15 ? 'lines+markers' : 'lines',
        name: rvLabel,
        x: dates,
        y: rvs,
        yaxis: 'y',
        line: { color: QUANT_COLORS.sky, width: 1.6, dash: 'dash' },
        hovertemplate: `${rvLabel}: %{y:.1f}%<extra></extra>`,
      },
    ];

    if (zReady) {
      list.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Spread 20d Z',
        x: dates,
        y: zs,
        yaxis: 'y2',
        line: { color: isDark ? '#D4D4DC' : '#3F3F46', width: 1.4 },
        hovertemplate: 'Spread Z: %{y:+.2f}σ<extra></extra>',
      });
    }

    const shapes: NonNullable<Layout['shapes']> = [];
    if (zReady) {
      shapes.push(
        {
          type: 'line',
          xref: 'paper',
          yref: 'y2',
          x0: 0,
          x1: 1,
          y0: 0,
          y1: 0,
          line: { color: t.zeroLine, width: 1, dash: 'dash' },
        },
        {
          type: 'line',
          xref: 'paper',
          yref: 'y2',
          x0: 0,
          x1: 1,
          y0: 1,
          y1: 1,
          line: { color: t.grid, width: 1, dash: 'dot' },
        },
        {
          type: 'line',
          xref: 'paper',
          yref: 'y2',
          x0: 0,
          x1: 1,
          y0: -1,
          y1: -1,
          line: { color: t.grid, width: 1, dash: 'dot' },
        }
      );
    }

    let bucket: number | null = null;
    for (let i = 0; i < spread.length; i++) {
      if (spread[i] < 0 && bucket === null) bucket = i;
      if (spread[i] >= 0 && bucket !== null) {
        shapes.push({
          type: 'rect',
          xref: 'x',
          yref: 'y',
          x0: dates[bucket],
          x1: dates[i],
          y0: 0,
          y1: Math.max(...ivs, ...rvs) * 1.15,
          fillcolor: 'rgba(222, 192, 132, 0.11)',
          line: { width: 0 },
          layer: 'below',
        });
        bucket = null;
      }
    }
    if (bucket !== null) {
      shapes.push({
        type: 'rect',
        xref: 'x',
        yref: 'y',
        x0: dates[bucket],
        x1: dates[dates.length - 1],
        y0: 0,
        y1: Math.max(...ivs, ...rvs) * 1.15,
        fillcolor: 'rgba(222, 192, 132, 0.11)',
        line: { width: 0 },
        layer: 'below',
      });
    }

    const chartLayout: Partial<Layout> = {
      shapes,
      margin: { l: 50, r: 16, t: 30, b: 22 },
      yaxis: {
        domain: zReady ? [0.35, 1] : [0, 1],
        ticksuffix: '%',
        title: { text: 'Vol %', font: { size: 10, color: t.textMuted } },
        gridcolor: t.grid,
      },
      ...(zReady
        ? {
            yaxis2: {
              domain: [0, 0.27],
              range: [-2.6, 2.6],
              ticksuffix: 'σ',
              title: { text: 'Z-score', font: { size: 10, color: t.textMuted } },
              gridcolor: t.grid,
              tickfont: { size: 10, color: t.textMuted },
            },
          }
        : {}),
    };

    return { traces: list, layout: chartLayout };
  }, [data, spread, spreadZ, ivLabel, rvLabel, isDark]);

  if (data.length < 2) {
    const accrued = points.filter((p) => p.iv30 !== null).length;
    return (
      <div className="flex h-44 flex-col items-center justify-center gap-1 rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] px-6 text-center">
        <p className="font-mono text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500 dark:text-zinc-400">
          {accrued} accrued IV session{accrued === 1 ? '' : 's'}
        </p>
        <p className="max-w-md text-[11px] leading-snug text-zinc-400 dark:text-zinc-500">
          The RV–IV spread needs overlapping IV + RV on the same sessions. The store accrues one
          snapshot per day you open the chain — z-score meaningful from ~15 sessions, stable by
          ~60. Nothing is interpolated in the meantime.
        </p>
      </div>
    );
  }

  const lastSpread = spread[spread.length - 1];
  const lastZ = spreadZ[spreadZ.length - 1];

  return (
    <div>
      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        rangeSlider={data.length > 20}
        hovermode="x unified"
        showLegend
        height={320}
        uirevisionKey={`ivrv-spread-${data.length}`}
        exportFilename="iv-rv-spread"
        showWheelHint
      />
      <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
        <span>Gold shading: IV &lt; RV (realized outrunning implied)</span>
        <span>
          spread {lastSpread >= 0 ? '+' : ''}
          {(lastSpread * 100).toFixed(1)} pts · z{' '}
          {lastZ !== null && lastZ !== undefined
            ? `${lastZ >= 0 ? '+' : ''}${lastZ.toFixed(2)}`
            : 'n/a (<20 sessions)'}{' '}
          · direct convention (RV √252 · IV √365)
        </span>
      </div>
    </div>
  );
}
