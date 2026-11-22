'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens } from '@/lib/plotly-theme';

export const REGIME_PALETTES: Record<number, string[]> = {
  2: ['rgba(63,178,127,0.9)', 'rgba(224,101,95,0.9)'],
  3: ['rgba(63,178,127,0.9)', 'rgba(222,192,132,0.9)', 'rgba(224,101,95,0.9)'],
  4: ['rgba(63,178,127,0.9)', 'rgba(163,196,90,0.85)', 'rgba(222,192,132,0.9)', 'rgba(224,101,95,0.9)'],
};

export const REGIME_NAMES: Record<number, string[]> = {
  2: ['Calm', 'Stress'],
  3: ['Calm', 'Normal', 'Crisis'],
  4: ['Calm', 'Watchful', 'Normal', 'Crisis'],
};

export function regimePalette(k: number): string[] {
  return REGIME_PALETTES[Math.max(2, Math.min(4, k))];
}

export function regimeNames(k: number): string[] {
  return REGIME_NAMES[Math.max(2, Math.min(4, k))];
}

interface RegimeTimelineProps {
  /** smoothed posteriors [T][K], ordered calm→crisis */
  posteriors: number[][];
  /** optional explicit labels; defaults to Calm/Normal… ordering */
  labels?: string[];
  className?: string;
  height?: number;
  /** Max bars to render (decimates uniformly) */
  maxBars?: number;
  /** session date per posterior — dates[i] is the date for posteriors[i].
   *  Note: posterior index ≠ bar index (features start at bars[1] and skip
   *  invalid bars), so callers must collect dates alongside the features. */
  dates?: string[];
}

export function RegimeTimeline({
  posteriors,
  labels,
  className,
  height = 84,
  maxBars = 600,
  dates,
}: RegimeTimelineProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const K = posteriors[0]?.length ?? 0;
  const names = useMemo(() => labels ?? regimeNames(K), [labels, K]);
  const palette = useMemo(() => regimePalette(K), [K]);

  const { traces, layout } = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (posteriors.length === 0 || K === 0) return { traces: [], layout: {} };
    const stride = Math.max(1, Math.ceil(posteriors.length / maxBars));
    const xs: (string | number)[] = [];
    const perState: number[][] = Array.from({ length: K }, () => []);

    for (let i = 0; i < posteriors.length; i += stride) {
      xs.push(dates?.[i] ?? i);
      const row = posteriors[i];
      for (let k = 0; k < K; k++) {
        perState[k].push((row[k] ?? 0) * 100);
      }
    }

    const dataTraces: Data[] = Array.from({ length: K }, (_, k) => ({
      type: 'scatter',
      mode: 'none',
      stackgroup: 'regime',
      groupnorm: 'percent',
      name: names[k] ?? `State ${k + 1}`,
      x: xs,
      y: perState[k],
      fillcolor: palette[k],
      line: { width: 0, color: palette[k] },
      hovertemplate: `%{y:.1f}%<extra>${names[k] ?? `State ${k + 1}`}</extra>`,
    }));

    return {
      traces: dataTraces,
      layout: {
        margin: { l: 30, r: 8, t: 6, b: 22 },
        yaxis: {
          range: [0, 100],
          tickvals: [0, 50, 100],
          ticksuffix: '%',
          gridcolor: tokens.grid,
          fixedrange: true,
        },
      },
    };
  }, [posteriors, K, maxBars, dates, names, palette, tokens]);

  const effectiveHeight = Math.max(96, height);

  return (
    <div className={`w-full ${className ?? ''}`}>
      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="x unified"
        autoScaleYOnWheel={false}
        height={effectiveHeight}
        exportFilename="quantmetrics-hmm-regime-timeline"
        uirevisionKey={`reg-${posteriors.length}-${K}`}
      />
      <div className="mt-1 flex flex-wrap items-center justify-between gap-2 font-mono text-[10px] text-zinc-500">
        <span>{posteriors.length.toLocaleString()} obs · Wheel: Pan X · Ctrl+Wheel: Zoom</span>
        <div className="flex items-center gap-2.5">
          {names.map((n, k) => (
            <span key={k} className="flex items-center gap-1">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: palette[k] }}
              />
              {n}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
