'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { AcfPacfResult } from '@/lib/quant-engine';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

interface AcfPacfChartProps {
  acfPacf: AcfPacfResult;
}

export function AcfPacfChart({ acfPacf }: AcfPacfChartProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [activeTab, setActiveTab] = useState<'acf' | 'pacf'>('acf');

  const { lags, acf, pacf, confidenceUpper, confidenceLower } = acfPacf;
  const values = activeTab === 'acf' ? acf : pacf;

  const { traces, layout, significantLags } = useMemo<{
    traces: Data[];
    layout: Partial<Layout>;
    significantLags: number;
  }>(() => {
    if (!lags || lags.length === 0) {
      return { traces: [], layout: {}, significantLags: 0 };
    }
    const sigCount = values.filter((v) => v > confidenceUpper || v < confidenceLower).length;
    const colors = values.map((v) =>
      v > confidenceUpper || v < confidenceLower ? QUANT_COLORS.goldLight : tokens.textMuted
    );

    const maxLimit = Math.max(
      0.4,
      Math.min(1.0, Math.max(...values.map(Math.abs), confidenceUpper * 1.6) * 1.15)
    );

    const label = activeTab === 'acf' ? 'ACF' : 'PACF';

    return {
      traces: [
        {
          type: 'bar',
          name: `${label} Stem`,
          x: lags,
          y: values,
          width: 0.22,
          marker: { color: colors },
          showlegend: false,
          hoverinfo: 'skip',
        },
        {
          type: 'scatter',
          mode: 'markers',
          name: `${label} (r_t²)`,
          x: lags,
          y: values,
          marker: { color: colors, size: 7.5 },
          hovertemplate: `Lag %{x} · ${label} %{y:+.3f}<extra></extra>`,
        },
      ],
      layout: {
        shapes: [
          {
            type: 'rect',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: confidenceLower,
            y1: confidenceUpper,
            fillcolor: 'rgba(222, 192, 132, 0.08)',
            line: { width: 0 },
            layer: 'below',
          },
          {
            type: 'line',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: confidenceUpper,
            y1: confidenceUpper,
            line: { color: QUANT_COLORS.goldLight, width: 1.1, dash: 'dash' },
          },
          {
            type: 'line',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: confidenceLower,
            y1: confidenceLower,
            line: { color: QUANT_COLORS.goldLight, width: 1.1, dash: 'dash' },
          },
        ],
        annotations: [
          {
            xref: 'paper',
            yref: 'y',
            x: 1,
            y: confidenceUpper,
            text: `+${confidenceUpper.toFixed(2)} (95%)`,
            showarrow: false,
            xanchor: 'right',
            yanchor: 'bottom',
            font: { size: 9.5, color: QUANT_COLORS.goldLight },
          },
        ],
        xaxis: {
          title: { text: 'Lag k', font: { size: 10, color: tokens.textMuted } },
          dtick: 1,
          gridcolor: tokens.grid,
        },
        yaxis: {
          range: [-maxLimit, maxLimit],
          zeroline: true,
          zerolinecolor: tokens.zeroLine,
          zerolinewidth: 1,
          gridcolor: tokens.grid,
        },
      },
      significantLags: sigCount,
    };
  }, [lags, values, confidenceUpper, confidenceLower, activeTab, tokens]);

  if (!lags || lags.length === 0) {
    return (
      <div className="h-56 flex items-center justify-center border border-[var(--hairline)] rounded-lg text-zinc-500 text-xs">
        Insufficient return series for ACF/PACF computation
      </div>
    );
  }

  return (
    <div className="panel p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
              {activeTab === 'acf' ? 'Autocorrelation (ACF)' : 'Partial Autocorrelation (PACF)'} of
              Squared Returns
            </h3>
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              · Volatility Clustering Diagnostics
            </span>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Dashed bounds denote Bartlett&apos;s 95% white-noise confidence interval (±1.96/√N)
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-zinc-100 dark:bg-zinc-800/80 rounded-lg">
          <button
            type="button"
            onClick={() => setActiveTab('acf')}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              activeTab === 'acf'
                ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
            }`}
          >
            ACF (r_t²)
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('pacf')}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              activeTab === 'pacf'
                ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
            }`}
          >
            PACF (r_t²)
          </button>
        </div>
      </div>

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="x unified"
        height={250}
        exportFilename={`quantmetrics-${activeTab}`}
        uirevisionKey={`acf-${activeTab}-${lags.length}`}
        showWheelHint
      />

      <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs text-zinc-500 dark:text-zinc-400 mt-2 pt-2 border-t border-[var(--hairline)] gap-2">
        <div className="flex items-center gap-2">
          <span>Significant Lags (&gt;95% bound):</span>
          <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100 tabular-nums">
            {significantLags} of {lags.length}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {significantLags > 2 ? (
            <span className="text-gold-600 dark:text-gold-400 font-medium">
              Persistent ARCH effects: Volatility clustering confirmed (GARCH recommended)
            </span>
          ) : (
            <span className="text-gold-600 dark:text-gold-400 font-medium">
              Weak autocorrelation: Return variance is approximately homoskedastic
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
