'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { RegressionSummary } from '@/lib/quant-engine';
import { SERIES_COLORS } from '@/lib/chart-palette';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

interface ResidualDiagnosticsChartProps {
  summary: RegressionSummary;
}

export function ResidualDiagnosticsChart({ summary }: ResidualDiagnosticsChartProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [activeTab, setActiveTab] = useState<'qq' | 'fitted' | 'histogram' | 'rolling'>('qq');

  const { residuals, rollingBetas, regressionType, independentTickers } = summary;

  const { traces, layout, isTimeSeries } = useMemo<{
    traces: Data[];
    layout: Partial<Layout>;
    isTimeSeries: boolean;
  }>(() => {
    if (!residuals || residuals.length === 0) {
      return { traces: [], layout: {}, isTimeSeries: false };
    }

    if (activeTab === 'qq') {
      const theo = residuals.map((r) => r.theoreticalQuantile);
      const samp = residuals.map((r) => r.standardizedResidual);
      const bound = Math.max(
        3.2,
        Math.max(...theo.map(Math.abs)),
        Math.max(...samp.map(Math.abs))
      );

      return {
        isTimeSeries: false,
        traces: [
          {
            type: 'scatter',
            mode: 'lines',
            name: 'Normal 45° Reference',
            x: [-bound, bound],
            y: [-bound, bound],
            line: { color: QUANT_COLORS.goldLight, width: 1.5, dash: 'dash' },
            hoverinfo: 'skip',
          },
          {
            type: 'scatter',
            mode: 'markers',
            name: 'Standardized Residuals',
            x: theo,
            y: samp,
            marker: { color: QUANT_COLORS.slate, size: 5.5, opacity: 0.78 },
            hovertemplate: 'Theoretical %{x:.2f} · Sample %{y:.2f}<extra>Q-Q</extra>',
          },
        ],
        layout: {
          xaxis: {
            title: { text: 'Theoretical Quantiles N(0, 1)', font: { size: 10, color: tokens.textMuted } },
            zeroline: true,
            zerolinecolor: tokens.zeroLine,
            gridcolor: tokens.grid,
          },
          yaxis: {
            title: { text: 'Standardized Residuals (e_i / s)', font: { size: 10, color: tokens.textMuted } },
            zeroline: true,
            zerolinecolor: tokens.zeroLine,
            gridcolor: tokens.grid,
          },
        },
      };
    }

    if (activeTab === 'fitted') {
      return {
        isTimeSeries: false,
        traces: [
          {
            type: 'scatter',
            mode: 'markers',
            name: 'Residuals vs Fitted',
            x: residuals.map((r) => r.fitted * 100),
            y: residuals.map((r) => r.residual * 100),
            marker: { color: QUANT_COLORS.slate, size: 5.5, opacity: 0.72 },
            hovertemplate: 'Fitted %{x:+.2f}% · Resid %{y:+.2f}%<extra></extra>',
          },
        ],
        layout: {
          shapes: [
            {
              type: 'line',
              xref: 'paper',
              yref: 'y',
              x0: 0,
              x1: 1,
              y0: 0,
              y1: 0,
              line: { color: QUANT_COLORS.positive, width: 1.2, dash: 'dash' },
            },
          ],
          xaxis: {
            title: { text: 'Fitted Value (Ŷ_t)', font: { size: 10, color: tokens.textMuted } },
            ticksuffix: '%',
            gridcolor: tokens.grid,
          },
          yaxis: {
            title: { text: 'Residual (e_t)', font: { size: 10, color: tokens.textMuted } },
            ticksuffix: '%',
            gridcolor: tokens.grid,
          },
        },
      };
    }

    if (activeTab === 'histogram') {
      return {
        isTimeSeries: false,
        traces: [
          {
            type: 'histogram',
            name: 'Residual Frequency',
            x: residuals.map((r) => r.residual * 100),
            nbinsx: 30,
            marker: {
              color: 'rgba(139, 139, 147, 0.7)',
              line: { color: QUANT_COLORS.goldLight, width: 0.6 },
            },
            hovertemplate: 'Bin %{x:.2f}% · %{y} obs<extra>Residuals</extra>',
          },
        ],
        layout: {
          bargap: 0.06,
          xaxis: {
            title: { text: 'Residual (e_t)', font: { size: 10, color: tokens.textMuted } },
            ticksuffix: '%',
            gridcolor: tokens.grid,
          },
          yaxis: {
            title: { text: 'Count', font: { size: 10, color: tokens.textMuted } },
            rangemode: 'tozero',
            gridcolor: tokens.grid,
          },
        },
      };
    }

    // Rolling Betas time series
    const rb = rollingBetas ?? [];
    const dates = rb.map((r) => r.date);
    const betaTraces: Data[] = independentTickers.map((tkr, tIdx) => ({
      type: 'scatter',
      mode: 'lines',
      name: `Beta (${tkr})`,
      x: dates,
      y: rb.map((r) => r.betas[tkr] ?? null),
      line: { color: SERIES_COLORS[tIdx % SERIES_COLORS.length], width: 1.8 },
      hovertemplate: `%{y:.3f}<extra>β (${tkr})</extra>`,
    }));

    return {
      isTimeSeries: true,
      traces: betaTraces,
      layout: {
        shapes: [
          {
            type: 'line',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: 0,
            y1: 0,
            line: { color: tokens.zeroLine, width: 1, dash: 'dot' },
          },
          {
            type: 'line',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: 1,
            y1: 1,
            line: { color: QUANT_COLORS.goldLight, width: 1, dash: 'dot' },
          },
        ],
        yaxis: {
          title: { text: 'Rolling β', font: { size: 10, color: tokens.textMuted } },
          gridcolor: tokens.grid,
        },
      },
    };
  }, [activeTab, residuals, rollingBetas, independentTickers, tokens]);

  if (!residuals || residuals.length === 0) return null;

  return (
    <div className="panel p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
              Model Diagnostics &amp; Residual Analysis
            </h3>
            <span className="text-xs text-zinc-500 dark:text-zinc-400">· {regressionType}</span>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Evaluate normality, homoskedasticity, and time-varying parameter stability
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-1 p-1 bg-zinc-100 dark:bg-zinc-800/80 rounded-lg text-xs">
          {(
            [
              { id: 'qq', label: 'Normal Q-Q Plot' },
              { id: 'fitted', label: 'Residual vs Fitted' },
              { id: 'histogram', label: 'Residual Distribution' },
              ...(rollingBetas && rollingBetas.length > 0
                ? [{ id: 'rolling', label: 'Rolling Betas' }]
                : []),
            ] as const
          ).map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id as typeof activeTab)}
              className={`px-3 py-1 font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeTab === tab.id
                  ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-xs'
                  : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {activeTab === 'histogram' && (
        <div className="flex flex-wrap items-center justify-around gap-2 text-xs mb-3 font-mono text-zinc-600 dark:text-zinc-400">
          <span>
            Durbin-Watson:{' '}
            <strong className="text-zinc-900 dark:text-zinc-100 tabular-nums">
              {summary.durbinWatson.toFixed(3)}
            </strong>{' '}
            (2.0 = no serial corr)
          </span>
          <span>
            Regression SE:{' '}
            <strong className="text-zinc-900 dark:text-zinc-100 tabular-nums">
              {(summary.standardErrorOfRegression * 100).toFixed(3)}%
            </strong>
          </span>
        </div>
      )}

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        rangeSlider={isTimeSeries}
        hovermode={isTimeSeries ? 'x unified' : 'closest'}
        showLegend={isTimeSeries}
        height={isTimeSeries ? 310 : 285}
        exportFilename={`quantmetrics-residuals-${activeTab}`}
        uirevisionKey={`resid-${activeTab}-${residuals.length}`}
        showWheelHint
      />

      {activeTab === 'qq' && (
        <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 text-center">
          Points conforming tightly to the diagonal line indicate normal residual distribution.
          Deviations in tails reflect market kurtosis/fat tails.
        </p>
      )}
      {activeTab === 'fitted' && (
        <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 text-center">
          Residuals randomly scattered around the zero line without a funnel shape signify
          homoskedasticity and absence of structural misspecification.
        </p>
      )}
    </div>
  );
}
