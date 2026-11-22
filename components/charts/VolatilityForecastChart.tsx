'use client';

import React, { useMemo, useState } from 'react';
import type { Annotations, Data, Layout, Shape } from 'plotly.js';
import { GarchModelSummary, GarchForecastPoint } from '@/lib/quant-engine';
import {
  getHmmRegimeToken,
  type HmmStateIdx,
  type VolRegimePerfRow,
} from '@/lib/hmm-regime-overlay';
import {
  HmmRegimeOverlayToggle,
  HmmRegimeScorecard,
} from '@/components/charts/HmmRegimeOverlayControls';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import {
  buildHmmRegimeShapes,
  getPlotlyThemeTokens,
  QUANT_COLORS,
} from '@/lib/plotly-theme';

interface VolatilityForecastChartProps {
  dates: string[];
  historicalVol: (number | null)[];
  /** The selected model's conditional vol series (EWMA / GJR-GARCH / HMM regime-mixture / TCN) */
  modelVol: number[];
  modelLabel: string;
  forecastPoints: GarchForecastPoint[];
  modelSummary: GarchModelSummary;
  /** Optional strictly causal dominant HMM state per historical return session */
  hmmStates?: HmmStateIdx[];
  /** Optional causal HMM posterior probability per historical return session */
  hmmProbs?: number[];
  /** Index into historical series where OOS holdout begins */
  holdoutStartIdx?: number;
  /** Per-regime 1d OOS performance rows (Calm / Watchful / Crisis) */
  regimePerformance?: VolRegimePerfRow[];
}

/**
 * Forward projection chart: 30d historical vol, the selected model's conditional vol line,
 * forward variance cones (68% / 95%), and causal HMM volatility regime overlays.
 */
export function VolatilityForecastChart({
  dates,
  historicalVol,
  modelVol,
  modelLabel,
  forecastPoints,
  modelSummary,
  hmmStates,
  hmmProbs,
  holdoutStartIdx,
  regimePerformance,
}: VolatilityForecastChartProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [showModel, setShowModel] = useState(true);
  const [showHist, setShowHist] = useState(true);
  const [showRegimeOverlay, setShowRegimeOverlay] = useState(true);

  const hasHmmOverlay = Boolean(hmmStates && hmmStates.length > 0);
  const nHist = dates.length;
  const nFc = forecastPoints.length;
  const totalPoints = nHist + nFc;
  const flatProjection = modelSummary.persistence >= 1;
  const lastModelVol = modelVol[nHist - 1] || forecastPoints[0]?.annualizedVol || 0.2;

  const { traces, layout } = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (totalPoints === 0) return { traces: [], layout: {} };

    const splitDate = dates[nHist - 1];
    const fcDates = [splitDate, ...forecastPoints.map((p) => p.date)];
    const fcMid = [lastModelVol * 100, ...forecastPoints.map((p) => p.annualizedVol * 100)];
    const fcUp95 = [lastModelVol * 100, ...forecastPoints.map((p) => p.upper95 * 100)];
    const fcLo95 = [lastModelVol * 100, ...forecastPoints.map((p) => p.lower95 * 100)];
    const fcUp68 = [lastModelVol * 100, ...forecastPoints.map((p) => p.upper68 * 100)];
    const fcLo68 = [lastModelVol * 100, ...forecastPoints.map((p) => p.lower68 * 100)];

    const dataTraces: Data[] = [
      // 95% confidence band
      {
        type: 'scatter',
        mode: 'lines',
        name: '95% Lower',
        x: fcDates,
        y: fcLo95,
        line: { width: 0 },
        showlegend: false,
        hoverinfo: 'skip',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: '95% Cone',
        x: fcDates,
        y: fcUp95,
        fill: 'tonexty',
        fillcolor: 'rgba(222, 192, 132, 0.14)',
        line: { width: 0 },
        hovertemplate: '%{y:.2f}%<extra>95% Upper</extra>',
      },
      // 68% confidence band
      {
        type: 'scatter',
        mode: 'lines',
        name: '68% Lower',
        x: fcDates,
        y: fcLo68,
        line: { width: 0 },
        showlegend: false,
        hoverinfo: 'skip',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: '68% Cone',
        x: fcDates,
        y: fcUp68,
        fill: 'tonexty',
        fillcolor: 'rgba(222, 192, 132, 0.24)',
        line: { width: 0 },
        hovertemplate: '%{y:.2f}%<extra>68% Upper</extra>',
      },
    ];

    if (showHist) {
      dataTraces.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Historical (30d RV)',
        x: dates,
        y: historicalVol.map((v) => (v !== null ? v * 100 : null)),
        line: { color: tokens.textMuted, width: 1.2, dash: 'dot' },
        connectgaps: false,
        hovertemplate: '%{y:.2f}%<extra>RV30</extra>',
      });
    }

    if (showModel) {
      const regimeCustom = dates.map((_, i) => {
        const st = hmmStates?.[i];
        const pr = hmmProbs?.[i];
        if (st === undefined) return '';
        const tok = getHmmRegimeToken(st);
        const pct = pr !== undefined ? ` (${Math.round(pr * 100)}%)` : '';
        return ` · ${tok.label}${pct}`;
      });

      dataTraces.push({
        type: 'scatter',
        mode: 'lines',
        name: modelLabel,
        x: dates,
        y: modelVol.map((v) => v * 100),
        customdata: regimeCustom,
        line: { color: QUANT_COLORS.goldLight, width: 1.9 },
        hovertemplate: `%{y:.2f}%%{customdata}<extra>${modelLabel}</extra>`,
      });

      // Color-coded OOS Prediction Points by Causal HMM Regime
      if (hasHmmOverlay && showRegimeOverlay && hmmStates && holdoutStartIdx !== undefined) {
        const start = Math.max(0, holdoutStartIdx);
        const oosLen = Math.max(1, nHist - start);
        const stride = Math.max(1, Math.ceil(oosLen / 56));
        const ox: string[] = [];
        const oy: number[] = [];
        const oc: string[] = [];
        const ot: string[] = [];
        for (let i = start; i < nHist; i += stride) {
          const st = hmmStates[i];
          const val = modelVol[i];
          if (st === undefined || !Number.isFinite(val)) continue;
          const tok = getHmmRegimeToken(st);
          ox.push(dates[i]);
          oy.push(val * 100);
          oc.push(tok.color);
          const pr = hmmProbs?.[i];
          ot.push(`${tok.label}${pr !== undefined ? ` (${Math.round(pr * 100)}%)` : ''}`);
        }
        if (ox.length > 0) {
          dataTraces.push({
            type: 'scatter',
            mode: 'markers',
            name: 'OOS Regime Origin',
            x: ox,
            y: oy,
            customdata: ot,
            marker: {
              color: oc,
              size: 6,
              line: { color: isDark ? '#08080A' : '#FFFFFF', width: 1 },
            },
            hovertemplate: '%{y:.2f}% · %{customdata}<extra>OOS Origin</extra>',
          });
        }
      }
    }

    // Forward trajectory dashed line
    dataTraces.push({
      type: 'scatter',
      mode: 'lines',
      name: `${nFc}d Forecast`,
      x: fcDates,
      y: fcMid,
      line: { color: QUANT_COLORS.gold, width: 2, dash: 'dash' },
      hovertemplate: '%{y:.2f}%<extra>Forecast</extra>',
    });

    const shapes: Partial<Shape>[] = [
      ...buildHmmRegimeShapes(dates, hmmStates ?? [], hasHmmOverlay && showRegimeOverlay, 0.12),
      // Split Today boundary
      {
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: splitDate,
        x1: splitDate,
        y0: 0,
        y1: 1,
        line: { color: tokens.textMuted, width: 1.1, dash: 'dot' },
      },
    ];

    const annotations: Partial<Annotations>[] = [
      {
        x: splitDate,
        y: 1.0,
        xref: 'x',
        yref: 'paper',
        text: 'Forecast →',
        showarrow: false,
        xanchor: 'left',
        yanchor: 'top',
        xshift: 6,
        font: { size: 10, color: QUANT_COLORS.goldLight },
      },
    ];

    if (
      hasHmmOverlay &&
      showRegimeOverlay &&
      holdoutStartIdx !== undefined &&
      holdoutStartIdx > 0 &&
      holdoutStartIdx < nHist
    ) {
      const oosDate = dates[holdoutStartIdx];
      shapes.push({
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: oosDate,
        x1: oosDate,
        y0: 0,
        y1: 1,
        line: { color: QUANT_COLORS.goldLight, width: 1, dash: 'dot' },
      });
      annotations.push({
        x: oosDate,
        y: 0.96,
        xref: 'x',
        yref: 'paper',
        text: 'OOS Holdout',
        showarrow: false,
        xanchor: 'left',
        yanchor: 'top',
        xshift: 4,
        font: { size: 9.5, color: QUANT_COLORS.goldLight },
      });
    }

    if (!flatProjection) {
      const vlPct = modelSummary.longRunAnnualizedVol * 100;
      shapes.push({
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: vlPct,
        y1: vlPct,
        line: { color: QUANT_COLORS.positive, width: 1, dash: 'dash' },
      });
      annotations.push({
        xref: 'paper',
        yref: 'y',
        x: 1,
        y: vlPct,
        text: `V_L: ${vlPct.toFixed(1)}%`,
        showarrow: false,
        xanchor: 'right',
        yanchor: 'bottom',
        font: { size: 9.5, color: QUANT_COLORS.positive },
      });
    }

    return {
      traces: dataTraces,
      layout: {
        shapes,
        annotations,
        yaxis: {
          ticksuffix: '%',
          rangemode: 'tozero',
          gridcolor: tokens.grid,
        },
      },
    };
  }, [
    totalPoints,
    dates,
    nHist,
    nFc,
    forecastPoints,
    lastModelVol,
    showHist,
    showModel,
    historicalVol,
    modelVol,
    modelLabel,
    hmmStates,
    hmmProbs,
    hasHmmOverlay,
    showRegimeOverlay,
    holdoutStartIdx,
    flatProjection,
    modelSummary,
    isDark,
    tokens,
  ]);

  if (totalPoints === 0) return null;

  return (
    <div className="panel p-5">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
              Volatility Forecast Projection
            </h3>
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              · forward variance projection
            </span>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            {nFc}-day {modelLabel} projection with 68% and 95% confidence cones
          </p>
        </div>

        {/* Series Visibility Toggles */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <button
            type="button"
            onClick={() => setShowModel(!showModel)}
            className={`px-2.5 py-1 rounded-md font-medium transition-colors border whitespace-nowrap shrink-0 cursor-pointer ${
              showModel
                ? 'bg-gold-500/10 border-gold-500/30 text-gold-600 dark:text-gold-400'
                : 'border-zinc-200 dark:border-zinc-800 text-zinc-400'
            }`}
          >
            {modelLabel}
          </button>
          <button
            type="button"
            onClick={() => setShowHist(!showHist)}
            className={`px-2.5 py-1 rounded-md font-medium transition-colors border whitespace-nowrap shrink-0 cursor-pointer ${
              showHist
                ? 'bg-zinc-500/10 border-zinc-500/30 text-zinc-600 dark:text-zinc-400'
                : 'border-zinc-200 dark:border-zinc-800 text-zinc-400'
            }`}
          >
            Historical (30d)
          </button>
        </div>
      </div>

      {/* HMM Regime Overlay Controls */}
      {hasHmmOverlay && (
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5 pb-2 border-b border-[var(--hairline)]">
          <HmmRegimeOverlayToggle
            enabled={showRegimeOverlay}
            onToggle={() => setShowRegimeOverlay((v) => !v)}
            caption="OOS points colored by origin t−1"
          />
        </div>
      )}

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        rangeSlider
        hovermode="x unified"
        height={320}
        exportFilename="quantmetrics-vol-forecast"
        uirevisionKey={`vf-${dates[0]}-${dates.length}-${modelLabel}-${nFc}`}
        showWheelHint
      />

      {/* Per-Regime OOS Volatility Accuracy Scorecard */}
      {regimePerformance && regimePerformance.length > 0 && (
        <HmmRegimeScorecard
          className="mt-4"
          title="TCN 1d OOS Volatility Accuracy by Causal HMM Regime"
          subtitle="Grouped by dominant forward-filtered HMM state at prediction origin (t−1)"
          columns={[
            { key: 'qlike', label: 'Mean QLIKE' },
            { key: 'mse', label: 'MSE log-RV' },
            { key: 'predVol', label: 'Mean Pred Vol' },
            { key: 'realVol', label: 'Mean Realized Vol' },
          ]}
          rows={regimePerformance.map((r) => ({
            state: r.state,
            count: r.count,
            cells: {
              qlike: r.meanQlike !== null ? r.meanQlike.toFixed(3) : '—',
              mse: r.meanMseLogVar !== null ? r.meanMseLogVar.toFixed(3) : '—',
              predVol: r.meanPredVol !== null ? `${(r.meanPredVol * 100).toFixed(1)}%` : '—',
              realVol: r.meanRealizedVol !== null ? `${(r.meanRealizedVol * 100).toFixed(1)}%` : '—',
            },
          }))}
        />
      )}

      {/* Model Diagnostic Footnotes & Parameter Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4 pt-3 border-t border-[var(--hairline)] text-xs">
        <div className="panel-inset p-2.5 rounded-lg">
          <span className="text-zinc-500 dark:text-zinc-400 block">Model Weights (α, β)</span>
          <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100 tabular-nums text-xs">
            {modelSummary.alpha.toFixed(3)} · {modelSummary.beta.toFixed(3)}
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            ω: {modelSummary.omega.toExponential(2)}
          </span>
        </div>

        <div className="panel-inset p-2.5 rounded-lg">
          <span className="text-zinc-500 dark:text-zinc-400 block">Shock Persistence (α+β)</span>
          <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100 tabular-nums text-xs">
            {flatProjection ? '1.0000' : modelSummary.persistence.toFixed(4)}
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            {flatProjection
              ? 'No mean reversion'
              : `Half-life: ${modelSummary.halfLife ? `${modelSummary.halfLife.toFixed(1)} days` : 'Infinite'}`}
          </span>
        </div>

        <div className="panel-inset p-2.5 rounded-lg">
          <span className="text-zinc-500 dark:text-zinc-400 block">Projection Level (start)</span>
          <span className="font-mono font-bold text-gold-600 dark:text-gold-400 tabular-nums text-xs">
            {(lastModelVol * 100).toFixed(2)}%
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            {flatProjection
              ? 'flat-forward from latest EWMA vol'
              : `reverts toward V_L ${(modelSummary.longRunAnnualizedVol * 100).toFixed(1)}%`}
          </span>
        </div>

        <div className="panel-inset p-2.5 rounded-lg">
          <span className="text-zinc-500 dark:text-zinc-400 block">Projection Horizon</span>
          <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100 tabular-nums text-xs">
            {nFc} trading days
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            68% / 95% cones widen as √h
          </span>
        </div>
      </div>
    </div>
  );
}
