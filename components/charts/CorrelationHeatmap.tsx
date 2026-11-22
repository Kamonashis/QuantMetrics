'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { EmptyState } from '@/components/ui/empty-state';
import { CorrelationMatrixResult } from '@/lib/quant-engine';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

interface CorrelationHeatmapProps {
  result: CorrelationMatrixResult;
  returnsByTicker: { [ticker: string]: number[] };
}

export function CorrelationHeatmap({
  result,
  returnsByTicker,
}: CorrelationHeatmapProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [selectedPair, setSelectedPair] = useState<{ tA: string; tB: string } | null>(null);

  const { tickers, matrix, avgCorrelation, strongestPair, weakestPair } = result;

  const activePair = useMemo(
    () => selectedPair || (tickers.length >= 2 ? { tA: tickers[0], tB: tickers[1] } : null),
    [selectedPair, tickers]
  );

  const { heatTraces, heatLayout } = useMemo<{
    heatTraces: Data[];
    heatLayout: Partial<Layout>;
  }>(() => {
    if (!tickers || tickers.length === 0) return { heatTraces: [], heatLayout: {} };
    const textMatrix = matrix.map((row) => row.map((v) => v.toFixed(2)));
    return {
      heatTraces: [
        {
          type: 'heatmap',
          x: tickers,
          y: tickers,
          z: matrix,
          zmin: -1,
          zmax: 1,
          colorscale: [
            [0, '#E0655F'],
            [0.5, isDark ? '#1B1B22' : '#F4F4F5'],
            [1, '#3FB27F'],
          ],
          text: textMatrix as unknown as string[],
          texttemplate: '%{text}',
          textfont: { size: 11, family: 'IBM Plex Mono, monospace' },
          hovertemplate: '%{y} vs %{x}: <b>r = %{z:.3f}</b><extra></extra>',
          showscale: true,
          colorbar: {
            thickness: 12,
            len: 0.85,
            tickfont: { size: 10, color: tokens.textMuted },
          },
        },
      ],
      heatLayout: {
        margin: { l: 56, r: 24, t: 14, b: 44 },
        xaxis: { side: 'bottom', tickfont: { size: 11, color: tokens.text } },
        yaxis: { autorange: 'reversed', tickfont: { size: 11, color: tokens.text } },
      },
    };
  }, [tickers, matrix, isDark, tokens]);

  const { scatterTraces, scatterLayout, activeCorr } = useMemo<{
    scatterTraces: Data[];
    scatterLayout: Partial<Layout>;
    activeCorr: number | null;
  }>(() => {
    if (!activePair) return { scatterTraces: [], scatterLayout: {}, activeCorr: null };
    const pairReturnsA = returnsByTicker[activePair.tA] || [];
    const pairReturnsB = returnsByTicker[activePair.tB] || [];
    const len = Math.min(pairReturnsA.length, pairReturnsB.length);
    const xs = pairReturnsA.slice(0, len).map((v) => v * 100);
    const ys = pairReturnsB.slice(0, len).map((v) => v * 100);

    // OLS slope fit for regression overlay
    let meanX = 0;
    let meanY = 0;
    for (let i = 0; i < len; i++) {
      meanX += xs[i];
      meanY += ys[i];
    }
    meanX /= Math.max(1, len);
    meanY /= Math.max(1, len);
    let cov = 0;
    let varX = 0;
    for (let i = 0; i < len; i++) {
      const dx = xs[i] - meanX;
      cov += dx * (ys[i] - meanY);
      varX += dx * dx;
    }
    const slope = varX > 1e-12 ? cov / varX : 0;
    const intercept = meanY - slope * meanX;
    const minX = xs.length ? Math.min(...xs) : -5;
    const maxX = xs.length ? Math.max(...xs) : 5;

    const corr =
      matrix[tickers.indexOf(activePair.tA)]?.[tickers.indexOf(activePair.tB)] ?? null;

    return {
      activeCorr: corr,
      scatterTraces: [
        {
          type: 'scatter',
          mode: 'markers',
          name: 'Daily Returns',
          x: xs,
          y: ys,
          marker: { color: QUANT_COLORS.slate, size: 5.5, opacity: 0.7 },
          hovertemplate: `${activePair.tA} %{x:+.2f}% · ${activePair.tB} %{y:+.2f}%<extra></extra>`,
        },
        {
          type: 'scatter',
          mode: 'lines',
          name: `OLS Slope (${slope.toFixed(2)})`,
          x: [minX, maxX],
          y: [intercept + slope * minX, intercept + slope * maxX],
          line: { color: QUANT_COLORS.goldLight, width: 2 },
          hoverinfo: 'skip',
        },
      ],
      scatterLayout: {
        margin: { l: 50, r: 14, t: 12, b: 40 },
        xaxis: {
          title: { text: `${activePair.tA} Return`, font: { size: 10, color: tokens.textMuted } },
          ticksuffix: '%',
          zeroline: true,
          zerolinecolor: tokens.zeroLine,
          gridcolor: tokens.grid,
        },
        yaxis: {
          title: { text: `${activePair.tB} Return`, font: { size: 10, color: tokens.textMuted } },
          ticksuffix: '%',
          zeroline: true,
          zerolinecolor: tokens.zeroLine,
          gridcolor: tokens.grid,
        },
      },
    };
  }, [activePair, returnsByTicker, matrix, tickers, tokens]);

  if (!tickers || tickers.length === 0) {
    return (
      <EmptyState
        title="No correlation matrix available"
        detail="Input at least two tickers with overlapping history"
        className="h-64"
      />
    );
  }

  return (
    <div className="panel p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
              Cross-Asset Return Correlation Matrix
            </h3>
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              · {tickers.length} Assets Synchronized
            </span>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Click any cell on the Plotly heatmap to inspect the pairwise co-movement scatter and
            OLS regression slope
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs">
          <div>
            <span className="text-zinc-500 dark:text-zinc-400">Avg Correlation:</span>{' '}
            <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100 tabular-nums">
              {avgCorrelation.toFixed(2)}
            </span>
          </div>
          <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">
            ·
          </span>
          <div>
            <span className="text-zinc-500 dark:text-zinc-400">Max Co-movement:</span>{' '}
            <span className="font-mono font-bold text-gold-600 dark:text-gold-300 tabular-nums">
              {strongestPair.tickerA}/{strongestPair.tickerB} ({strongestPair.corr.toFixed(2)})
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Interactive Plotly Correlation Heatmap */}
        <div className="lg:col-span-7">
          <PlotlyChart
            data={heatTraces}
            layout={heatLayout}
            plotType="heatmap"
            enableWheelPan={false}
            hovermode="closest"
            height={Math.max(290, Math.min(440, tickers.length * 46 + 70))}
            exportFilename="quantmetrics-correlation-heatmap"
            uirevisionKey={`corr-hm-${tickers.join('-')}`}
            onClick={(ev) => {
              const pts = (ev as { points?: Array<{ x?: string; y?: string }> })?.points;
              const p = pts?.[0];
              if (p?.x && p?.y && p.x !== p.y) {
                setSelectedPair({ tA: String(p.y), tB: String(p.x) });
              }
            }}
          />
          <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400 mt-2">
            <span>Click any off-diagonal cell to update scatter</span>
            <span>
              Least Correlated:{' '}
              <span className="font-mono text-rose-500 tabular-nums">
                {weakestPair.tickerA}/{weakestPair.tickerB} ({weakestPair.corr.toFixed(2)})
              </span>
            </span>
          </div>
        </div>

        {/* Interactive Plotly Pairwise Return Scatter */}
        <div className="lg:col-span-5 panel-inset p-4 rounded-lg">
          {activePair ? (
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                  {activePair.tA} vs {activePair.tB} Return Scatter
                </span>
                <span className="text-xs font-mono text-gold-600 dark:text-gold-300 font-bold tabular-nums">
                  r = {activeCorr !== null ? activeCorr.toFixed(3) : '—'}
                </span>
              </div>

              <PlotlyChart
                data={scatterTraces}
                layout={scatterLayout}
                plotType="2d"
                hovermode="closest"
                height={260}
                exportFilename={`quantmetrics-scatter-${activePair.tA}-${activePair.tB}`}
                uirevisionKey={`scat-${activePair.tA}-${activePair.tB}`}
                showWheelHint
              />

              <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-2">
                OLS slope reflects return sensitivity between {activePair.tA} and {activePair.tB}.
                Tight diagonal clustering indicates systemic co-movement.
              </p>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-xs text-zinc-400">
              Select an asset pair to view return scatter
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
