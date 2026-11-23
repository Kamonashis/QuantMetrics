'use client';

import React, { useMemo, useState } from 'react';
import type { Annotations, Data, Layout, Shape } from 'plotly.js';
import { Activity, BarChart3 } from 'lucide-react';
import { EmptyState } from '@/components/ui/empty-state';
import { PriceBar } from '@/lib/quant-engine';
import { computeVolumeProfile, type VolumeProfileResult } from '@/lib/volume-profile';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

const defaultPriceFormatter = (v: number) => `$${v.toFixed(2)}`;

export interface StructuralBreakMarker {
  date: string;
  index?: number;
  reason: string;
  severity?: 'High' | 'Moderate' | 'Low' | string;
}

export interface CusumSqPoint {
  date: string;
  s: number;
  upperBand: number;
  lowerBand: number;
}

interface TimeSeriesChartProps {
  bars: PriceBar[];
  /** Provenance of the bars/volume used for the profile estimate. */
  barsSource?: string | null;
  title?: string;
  subtitle?: string;
  showVolume?: boolean;
  valueFormatter?: (val: number) => string;
  breakPoints?: StructuralBreakMarker[];
  cusumSqPoints?: CusumSqPoint[];
}

export function TimeSeriesChart({
  bars,
  barsSource,
  title = 'Historical Price',
  subtitle,
  showVolume = true,
  valueFormatter = defaultPriceFormatter,
  breakPoints = [],
  cusumSqPoints = [],
}: TimeSeriesChartProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [showBreaks, setShowBreaks] = useState<boolean>(true);
  const [showCusumCurve, setShowCusumCurve] = useState<boolean>(false);
  const [showVolumeProfile, setShowVolumeProfile] = useState<boolean>(true);

  const validBars = useMemo(() => bars || [], [bars]);

  // OHLC-derived Volume-at-Price estimate over the bars currently on screen.
  const volumeProfile = useMemo<VolumeProfileResult | null>(
    () => (showVolumeProfile ? computeVolumeProfile(validBars, { bins: 48 }) : null),
    [showVolumeProfile, validBars]
  );

  const resolvedBreakPoints = useMemo(() => {
    if (!breakPoints || breakPoints.length === 0 || validBars.length === 0) return [];
    return breakPoints
      .map((bp) => {
        let idx = validBars.findIndex((b) => b.date === bp.date);
        if (idx === -1 && bp.index !== undefined && bp.index >= 0 && bp.index < validBars.length) {
          idx = bp.index;
        }
        if (idx === -1) {
          let closestDist = Infinity;
          let closestIdx = -1;
          const targetTime = new Date(bp.date).getTime();
          for (let i = 0; i < validBars.length; i++) {
            const t = new Date(validBars[i].date).getTime();
            const dist = Math.abs(t - targetTime);
            if (dist < closestDist) {
              closestDist = dist;
              closestIdx = i;
            }
          }
          idx = closestIdx;
        }
        if (idx === -1) return null;
        return {
          ...bp,
          index: idx,
          bar: validBars[idx],
        };
      })
      .filter((bp): bp is NonNullable<typeof bp> => bp !== null);
  }, [breakPoints, validBars]);

  const isUp = validBars.length > 0 && validBars[validBars.length - 1].close >= validBars[0].close;
  const strokeColor = isUp ? QUANT_COLORS.positive : QUANT_COLORS.negative;
  const fillRgba = isUp ? 'rgba(63, 178, 127, 0.14)' : 'rgba(224, 101, 95, 0.14)';

  const lastBar = validBars[validBars.length - 1];

  const { priceTraces, priceLayout } = useMemo<{
    priceTraces: Data[];
    priceLayout: Partial<Layout>;
  }>(() => {
    if (validBars.length === 0) return { priceTraces: [], priceLayout: {} };
    const dates = validBars.map((b) => b.date);
    const closes = validBars.map((b) => b.close);
    const volumes = validBars.map((b) => b.volume / 1e6);
    const volColors = validBars.map((b, i) => {
      const up = i === 0 || b.close >= validBars[i - 1].close;
      return up ? 'rgba(63, 178, 127, 0.42)' : 'rgba(224, 101, 95, 0.42)';
    });

    const traces: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Close',
        x: dates,
        y: closes,
        yaxis: 'y',
        line: { color: strokeColor, width: 1.8 },
        fill: 'tozeroy',
        fillcolor: fillRgba,
        hovertemplate: '%{y:,.2f}<extra>Close</extra>',
      },
    ];

    if (showVolume) {
      traces.push({
        type: 'bar',
        name: 'Volume (M)',
        x: dates,
        y: volumes,
        yaxis: 'y2',
        marker: { color: volColors },
        hovertemplate: '%{y:.2f}M<extra>Volume</extra>',
      });
    }

    const shapes: Partial<Shape>[] = [];
    const annotations: Partial<Annotations>[] = [];

    if (showBreaks && resolvedBreakPoints.length > 0) {
      traces.push({
        type: 'scatter',
        mode: 'markers',
        name: 'Regime Break',
        x: resolvedBreakPoints.map((bp) => bp.bar.date),
        y: resolvedBreakPoints.map((bp) => bp.bar.close),
        customdata: resolvedBreakPoints.map((bp) => [bp.reason, bp.severity ?? 'Significant']),
        yaxis: 'y',
        marker: {
          color: QUANT_COLORS.goldLight,
          size: 8,
          symbol: 'diamond',
          line: { color: isDark ? '#08080A' : '#FFFFFF', width: 1.2 },
        },
        hovertemplate:
          '<b>Regime Break</b><br>Close: %{y:,.2f}<br>%{customdata[0]} (%{customdata[1]})<extra></extra>',
      });

      for (const bp of resolvedBreakPoints) {
        shapes.push({
          type: 'line',
          xref: 'x',
          yref: 'paper',
          x0: bp.bar.date,
          x1: bp.bar.date,
          y0: 0,
          y1: 1,
          line: { color: QUANT_COLORS.goldLight, width: 1.1, dash: 'dash' },
        });
        annotations.push({
          x: bp.bar.date,
          y: 1.01,
          xref: 'x',
          yref: 'paper',
          text: bp.bar.date.length > 5 ? bp.bar.date.slice(5) : bp.bar.date,
          showarrow: false,
          yanchor: 'bottom',
          font: { size: 9, color: QUANT_COLORS.goldLight },
        });
      }
    }

    if (volumeProfile) {
      const { poc, valueArea, bins, hvn } = volumeProfile;
      const maxVolume = bins.reduce((max, b) => Math.max(max, b.volume), 0);
      const maxBarWidth = 0.2; // fraction of plot width reserved for the profile

      // Value-area band spans the full chart behind the price trace.
      if (valueArea) {
        shapes.push({
          type: 'rect',
          xref: 'paper',
          yref: 'y',
          x0: 0,
          x1: 1,
          y0: valueArea.low,
          y1: valueArea.high,
          fillcolor: 'rgba(222,192,132,0.05)',
          line: { width: 0 },
          layer: 'below',
        });
      }

      // Right-anchored horizontal bars: width proportional to volume at price.
      if (maxVolume > 0) {
        for (const bin of bins) {
          if (bin.volume <= 0) continue;
          const width = (bin.volume / maxVolume) * maxBarWidth;
          const isPoc = poc !== null && bin.mid === poc.mid;
          const isHvn = hvn.some((node) => node.mid === bin.mid);
          shapes.push({
            type: 'rect',
            xref: 'paper',
            yref: 'y',
            x0: 1 - width,
            x1: 1,
            y0: bin.lower,
            y1: bin.upper,
            fillcolor: isPoc
              ? 'rgba(222,192,132,0.5)'
              : isHvn
                ? 'rgba(222,192,132,0.3)'
                : 'rgba(222,192,132,0.16)',
            line: { width: 0 },
            layer: 'below',
          });
        }
      }

      const addLevel = (y: number, label: string, color: string, dash: 'solid' | 'dash' | 'dot') => {
        shapes.push({
          type: 'line',
          xref: 'paper',
          yref: 'y',
          x0: 0,
          x1: 1,
          y0: y,
          y1: y,
          line: { color, width: 1.1, dash },
        });
        annotations.push({
          x: 1,
          y,
          xref: 'paper',
          yref: 'y',
          text: label,
          showarrow: false,
          xanchor: 'right',
          yanchor: 'bottom',
          bgcolor: isDark ? 'rgba(8,8,10,0.78)' : 'rgba(245,244,241,0.82)',
          font: { size: 9, color },
        });
      };

      if (poc) {
        addLevel(poc.mid, `POC ${valueFormatter(poc.mid)}`, QUANT_COLORS.goldLight, 'solid');
      }
      if (valueArea && poc && valueArea.low < poc.mid) {
        addLevel(valueArea.low, `VA ${valueFormatter(valueArea.low)}`, 'rgba(222,192,132,0.7)', 'dash');
      }
      if (valueArea && poc && valueArea.high > poc.mid) {
        addLevel(valueArea.high, `VA ${valueFormatter(valueArea.high)}`, 'rgba(222,192,132,0.7)', 'dash');
      }
    }

    const layout: Partial<Layout> = {
      shapes,
      annotations,
      bargap: 0.12,
      yaxis: {
        domain: showVolume ? [0.24, 1] : [0, 1],
        gridcolor: tokens.grid,
        zeroline: false,
        tickfont: { size: 10.5, color: tokens.textMuted },
      },
      ...(showVolume
        ? {
            yaxis2: {
              domain: [0, 0.18],
              gridcolor: tokens.grid,
              zeroline: false,
              showticklabels: true,
              nticks: 3,
              ticksuffix: 'M',
              tickfont: { size: 9.5, color: tokens.textMuted },
              fixedrange: true,
            },
          }
        : {}),
    };

    return { priceTraces: traces, priceLayout: layout };
  }, [
    validBars,
    strokeColor,
    fillRgba,
    showVolume,
    showBreaks,
    resolvedBreakPoints,
    volumeProfile,
    valueFormatter,
    isDark,
    tokens,
  ]);

  const { cusumTraces, cusumLayout } = useMemo<{
    cusumTraces: Data[];
    cusumLayout: Partial<Layout>;
  }>(() => {
    if (!showCusumCurve || cusumSqPoints.length === 0) {
      return { cusumTraces: [], cusumLayout: {} };
    }
    const n = cusumSqPoints.length;
    const dates = cusumSqPoints.map((p) => p.date);
    const expected = cusumSqPoints.map((_, idx) => idx / Math.max(1, n - 1));
    const upper = cusumSqPoints.map((p) => p.upperBand);
    const lower = cusumSqPoints.map((p) => p.lowerBand);
    const sk = cusumSqPoints.map((p) => p.s);

    const traces: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: 'E[S_k] = k/n',
        x: dates,
        y: expected,
        line: { color: tokens.textMuted, width: 1, dash: 'dot' },
        hovertemplate: '%{y:.3f}<extra>Expected k/n</extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: '+c_α (95%)',
        x: dates,
        y: upper,
        line: { color: QUANT_COLORS.goldLight, width: 1, dash: 'dash' },
        hovertemplate: '%{y:.3f}<extra>Upper 95%</extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: '−c_α (95%)',
        x: dates,
        y: lower,
        line: { color: QUANT_COLORS.goldLight, width: 1, dash: 'dash' },
        fill: 'tonexty',
        fillcolor: 'rgba(222, 192, 132, 0.08)',
        hovertemplate: '%{y:.3f}<extra>Lower 95%</extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'S_k',
        x: dates,
        y: sk,
        line: { color: QUANT_COLORS.positive, width: 1.8 },
        hovertemplate: '%{y:.3f}<extra>CUSUM S_k</extra>',
      },
    ];

    const shapes: Partial<Shape>[] = showBreaks
      ? resolvedBreakPoints.map((bp) => ({
          type: 'line',
          xref: 'x',
          yref: 'paper',
          x0: bp.bar.date,
          x1: bp.bar.date,
          y0: 0,
          y1: 1,
          line: { color: QUANT_COLORS.goldLight, width: 1, dash: 'dot' },
        }))
      : [];

    return {
      cusumTraces: traces,
      cusumLayout: {
        shapes,
        yaxis: {
          range: [-0.05, 1.05],
          tickvals: [0, 0.5, 1],
          gridcolor: tokens.grid,
          zeroline: false,
        },
      },
    };
  }, [showCusumCurve, cusumSqPoints, showBreaks, resolvedBreakPoints, tokens]);

  if (!bars || bars.length === 0) {
    return (
      <EmptyState
        title="No price data available"
        detail="The active ticker has no bars in the current cache"
        className="h-64"
      />
    );
  }

  return (
    <div className="panel p-5 space-y-4">
      {/* Top Header Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[var(--hairline)]">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
              {title}
            </h3>
            {subtitle && (
              <span className="text-xs text-zinc-500 dark:text-zinc-400">· {subtitle}</span>
            )}
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Historical trajectory across {bars.length.toLocaleString()} observations
            {resolvedBreakPoints.length > 0 && (
              <span className="text-gold-600 dark:text-gold-400 font-medium">
                {' '}· {resolvedBreakPoints.length} CUSUM regime shift(s) mapped
              </span>
            )}
          </p>
        </div>

        {/* Feature Toggles & Latest Price */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {resolvedBreakPoints.length > 0 && (
            <button
              type="button"
              onClick={() => setShowBreaks((prev) => !prev)}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                showBreaks
                  ? 'bg-gold-500/15 text-gold-700 dark:text-gold-300 border border-gold-500/40'
                  : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 dark:text-zinc-400 border border-transparent hover:text-zinc-800 dark:hover:text-zinc-200'
              }`}
              title="Toggle volatility regime shift break lines on chart"
            >
              <span
                className={`w-2 h-2 rounded-full ${showBreaks ? 'bg-gold-500' : 'bg-zinc-400'}`}
              />
              <span>Break Points ({resolvedBreakPoints.length})</span>
            </button>
          )}

          {validBars.length >= 10 && (
            <button
              type="button"
              onClick={() => setShowVolumeProfile((prev) => !prev)}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                showVolumeProfile
                  ? 'bg-gold-500/15 text-gold-700 dark:text-gold-300 border border-gold-500/40'
                  : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 dark:text-zinc-400 border border-transparent hover:text-zinc-800 dark:hover:text-zinc-200'
              }`}
              title="Toggle the Volume-at-Price profile and liquidity levels"
            >
              <BarChart3 className="w-3.5 h-3.5" />
              <span>Volume Profile</span>
            </button>
          )}

          {cusumSqPoints.length > 0 && (
            <button
              type="button"
              onClick={() => setShowCusumCurve((prev) => !prev)}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                showCusumCurve
                  ? 'bg-gold-500/15 text-gold-700 dark:text-gold-300 border border-gold-500/40'
                  : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 dark:text-zinc-400 border border-transparent hover:text-zinc-800 dark:hover:text-zinc-200'
              }`}
              title="Toggle CUSUM of squares diagnostic curve"
            >
              <Activity className="w-3.5 h-3.5" />
              <span>CUSUM S_k Band</span>
            </button>
          )}

          {lastBar && (
            <div className="flex items-baseline gap-2.5 text-right pl-1">
              <div>
                <span className="text-xs text-zinc-500 dark:text-zinc-400 mr-1.5">
                  {lastBar.date}:
                </span>
                <span className="font-mono text-sm sm:text-base font-bold text-zinc-900 dark:text-zinc-100 tabular-nums">
                  {valueFormatter(lastBar.close)}
                </span>
              </div>
              {showVolume && (
                <div className="hidden sm:block text-xs text-zinc-500 dark:text-zinc-400">
                  Vol:{' '}
                  <span className="font-mono tabular-nums">
                    {(lastBar.volume / 1e6).toFixed(2)}M
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Primary Interactive Plotly Price & Volume Chart */}
      <PlotlyChart
        data={priceTraces}
        layout={priceLayout}
        plotType="2d"
        size="primary"
        title={title}
        subtitle={subtitle}
        rangeSlider
        hovermode="x unified"
        exportFilename="quantmetrics-price-history"
        uirevisionKey={`ts-${validBars[0]?.date}-${validBars.length}`}
        showWheelHint
      />

      {volumeProfile && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 pt-2 border-t border-[var(--hairline)] text-[10px] font-mono text-zinc-500 dark:text-zinc-400">
          <span className="text-gold-700 dark:text-gold-400 font-semibold">
            VOLUME AT PRICE · OHLC-DERIVED ESTIMATE
          </span>
          {volumeProfile.poc && <span>POC {valueFormatter(volumeProfile.poc.mid)}</span>}
          {volumeProfile.valueArea && (
            <span>
              70% VA {valueFormatter(volumeProfile.valueArea.low)} –{' '}
              {valueFormatter(volumeProfile.valueArea.high)}
            </span>
          )}
          <span>
            {volumeProfile.bins.length} bins · {volumeProfile.barCount} bars
          </span>
          {barsSource && barsSource !== 'live_market' && (
            <span className="text-amber-600 dark:text-amber-400">synthetic volume — illustrative</span>
          )}
        </div>
      )}

      {/* Optional Interactive CUSUM S_k Trajectory Sub-Chart */}
      {showCusumCurve && cusumSqPoints.length > 0 && (
        <div className="pt-3 border-t border-[var(--hairline)] space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-gold-500" />
              <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                Recursive CUSUM of Squares Trajectory (S_k)
              </span>
              <span className="text-[10px] text-zinc-500 dark:text-zinc-400">
                Brown–Durbin–Evans / Inclán–Tiao 95% Confidence Bounds
              </span>
            </div>
          </div>

          <PlotlyChart
            data={cusumTraces}
            layout={cusumLayout}
            plotType="2d"
            size="compact"
            title="Recursive CUSUM of Squares (S_k)"
            subtitle="Brown–Durbin–Evans / Inclán–Tiao 95% Confidence Bounds"
            hovermode="x unified"
            showLegend
            exportFilename="quantmetrics-cusum-sq"
            uirevisionKey={`cusum-${cusumSqPoints[0]?.date}-${cusumSqPoints.length}`}
          />
        </div>
      )}
    </div>
  );
}
