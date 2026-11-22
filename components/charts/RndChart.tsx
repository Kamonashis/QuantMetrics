'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { QUANT_COLORS } from '@/lib/plotly-theme';

/** Structural subset of an RND payload — SVI-fitted (`RndOut`) or market-price (`MarketRnd`). */
export interface DensityCurve {
  strikes: number[];
  density: number[];
}

/**
 * Interactive Plotly Risk-Neutral Density (Breeden–Litzenberger) vs realized 21d log-return histogram.
 */
export function RndChart({
  rnd,
  fittedRnd,
  spot,
  realizedReturns,
  primaryLabel = 'RND (Breeden–Litzenberger)',
}: {
  rnd: DensityCurve;
  /** optional second curve, drawn dashed — e.g. the SVI-fitted density for comparison */
  fittedRnd?: DensityCurve | null;
  spot: number;
  realizedReturns: number[];
  primaryLabel?: string;
}) {
  const isDark = usePlotlyDarkTheme();

  const { traces, layout } = useMemo(() => {
    const kbd = rnd.strikes.map((K, i) => ({
      kPct: Math.log(K / spot) * 100,
      k: Math.log(K / spot),
      q: rnd.density[i] * K,
    }));
    const rndArea = kbd.reduce(
      (s, p, i) => (i === 0 ? s : s + ((kbd[i - 1].q + p.q) / 2) * (p.k - kbd[i - 1].k)),
      0
    );

    const fittedKbd =
      fittedRnd && fittedRnd.strikes.length >= 2
        ? fittedRnd.strikes.map((K, i) => ({
            kPct: Math.log(K / spot) * 100,
            k: Math.log(K / spot),
            q: fittedRnd.density[i] * K,
          }))
        : null;
    const fittedArea =
      fittedKbd?.reduce(
        (s, p, i) =>
          i === 0 ? s : s + ((fittedKbd[i - 1].q + p.q) / 2) * (p.k - fittedKbd[i - 1].k),
        0
      ) ?? 0;

    const list: Data[] = [];

    if (realizedReturns.length >= 20) {
      const sorted = [...realizedReturns].sort((a, b) => a - b);
      const lo = sorted[Math.floor(sorted.length * 0.01)];
      const hi = sorted[Math.floor(sorted.length * 0.99)];
      const nBins = 41;
      const binWidth = Math.max(1e-5, (hi - lo) / nBins);
      const bins = Array.from({ length: nBins }, (_, i) => ({
        centerPct: (lo + (i + 0.5) * binWidth) * 100,
        count: 0,
      }));
      for (const r of realizedReturns) {
        const idx = Math.min(nBins - 1, Math.max(0, Math.floor((r - lo) / binWidth)));
        bins[idx].count++;
      }
      list.push({
        type: 'bar',
        name: 'Realized 21d Returns',
        x: bins.map((b) => b.centerPct),
        y: bins.map((b) => b.count / (Math.max(1, realizedReturns.length) * binWidth)),
        customdata: bins.map((b) => b.count),
        marker: {
          color: isDark ? 'rgba(163, 163, 173, 0.30)' : 'rgba(113, 113, 122, 0.25)',
          line: { color: isDark ? 'rgba(163,163,173,0.4)' : 'rgba(113,113,122,0.35)', width: 0.5 },
        },
        hovertemplate:
          'Return %{x:+.1f}%<br>Density: %{y:.2f} (%{customdata} obs)<extra></extra>',
      });
    }

    list.push({
      type: 'scatter',
      mode: 'lines',
      name: primaryLabel,
      x: kbd.map((p) => p.kPct),
      y: kbd.map((p) => p.q / Math.max(rndArea, 1e-9)),
      line: { color: QUANT_COLORS.goldLight, width: 2.2 },
      fill: 'tozeroy',
      fillcolor: 'rgba(222, 192, 132, 0.14)',
      hovertemplate: `${primaryLabel}<br>Return %{x:+.1f}% · Density %{y:.2f}<extra></extra>`,
    });

    if (fittedKbd && fittedArea > 0) {
      list.push({
        type: 'scatter',
        mode: 'lines',
        name: 'RND (SVI Fit)',
        x: fittedKbd.map((p) => p.kPct),
        y: fittedKbd.map((p) => p.q / Math.max(fittedArea, 1e-9)),
        line: {
          color: isDark ? 'rgba(228,228,231,0.78)' : 'rgba(39,39,42,0.72)',
          width: 1.5,
          dash: 'dash',
        },
        hovertemplate: 'SVI Fit<br>Return %{x:+.1f}% · Density %{y:.2f}<extra></extra>',
      });
    }

    const chartLayout: Partial<Layout> = {
      bargap: 0.06,
      shapes: [
        {
          type: 'line',
          xref: 'x',
          yref: 'paper',
          x0: 0,
          x1: 0,
          y0: 0,
          y1: 1,
          line: { color: 'rgba(148,148,160,0.4)', width: 1, dash: 'dot' },
        },
      ],
    };

    return { traces: list, layout: chartLayout };
  }, [rnd, fittedRnd, spot, realizedReturns, primaryLabel, isDark]);

  return (
    <PlotlyChart
      data={traces}
      layout={layout}
      plotType="2d"
      hovermode="x unified"
      showLegend
      xAxisTitle="Log-Return k = ln(K / S₀)"
      xTickSuffix="%"
      yAxisTitle="Probability Density"
      height={285}
      uirevisionKey={`rnd-${spot.toFixed(1)}-${primaryLabel}`}
      exportFilename="risk-neutral-density"
      showWheelHint
    />
  );
}
