'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import type { MarketRnd } from '@/lib/rnd';
import { PlotlyChart } from '@/components/charts/PlotlyChart';
import { QUANT_COLORS } from '@/lib/plotly-theme';

/**
 * Interactive Plotly cumulative distribution function (CDF) of the market-price RND.
 */
export function RndCdfChart({ rnd }: { rnd: MarketRnd }) {
  const { traces, layout, tail } = useMemo(() => {
    const pts = rnd.strikes.map((K, i) => ({
      k: Math.log(K / rnd.spot),
      kPct: Math.log(K / rnd.spot) * 100,
      strike: K,
      p: rnd.cdf[i],
      pPct: rnd.cdf[i] * 100,
    }));

    const cdfAt = (K: number): number => {
      if (pts.length === 0) return 0;
      const k = Math.log(K / rnd.spot);
      if (k <= pts[0].k) return pts[0].p;
      if (k >= pts[pts.length - 1].k) return pts[pts.length - 1].p;
      for (let i = 1; i < pts.length; i++) {
        if (k <= pts[i].k) {
          const f = (k - pts[i - 1].k) / Math.max(1e-12, pts[i].k - pts[i - 1].k);
          return pts[i - 1].p + f * (pts[i].p - pts[i - 1].p);
        }
      }
      return 1;
    };

    const list: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Market-Price CDF',
        x: pts.map((p) => p.kPct),
        y: pts.map((p) => p.pPct),
        customdata: pts.map((p) => p.strike),
        line: { color: QUANT_COLORS.goldLight, width: 2 },
        fill: 'tozeroy',
        fillcolor: 'rgba(222, 192, 132, 0.12)',
        hovertemplate:
          '<b>Return %{x:+.1f}% (K $%{customdata:.2f})</b><br>P(S_T ≤ K): %{y:.1f}%<extra></extra>',
      },
    ];

    const thresholds = [
      { kPct: Math.log(0.8) * 100, label: '−20%' },
      { kPct: Math.log(0.9) * 100, label: '−10%' },
      { kPct: 0, label: 'Spot' },
      { kPct: Math.log(1.1) * 100, label: '+10%' },
      { kPct: Math.log(1.2) * 100, label: '+20%' },
    ];

    const shapes: NonNullable<Layout['shapes']> = [
      {
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: 50,
        y1: 50,
        line: { color: 'rgba(148,148,160,0.35)', width: 1, dash: 'dot' },
      },
      ...thresholds.map((th) => ({
        type: 'line' as const,
        xref: 'x' as const,
        yref: 'paper' as const,
        x0: th.kPct,
        x1: th.kPct,
        y0: 0,
        y1: 1,
        line: {
          color: th.kPct === 0 ? 'rgba(222,192,132,0.45)' : 'rgba(148,148,160,0.2)',
          width: 1,
          dash: 'dot' as const,
        },
      })),
    ];

    const annotations: NonNullable<Layout['annotations']> = thresholds.map((th) => ({
      x: th.kPct,
      y: 0.99,
      xref: 'x',
      yref: 'paper',
      text: th.label,
      showarrow: false,
      font: { size: 9, color: th.kPct === 0 ? QUANT_COLORS.goldLight : QUANT_COLORS.slate },
      yanchor: 'top',
    }));

    return {
      traces: list,
      layout: {
        shapes,
        annotations,
        yaxis: { range: [0, 102], ticksuffix: '%' },
        margin: { l: 48, r: 14, t: 14, b: 36 },
      } as Partial<Layout>,
      tail: {
        deep: cdfAt(rnd.spot * 0.8),
        ten: cdfAt(rnd.spot * 0.9),
        spot: cdfAt(rnd.spot),
        up10: cdfAt(rnd.spot * 1.1),
        up20: cdfAt(rnd.spot * 1.2),
      },
    };
  }, [rnd]);

  return (
    <div>
      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="x unified"
        xAxisTitle="Log-Return k = ln(K / S₀)"
        xTickSuffix="%"
        yAxisTitle="Cumulative Prob"
        height={210}
        uirevisionKey={`rnd-cdf-${rnd.spot.toFixed(1)}`}
        exportFilename="risk-neutral-cdf"
        showWheelHint
      />
      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
        <span>CDF of market-price RND</span>
        <span className="tabular-nums">P(&lt;0.8S) {(tail.deep * 100).toFixed(1)}%</span>
        <span className="tabular-nums">P(&lt;0.9S) {(tail.ten * 100).toFixed(1)}%</span>
        <span className="tabular-nums">P(&lt;S) {(tail.spot * 100).toFixed(1)}%</span>
        <span className="tabular-nums">P(&lt;1.1S) {(tail.up10 * 100).toFixed(1)}%</span>
        <span className="tabular-nums">P(&lt;1.2S) {(tail.up20 * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}
