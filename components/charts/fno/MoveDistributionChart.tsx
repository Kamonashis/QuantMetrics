'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { EmptyState } from '@/components/ui/empty-state';
import { PlotlyChart } from '@/components/charts/PlotlyChart';
import { QUANT_COLORS } from '@/lib/plotly-theme';

/**
 * Interactive Plotly distribution of historical horizon |moves| with implied
 * straddle move and optional TCN-Attention move forecast rules.
 */
export function MoveDistributionChart({
  samples,
  impliedPct,
  tcnMovePct = null,
  horizonLabel,
}: {
  samples: number[];
  impliedPct: number | null;
  tcnMovePct?: number | null;
  horizonLabel: string;
}) {
  const { traces, layout, hasData } = useMemo(() => {
    if (samples.length < 20) {
      return { traces: [] as Data[], layout: {} as Partial<Layout>, hasData: false };
    }

    const sorted = [...samples].sort((a, b) => a - b);
    const p25 = sorted[Math.floor(sorted.length * 0.25)] * 100;
    const p50 = sorted[Math.floor(sorted.length * 0.5)] * 100;
    const p75 = sorted[Math.floor(sorted.length * 0.75)] * 100;
    const maxPct = sorted[sorted.length - 1] * 100;
    const capPct = Math.min(maxPct, p75 * 2.5);
    const nBins = 30;
    const binWidth = Math.max(1e-4, capPct / nBins);
    const bins = Array.from({ length: nBins }, (_, i) => ({
      x0: i * binWidth,
      x1: (i + 1) * binWidth,
      mid: (i + 0.5) * binWidth,
      count: 0,
    }));
    for (const s of sorted) {
      const pct = s * 100;
      if (pct >= capPct) continue;
      const idx = Math.min(nBins - 1, Math.floor(pct / binWidth));
      bins[idx].count++;
    }

    const list: Data[] = [
      {
        type: 'bar',
        name: 'Realized |Move|',
        x: bins.map((b) => b.mid),
        y: bins.map((b) => b.count),
        customdata: bins.map((b) => [b.x0.toFixed(1), b.x1.toFixed(1)]),
        marker: {
          color: 'rgba(222, 192, 132, 0.42)',
          line: { color: 'rgba(222, 192, 132, 0.7)', width: 0.7 },
        },
        hovertemplate:
          '<b>|Move| %{customdata[0]}% – %{customdata[1]}%</b><br>%{y} windows<extra></extra>',
      },
    ];

    const shapes: NonNullable<Layout['shapes']> = [];
    const annotations: NonNullable<Layout['annotations']> = [];

    [
      { val: p25, label: 'P25' },
      { val: p50, label: 'P50' },
      { val: p75, label: 'P75' },
    ].forEach((q) => {
      shapes.push({
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: q.val,
        x1: q.val,
        y0: 0,
        y1: 1,
        line: { color: 'rgba(148,148,160,0.28)', width: 1, dash: 'dot' },
      });
      annotations.push({
        x: q.val,
        y: 0.99,
        xref: 'x',
        yref: 'paper',
        text: q.label,
        showarrow: false,
        yanchor: 'top',
        font: { size: 9, color: QUANT_COLORS.slate },
      });
    });

    if (impliedPct !== null && impliedPct > 0) {
      const xPos = Math.min(impliedPct, capPct);
      shapes.push({
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: xPos,
        x1: xPos,
        y0: 0,
        y1: 1,
        line: { color: QUANT_COLORS.goldLight, width: 2 },
      });
      annotations.push({
        x: xPos,
        y: 0.88,
        xref: 'x',
        yref: 'paper',
        text: `Implied ${impliedPct.toFixed(1)}%`,
        showarrow: false,
        xanchor: 'left',
        xshift: 4,
        font: { size: 9.5, color: QUANT_COLORS.goldLight },
      });
    }

    if (tcnMovePct !== null && tcnMovePct > 0) {
      const xPos = Math.min(tcnMovePct, capPct);
      shapes.push({
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: xPos,
        x1: xPos,
        y0: 0,
        y1: 1,
        line: { color: QUANT_COLORS.positive, width: 2, dash: 'dash' },
      });
      annotations.push({
        x: xPos,
        y: 0.76,
        xref: 'x',
        yref: 'paper',
        text: `TCN ${tcnMovePct.toFixed(1)}%`,
        showarrow: false,
        xanchor: 'left',
        xshift: 4,
        font: { size: 9.5, color: QUANT_COLORS.positive },
      });
    }

    return {
      traces: list,
      layout: { bargap: 0.08, shapes, annotations } as Partial<Layout>,
      hasData: true,
    };
  }, [samples, impliedPct, tcnMovePct]);

  if (!hasData) {
    return (
      <EmptyState
        title="Not enough history in this sample"
        detail="Need at least 20 realized-move windows"
        className="h-40"
      />
    );
  }

  return (
    <div>
      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="x unified"
        xAxisTitle="Absolute Horizon Move (%)"
        xTickSuffix="%"
        yAxisTitle="Sessions"
        height={240}
        uirevisionKey={`move-dist-${horizonLabel}-${samples.length}`}
        exportFilename="move-distribution"
        showWheelHint
      />
      <p className="mt-1 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
        {horizonLabel} · n = {samples.length.toLocaleString()} windows · capped at P75×2.5 ·
        implied = 0.8×ATM straddle
        {tcnMovePct !== null ? ' · TCN = 0.8×σ_TCN(T)√(T/252)' : ''}
      </p>
    </div>
  );
}
