'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import type { SliceOut } from '@/components/views/volatility/types';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { EmptyState } from '@/components/ui/empty-state';

const GOLD_HEAT_SCALE: [number, string][] = [
  [0.0, 'rgba(24, 36, 54, 0.85)'],
  [0.35, 'rgba(63, 178, 127, 0.78)'],
  [0.7, 'rgba(201, 164, 92, 0.90)'],
  [1.0, 'rgba(224, 101, 95, 0.96)'],
];

/**
 * Interactive Plotly Implied-Vol Surface Heatmap: expiries (rows) × log-moneyness k buckets (cols).
 * Gaps remain null/transparent — the surface is only as real as the observed quotes.
 */
export function VolSurfaceHeatmap({ slices }: { slices: SliceOut[] }) {
  const isDark = usePlotlyDarkTheme();

  const heatModel = useMemo(() => {
    if (!slices || slices.length === 0) return null;

    const buckets: { k0: number; label: string }[] = [];
    for (let k0 = -0.45; k0 < 0.45; k0 += 0.1) {
      buckets.push({ k0: Number(k0.toFixed(2)), label: `${k0 >= 0 ? '+' : ''}${k0.toFixed(1)}` });
    }

    const sortedSlices = [...slices].sort((a, b) => a.dte - b.dte);
    const yLabels = sortedSlices.map((s) => `${s.dte}d (${s.expirationDate.slice(5)})`);
    const xLabels = buckets.map((b) => b.label);

    const z: (number | null)[][] = [];
    const text: string[][] = [];
    const customdata: string[][] = [];

    for (const slice of sortedSlices) {
      const zRow: (number | null)[] = [];
      const tRow: string[] = [];
      const cRow: string[] = [];
      for (const b of buckets) {
        const inBucket = slice.points.filter((p) => p.k >= b.k0 && p.k < b.k0 + 0.1);
        if (inBucket.length === 0) {
          zRow.push(null);
          tRow.push('');
          cRow.push(`k ∈ [${b.label}, ${(b.k0 + 0.1).toFixed(1)})<br>No usable quotes`);
        } else {
          const avgIvPct = (inBucket.reduce((acc, p) => acc + p.iv, 0) / inBucket.length) * 100;
          zRow.push(Number(avgIvPct.toFixed(2)));
          tRow.push(`${avgIvPct.toFixed(0)}%`);
          cRow.push(
            `k ∈ [${b.label}, ${(b.k0 + 0.1).toFixed(1)})<br>Mean IV: <b>${avgIvPct.toFixed(1)}%</b> · ${inBucket.length} quote${inBucket.length > 1 ? 's' : ''}`
          );
        }
      }
      z.push(zRow);
      text.push(tRow);
      customdata.push(cRow);
    }

    return { xLabels, yLabels, z, text, customdata, rowCount: sortedSlices.length };
  }, [slices]);

  const data = useMemo<Data[]>(() => {
    if (!heatModel) return [];
    return [
      {
        type: 'heatmap',
        x: heatModel.xLabels,
        y: heatModel.yLabels,
        z: heatModel.z,
        text: heatModel.text as unknown as string[],
        texttemplate: '%{text}',
        textfont: {
          size: 10,
          color: isDark ? '#F4F4F7' : '#18181B',
        },
        customdata: heatModel.customdata as unknown as string[],
        colorscale: GOLD_HEAT_SCALE,
        hoverongaps: false,
        xgap: 2,
        ygap: 2,
        colorbar: {
          title: { text: 'IV %', font: { size: 10 } },
          ticksuffix: '%',
          thickness: 12,
          outlinewidth: 0,
        },
        hovertemplate: '<b>%{y}</b><br>%{customdata}<extra></extra>',
      } as Data,
    ];
  }, [heatModel, isDark]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      margin: { l: 88, r: 24, t: 14, b: 42 },
      xaxis: {
        title: { text: 'Log-Moneyness Bucket k = ln(K/F)' },
        type: 'category',
        side: 'bottom',
      },
      yaxis: {
        title: { text: 'Expiry' },
        type: 'category',
        autorange: 'reversed',
      },
    }),
    []
  );

  if (!heatModel) {
    return (
      <EmptyState
        title="No options slices available"
        detail="Heatmap populates once usable option quotes load"
        className="h-56"
      />
    );
  }

  const dynamicHeight = Math.max(260, Math.min(520, heatModel.rowCount * 36 + 96));

  return (
    <div className="panel p-3">
      <PlotlyChart
        data={data}
        layout={layout}
        plotType="heatmap"
        hovermode="closest"
        enableWheelPan={false}
        height={dynamicHeight}
        exportFilename="vol-surface-heatmap"
      />
    </div>
  );
}
