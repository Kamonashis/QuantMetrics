'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import type {
  GexExpiryBucketId,
  GexExpiryBucketOut,
  GexStrikeExpiryMatrixOut,
} from '@/components/views/volatility/types';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { EmptyState } from '@/components/ui/empty-state';
import { SegmentedControl } from '@/components/ui/lab-ui';
import { getPlotlyThemeTokens } from '@/lib/plotly-theme';
import { cn } from '@/lib/utils';

export type GexBucketFilter = 'all' | GexExpiryBucketId;

export interface GexExpiryHeatmapProps {
  expiryBuckets?: GexExpiryBucketOut[];
  matrix?: GexStrikeExpiryMatrixOut;
  spot: number;
  convention: 'squeeze' | 'inverse';
  serverConvention?: 'squeeze' | 'inverse';
  include0Dte: boolean;
  selectedBucket: GexBucketFilter;
  onSelectBucket: (bucket: GexBucketFilter) => void;
}

const fmtSignedUsd = (v: number | null | undefined): string => {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const sign = v > 0 ? '+' : v < 0 ? '-' : '';
  const abs = Math.abs(v);
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(0)}K`;
  return `${sign}$${abs.toFixed(0)}`;
};

const BUCKET_OPTIONS: { value: GexBucketFilter; label: string }[] = [
  { value: 'all', label: 'All ≤45d' },
  { value: '0dte', label: '0-DTE' },
  { value: 'weekly', label: 'Weekly' },
  { value: 'monthly', label: 'Monthly' },
  { value: 'far', label: 'Far' },
];

/**
 * Expiry-bucketed GEX summary cards + interactive bucket filter + 2D Plotly
 * Strike × Expiry Net GEX Heatmap (signed diverging gold/emerald vs rose/zinc).
 */
export function GexExpiryHeatmap({
  expiryBuckets,
  matrix,
  spot,
  convention,
  serverConvention = 'squeeze',
  include0Dte,
  selectedBucket,
  onSelectBucket,
}: GexExpiryHeatmapProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const convMult = convention === serverConvention ? 1 : -1;

  const buckets = useMemo(() => {
    if (!expiryBuckets || expiryBuckets.length === 0) return [];
    return expiryBuckets.map((b) => ({
      ...b,
      callGex: b.callGex * convMult,
      putGex: b.putGex * convMult,
      totalGex: b.totalGex * convMult,
      totalVolumeGex: b.totalVolumeGex * convMult,
      totalCharm: b.totalCharm * convMult,
      totalVanna: b.totalVanna * convMult,
      excluded: !include0Dte && b.bucket === '0dte',
    }));
  }, [expiryBuckets, convMult, include0Dte]);

  const heatModel = useMemo(() => {
    if (!matrix || matrix.strikes.length === 0 || matrix.expiries.length === 0) return null;

    const expIndices: number[] = [];
    for (let j = 0; j < matrix.expiries.length; j++) {
      const exp = matrix.expiries[j];
      if (!include0Dte && (exp.dte <= 0 || exp.bucket === '0dte')) continue;
      if (selectedBucket !== 'all' && exp.bucket !== selectedBucket) continue;
      expIndices.push(j);
    }
    if (expIndices.length === 0) return null;

    const activeExpiries = expIndices.map((j) => matrix.expiries[j]);
    const xLabels = activeExpiries.map((e) => `${e.dte}d (${e.expiry.slice(5)})`);
    const yLabels = matrix.strikes.map((k) => `$${k}`);

    let maxAbsRaw = 0;
    for (let i = 0; i < matrix.strikes.length; i++) {
      for (const j of expIndices) {
        const v = Math.abs((matrix.netGex[i]?.[j] ?? 0) * convMult);
        if (v > maxAbsRaw) maxAbsRaw = v;
      }
    }

    const useK = maxAbsRaw > 0 && maxAbsRaw < 1e6;
    const div = useK ? 1e3 : 1e6;
    const unit = useK ? 'K' : 'M';
    const maxAbsScaled = Math.max(maxAbsRaw / div, 0.01);

    const z: (number | null)[][] = [];
    const text: string[][] = [];
    const customdata: string[][] = [];

    for (let i = 0; i < matrix.strikes.length; i++) {
      const strike = matrix.strikes[i];
      const zRow: (number | null)[] = [];
      const tRow: string[] = [];
      const cRow: string[] = [];
      for (let idx = 0; idx < expIndices.length; idx++) {
        const j = expIndices[idx];
        const exp = matrix.expiries[j];
        const net = (matrix.netGex[i]?.[j] ?? 0) * convMult;
        const call = (matrix.callGex[i]?.[j] ?? 0) * convMult;
        const put = (matrix.putGex[i]?.[j] ?? 0) * convMult;
        const scaled = net / div;
        if (Math.abs(net) < 1e-6 && Math.abs(call) < 1e-6 && Math.abs(put) < 1e-6) {
          zRow.push(0);
          tRow.push('');
        } else {
          zRow.push(Number(scaled.toFixed(3)));
          tRow.push(
            Math.abs(net) >= maxAbsRaw * 0.06
              ? `${scaled >= 0 ? '+' : ''}${Math.abs(scaled) >= 10 ? scaled.toFixed(0) : scaled.toFixed(1)}`
              : ''
          );
        }
        cRow.push(
          `Strike $${strike} · ${exp.expiry} (${exp.dte}d, ${exp.bucket})<br>` +
            `Net GEX: <b>${fmtSignedUsd(net)}</b><br>` +
            `Call: ${fmtSignedUsd(call)} · Put: ${fmtSignedUsd(put)}`
        );
      }
      z.push(zRow);
      text.push(tRow);
      customdata.push(cRow);
    }

    // Closest strike to spot for visual caption
    const nearestStrike = matrix.strikes.reduce((best, s) =>
      Math.abs(s - spot) < Math.abs(best - spot) ? s : best
    );

    return {
      xLabels,
      yLabels,
      z,
      text,
      customdata,
      unit,
      maxAbsScaled,
      rowCount: matrix.strikes.length,
      colCount: activeExpiries.length,
      nearestStrike,
    };
  }, [matrix, include0Dte, selectedBucket, convMult, spot]);

  const { traces, layout } = useMemo(() => {
    if (!heatModel) return { traces: [] as Data[], layout: {} as Partial<Layout> };
    const neutral = isDark ? '#18181E' : '#F2F1EE';
    const colorscale: [number, string][] = [
      [0.0, '#CF4B45'],
      [0.25, 'rgba(224, 101, 95, 0.50)'],
      [0.5, neutral],
      [0.75, 'rgba(63, 178, 127, 0.62)'],
      [1.0, '#DEC084'],
    ];

    const list: Data[] = [
      {
        type: 'heatmap',
        x: heatModel.xLabels,
        y: heatModel.yLabels,
        z: heatModel.z,
        zmin: -heatModel.maxAbsScaled,
        zmax: heatModel.maxAbsScaled,
        zmid: 0,
        text: heatModel.text as unknown as string[],
        texttemplate: '%{text}',
        textfont: {
          size: 9.5,
          color: isDark ? '#F4F4F7' : '#18181B',
        },
        customdata: heatModel.customdata as unknown as string[],
        colorscale,
        xgap: 1.5,
        ygap: 1.5,
        colorbar: {
          title: { text: `$${heatModel.unit}/1%`, font: { size: 10, color: tokens.textMuted } },
          ticksuffix: heatModel.unit,
          thickness: 12,
          outlinewidth: 0,
          tickfont: { size: 9.5, color: tokens.textMuted },
        },
        hovertemplate: '%{customdata}<extra></extra>',
      } as Data,
    ];

    const spotLabel = `$${heatModel.nearestStrike}`;
    const shapes: NonNullable<Layout['shapes']> = [
      {
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: spotLabel,
        y1: spotLabel,
        line: { color: 'rgba(222, 192, 132, 0.65)', width: 1.1, dash: 'dot' },
      },
    ];

    return {
      traces: list,
      layout: {
        shapes,
        margin: { l: 64, r: 24, t: 14, b: 56 },
        xaxis: {
          title: { text: 'Expiration (DTE)', font: { size: 10.5, color: tokens.textMuted } },
          type: 'category',
          tickangle: -25,
          tickfont: { size: 10, color: tokens.textMuted },
        },
        yaxis: {
          title: { text: 'Strike', font: { size: 10.5, color: tokens.textMuted } },
          type: 'category',
          tickfont: { size: 10, color: tokens.textMuted },
        },
      } as Partial<Layout>,
    };
  }, [heatModel, isDark, tokens]);

  const heatmapHeight = heatModel
    ? Math.max(300, Math.min(540, heatModel.rowCount * 18 + 90))
    : 320;

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 border-b border-[var(--hairline)] pb-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="kicker mb-1">Term &amp; Tenor Decomposition</p>
          <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
            Expiry-Bucketed GEX &amp; Strike × Expiry Matrix
          </h3>
        </div>
        <SegmentedControl
          options={BUCKET_OPTIONS}
          value={selectedBucket}
          onChange={onSelectBucket}
          ariaLabel="Filter gamma profile by expiry bucket"
        />
      </div>

      {/* Expiry bucket summary cards */}
      {buckets.length > 0 && (
        <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 xl:grid-cols-4">
          {buckets.map((b) => {
            const active = selectedBucket === b.bucket;
            return (
              <button
                key={b.bucket}
                type="button"
                onClick={() => onSelectBucket(active ? 'all' : b.bucket)}
                className={cn(
                  'panel-inset cursor-pointer p-3 text-left transition-all',
                  active
                    ? 'border-gold-500/60 ring-1 ring-gold-500/35'
                    : 'hover:border-[var(--hairline-strong)]',
                  b.excluded && 'opacity-50'
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-[10px] font-bold uppercase tracking-wider text-zinc-800 dark:text-zinc-100">
                    {b.label}
                  </span>
                  <span className="rounded border border-[var(--hairline)] px-1.5 py-0.5 font-mono text-[9px] text-zinc-500 dark:text-zinc-400">
                    {b.dteRange}
                  </span>
                </div>

                <div className="mt-1.5 flex items-baseline justify-between gap-2">
                  <span
                    className={cn(
                      'font-mono text-lg font-bold tabular-nums',
                      b.totalGex >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'
                    )}
                  >
                    {fmtSignedUsd(b.totalGex)}
                  </span>
                  <span className="font-mono text-[10px] tabular-nums text-zinc-500 dark:text-zinc-400">
                    {(b.shareOfAbsGex * 100).toFixed(0)}% gross
                  </span>
                </div>

                <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-white/[0.06]">
                  <div
                    className={cn(
                      'h-full rounded-full',
                      b.totalGex >= 0 ? 'bg-gold-400/80' : 'bg-rose-400/80'
                    )}
                    style={{ width: `${Math.min(100, Math.max(0, b.shareOfAbsGex * 100))}%` }}
                  />
                </div>

                <div className="mt-2 flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5 font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
                  <span>C {fmtSignedUsd(b.callGex)} · P {fmtSignedUsd(b.putGex)}</span>
                  <span>{b.oi.toLocaleString()} OI</span>
                </div>
                {b.excluded && (
                  <div className="mt-1 font-mono text-[8.5px] uppercase tracking-wider text-amber-600 dark:text-amber-400">
                    Excluded by 0-DTE toggle
                  </div>
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* Strike × Expiry 2D Heatmap */}
      {heatModel ? (
        <ChartFullscreen
          title="Strike × Expiry Net GEX Matrix"
          subtitle={`${convention} convention · ${heatModel.colCount} expiries × ${heatModel.rowCount} top strikes`}
        >
          <div>
            <PlotlyChart
              data={traces}
              layout={layout}
              plotType="heatmap"
              hovermode="closest"
              enableWheelPan={false}
              height={heatmapHeight}
              uirevisionKey={`gex-hm-${convention}-${include0Dte ? 1 : 0}-${selectedBucket}-${heatModel.colCount}`}
              exportFilename="gex-strike-expiry-heatmap"
            />
            <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
              <span>
                Gold / emerald = positive net dealer gamma · Rose = negative net dealer gamma · Dotted line = strike nearest spot (${heatModel.nearestStrike})
              </span>
              <span>
                {heatModel.colCount} expiries · {heatModel.rowCount} highest gross-GEX strikes (≤ 90 DTE)
              </span>
            </div>
          </div>
        </ChartFullscreen>
      ) : (
        <EmptyState
          title="No expiries match the active filter"
          detail={
            !include0Dte && selectedBucket === '0dte'
              ? 'Re-enable Incl. 0-DTE or select Weekly / Monthly / Far'
              : 'Requires strike × expiry GEX rows in this bucket'
          }
          className="h-56"
        />
      )}
    </div>
  );
}
