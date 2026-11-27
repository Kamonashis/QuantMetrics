'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import type { GexStrikeOut } from '@/components/views/volatility/types';
import { PlotlyChart } from '@/components/charts/PlotlyChart';
import { QUANT_COLORS } from '@/lib/plotly-theme';
import { EmptyState } from '@/components/ui/empty-state';
import { SegmentedControl } from '@/components/ui/lab-ui';

export type GammaProfileMetric = 'gex' | 'volumeGex' | 'charm' | 'vanna';

export type GexStrikePoint = Pick<GexStrikeOut, 'strike' | 'callOi' | 'putOi' | 'gex'> &
  Partial<GexStrikeOut>;

export interface GammaProfileChartProps {
  points: GexStrikePoint[];
  spot: number;
  flip: number | null;
  maxBars?: number;
  convention?: 'squeeze' | 'inverse';
  serverConvention?: 'squeeze' | 'inverse';
  metric?: GammaProfileMetric;
  onMetricChange?: (metric: GammaProfileMetric) => void;
  showMetricSelector?: boolean;
  bucketLabel?: string;
  callWall?: number | null;
  putWall?: number | null;
  gammaCallWall?: number | null;
  gammaPutWall?: number | null;
}

const METRIC_OPTIONS: { value: GammaProfileMetric; label: string }[] = [
  { value: 'gex', label: 'OI GEX' },
  { value: 'volumeGex', label: 'Volume GEX' },
  { value: 'charm', label: 'Charm' },
  { value: 'vanna', label: 'Vanna' },
];

const METRIC_META: Record<
  GammaProfileMetric,
  { title: string; posLabel: string; negLabel: string; unitHint: string }
> = {
  gex: {
    title: 'Dealer OI GEX',
    posLabel: 'Long Gamma',
    negLabel: 'Short Gamma',
    unitHint: 'per 1% move',
  },
  volumeGex: {
    title: 'Intraday Volume-GEX',
    posLabel: 'Positive Vol-GEX',
    negLabel: 'Negative Vol-GEX',
    unitHint: 'per 1% move',
  },
  charm: {
    title: 'Charm (1d Drift)',
    posLabel: '+Delta / day',
    negLabel: '-Delta / day',
    unitHint: 'per calendar day',
  },
  vanna: {
    title: 'Vanna (+1 vol pt)',
    posLabel: '+Delta / +1σv',
    negLabel: '-Delta / +1σv',
    unitHint: 'per +1 vol pt',
  },
};

function readMetricParts(
  p: GexStrikePoint,
  metric: GammaProfileMetric,
  convMult: number
): { net: number; call: number; put: number } {
  if (metric === 'volumeGex') {
    const hasSplit = p.callVolumeGex !== undefined && p.putVolumeGex !== undefined;
    const call = (p.callVolumeGex ?? 0) * convMult;
    const put = (p.putVolumeGex ?? 0) * convMult;
    const net = hasSplit ? call + put : (p.volumeGex ?? 0) * convMult;
    return { net, call, put };
  }
  if (metric === 'charm') {
    const hasSplit = p.callCharm !== undefined && p.putCharm !== undefined;
    const call = (p.callCharm ?? 0) * convMult;
    const put = (p.putCharm ?? 0) * convMult;
    const net = hasSplit ? call + put : (p.charm ?? 0) * convMult;
    return { net, call, put };
  }
  if (metric === 'vanna') {
    const hasSplit = p.callVanna !== undefined && p.putVanna !== undefined;
    const call = (p.callVanna ?? 0) * convMult;
    const put = (p.putVanna ?? 0) * convMult;
    const net = hasSplit ? call + put : (p.vanna ?? 0) * convMult;
    return { net, call, put };
  }
  const hasSplit = p.callGex !== undefined && p.putGex !== undefined;
  const call = (p.callGex ?? (p.gex >= 0 ? p.gex : 0)) * convMult;
  const put = (p.putGex ?? (p.gex < 0 ? p.gex : 0)) * convMult;
  const net = hasSplit ? call + put : p.gex * convMult;
  return { net, call, put };
}

/**
 * Interactive Plotly exposure by strike: diverging horizontal bars across
 * OI GEX, Volume GEX, Charm, and Vanna with convention flipping and Spot/Flip/Wall rules.
 */
export function GammaProfileChart({
  points,
  spot,
  flip,
  maxBars = 34,
  convention = 'squeeze',
  serverConvention = 'squeeze',
  metric,
  onMetricChange,
  showMetricSelector = true,
  bucketLabel,
  callWall = null,
  putWall = null,
  gammaCallWall = null,
  gammaPutWall = null,
}: GammaProfileChartProps) {
  const [internalMetric, setInternalMetric] = useState<GammaProfileMetric>('gex');
  const activeMetric = metric ?? internalMetric;
  const handleMetricChange = (next: GammaProfileMetric) => {
    if (onMetricChange) onMetricChange(next);
    else setInternalMetric(next);
  };

  const convMult = convention === serverConvention ? 1 : -1;

  const { trimmed, div, unit } = useMemo(() => {
    const sorted = [...points].sort((a, b) => a.strike - b.strike);
    const enriched = sorted.map((p) => {
      const parts = readMetricParts(p, activeMetric, convMult);
      return { p, ...parts };
    });

    let maxAbs = 0;
    for (const row of enriched) {
      const a = Math.abs(row.net);
      if (a > maxAbs) maxAbs = a;
    }

    if (maxAbs <= 1e-8) {
      return { trimmed: [] as typeof enriched, div: 1e6, unit: 'M' as const };
    }

    const threshold = maxAbs * 0.02;
    const meaningful = enriched.filter((r) => Math.abs(r.net) >= threshold);

    let sliced = meaningful;
    if (meaningful.length > maxBars) {
      // Center window around the strike nearest current spot so ATM & walls stay visible
      let nearestIdx = 0;
      let bestDist = Infinity;
      for (let i = 0; i < meaningful.length; i++) {
        const d = Math.abs(meaningful[i].p.strike - spot);
        if (d < bestDist) {
          bestDist = d;
          nearestIdx = i;
        }
      }
      const half = Math.floor(maxBars / 2);
      const start = Math.max(0, Math.min(meaningful.length - maxBars, nearestIdx - half));
      sliced = meaningful.slice(start, start + maxBars);
    }

    const useK = maxAbs < 1e6;
    return {
      trimmed: sliced,
      div: useK ? 1e3 : 1e6,
      unit: (useK ? 'K' : 'M') as 'K' | 'M',
    };
  }, [points, activeMetric, convMult, maxBars, spot]);

  const meta = METRIC_META[activeMetric];

  const { traces, layout } = useMemo(() => {
    if (trimmed.length === 0) {
      return { traces: [] as Data[], layout: {} as Partial<Layout> };
    }

    const pos = trimmed.map((r) => (r.net >= 0 ? r.net / div : 0));
    const neg = trimmed.map((r) => (r.net < 0 ? r.net / div : 0));
    const strikes = trimmed.map((r) => r.p.strike);
    const isVol = activeMetric === 'volumeGex';
    const custom = trimmed.map((r) => [
      isVol ? (r.p.callVolume ?? 0).toLocaleString() : r.p.callOi.toLocaleString(),
      isVol ? (r.p.putVolume ?? 0).toLocaleString() : r.p.putOi.toLocaleString(),
      (r.net / div).toFixed(2),
      (r.call / div).toFixed(2),
      (r.put / div).toFixed(2),
    ]);

    const countLabel = isVol ? 'Vol' : 'OI';

    const list: Data[] = [
      {
        type: 'bar',
        orientation: 'h',
        name: meta.posLabel,
        y: strikes,
        x: pos,
        customdata: custom,
        marker: { color: 'rgba(222, 192, 132, 0.82)' },
        hovertemplate:
          `<b>Strike $%{y}</b><br>Net: %{customdata[2]}${unit}<br>` +
          `Call: %{customdata[3]}${unit} · Put: %{customdata[4]}${unit}<br>` +
          `Call ${countLabel} %{customdata[0]} · Put ${countLabel} %{customdata[1]}<extra></extra>`,
      },
      {
        type: 'bar',
        orientation: 'h',
        name: meta.negLabel,
        y: strikes,
        x: neg,
        customdata: custom,
        marker: { color: 'rgba(224, 101, 95, 0.78)' },
        hovertemplate:
          `<b>Strike $%{y}</b><br>Net: %{customdata[2]}${unit}<br>` +
          `Call: %{customdata[3]}${unit} · Put: %{customdata[4]}${unit}<br>` +
          `Call ${countLabel} %{customdata[0]} · Put ${countLabel} %{customdata[1]}<extra></extra>`,
      },
    ];

    const shapes: NonNullable<Layout['shapes']> = [
      {
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: 0,
        x1: 0,
        y0: 0,
        y1: 1,
        line: { color: 'rgba(148, 148, 160, 0.4)', width: 1, dash: 'dot' },
      },
    ];

    const minStrike = strikes[0];
    const maxStrike = strikes[strikes.length - 1];
    const loBound = minStrike * 0.94;
    const hiBound = maxStrike * 1.06;

    const annotations: NonNullable<Layout['annotations']> = [];
    const addHLine = (
      val: number | null | undefined,
      label: string,
      color: string,
      dash: 'dash' | 'dot' | 'dashdot',
      xAnchor: 'left' | 'right',
      yAnchor: 'top' | 'bottom' = 'bottom'
    ) => {
      if (val === null || val === undefined || !Number.isFinite(val) || val <= 0) return;
      if (val < loBound || val > hiBound) return;
      shapes.push({
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: val,
        y1: val,
        line: { color, width: 1.25, dash },
      });
      annotations.push({
        x: xAnchor === 'right' ? 0.99 : 0.01,
        y: val,
        xref: 'paper',
        yref: 'y',
        text: `${label} $${val.toFixed(val >= 100 ? 0 : 2)}`,
        showarrow: false,
        xanchor: xAnchor,
        yanchor: yAnchor,
        font: { size: 9.5, color },
      });
    };

    addHLine(spot, 'Spot', QUANT_COLORS.goldLight, 'dash', 'right', 'bottom');
    addHLine(flip, 'Flip', QUANT_COLORS.amber, 'dot', 'left', 'bottom');

    const sameCw = callWall !== null && gammaCallWall !== null && callWall === gammaCallWall;
    const samePw = putWall !== null && gammaPutWall !== null && putWall === gammaPutWall;

    addHLine(callWall, sameCw ? 'CW / γCW' : 'CW', QUANT_COLORS.positive, 'dash', 'left', 'top');
    if (!sameCw) {
      addHLine(gammaCallWall, 'γCW', QUANT_COLORS.gold, 'dashdot', 'right', 'top');
    }
    addHLine(putWall, samePw ? 'PW / γPW' : 'PW', QUANT_COLORS.negative, 'dash', 'left', 'top');
    if (!samePw) {
      addHLine(gammaPutWall, 'γPW', QUANT_COLORS.amber, 'dashdot', 'right', 'top');
    }

    return {
      traces: list,
      layout: {
        barmode: 'relative',
        shapes,
        annotations,
        margin: { l: 54, r: 16, t: 28, b: 42 },
      } as Partial<Layout>,
    };
  }, [trimmed, div, unit, activeMetric, meta, spot, flip, callWall, putWall, gammaCallWall, gammaPutWall]);

  const height = Math.max(300, Math.min(540, Math.max(trimmed.length, 10) * 16 + 76));

  return (
    <div>
      {showMetricSelector && (
        <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
          <SegmentedControl
            options={METRIC_OPTIONS}
            value={activeMetric}
            onChange={handleMetricChange}
            ariaLabel="Gamma profile metric"
          />
          {bucketLabel && (
            <span className="rounded border border-[var(--hairline)] bg-[var(--bg-inset)] px-2 py-0.5 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
              {bucketLabel}
            </span>
          )}
        </div>
      )}

      {trimmed.length === 0 ? (
        <EmptyState
          title={`No ${meta.title.toLowerCase()} bars to plot`}
          detail={
            activeMetric === 'volumeGex'
              ? 'Requires positive intraday option volume in this bucket'
              : 'Requires open interest across strikes in the selected filter'
          }
          className="h-56"
        />
      ) : (
        <PlotlyChart
          data={traces}
          layout={layout}
          plotType="2d"
          hovermode="y unified"
          showLegend
          xAxisTitle={`${meta.title} ($${unit} ${meta.unitHint})`}
          xTickSuffix={unit}
          yAxisTitle="Strike"
          yTickPrefix="$"
          height={height}
          uirevisionKey={`gex-${activeMetric}-${convention}-${bucketLabel ?? 'all'}-${spot.toFixed(1)}-${trimmed.length}`}
          exportFilename={`gamma-profile-${activeMetric}`}
          showWheelHint
        />
      )}

      <div className="mt-1 flex flex-wrap items-center gap-3 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
        <span>{convention === 'squeeze' ? 'Squeeze convention (long C / short P)' : 'Inverse convention (short C / long P)'}</span>
        <span>·</span>
        <span>
          Spot ${spot.toFixed(2)} · {flip ? `Flip ≈ $${flip.toFixed(2)}` : 'No flip in grid'}
        </span>
      </div>
    </div>
  );
}
