'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import type { IntradayBar } from '@/lib/quant-engine';
import { EmptyState } from '@/components/ui/empty-state';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { SegmentedControl } from '@/components/ui/lab-ui';
import { ProvenanceBadge } from '@/components/ui/provenance-badge';
import { QUANT_COLORS } from '@/lib/plotly-theme';

export interface TrapPriceChartProps {
  closes: { date: string; close: number }[];
  spot: number;
  callWall: number | null;
  putWall: number | null;
  flip: number | null;
  gammaCallWall?: number | null;
  gammaPutWall?: number | null;
  intradayBars?: IntradayBar[];
  intradaySource?: string | null;
}

type PriceViewMode = 'daily' | 'intraday';

/**
 * Interactive Plotly price series (Daily 180d or latest Intraday 1m session)
 * overlaid with OI Call/Put Walls, Gamma Walls, and Zero-Gamma Flip.
 */
export function TrapPriceChart({
  closes,
  spot,
  callWall,
  putWall,
  flip,
  gammaCallWall = null,
  gammaPutWall = null,
  intradayBars,
  intradaySource = null,
}: TrapPriceChartProps) {
  const isDark = usePlotlyDarkTheme();
  const [viewMode, setViewMode] = useState<PriceViewMode>('daily');

  const hasIntraday = Array.isArray(intradayBars) && intradayBars.length >= 10;
  const effectiveMode: PriceViewMode = viewMode === 'intraday' && hasIntraday ? 'intraday' : 'daily';

  const dailySlice = useMemo(() => closes.slice(-180), [closes]);

  const { intradaySlice, intradaySessionDate } = useMemo(() => {
    if (!intradayBars || intradayBars.length === 0) {
      return { intradaySlice: [] as IntradayBar[], intradaySessionDate: null as string | null };
    }
    const lastDate = intradayBars[intradayBars.length - 1].date;
    const latestDay = intradayBars.filter((b) => b.date === lastDate);
    if (latestDay.length >= 30) {
      return { intradaySlice: latestDay, intradaySessionDate: lastDate };
    }
    return {
      intradaySlice: intradayBars.slice(-390),
      intradaySessionDate: lastDate,
    };
  }, [intradayBars]);

  const { traces, layout, hasEnoughData } = useMemo(() => {
    const activeSeries =
      effectiveMode === 'intraday'
        ? intradaySlice.map((b) => ({ x: `${b.date} ${b.time}`, y: b.close }))
        : dailySlice.map((d) => ({ x: d.date, y: d.close }));

    if (activeSeries.length < 10) {
      return { traces: [] as Data[], layout: {} as Partial<Layout>, hasEnoughData: false };
    }

    const xs = activeSeries.map((d) => d.x);
    const ys = activeSeries.map((d) => d.y);
    let minY = Infinity;
    let maxY = -Infinity;
    for (const v of ys) {
      if (v < minY) minY = v;
      if (v > maxY) maxY = v;
    }

    const spotInRange = spot > 0 && Number.isFinite(spot) && spot >= minY * 0.5 && spot <= maxY * 2.0;

    const list: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: effectiveMode === 'intraday' ? '1m Close' : 'Daily Close',
        x: xs,
        y: ys,
        line: {
          color: isDark ? '#ECECF1' : '#18181B',
          width: 1.8,
        },
        fill: 'tozeroy',
        fillcolor: 'rgba(222, 192, 132, 0.08)',
        hovertemplate: '<b>%{x}</b><br>Close: $%{y:.2f}<extra></extra>',
      },
      ...(spotInRange
        ? [
            {
              type: 'scatter' as const,
              mode: 'markers' as const,
              name: 'Chain Spot',
              x: [xs[xs.length - 1]],
              y: [spot],
              marker: {
                size: 7,
                color: QUANT_COLORS.goldLight,
                line: { color: isDark ? '#0B0B0D' : '#FFFFFF', width: 1.5 },
              },
              hovertemplate: '<b>Chain Spot</b><br>$%{y:.2f}<extra></extra>',
            },
          ]
        : []),
    ];

    const shapes: NonNullable<Layout['shapes']> = [];
    const annotations: NonNullable<Layout['annotations']> = [];
    const loBound = minY * 0.84;
    const hiBound = maxY * 1.16;

    const addLevel = (
      val: number | null | undefined,
      color: string,
      label: string,
      dash: 'dash' | 'dot' | 'dashdot',
      xAnchor: 'left' | 'right' = 'right',
      yAnchor: 'top' | 'bottom' = 'bottom'
    ) => {
      if (val === null || val === undefined || !Number.isFinite(val)) return;
      if (val < loBound || val > hiBound) return;
      shapes.push({
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: val,
        y1: val,
        line: { color, width: 1.2, dash },
      });
      annotations.push({
        x: xAnchor === 'right' ? 0.99 : 0.01,
        y: val,
        xref: 'paper',
        yref: 'y',
        text: `${label} $${val.toFixed(1)}`,
        showarrow: false,
        xanchor: xAnchor,
        yanchor: yAnchor,
        font: { size: 9.5, color },
      });
    };

    const sameCw = callWall !== null && gammaCallWall !== null && callWall === gammaCallWall;
    const samePw = putWall !== null && gammaPutWall !== null && putWall === gammaPutWall;

    addLevel(callWall, QUANT_COLORS.goldLight, sameCw ? 'CW / γCW' : 'CW', 'dash', 'right', 'bottom');
    if (!sameCw) {
      addLevel(gammaCallWall, QUANT_COLORS.positive, 'γCW', 'dashdot', 'left', 'bottom');
    }
    addLevel(putWall, QUANT_COLORS.negative, samePw ? 'PW / γPW' : 'PW', 'dash', 'right', 'top');
    if (!samePw) {
      addLevel(gammaPutWall, QUANT_COLORS.amber, 'γPW', 'dashdot', 'left', 'top');
    }
    addLevel(flip, QUANT_COLORS.slate, 'FLIP', 'dot', 'left', 'bottom');

    return {
      hasEnoughData: true,
      traces: list,
      layout: {
        shapes,
        annotations,
        margin: { l: 50, r: 16, t: 14, b: 30 },
      } as Partial<Layout>,
    };
  }, [effectiveMode, intradaySlice, dailySlice, spot, callWall, putWall, gammaCallWall, gammaPutWall, flip, isDark]);

  const levelsSummary = [
    { label: 'CW', val: callWall, color: 'text-gold-700 dark:text-gold-300' },
    { label: 'γCW', val: gammaCallWall, color: 'text-[var(--positive)]' },
    { label: 'Flip', val: flip, color: 'text-amber-600 dark:text-amber-400' },
    { label: 'γPW', val: gammaPutWall, color: 'text-amber-600 dark:text-amber-300' },
    { label: 'PW', val: putWall, color: 'text-[var(--negative)]' },
  ].filter((l) => l.val !== null && l.val !== undefined && Number.isFinite(l.val));

  return (
    <div>
      {hasIntraday && (
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <SegmentedControl
            options={[
              { value: 'daily' as const, label: 'Daily 180d' },
              { value: 'intraday' as const, label: 'Intraday' },
            ]}
            value={effectiveMode}
            onChange={setViewMode}
            ariaLabel="Price overlay timeframe"
          />
          {effectiveMode === 'intraday' && (
            <div className="flex flex-wrap items-center gap-2">
              <ProvenanceBadge daily={intradaySource} intraday={intradaySource} compact />
              <span className="font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
                {intradaySource === 'live_market'
                  ? `Live 1m (${intradaySessionDate ?? 'latest'} · ${intradaySlice.length} bars) · levels are EOD snapshot`
                  : `Synthetic GBM intraday path (${intradaySessionDate ?? 'sim'}) — illustrative only; levels are EOD snapshot`}
              </span>
            </div>
          )}
        </div>
      )}

      {!hasEnoughData ? (
        <EmptyState
          title="Not enough price history"
          detail={
            effectiveMode === 'intraday'
              ? 'Intraday overlay needs ≥ 10 1-minute bars'
              : 'The trap overlay needs a daily close series'
          }
          className="h-44"
        />
      ) : (
        <PlotlyChart
          data={traces}
          layout={layout}
          plotType="2d"
          hovermode="x unified"
          yTickPrefix="$"
          height={230}
          uirevisionKey={`trap-${effectiveMode}-${effectiveMode === 'intraday' ? intradaySlice.length : dailySlice.length}-${spot.toFixed(1)}`}
          exportFilename={`trap-price-levels-${effectiveMode}`}
          showWheelHint
        />
      )}

      {levelsSummary.length > 0 && (
        <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
          {levelsSummary.map((lvl) => {
            const distPct = spot > 0 && lvl.val !== null && lvl.val !== undefined ? ((lvl.val / spot - 1) * 100) : null;
            return (
              <span key={lvl.label} className="tabular-nums">
                <strong className={lvl.color}>{lvl.label}</strong> ${(lvl.val as number).toFixed(1)}
                {distPct !== null ? ` (${distPct >= 0 ? '+' : ''}${distPct.toFixed(1)}%)` : ''}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
