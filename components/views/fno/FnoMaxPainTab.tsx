'use client';

import React, { useMemo } from 'react';
import { Crosshair, Info } from 'lucide-react';
import type { Data, Layout } from 'plotly.js';
import type { PanelPayload } from './types';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { PlotlyChart } from '@/components/charts/PlotlyChart';
import { QUANT_COLORS } from '@/lib/plotly-theme';

export function FnoMaxPainTab({ panel, symbol }: { panel: PanelPayload | null; symbol: string }) {
  const pain = panel?.maxPain ?? null;
  const spot = panel?.quote?.price ?? 0;
  const bars = useMemo(() => pain?.curve ?? [], [pain]);
  const maxPainStrike = pain?.maxPainStrike ?? null;

  const { traces, layout } = useMemo(() => {
    if (bars.length === 0) {
      return { traces: [] as Data[], layout: {} as Partial<Layout> };
    }
    const list: Data[] = [
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Aggregate Buyer Payout',
        x: bars.map((b) => b.strike),
        y: bars.map((b) => b.pain / 1e6),
        line: { color: QUANT_COLORS.goldLight, width: 2 },
        marker: { size: 4, color: QUANT_COLORS.goldLight },
        fill: 'tozeroy',
        fillcolor: 'rgba(222, 192, 132, 0.13)',
        hovertemplate: '<b>Strike $%{x}</b><br>Payout: $%{y:.2f}M<extra></extra>',
      },
    ];

    const shapes: NonNullable<Layout['shapes']> = [];
    const annotations: NonNullable<Layout['annotations']> = [];

    if (maxPainStrike !== null) {
      shapes.push({
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: maxPainStrike,
        x1: maxPainStrike,
        y0: 0,
        y1: 1,
        line: { color: QUANT_COLORS.goldLight, width: 1.5, dash: 'dash' },
      });
      annotations.push({
        x: maxPainStrike,
        y: 0.98,
        xref: 'x',
        yref: 'paper',
        text: `Max Pain $${maxPainStrike}`,
        showarrow: false,
        xanchor: 'left',
        xshift: 4,
        font: { size: 9.5, color: QUANT_COLORS.goldLight },
      });
    }

    if (spot > 0) {
      shapes.push({
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: spot,
        x1: spot,
        y0: 0,
        y1: 1,
        line: { color: QUANT_COLORS.slate, width: 1.3, dash: 'dot' },
      });
      annotations.push({
        x: spot,
        y: 0.86,
        xref: 'x',
        yref: 'paper',
        text: `Spot $${spot.toFixed(1)}`,
        showarrow: false,
        xanchor: 'right',
        xshift: -4,
        font: { size: 9.5, color: QUANT_COLORS.slate },
      });
    }

    return { traces: list, layout: { shapes, annotations } as Partial<Layout> };
  }, [bars, maxPainStrike, spot]);

  return (
    <div className="space-y-4">
      {/* Summary row */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          {
            label: 'Max Pain',
            value: maxPainStrike !== null ? String(maxPainStrike) : '—',
            tone: 'text-gold-700 dark:text-gold-300',
          },
          {
            label: 'Distance',
            value:
              pain?.distancePct !== null && pain?.distancePct !== undefined
                ? `${pain.distancePct >= 0 ? '+' : ''}${pain.distancePct.toFixed(1)}%`
                : '—',
            tone: 'text-zinc-900 dark:text-zinc-50',
          },
          {
            label: 'Basin width',
            value:
              pain?.basinWidthPct !== null && pain?.basinWidthPct !== undefined
                ? `${pain.basinWidthPct.toFixed(1)}%`
                : '—',
            tone: 'text-zinc-900 dark:text-zinc-50',
          },
          {
            label: 'Spot',
            value: spot > 0 ? spot.toFixed(2) : '—',
            tone: 'text-zinc-900 dark:text-zinc-50',
          },
        ].map((s) => (
          <div key={s.label} className="panel-inset p-4">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">
              {s.label}
            </div>
            <div className={`mt-1 font-mono text-2xl font-bold tabular-nums ${s.tone}`}>
              {s.value}
            </div>
          </div>
        ))}
      </div>

      {/* Pain curve */}
      <div className="panel p-5">
        <div className="mb-3 flex flex-col justify-between gap-2 border-b border-[var(--hairline)] pb-3 sm:flex-row sm:items-end">
          <div>
            <p className="kicker mb-1 flex items-center gap-2">
              <Crosshair className="h-3.5 w-3.5" /> Pain Curve
            </p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Payout to option buyers at expiration, by strike — minimum marked
            </h3>
          </div>
          <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
            basin width = strikes ≤ 105% of min pain · distance = |spot − pain| / spot
          </span>
        </div>

        {bars.length === 0 ? (
          <div className="text-xs text-zinc-400">
            Chain too thin to build a pain curve — insufficient OI at nearby strikes.
          </div>
        ) : (
          <ChartFullscreen title={`${symbol} Max-Pain Curve`}>
            <PlotlyChart
              data={traces}
              layout={layout}
              plotType="2d"
              hovermode="x unified"
              xAxisTitle="Expiration Strike"
              xTickFormat="$.0f"
              yAxisTitle="Buyer Payout ($M)"
              yTickPrefix="$"
              yTickSuffix="M"
              height={265}
              uirevisionKey={`max-pain-${symbol}-${bars.length}`}
              exportFilename={`${symbol}-max-pain`}
              showWheelHint
            />
          </ChartFullscreen>
        )}
      </div>

      {/* Honesty footer */}
      <div className="flex items-start gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 py-2 text-[10px] leading-relaxed text-zinc-500 dark:text-zinc-400">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-400" />
        <span>
          Max pain is a <strong>heuristic, not a law</strong> — settlement is drawn toward pain only
          when open interest is concentrated and the basin is narrow. Treat the basin width, not the
          point estimate, as the signal: a flat basin (current width{' '}
          {pain?.basinWidthPct?.toFixed(1) ?? '—'}%) means the pin itself is arbitrary and you should
          not trade toward it. Historical hit rate will accrue with daily snapshots.
        </span>
      </div>
    </div>
  );
}
