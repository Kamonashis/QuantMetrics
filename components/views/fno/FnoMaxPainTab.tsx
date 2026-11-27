'use client';

import React, { useMemo } from 'react';
import { Crosshair, Info, Magnet } from 'lucide-react';
import type { Data, Layout } from 'plotly.js';
import type { PanelPayload } from './types';
import type { TrapPayload } from '@/components/views/volatility/types';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { PlotlyChart } from '@/components/charts/PlotlyChart';
import { QUANT_COLORS } from '@/lib/plotly-theme';

const fmtDist = (level: number | null | undefined, spot: number) => {
  if (level === null || level === undefined || !isFinite(level) || spot <= 0) return '—';
  const pct = (level / spot - 1) * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`;
};

export function FnoMaxPainTab({
  panel,
  symbol,
  trap = null,
}: {
  panel: PanelPayload | null;
  symbol: string;
  trap?: TrapPayload | null;
}) {
  const pain = panel?.maxPain ?? null;
  const spot = panel?.quote?.price ?? 0;
  const bars = useMemo(() => pain?.curve ?? [], [pain]);
  const maxPainStrike = pain?.maxPainStrike ?? null;
  const activeTrap = panel?.trap ?? trap ?? null;
  const trapReady = activeTrap?.available === true;
  const finiteOrNull = (v: number | null | undefined) =>
    typeof v === 'number' && isFinite(v) ? v : null;
  const callWall = trapReady ? finiteOrNull(activeTrap.walls.call?.strike) : null;
  const putWall = trapReady ? finiteOrNull(activeTrap.walls.put?.strike) : null;
  const gammaCallWall = trapReady ? finiteOrNull(activeTrap.gammaWalls?.call?.strike) : null;
  const gammaPutWall = trapReady ? finiteOrNull(activeTrap.gammaWalls?.put?.strike) : null;
  const gammaFlip = trapReady ? finiteOrNull(activeTrap.gammaFlip?.flip) : null;
  const gexRegime = trapReady ? activeTrap.regime : 'unknown';

  const { traces, layout } = useMemo(() => {
    if (bars.length === 0) {
      return { traces: [] as Data[], layout: {} as Partial<Layout> };
    }
    const minCurve = Math.min(...bars.map((b) => b.strike));
    const maxCurve = Math.max(...bars.map((b) => b.strike));
    const inCurve = (v: number | null): v is number =>
      typeof v === 'number' && isFinite(v) && v >= minCurve * 0.95 && v <= maxCurve * 1.05;

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

    const addVLine = (
      x: number,
      text: string,
      color: string,
      dash: 'dash' | 'dot' | 'dashdot',
      yPaper: number,
      anchor: 'left' | 'right' = 'left'
    ) => {
      shapes.push({
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: x,
        x1: x,
        y0: 0,
        y1: 1,
        line: { color, width: 1.4, dash },
      });
      annotations.push({
        x,
        y: yPaper,
        xref: 'x',
        yref: 'paper',
        text,
        showarrow: false,
        xanchor: anchor,
        xshift: anchor === 'left' ? 4 : -4,
        font: { size: 9.5, color },
      });
    };

    if (maxPainStrike !== null) {
      addVLine(maxPainStrike, `Max Pain $${maxPainStrike}`, QUANT_COLORS.goldLight, 'dash', 0.98, 'left');
    }

    if (spot > 0) {
      addVLine(spot, `Spot $${spot.toFixed(1)}`, QUANT_COLORS.slate, 'dot', 0.86, 'right');
    }

    if (inCurve(gammaFlip)) {
      addVLine(gammaFlip, `γ Flip $${gammaFlip.toFixed(1)}`, QUANT_COLORS.amber, 'dot', 0.74, 'left');
    }

    if (inCurve(callWall)) {
      addVLine(callWall, `Call Wall $${callWall}`, QUANT_COLORS.positive, 'dash', 0.62, 'left');
    }

    if (inCurve(putWall)) {
      addVLine(putWall, `Put Wall $${putWall}`, QUANT_COLORS.negative, 'dash', 0.50, 'right');
    }

    return { traces: list, layout: { shapes, annotations } as Partial<Layout> };
  }, [bars, maxPainStrike, spot, gammaFlip, callWall, putWall]);

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
              Payout to option buyers at expiration, by strike — overlaid with γ Flip &amp; Walls
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

      {/* Max Pain vs. Gamma Flip & Walls comparison card */}
      <div className="panel p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
          <div>
            <p className="kicker mb-1 flex items-center gap-2">
              <Magnet className="h-3.5 w-3.5" /> Pin vs. Acceleration
            </p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Max Pain vs. Gamma Flip &amp; Walls
            </h3>
          </div>
          <span
            className={`rounded border px-2 py-0.5 font-mono text-[9.5px] font-bold uppercase tracking-wider ${
              gexRegime === 'long'
                ? 'border-emerald-500/40 bg-emerald-500/10 text-[var(--positive)]'
                : gexRegime === 'short'
                  ? 'border-rose-500/40 bg-rose-500/10 text-[var(--negative)]'
                  : 'border-[var(--hairline)] bg-[var(--bg-inset)] text-zinc-500'
            }`}
          >
            {gexRegime === 'long'
              ? 'Long γ · Pinning bias'
              : gexRegime === 'short'
                ? 'Short γ · Acceleration bias'
                : 'γ Regime unavailable'}
          </span>
        </div>

        {!trapReady ? (
          <p className="font-mono text-[10.5px] text-zinc-500 dark:text-zinc-400">
            Gamma Flip &amp; Dealer Walls unavailable right now — refresh the chain to compare pinning vs. dealer-hedging flow.
          </p>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4 font-mono">
              <div className="panel-inset p-3">
                <div className="text-[9px] uppercase tracking-wider text-zinc-400">Max Pain</div>
                <div className="mt-1 text-base font-bold tabular-nums text-gold-700 dark:text-gold-300">
                  {maxPainStrike !== null ? `$${maxPainStrike}` : '—'}
                </div>
                <div className="text-[9.5px] text-zinc-500">
                  {fmtDist(maxPainStrike, spot)} vs spot
                </div>
              </div>
              <div className="panel-inset p-3">
                <div className="text-[9px] uppercase tracking-wider text-zinc-400">Gamma Flip</div>
                <div className="mt-1 text-base font-bold tabular-nums text-amber-600 dark:text-amber-300">
                  {gammaFlip !== null ? `$${gammaFlip.toFixed(1)}` : '—'}
                </div>
                <div className="text-[9.5px] text-zinc-500">
                  {fmtDist(gammaFlip, spot)} vs spot
                </div>
              </div>
              <div className="panel-inset p-3">
                <div className="text-[9px] uppercase tracking-wider text-zinc-400">Call Wall (OI / γ)</div>
                <div className="mt-1 text-base font-bold tabular-nums text-[var(--positive)]">
                  {callWall !== null ? `$${callWall}` : '—'}
                  {gammaCallWall !== null && gammaCallWall !== callWall ? ` / $${gammaCallWall}` : ''}
                </div>
                <div className="text-[9.5px] text-zinc-500">
                  {fmtDist(callWall ?? gammaCallWall, spot)} vs spot
                </div>
              </div>
              <div className="panel-inset p-3">
                <div className="text-[9px] uppercase tracking-wider text-zinc-400">Put Wall (OI / γ)</div>
                <div className="mt-1 text-base font-bold tabular-nums text-[var(--negative)]">
                  {putWall !== null ? `$${putWall}` : '—'}
                  {gammaPutWall !== null && gammaPutWall !== putWall ? ` / $${gammaPutWall}` : ''}
                </div>
                <div className="text-[9.5px] text-zinc-500">
                  {fmtDist(putWall ?? gammaPutWall, spot)} vs spot
                </div>
              </div>
            </div>
            <p className="text-[11px] leading-relaxed text-zinc-600 dark:text-zinc-300">
              {gexRegime === 'long' ? (
                <>
                  <strong>Under Long γ (above flip, assumed squeeze):</strong> dealers hedge by selling rallies and buying dips, which suppresses realized volatility and pulls spot toward high-OI walls and Max Pain into expiration
                  {maxPainStrike !== null && gammaFlip !== null
                    ? maxPainStrike >= gammaFlip
                      ? ` — Max Pain ($${maxPainStrike}) sits inside the Long-γ zone, reinforcing pinning behavior.`
                      : ` — note Max Pain ($${maxPainStrike}) sits below the Flip ($${gammaFlip.toFixed(1)}); pinning loses force if spot breaks under the Flip.`
                    : '.'}
                </>
              ) : gexRegime === 'short' ? (
                <>
                  <strong>Under Short γ (below/past flip, assumed squeeze):</strong> dealers hedge in the direction of spot moves (selling drops, buying rallies), so price accelerates away from strikes rather than pinning smoothly to Max Pain
                  {maxPainStrike !== null ? ` ($${maxPainStrike})` : ''}. Watch the Put Wall{putWall !== null ? ` ($${putWall})` : ''} and Flip{gammaFlip !== null ? ` ($${gammaFlip.toFixed(1)})` : ''} for breakaway vs. snapback.
                </>
              ) : (
                <>
                  Under Long γ, dealer counter-hedging dampens swings and favors pinning toward Max Pain and OI walls; under Short γ, pro-cyclical dealer hedging accelerates directional moves past walls rather than pinning.
                </>
              )}
            </p>
          </div>
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
