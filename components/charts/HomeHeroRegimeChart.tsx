'use client';

import React, { useMemo, useState } from 'react';
import { Layers } from 'lucide-react';
import type { Data, Layout } from 'plotly.js';
import {
  getHmmRegimeToken,
  HMM_REGIME_TOKENS,
  type HmmStateIdx,
} from '@/lib/hmm-regime-overlay';
import type { ActiveHmmSeriesResult } from '@/lib/home-universe';
import type { PriceBar } from '@/lib/quant-engine';
import { ProvenanceBadge } from '@/components/ui/provenance-badge';
import { PlotlyChart } from '@/components/charts/PlotlyChart';
import { buildHmmRegimeShapes, QUANT_COLORS } from '@/lib/plotly-theme';

type HorizonKey = '3M' | '6M' | '1Y' | 'ALL';

const HORIZON_BARS: Record<HorizonKey, number> = {
  '3M': 63,
  '6M': 126,
  '1Y': 252,
  ALL: 504,
};

interface PlotPoint {
  date: string;
  close: number;
  retPct: number;
  state: HmmStateIdx | null;
  prob: number | null;
}

export interface HomeHeroRegimeChartProps {
  ticker: string;
  companyName: string;
  bars: PriceBar[];
  barsSource?: string | null;
  livePrice?: number | null;
  liveChangePct?: number | null;
  liveAsOf?: string | null;
  pending?: boolean;
  hmmSeries: ActiveHmmSeriesResult | null;
  garchPersistence: number | null;
  ewmaVol: number | null;
  var99: number | null;
}

function formatDateShort(dateStr: string): string {
  if (!dateStr || dateStr.length < 10) return dateStr;
  const parts = dateStr.slice(0, 10).split('-');
  if (parts.length !== 3) return dateStr;
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const mIdx = Number(parts[1]) - 1;
  const mon = months[mIdx] ?? parts[1];
  return `${mon} ${Number(parts[2])}, ${parts[0].slice(2)}`;
}

export function HomeHeroRegimeChart({
  ticker,
  companyName,
  bars,
  barsSource,
  livePrice,
  liveChangePct,
  liveAsOf,
  pending = false,
  hmmSeries,
  garchPersistence,
  ewmaVol,
  var99,
}: HomeHeroRegimeChartProps) {
  const [horizon, setHorizon] = useState<HorizonKey>('6M');
  const [showRegimeBands, setShowRegimeBands] = useState(true);

  const plotPoints = useMemo<PlotPoint[]>(() => {
    const count = HORIZON_BARS[horizon];
    if (hmmSeries && hmmSeries.points.length >= 2) {
      const sliced = hmmSeries.points.slice(-count);
      return sliced.map((p) => ({
        date: p.date,
        close: p.close,
        retPct: p.retPct,
        state: p.state,
        prob: p.prob,
      }));
    }
    if (!bars || bars.length < 2) return [];
    const sliced = bars.slice(-count);
    return sliced.map((b, i) => {
      const prev = i > 0 ? sliced[i - 1].close : b.close;
      const pct = prev > 0 ? ((b.close - prev) / prev) * 100 : 0;
      return {
        date: b.date,
        close: b.close,
        retPct: pct,
        state: null,
        prob: null,
      };
    });
  }, [hmmSeries, bars, horizon]);

  const { traces, layout } = useMemo(() => {
    if (plotPoints.length < 2) {
      return { traces: [] as Data[], layout: {} as Partial<Layout> };
    }

    const dates = plotPoints.map((p) => p.date);
    const closes = plotPoints.map((p) => p.close);
    const states = plotPoints.map((p) => p.state);
    const custom = plotPoints.map((p) => {
      const tok = p.state !== null ? getHmmRegimeToken(p.state) : null;
      const regimeText = tok
        ? `${tok.label}${p.prob !== null ? ` (${Math.round(p.prob * 100)}%)` : ''}`
        : '—';
      const sign = p.retPct >= 0 ? '+' : '';
      return [`${sign}${p.retPct.toFixed(2)}%`, regimeText];
    });

    const list: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: `${ticker} Close`,
        x: dates,
        y: closes,
        customdata: custom,
        line: { color: QUANT_COLORS.gold, width: 2 },
        fill: 'tozeroy',
        fillcolor: 'rgba(201, 164, 92, 0.14)',
        hovertemplate:
          '<b>$%{y:.2f}</b> (%{customdata[0]} 1d)<br>HMM: %{customdata[1]}<extra></extra>',
      },
    ];

    const shapes = buildHmmRegimeShapes(dates, states, showRegimeBands, 0.14);

    return {
      traces: list,
      layout: {
        shapes,
        margin: { l: 48, r: 14, t: 10, b: 18 },
      } as Partial<Layout>,
    };
  }, [plotPoints, ticker, showRegimeBands]);

  const lastPoint = plotPoints.length > 0 ? plotPoints[plotPoints.length - 1] : null;
  const firstPoint = plotPoints.length > 0 ? plotPoints[0] : null;

  const periodReturnPct =
    firstPoint && lastPoint && firstPoint.close > 0
      ? ((lastPoint.close - firstPoint.close) / firstPoint.close) * 100
      : null;

  const displayPrice =
    !pending && livePrice != null && Number.isFinite(livePrice)
      ? livePrice
      : !pending && lastPoint
        ? lastPoint.close
        : null;

  const displayChangePct =
    !pending && liveChangePct != null && Number.isFinite(liveChangePct)
      ? liveChangePct
      : !pending && lastPoint
        ? lastPoint.retPct
        : null;

  const displayState = lastPoint?.state ?? hmmSeries?.latestState ?? null;
  const displayProb = lastPoint?.prob ?? hmmSeries?.latestProb ?? null;
  const regimeToken = displayState !== null ? getHmmRegimeToken(displayState) : null;

  return (
    <div className="panel w-full flex flex-col justify-between overflow-hidden">
      {/* Header: Active Symbol + Controls */}
      <div className="px-4 sm:px-5 pt-4 pb-3 border-b border-[var(--hairline)] flex flex-wrap items-center justify-between gap-2.5">
        <div className="min-w-0 flex items-center gap-2.5">
          <div className="min-w-0 flex items-center gap-2">
            <span className="font-mono text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
              {ticker}
            </span>
            <span className="text-xs text-zinc-500 dark:text-zinc-400 truncate max-w-[160px] sm:max-w-[220px]">
              {companyName}
            </span>
            {barsSource && <ProvenanceBadge daily={barsSource} compact />}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {/* Horizon Segmented Control */}
          <div
            role="group"
            aria-label="Chart horizon"
            className="inline-flex items-center rounded-md border border-[var(--hairline)] bg-[var(--bg-inset)] p-0.5"
          >
            {(['3M', '6M', '1Y', 'ALL'] as HorizonKey[]).map((key) => {
              const active = horizon === key;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => setHorizon(key)}
                  aria-pressed={active}
                  className={`px-2 py-0.5 text-[11px] font-mono rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    active
                      ? 'bg-white dark:bg-[#1E1E24] text-zinc-900 dark:text-gold-300 shadow-xs'
                      : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
                  }`}
                >
                  {key}
                </button>
              );
            })}
          </div>

          {/* HMM Regime Band Toggle */}
          <button
            type="button"
            onClick={() => setShowRegimeBands((v) => !v)}
            aria-pressed={showRegimeBands}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-mono border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              showRegimeBands
                ? 'border-gold-500/40 bg-gold-500/10 text-gold-700 dark:text-gold-300'
                : 'border-[var(--hairline)] bg-transparent text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
            }`}
          >
            <Layers className="w-3 h-3 shrink-0" />
            <span>HMM Bands</span>
          </button>
        </div>
      </div>

      {/* Live Quote & Current Causal Regime Summary */}
      <div className="px-4 sm:px-5 pt-3 pb-1 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <div className="flex items-baseline gap-2.5 flex-wrap">
          <span className="font-mono text-2xl sm:text-[26px] font-semibold tracking-tight text-zinc-900 dark:text-zinc-50 tabular-nums">
            {displayPrice != null ? `$${displayPrice.toFixed(2)}` : '—'}
          </span>
          {displayChangePct != null ? (
            <span
              className={`font-mono text-xs font-medium tabular-nums ${
                displayChangePct >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'
              }`}
            >
              {displayChangePct >= 0 ? '+' : ''}
              {displayChangePct.toFixed(2)}% 1d
            </span>
          ) : (
            <span className="font-mono text-xs text-zinc-400 tabular-nums">—</span>
          )}
          {periodReturnPct != null && (
            <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400 tabular-nums">
              · {horizon}{' '}
              <span
                className={
                  periodReturnPct >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'
                }
              >
                {periodReturnPct >= 0 ? '+' : ''}
                {periodReturnPct.toFixed(1)}%
              </span>
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 text-[11px] font-mono tabular-nums">
          <span className="text-zinc-500 dark:text-zinc-400">
            {liveAsOf ? formatDateShort(liveAsOf) : lastPoint ? formatDateShort(lastPoint.date) : '—'}
          </span>
          {regimeToken && (
            <>
              <span className="text-zinc-300 dark:text-zinc-700">·</span>
              <span className={`inline-flex items-center gap-1.5 font-medium ${regimeToken.textClass}`}>
                <span
                  className="inline-block w-1.5 h-1.5 rounded-full shrink-0"
                  style={{ backgroundColor: regimeToken.color }}
                />
                {regimeToken.label}
                {displayProb != null && (
                  <span className="opacity-75">({Math.round(displayProb * 100)}%)</span>
                )}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Interactive Plotly Price & Causal HMM Chart */}
      <div className="px-2 sm:px-3 pt-1">
        {plotPoints.length >= 2 ? (
          <PlotlyChart
            data={traces}
            layout={layout}
            plotType="2d"
            size="secondary"
            title={`${ticker} · Price & Causal Volatility Regimes`}
            subtitle={`${horizon} horizon`}
            rangeSlider
            hovermode="x unified"
            yTickPrefix="$"
            uirevisionKey={`hero-${ticker}-${horizon}`}
            exportFilename={`${ticker}-hero-regime-chart`}
          />
        ) : (
          <div className="h-[340px] flex items-center justify-center text-xs font-mono text-zinc-500">
            Loading daily price series…
          </div>
        )}

        {/* Inline Unboxed Regime Legend & Wheel Hint */}
        <div className="px-2 pb-2.5 flex flex-wrap items-center justify-between gap-2 text-[10.5px] font-mono text-zinc-500 dark:text-zinc-400">
          <div className="flex items-center gap-3">
            {HMM_REGIME_TOKENS.map((tok) => (
              <span key={tok.state} className="inline-flex items-center gap-1.5">
                <span
                  className="inline-block w-2 h-2 rounded-[2px] shrink-0"
                  style={{ backgroundColor: tok.color, opacity: showRegimeBands ? 0.85 : 0.3 }}
                />
                <span>{tok.label}</span>
              </span>
            ))}
          </div>
          <span className="text-[10px] text-zinc-400 dark:text-zinc-500">
            Wheel: Pan X · Ctrl+Wheel: Zoom · Double-click: Reset
          </span>
        </div>
      </div>

      {/* Compact Hairline-Divided Econometric Footer */}
      <div className="border-t border-[var(--hairline)] bg-[var(--bg-inset)]/60 grid grid-cols-3 divide-x divide-[var(--hairline)]">
        <div className="px-3.5 sm:px-4 py-2.5">
          <div className="text-[10px] font-mono uppercase tracking-[0.08em] text-zinc-500 dark:text-zinc-400">
            GARCH α+β
          </div>
          <div className="mt-0.5 font-mono text-sm font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
            {garchPersistence != null ? garchPersistence.toFixed(4) : '—'}
          </div>
        </div>
        <div className="px-3.5 sm:px-4 py-2.5">
          <div className="text-[10px] font-mono uppercase tracking-[0.08em] text-zinc-500 dark:text-zinc-400">
            EWMA λ=0.94
          </div>
          <div className="mt-0.5 font-mono text-sm font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
            {ewmaVol != null ? `${(ewmaVol * 100).toFixed(1)}%` : '—'}
          </div>
        </div>
        <div className="px-3.5 sm:px-4 py-2.5">
          <div className="text-[10px] font-mono uppercase tracking-[0.08em] text-zinc-500 dark:text-zinc-400">
            1d 99% VaR
          </div>
          <div className="mt-0.5 font-mono text-sm font-semibold text-[var(--negative)] tabular-nums">
            {var99 != null ? `-${(var99 * 100).toFixed(2)}%` : '—'}
          </div>
        </div>
      </div>
    </div>
  );
}
