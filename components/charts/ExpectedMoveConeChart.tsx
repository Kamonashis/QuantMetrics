'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import type { PriceBar } from '@/lib/quant-engine';
import { PlotlyChart } from '@/components/charts/PlotlyChart';
import { QUANT_COLORS } from '@/lib/plotly-theme';

function addTradingDays(dateStr: string, tradingDays: number): string {
  const d = new Date(`${dateStr.slice(0, 10)}T12:00:00Z`);
  if (!Number.isFinite(d.getTime())) return dateStr;
  let added = 0;
  while (added < tradingDays) {
    d.setUTCDate(d.getUTCDate() + 1);
    const day = d.getUTCDay();
    if (day !== 0 && day !== 6) added++;
  }
  return d.toISOString().slice(0, 10);
}

/**
 * Interactive Plotly price trajectory with a forward expected-move cone (±1σ / ±2σ)
 * fanning from spot using ±k·S₀·IV30·√(t/252).
 */
export function ExpectedMoveConeChart({
  bars,
  iv30,
  horizonTd = 21,
  callWall = null,
  putWall = null,
  gammaFlip = null,
  gexRegime = null,
}: {
  bars: PriceBar[];
  iv30: number | null;
  horizonTd?: number;
  callWall?: number | null;
  putWall?: number | null;
  gammaFlip?: number | null;
  gexRegime?: 'long' | 'short' | 'unknown' | null;
}) {
  const hist = useMemo(() => (bars ?? []).slice(-180), [bars]);
  const spot = hist.length > 0 ? hist[hist.length - 1].close : null;

  const { traces, layout } = useMemo(() => {
    if (iv30 === null || !spot || hist.length < 2 || !Number.isFinite(spot) || spot <= 0) {
      return { traces: [] as Data[], layout: {} as Partial<Layout> };
    }

    const lastDate = hist[hist.length - 1].date;
    const histDates = hist.map((b) => b.date);
    const histCloses = hist.map((b) => b.close);

    const futDates: string[] = [lastDate];
    const up1: number[] = [spot];
    const dn1: number[] = [spot];
    const up2: number[] = [spot];
    const dn2: number[] = [spot];
    const mid: number[] = [spot];

    for (let f = 1; f <= horizonTd; f++) {
      const dt = addTradingDays(lastDate, f);
      const w1 = spot * iv30 * Math.sqrt(f / 252);
      futDates.push(dt);
      up1.push(spot + w1);
      dn1.push(Math.max(0, spot - w1));
      up2.push(spot + 2 * w1);
      dn2.push(Math.max(0, spot - 2 * w1));
      mid.push(spot);
    }

    const list: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: '−2σ Bound',
        x: futDates,
        y: dn2,
        line: { color: 'rgba(222,192,132,0.45)', width: 1, dash: 'dot' },
        showlegend: false,
        hovertemplate: '−2σ: $%{y:.2f}<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: '±2σ Cone',
        x: futDates,
        y: up2,
        fill: 'tonexty',
        fillcolor: 'rgba(222, 192, 132, 0.08)',
        line: { color: 'rgba(222,192,132,0.45)', width: 1, dash: 'dot' },
        hovertemplate: '+2σ: $%{y:.2f}<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: '−1σ Bound',
        x: futDates,
        y: dn1,
        line: { color: 'rgba(222,192,132,0.85)', width: 1.3, dash: 'dash' },
        showlegend: false,
        hovertemplate: '−1σ: $%{y:.2f}<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: '±1σ Cone',
        x: futDates,
        y: up1,
        fill: 'tonexty',
        fillcolor: 'rgba(222, 192, 132, 0.16)',
        line: { color: 'rgba(222,192,132,0.85)', width: 1.3, dash: 'dash' },
        hovertemplate: '+1σ: $%{y:.2f}<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Drift-Free Spot',
        x: futDates,
        y: mid,
        line: { color: 'rgba(222,192,132,0.55)', width: 1, dash: 'longdash' },
        hovertemplate: 'Spot ref: $%{y:.2f}<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Historical Close',
        x: histDates,
        y: histCloses,
        line: { color: QUANT_COLORS.goldLight, width: 2 },
        hovertemplate: 'Close: $%{y:.2f}<extra></extra>',
      },
    ];

    const shapes: NonNullable<Layout['shapes']> = [
      {
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: lastDate,
        x1: lastDate,
        y0: 0,
        y1: 1,
        line: { color: 'rgba(148,148,160,0.45)', width: 1, dash: 'dash' },
      },
    ];

    const annotations: NonNullable<Layout['annotations']> = [
      {
        x: lastDate,
        y: 0.99,
        xref: 'x',
        yref: 'paper',
        text: 'Today',
        showarrow: false,
        font: { size: 9.5, color: QUANT_COLORS.slate },
        yanchor: 'top',
      },
    ];

    const dealerLevels: number[] = [];
    const addDealerLevel = (
      val: number | null,
      label: string,
      color: string,
      dash: 'dash' | 'dot'
    ) => {
      if (val === null || !Number.isFinite(val) || val <= 0) return;
      if (Math.abs(val / spot - 1) <= 0.35) {
        dealerLevels.push(val);
      }
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
        x: 0.01,
        y: val,
        xref: 'paper',
        yref: 'y',
        text: `${label} $${val.toFixed(1)}`,
        showarrow: false,
        xanchor: 'left',
        yanchor: 'bottom',
        font: { size: 9, color },
      });
    };

    addDealerLevel(callWall, 'CW', QUANT_COLORS.positive, 'dash');
    addDealerLevel(putWall, 'PW', QUANT_COLORS.negative, 'dash');
    addDealerLevel(gammaFlip, 'γ Flip', QUANT_COLORS.amber, 'dot');

    let yaxis: Partial<Layout['yaxis']> | undefined;
    if (dealerLevels.length > 0) {
      const allY = [...histCloses, ...up2, ...dn2, ...dealerLevels];
      const minY = Math.min(...allY);
      const maxY = Math.max(...allY);
      const span = Math.max(maxY - minY, spot * 0.02);
      const pad = span * 0.06;
      yaxis = { range: [Math.max(0, minY - pad), maxY + pad] };
    }

    return {
      traces: list,
      layout: { shapes, annotations, ...(yaxis ? { yaxis } : {}) } as Partial<Layout>,
    };
  }, [hist, spot, iv30, horizonTd, callWall, putWall, gammaFlip]);

  if (iv30 === null || spot === null || hist.length < 2) {
    return (
      <div className="flex h-44 flex-col items-center justify-center gap-1 rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] px-6 text-center">
        <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-500 dark:text-zinc-400">
          Cone needs live ATM IV
        </p>
        <p className="max-w-md text-[11px] leading-snug text-zinc-400 dark:text-zinc-500">
          The forward fan is priced off spot × IV30 × √(t/252). The option chain is unavailable
          right now, so no cone is drawn — rather than one built on a stale or synthetic IV.
        </p>
      </div>
    );
  }

  const oneSigma = (f: number) => spot * iv30 * Math.sqrt(f / 252);

  return (
    <div>
      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        rangeSlider
        hovermode="x unified"
        showLegend
        yTickPrefix="$"
        height={320}
        uirevisionKey={`cone-${spot.toFixed(1)}-${horizonTd}`}
        exportFilename="expected-move-cone"
        showWheelHint
      />
      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
        <span>
          spot <span className="text-zinc-900 dark:text-zinc-100">${spot.toFixed(2)}</span>
        </span>
        <span>
          IV30 <span className="text-zinc-900 dark:text-zinc-100">{(iv30 * 100).toFixed(1)}%</span>
        </span>
        <span>
          ±1σ @10td{' '}
          <span className="text-zinc-900 dark:text-zinc-100">${oneSigma(10).toFixed(1)}</span>
        </span>
        <span>
          ±1σ @21td{' '}
          <span className="text-zinc-900 dark:text-zinc-100">${oneSigma(21).toFixed(1)}</span>
        </span>
        {callWall != null && (
          <span>
            CW <span className="text-[var(--positive)]">${callWall}</span>
          </span>
        )}
        {putWall != null && (
          <span>
            PW <span className="text-[var(--negative)]">${putWall}</span>
          </span>
        )}
        {gammaFlip != null && (
          <span>
            Flip <span className="text-amber-600 dark:text-amber-300">${gammaFlip.toFixed(1)}</span>
          </span>
        )}
        {gexRegime && gexRegime !== 'unknown' && (
          <span className={gexRegime === 'long' ? 'text-[var(--positive)]' : 'text-[var(--negative)]'}>
            {gexRegime === 'long' ? 'Long γ' : 'Short γ'}
          </span>
        )}
        <span className="ml-auto">±k·σ·√(t/252) · no drift · European BSM</span>
      </div>
    </div>
  );
}
