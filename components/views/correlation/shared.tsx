'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

export type Pair = [string, string];

export const fmtCorr = (v: number | null | undefined, d = 2) => {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const r = Number(v.toFixed(d)) + 0;
  return `${r >= 0 ? '+' : ''}${r.toFixed(d)}`;
};

/** Fixed decimals without a signed zero ("-0.00"). */
export const num = (v: number, d = 2) => (Number.isFinite(v) ? (Number(v.toFixed(d)) + 0).toFixed(d) : '—');

export const fmtPct = (v: number, d = 1) => (Number.isFinite(v) ? `${(v * 100).toFixed(d)}%` : '—');

export function ordinal(pct: number): string {
  const n = Math.round(pct * 100);
  const mod100 = n % 100;
  const suffix = mod100 >= 11 && mod100 <= 13 ? 'th' : n % 10 === 1 ? 'st' : n % 10 === 2 ? 'nd' : n % 10 === 3 ? 'rd' : 'th';
  return `${n}${suffix}`;
}

/** Where a current reading sits in its own history. */
export function PercentileChip({ value }: { value: number }) {
  if (!Number.isFinite(value)) return null;
  const extreme = value >= 0.9 || value <= 0.1;
  return (
    <span
      title="Share of past readings at or below the current one"
      className={`inline-flex items-center rounded-md border px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-wider ${
        extreme
          ? 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400'
          : 'border-zinc-300 bg-zinc-100 text-zinc-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-400'
      }`}
    >
      {ordinal(value)} percentile
    </span>
  );
}

const selectClass =
  'cursor-pointer rounded-lg border border-zinc-200 bg-zinc-50 px-2 py-1 font-mono text-xs font-bold text-zinc-900 focus:outline-none focus:ring-1 focus:ring-gold-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100';

export function PairPicker({ tickers, pair, onChange }: { tickers: readonly string[]; pair: Pair; onChange: (p: Pair) => void }) {
  return (
    <div className="flex items-center gap-1.5 text-xs text-zinc-500 dark:text-zinc-400">
      <span>Pair</span>
      <select
        aria-label="First asset"
        value={pair[0]}
        onChange={(e) => onChange([e.target.value, e.target.value === pair[1] ? pair[0] : pair[1]])}
        className={selectClass}
      >
        {tickers.map((t) => (
          <option key={t} value={t}>
            {t}
          </option>
        ))}
      </select>
      <span aria-hidden="true">/</span>
      <select
        aria-label="Second asset"
        value={pair[1]}
        onChange={(e) => onChange([e.target.value === pair[0] ? pair[1] : pair[0], e.target.value])}
        className={selectClass}
      >
        {tickers.map((t) => (
          <option key={t} value={t}>
            {t}
          </option>
        ))}
      </select>
    </div>
  );
}

/**
 * Lower-triangle correlation heatmap. The colour scale is fitted to the data
 * (sequential when every value is positive, diverging otherwise) and the
 * colour bar always shows the range in use.
 */
export function MatrixHeatmap({
  tickers,
  matrix,
  selected,
  onSelect,
  exportName,
}: {
  tickers: readonly string[];
  matrix: (number | null)[][];
  selected?: Pair;
  onSelect?: (p: Pair) => void;
  exportName: string;
}) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const { traces, layout } = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    const n = tickers.length;
    // Lower triangle only: rows start at the second asset, columns stop before the last.
    const rows = tickers.slice(1);
    const cols = tickers.slice(0, n - 1);
    const z: (number | null)[][] = [];
    const text: string[][] = [];
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 1; i < n; i++) {
      const zr: (number | null)[] = [];
      const tr: string[] = [];
      for (let j = 0; j < n - 1; j++) {
        const raw = j < i ? matrix[i]?.[j] ?? null : null;
        const ok = raw !== null && Number.isFinite(raw);
        // Round first so a tiny negative never prints as "-0.00" or flips the scale to diverging.
        const v = ok ? Math.round((raw as number) * 100) / 100 + 0 : null;
        zr.push(v);
        tr.push(v !== null ? v.toFixed(2) : '');
        if (v !== null) {
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
      }
      z.push(zr);
      text.push(tr);
    }
    if (!Number.isFinite(lo)) {
      lo = 0;
      hi = 1;
    }
    const neutral = isDark ? '#1B1B22' : '#F1EFEA';
    const diverging = lo < 0;
    const bound = Math.min(1, Math.ceil(Math.max(Math.abs(lo), Math.abs(hi)) * 10) / 10);
    const zmin = diverging ? -bound : Math.max(0, Math.floor(lo * 10) / 10);
    const zmax = diverging ? bound : Math.min(1, Math.max(zmin + 0.1, Math.ceil(hi * 10) / 10));
    const shapes: Layout['shapes'] = [];
    if (selected) {
      const a = tickers.indexOf(selected[0]);
      const b = tickers.indexOf(selected[1]);
      if (a >= 0 && b >= 0 && a !== b) {
        const row = Math.max(a, b);
        const col = Math.min(a, b);
        shapes.push({
          type: 'rect',
          xref: 'x',
          yref: 'y',
          x0: col - 0.5,
          x1: col + 0.5,
          y0: row - 1.5,
          y1: row - 0.5,
          line: { color: isDark ? '#F4F4F5' : '#18181B', width: 2 },
        });
      }
    }
    return {
      traces: [
        {
          type: 'heatmap',
          x: cols,
          y: rows,
          z,
          zmin,
          zmax,
          colorscale: diverging
            ? [
                [0, isDark ? '#D4D4D8' : '#52525B'],
                [0.5, neutral],
                [1, QUANT_COLORS.gold],
              ]
            : [
                [0, neutral],
                [1, QUANT_COLORS.gold],
              ],
          text: text as unknown as string[],
          texttemplate: '%{text}',
          textfont: { size: 11 },
          hoverongaps: false,
          hovertemplate: '%{y} / %{x}: <b>%{z:.2f}</b><extra></extra>',
          xgap: 2,
          ygap: 2,
          showscale: true,
          colorbar: { thickness: 10, len: 0.8, outlinewidth: 0, tickfont: { size: 10, color: tokens.textMuted } },
        } as Data,
      ],
      layout: {
        margin: { l: 56, r: 8, t: 8, b: 40 },
        shapes,
        xaxis: { type: 'category', showgrid: false, tickangle: 0, fixedrange: true },
        yaxis: { type: 'category', showgrid: false, autorange: 'reversed', fixedrange: true },
      },
    };
  }, [tickers, matrix, selected, isDark, tokens]);

  return (
    <PlotlyChart
      data={traces}
      layout={layout}
      plotType="heatmap"
      enableWheelPan={false}
      hovermode="closest"
      height={Math.max(220, Math.min(520, (tickers.length - 1) * 46 + 70))}
      exportFilename={exportName}
      uirevisionKey={`${exportName}-${tickers.join('-')}`}
      onClick={(ev) => {
        const p = (ev as { points?: Array<{ x?: unknown; y?: unknown }> })?.points?.[0];
        if (onSelect && p?.x !== undefined && p?.y !== undefined && p.x !== p.y) onSelect([String(p.y), String(p.x)]);
      }}
    />
  );
}

export function DataTable({ children }: { children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full font-mono text-xs tabular-nums">{children}</table>
    </div>
  );
}

export const thClass = 'border-b border-[var(--hairline)] px-2 py-2 text-right font-semibold text-zinc-500 first:pl-0 first:text-left last:pr-0';
const tdBare = 'border-b border-[var(--hairline)] px-2 py-2 text-right first:pl-0 first:text-left last:pr-0';
export const tdClass = `${tdBare} text-zinc-700 dark:text-zinc-300`;
/** Table cell with an explicit text colour (avoids competing Tailwind colour classes). */
export const td = (tone?: string | false | null, extra = '') => `${tdBare} ${tone || 'text-zinc-700 dark:text-zinc-300'} ${extra}`;
