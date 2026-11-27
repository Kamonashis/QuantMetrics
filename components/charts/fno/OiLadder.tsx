'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { EmptyState } from '@/components/ui/empty-state';
import { PlotlyChart } from '@/components/charts/PlotlyChart';
import { QUANT_COLORS } from '@/lib/plotly-theme';

export interface OiLadderLevels {
  callWall?: number | null;
  putWall?: number | null;
  gammaCallWall?: number | null;
  gammaPutWall?: number | null;
  gammaFlip?: number | null;
  regime?: 'long' | 'short' | 'unknown' | null;
}

const fmtLevelPct = (level: number | null | undefined, spot: number) => {
  if (level === null || level === undefined || !isFinite(level) || spot <= 0) return null;
  const pct = (level / spot - 1) * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`;
};

/**
 * Interactive Plotly dual-sided OI butterfly ladder around spot:
 * Calls extending left in champagne gold, Puts extending right in crimson.
 */
export function OiLadder({
  rows,
  spot,
  levels = null,
}: {
  rows: { strike: number; callOi: number; putOi: number; totalOi: number }[];
  spot: number;
  levels?: OiLadderLevels | null;
}) {
  const { traces, layout } = useMemo(() => {
    if (rows.length === 0) {
      return { traces: [] as Data[], layout: {} as Partial<Layout> };
    }

    const sorted = [...rows].sort((a, b) => a.strike - b.strike);
    const strikes = sorted.map((r) => r.strike);
    const minStrike = strikes[0];
    const maxStrike = strikes[strikes.length - 1];
    const inBand = (v: number | null | undefined): v is number =>
      typeof v === 'number' && isFinite(v) && v > 0 && v >= minStrike * 0.98 && v <= maxStrike * 1.02;

    const list: Data[] = [
      {
        type: 'bar',
        orientation: 'h',
        name: 'Call OI',
        y: strikes,
        x: sorted.map((r) => -r.callOi),
        customdata: sorted.map((r) => [r.callOi.toLocaleString(), r.totalOi.toLocaleString()]),
        marker: {
          color: sorted.map((r) =>
            Math.abs(r.strike - spot) / Math.max(spot, 1e-6) < 0.025
              ? 'rgba(222, 192, 132, 0.96)'
              : 'rgba(222, 192, 132, 0.62)'
          ),
        },
        hovertemplate:
          '<b>Strike $%{y} · Calls</b><br>Call OI: %{customdata[0]}<br>Total OI: %{customdata[1]}<extra></extra>',
      },
      {
        type: 'bar',
        orientation: 'h',
        name: 'Put OI',
        y: strikes,
        x: sorted.map((r) => r.putOi),
        customdata: sorted.map((r) => [r.putOi.toLocaleString(), r.totalOi.toLocaleString()]),
        marker: {
          color: sorted.map((r) =>
            Math.abs(r.strike - spot) / Math.max(spot, 1e-6) < 0.025
              ? 'rgba(224, 101, 95, 0.96)'
              : 'rgba(224, 101, 95, 0.62)'
          ),
        },
        hovertemplate:
          '<b>Strike $%{y} · Puts</b><br>Put OI: %{customdata[0]}<br>Total OI: %{customdata[1]}<extra></extra>',
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
        line: { color: 'rgba(148,148,160,0.4)', width: 1, dash: 'dot' },
      },
      {
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: spot,
        y1: spot,
        line: { color: QUANT_COLORS.goldLight, width: 1.3, dash: 'dash' },
      },
    ];

    const annotations: NonNullable<Layout['annotations']> = [
      {
        x: 0.99,
        y: spot,
        xref: 'paper',
        yref: 'y',
        text: `Spot $${spot.toFixed(2)}`,
        showarrow: false,
        xanchor: 'right',
        yanchor: 'bottom',
        font: { size: 9.5, color: QUANT_COLORS.goldLight },
      },
    ];

    const addRefLine = (
      val: number,
      label: string,
      color: string,
      dash: 'dash' | 'dot' | 'dashdot',
      align: 'left' | 'right'
    ) => {
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
        x: align === 'left' ? 0.01 : 0.99,
        y: val,
        xref: 'paper',
        yref: 'y',
        text: label,
        showarrow: false,
        xanchor: align,
        yanchor: 'bottom',
        font: { size: 9, color },
      });
    };

    const cw = levels?.callWall ?? null;
    const pw = levels?.putWall ?? null;
    const gcw = levels?.gammaCallWall ?? null;
    const gpw = levels?.gammaPutWall ?? null;
    const flip = levels?.gammaFlip ?? null;

    if (inBand(cw)) {
      const sameGamma = inBand(gcw) && Math.abs(gcw - cw) < 1e-6;
      addRefLine(cw, sameGamma ? `CW / γCW $${cw}` : `CW $${cw}`, QUANT_COLORS.positive, 'dash', 'left');
    }
    if (inBand(gcw) && (!inBand(cw) || Math.abs(gcw - cw) >= 1e-6)) {
      addRefLine(gcw, `γCW $${gcw}`, QUANT_COLORS.sky, 'dashdot', 'right');
    }
    if (inBand(pw)) {
      const sameGamma = inBand(gpw) && Math.abs(gpw - pw) < 1e-6;
      addRefLine(pw, sameGamma ? `PW / γPW $${pw}` : `PW $${pw}`, QUANT_COLORS.negative, 'dash', 'left');
    }
    if (inBand(gpw) && (!inBand(pw) || Math.abs(gpw - pw) >= 1e-6)) {
      addRefLine(gpw, `γPW $${gpw}`, QUANT_COLORS.violet, 'dashdot', 'right');
    }
    if (inBand(flip)) {
      addRefLine(flip, `γ Flip $${flip.toFixed(1)}`, QUANT_COLORS.amber, 'dot', 'left');
    }

    const chartLayout: Partial<Layout> = {
      barmode: 'relative',
      margin: { l: 54, r: 18, t: 28, b: 40 },
      shapes,
      annotations,
    };

    return { traces: list, layout: chartLayout };
  }, [rows, spot, levels]);

  if (rows.length === 0) {
    return (
      <EmptyState
        title="No strikes in ±15% band"
        detail="The chain has no usable rows in this window"
        className="h-40"
      />
    );
  }

  const height = Math.max(290, Math.min(520, rows.length * 16 + 76));
  const hasLevels =
    levels &&
    (levels.callWall != null ||
      levels.putWall != null ||
      levels.gammaCallWall != null ||
      levels.gammaPutWall != null ||
      levels.gammaFlip != null);

  return (
    <div>
      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="y unified"
        showLegend
        xAxisTitle="← Call OI · Put OI →"
        yAxisTitle="Strike"
        yTickPrefix="$"
        height={height}
        uirevisionKey={`oi-ladder-${spot.toFixed(1)}-${rows.length}`}
        exportFilename="oi-ladder"
        showWheelHint
      />
      {hasLevels && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5 font-mono text-[9.5px]">
          {levels.regime && levels.regime !== 'unknown' && (
            <span
              className={`rounded border px-1.5 py-0.5 font-semibold uppercase tracking-wider ${
                levels.regime === 'long'
                  ? 'border-emerald-500/40 bg-emerald-500/10 text-[var(--positive)]'
                  : 'border-rose-500/40 bg-rose-500/10 text-[var(--negative)]'
              }`}
            >
              {levels.regime === 'long' ? 'Long γ' : 'Short γ'}
            </span>
          )}
          {levels.gammaFlip != null && (
            <span className="rounded border border-amber-500/35 bg-amber-500/10 px-1.5 py-0.5 tabular-nums text-amber-700 dark:text-amber-300">
              Flip ${levels.gammaFlip.toFixed(1)} ({fmtLevelPct(levels.gammaFlip, spot) ?? '—'})
            </span>
          )}
          {levels.callWall != null && (
            <span className="rounded border border-emerald-500/35 bg-emerald-500/10 px-1.5 py-0.5 tabular-nums text-emerald-700 dark:text-emerald-300">
              OI Call Wall ${levels.callWall} ({fmtLevelPct(levels.callWall, spot) ?? '—'})
            </span>
          )}
          {levels.putWall != null && (
            <span className="rounded border border-rose-500/35 bg-rose-500/10 px-1.5 py-0.5 tabular-nums text-rose-600 dark:text-rose-400">
              OI Put Wall ${levels.putWall} ({fmtLevelPct(levels.putWall, spot) ?? '—'})
            </span>
          )}
          {levels.gammaCallWall != null && (
            <span className="rounded border border-[var(--hairline)] bg-[var(--bg-inset)] px-1.5 py-0.5 tabular-nums text-zinc-700 dark:text-zinc-200">
              γ Call Wall ${levels.gammaCallWall} ({fmtLevelPct(levels.gammaCallWall, spot) ?? '—'})
            </span>
          )}
          {levels.gammaPutWall != null && (
            <span className="rounded border border-[var(--hairline)] bg-[var(--bg-inset)] px-1.5 py-0.5 tabular-nums text-zinc-700 dark:text-zinc-200">
              γ Put Wall ${levels.gammaPutWall} ({fmtLevelPct(levels.gammaPutWall, spot) ?? '—'})
            </span>
          )}
        </div>
      )}
    </div>
  );
}
