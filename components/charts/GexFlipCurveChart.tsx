'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import type { GexFlipPointOut } from '@/components/views/volatility/types';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';
import { EmptyState } from '@/components/ui/empty-state';

export interface GexFlipCurveChartProps {
  profile: GexFlipPointOut[];
  spot: number;
  flip: number | null;
  convention: 'squeeze' | 'inverse';
  serverConvention?: 'squeeze' | 'inverse';
  expectedMove30?: number | null;
  flipDistPct?: number | null;
  flipDistSigma?: number | null;
}

/**
 * Split a continuous (x, y) series into positive and negative fill-to-zero arrays
 * with linearly interpolated zero-crossings so area shading terminates cleanly at GEX = 0.
 */
function buildSignedFills(xs: number[], ys: number[]): {
  fx: number[];
  pos: number[];
  neg: number[];
} {
  const fx: number[] = [];
  const pos: number[] = [];
  const neg: number[] = [];
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];
    const y = ys[i];
    if (i > 0) {
      const px = xs[i - 1];
      const py = ys[i - 1];
      if ((py > 0 && y < 0) || (py < 0 && y > 0)) {
        const t = -py / (y - py);
        const zx = px + t * (x - px);
        fx.push(zx);
        pos.push(0);
        neg.push(0);
      }
    }
    fx.push(x);
    pos.push(y >= 0 ? y : 0);
    neg.push(y <= 0 ? y : 0);
  }
  return { fx, pos, neg };
}

/**
 * Interactive Plotly spot-ladder GEX(S) convexity curve across ±15% spot,
 * with active & optional mirror convention lines, Spot, Gamma Flip, and ±1σ EM band.
 */
export function GexFlipCurveChart({
  profile,
  spot,
  flip,
  convention,
  serverConvention = 'squeeze',
  expectedMove30 = null,
  flipDistPct = null,
  flipDistSigma = null,
}: GexFlipCurveChartProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [showMirror, setShowMirror] = useState(true);

  const model = useMemo(() => {
    if (!profile || profile.length < 2) return null;
    const sorted = [...profile].sort((a, b) => a.spot - b.spot);
    const serverMult = serverConvention === 'squeeze' ? 1 : -1;
    const convMult = convention === serverConvention ? 1 : -1;

    const spots: number[] = [];
    const activeRaw: number[] = [];
    const mirrorRaw: number[] = [];
    const callM: number[] = [];
    const putM: number[] = [];

    let maxAbsRaw = 0;
    for (const p of sorted) {
      const squeezeVal = p.gexSqueeze ?? p.gex * serverMult;
      const inverseVal = p.gexInverse ?? -squeezeVal;
      const act = convention === 'squeeze' ? squeezeVal : inverseVal;
      const mir = convention === 'squeeze' ? inverseVal : squeezeVal;
      spots.push(p.spot);
      activeRaw.push(act);
      mirrorRaw.push(mir);
      callM.push((p.callGex ?? 0) * convMult);
      putM.push((p.putGex ?? 0) * convMult);
      if (Math.abs(act) > maxAbsRaw) maxAbsRaw = Math.abs(act);
    }

    const useK = maxAbsRaw > 0 && maxAbsRaw < 1e6;
    const div = useK ? 1e3 : 1e6;
    const unit = useK ? 'K' : 'M';
    const active = activeRaw.map((v) => v / div);
    const mirror = mirrorRaw.map((v) => v / div);
    const custom = spots.map((_, i) => [
      active[i].toFixed(2),
      (callM[i] / div).toFixed(2),
      (putM[i] / div).toFixed(2),
      mirror[i].toFixed(2),
    ]);

    // Estimate local convexity / slope dGEX / d(1% spot) around spot
    let slopePerPct: number | null = null;
    if (spots.length >= 3 && spot > 0) {
      let nearestIdx = 0;
      let bestDist = Infinity;
      for (let i = 0; i < spots.length; i++) {
        const d = Math.abs(spots[i] - spot);
        if (d < bestDist) {
          bestDist = d;
          nearestIdx = i;
        }
      }
      const loIdx = Math.max(0, nearestIdx - 2);
      const hiIdx = Math.min(spots.length - 1, nearestIdx + 2);
      const dPct = ((spots[hiIdx] - spots[loIdx]) / spot) * 100;
      if (dPct > 0.05) {
        slopePerPct = (activeRaw[hiIdx] - activeRaw[loIdx]) / dPct;
      }
    }

    return { spots, active, mirror, custom, unit, slopePerPct };
  }, [profile, spot, convention, serverConvention]);

  const { traces, layout } = useMemo(() => {
    if (!model) return { traces: [] as Data[], layout: {} as Partial<Layout> };
    const { spots, active, mirror, custom, unit } = model;
    const { fx, pos, neg } = buildSignedFills(spots, active);

    const list: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        x: fx,
        y: pos,
        fill: 'tozeroy',
        fillcolor: 'rgba(63, 178, 127, 0.11)',
        line: { width: 0 },
        hoverinfo: 'skip',
        showlegend: false,
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: fx,
        y: neg,
        fill: 'tozeroy',
        fillcolor: 'rgba(224, 101, 95, 0.11)',
        line: { width: 0 },
        hoverinfo: 'skip',
        showlegend: false,
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: convention === 'squeeze' ? 'GEX(S) · Squeeze' : 'GEX(S) · Inverse',
        x: spots,
        y: active,
        customdata: custom,
        line: { color: QUANT_COLORS.goldLight, width: 2.3 },
        hovertemplate:
          `<b>Spot $%{x:.2f}</b><br>Net GEX: %{customdata[0]}${unit}<br>Calls: %{customdata[1]}${unit} · Puts: %{customdata[2]}${unit}<extra></extra>`,
      },
    ];

    if (showMirror) {
      list.push({
        type: 'scatter',
        mode: 'lines',
        name: convention === 'squeeze' ? 'Inverse mirror' : 'Squeeze mirror',
        x: spots,
        y: mirror,
        line: { color: QUANT_COLORS.slate, width: 1.3, dash: 'dash' },
        hovertemplate: `Mirror: %{y:.2f}${unit}<extra></extra>`,
      });
    }

    const shapes: NonNullable<Layout['shapes']> = [
      {
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: 0,
        y1: 0,
        line: { color: tokens.zeroLine, width: 1.2 },
      },
      {
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: spot,
        x1: spot,
        y0: 0,
        y1: 1,
        line: { color: QUANT_COLORS.goldLight, width: 1.4, dash: 'dash' },
      },
    ];

    const flipCloseToSpot =
      flip !== null && Number.isFinite(flip) && spot > 0 && Math.abs(flip - spot) / spot < 0.025;

    const annotations: NonNullable<Layout['annotations']> = [
      {
        x: spot,
        y: 0.99,
        xref: 'x',
        yref: 'paper',
        text: `Spot $${spot.toFixed(2)}`,
        showarrow: false,
        xanchor: flip !== null && flip > spot && flipCloseToSpot ? 'right' : 'left',
        yanchor: 'top',
        font: { size: 9.5, color: QUANT_COLORS.goldLight },
      },
    ];

    if (flip !== null && Number.isFinite(flip) && flip > 0) {
      shapes.push({
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: flip,
        x1: flip,
        y0: 0,
        y1: 1,
        line: { color: QUANT_COLORS.amber, width: 1.3, dash: 'dot' },
      });
      annotations.push({
        x: flip,
        y: flipCloseToSpot ? 0.87 : 0.99,
        xref: 'x',
        yref: 'paper',
        text: `Flip $${flip.toFixed(2)}`,
        showarrow: false,
        xanchor: flip > spot && flipCloseToSpot ? 'left' : 'right',
        yanchor: 'top',
        font: { size: 9.5, color: QUANT_COLORS.amber },
      });
    }

    if (expectedMove30 !== null && Number.isFinite(expectedMove30) && expectedMove30 > 0) {
      const minX = spots[0];
      const maxX = spots[spots.length - 1];
      const lo = Math.max(minX, spot - expectedMove30);
      const hi = Math.min(maxX, spot + expectedMove30);
      if (hi > lo) {
        shapes.push({
          type: 'rect',
          xref: 'x',
          yref: 'paper',
          x0: lo,
          x1: hi,
          y0: 0,
          y1: 1,
          fillcolor: 'rgba(222, 192, 132, 0.055)',
          line: { width: 0 },
          layer: 'below',
        });
      }
      if (spot - expectedMove30 >= minX) {
        shapes.push({
          type: 'line',
          xref: 'x',
          yref: 'paper',
          x0: spot - expectedMove30,
          x1: spot - expectedMove30,
          y0: 0,
          y1: 1,
          line: { color: 'rgba(148, 148, 160, 0.35)', width: 1, dash: 'dot' },
        });
        annotations.push({
          x: spot - expectedMove30,
          y: 0.04,
          xref: 'x',
          yref: 'paper',
          text: `-1σ EM`,
          showarrow: false,
          xanchor: 'left',
          yanchor: 'bottom',
          font: { size: 8.5, color: tokens.textMuted },
        });
      }
      if (spot + expectedMove30 <= maxX) {
        shapes.push({
          type: 'line',
          xref: 'x',
          yref: 'paper',
          x0: spot + expectedMove30,
          x1: spot + expectedMove30,
          y0: 0,
          y1: 1,
          line: { color: 'rgba(148, 148, 160, 0.35)', width: 1, dash: 'dot' },
        });
        annotations.push({
          x: spot + expectedMove30,
          y: 0.04,
          xref: 'x',
          yref: 'paper',
          text: `+1σ EM`,
          showarrow: false,
          xanchor: 'right',
          yanchor: 'bottom',
          font: { size: 8.5, color: tokens.textMuted },
        });
      }
    }

    return {
      traces: list,
      layout: {
        shapes,
        annotations,
        xaxis: { tickprefix: '$', title: { text: 'Hypothetical Spot (sticky-strike IV)' } },
        margin: { l: 54, r: 16, t: 28, b: 42 },
      } as Partial<Layout>,
    };
  }, [model, convention, showMirror, spot, flip, expectedMove30, tokens]);

  if (!model) {
    return (
      <EmptyState
        title="No spot-ladder GEX curve"
        detail="Requires a multi-strike options chain across ±15% spot"
        className="h-56"
      />
    );
  }

  const slopeFormatted =
    model.slopePerPct !== null && Number.isFinite(model.slopePerPct)
      ? `${model.slopePerPct >= 0 ? '+' : '-'}$${(
          Math.abs(model.slopePerPct) / (Math.abs(model.slopePerPct) >= 1e6 ? 1e6 : 1e3)
        ).toFixed(1)}${Math.abs(model.slopePerPct) >= 1e6 ? 'M' : 'K'} / +1% S`
      : '—';

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <span className="font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
          Convexity slope at spot: <strong className="text-zinc-800 dark:text-zinc-100">{slopeFormatted}</strong>
        </span>
        <button
          type="button"
          onClick={() => setShowMirror((v) => !v)}
          className={`cursor-pointer rounded-md border px-2 py-0.5 font-mono text-[10px] transition-colors ${
            showMirror
              ? 'border-gold-500/40 bg-gold-500/10 text-gold-800 dark:text-gold-200'
              : 'border-[var(--hairline)] text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-100'
          }`}
        >
          {showMirror ? 'Hide mirror curve' : 'Compare mirror curve'}
        </button>
      </div>

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="x unified"
        showLegend
        yAxisTitle={`Net GEX ($${model.unit} / 1% move)`}
        yTickSuffix={model.unit}
        height={320}
        uirevisionKey={`flip-curve-${spot.toFixed(1)}-${convention}-${showMirror ? 1 : 0}`}
        exportFilename="gex-spot-ladder"
        showWheelHint
      />

      <div className="mt-1 flex flex-wrap items-center gap-3 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
        <span>Above zero = Long γ pinning</span>
        <span>·</span>
        <span>Below zero = Short γ amplification</span>
        <span>·</span>
        <span>
          {flip !== null
            ? `Flip $${flip.toFixed(2)} (${flipDistPct !== null ? `${flipDistPct >= 0 ? '+' : ''}${(flipDistPct * 100).toFixed(2)}%` : '—'}${flipDistSigma !== null ? ` · ${flipDistSigma.toFixed(2)}σ` : ''})`
            : 'No zero-GEX crossing in ±15% grid'}
        </span>
        {expectedMove30 !== null && expectedMove30 > 0 && (
          <>
            <span>·</span>
            <span>30d ±1σ EM: ${(spot - expectedMove30).toFixed(1)}–${(spot + expectedMove30).toFixed(1)}</span>
          </>
        )}
      </div>
    </div>
  );
}
