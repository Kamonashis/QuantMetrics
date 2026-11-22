'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { EmptyState } from '@/components/ui/empty-state';
import type { SliceOut } from '@/components/views/volatility/types';
import {
  sviTotalVariance,
  fitLocalSmile,
  evalLocalSmileIv,
  type LocalSmileFit,
} from '@/lib/options';
import { AlgoSelect } from '@/components/ui/AlgoSelect';
import { PlotlyChart } from '@/components/charts/PlotlyChart';
import { QUANT_COLORS } from '@/lib/plotly-theme';

export type SmileFitMethod = 'svi' | 'ssvi' | 'spline';

interface VolatilitySmileChartProps {
  slice: SliceOut | null;
  fitMethod?: SmileFitMethod;
  onFitMethodChange?: (method: SmileFitMethod) => void;
}

const FIT_LABELS: Record<SmileFitMethod, string> = {
  svi: 'SVI',
  ssvi: 'SSVI',
  spline: 'Spline',
};

/**
 * Interactive Plotly smile slice: raw quote IVs (calls & puts) with SVI / SSVI / Natural Spline fit overlay.
 */
export function VolatilitySmileChart({
  slice,
  fitMethod = 'svi',
  onFitMethodChange,
}: VolatilitySmileChartProps) {
  const localFit = useMemo<LocalSmileFit | null>(() => {
    if (!slice || fitMethod === 'svi') return null;
    const pts = slice.points.map((p) => ({ k: p.k, iv: p.iv, weight: p.weight }));
    return fitLocalSmile(pts, slice.t, fitMethod);
  }, [slice, fitMethod]);

  const activeMethod: SmileFitMethod | 'svi-fallback' =
    fitMethod === 'svi' ? 'svi' : localFit ? fitMethod : 'svi-fallback';
  const usedViaFallback = fitMethod !== 'svi' && localFit === null;

  const { traces, layout, paramsSummary, rmseLabel, methodLabel } = useMemo(() => {
    if (!slice || slice.points.length < 3) {
      return {
        traces: [] as Data[],
        layout: {} as Partial<Layout>,
        paramsSummary: 'no fit available',
        rmseLabel: 'no fit',
        methodLabel: 'SVI',
      };
    }

    const T = slice.t;
    const fitIvAt = (k: number): number | null => {
      if (activeMethod === 'spline' || activeMethod === 'ssvi') {
        return localFit ? evalLocalSmileIv(localFit, k, T) : null;
      }
      if (slice.svi) {
        return Math.sqrt(
          Math.max(
            1e-12,
            sviTotalVariance(
              slice.svi as { a: number; b: number; rho: number; m: number; sigma: number },
              k
            )
          ) / T
        );
      }
      return null;
    };

    const ks = slice.points.map((p) => p.k);
    const kMin = Math.min(...ks);
    const kMax = Math.max(...ks);
    const padK = Math.max(0.02, (kMax - kMin) * 0.06);

    const calls = slice.points.filter((p) => p.type === 'call');
    const puts = slice.points.filter((p) => p.type === 'put');

    const curveK: number[] = [];
    const curveIvPct: number[] = [];
    for (let i = 0; i < 80; i++) {
      const k = kMin - padK + ((kMax + padK - (kMin - padK)) * i) / 79;
      const iv = fitIvAt(k);
      if (iv !== null && Number.isFinite(iv)) {
        curveK.push(k);
        curveIvPct.push(iv * 100);
      }
    }

    const mLabel =
      activeMethod === 'svi-fallback'
        ? 'SVI (fallback)'
        : FIT_LABELS[activeMethod as SmileFitMethod];

    const list: Data[] = [];

    if (puts.length > 0) {
      list.push({
        type: 'scatter',
        mode: 'markers',
        name: 'Put Quotes',
        x: puts.map((p) => p.k),
        y: puts.map((p) => p.iv * 100),
        customdata: puts.map((p) => [p.strike, p.mid, p.relSpread * 100]),
        marker: {
          size: 6,
          color: QUANT_COLORS.negative,
          opacity: 0.82,
        },
        hovertemplate:
          '<b>Put K $%{customdata[0]:.2f}</b><br>k = %{x:+.3f} · IV %{y:.2f}%<br>Mid $%{customdata[1]:.2f} · Spread %{customdata[2]:.1f}%<extra></extra>',
      });
    }

    if (calls.length > 0) {
      list.push({
        type: 'scatter',
        mode: 'markers',
        name: 'Call Quotes',
        x: calls.map((p) => p.k),
        y: calls.map((p) => p.iv * 100),
        customdata: calls.map((p) => [p.strike, p.mid, p.relSpread * 100]),
        marker: {
          size: 6,
          color: QUANT_COLORS.positive,
          opacity: 0.82,
        },
        hovertemplate:
          '<b>Call K $%{customdata[0]:.2f}</b><br>k = %{x:+.3f} · IV %{y:.2f}%<br>Mid $%{customdata[1]:.2f} · Spread %{customdata[2]:.1f}%<extra></extra>',
      });
    }

    if (curveK.length >= 2) {
      list.push({
        type: 'scatter',
        mode: 'lines',
        name: `${mLabel} Fit`,
        x: curveK,
        y: curveIvPct,
        line: { color: QUANT_COLORS.goldLight, width: 2.3 },
        hovertemplate: `${mLabel} Fit: %{y:.2f}% (k=%{x:+.3f})<extra></extra>`,
      });
    }

    let summary = 'no fit available';
    if (activeMethod === 'ssvi' && localFit?.model === 'ssvi') {
      summary = `ρ ${localFit.rho.toFixed(2)} · η ${localFit.eta.toFixed(2)} · γ ${localFit.gamma.toFixed(2)} · θ ${(localFit.atmTotalVariance * 100).toFixed(2)} (w₀)`;
    } else if (activeMethod === 'spline' && localFit?.model === 'spline') {
      summary = `${localFit.xs.length} knots · natural cubic in w = σ²·T`;
    } else if (slice.svi) {
      summary = `a ${slice.svi.a.toFixed(4)} · b ${slice.svi.b.toFixed(3)} · ρ ${slice.svi.rho.toFixed(2)} · m ${slice.svi.m.toFixed(3)} · σ ${slice.svi.sigma.toFixed(3)}`;
    }

    const rmse = localFit
      ? `${(localFit.rmseIv * 100).toFixed(2)} vol pts`
      : slice.svi
        ? `${(slice.svi.rmseIv * 100).toFixed(2)} vol pts`
        : 'no fit';

    const chartLayout: Partial<Layout> = {
      shapes: [
        {
          type: 'line',
          xref: 'x',
          yref: 'paper',
          x0: 0,
          x1: 0,
          y0: 0,
          y1: 1,
          line: { color: 'rgba(222, 192, 132, 0.45)', width: 1, dash: 'dot' },
        },
      ],
      annotations: [
        {
          x: 0,
          y: 0.98,
          xref: 'x',
          yref: 'paper',
          text: 'ATM (k=0)',
          showarrow: false,
          font: { size: 9.5, color: QUANT_COLORS.goldLight },
          xanchor: 'left',
          xshift: 4,
        },
      ],
    };

    return {
      traces: list,
      layout: chartLayout,
      paramsSummary: summary,
      rmseLabel: rmse,
      methodLabel: mLabel,
    };
  }, [slice, activeMethod, localFit]);

  if (!slice || slice.points.length < 3) {
    return (
      <EmptyState
        title="No usable quotes in this slice"
        detail="Wide or zero-bid quotes are filtered out rather than fitted"
        className="h-56"
      />
    );
  }

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="font-mono text-[10.5px] text-zinc-500 dark:text-zinc-400">
          {slice.expirationDate} ({slice.dte}d) · F ${slice.forward.toFixed(2)} · {methodLabel}{' '}
          RMSE {rmseLabel}
          {usedViaFallback && (
            <span className="ml-1.5 text-amber-600 dark:text-amber-400">
              (fell back to SVI)
            </span>
          )}
        </div>
        {onFitMethodChange && (
          <AlgoSelect
            ariaLabel="Smile fit method"
            value={fitMethod}
            onChange={(v) => onFitMethodChange(v as SmileFitMethod)}
            options={[
              { value: 'svi', label: 'SVI' },
              { value: 'ssvi', label: 'SSVI' },
              { value: 'spline', label: 'Spline' },
            ]}
          />
        )}
      </div>

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="closest"
        showLegend
        xAxisTitle="Log-Moneyness k = ln(K/F)"
        yAxisTitle="Implied Vol (%)"
        yTickSuffix="%"
        height={300}
        uirevisionKey={`smile-${slice.expirationDate}-${fitMethod}`}
        exportFilename={`vol-smile-${slice.expirationDate}`}
        showWheelHint
      />

      <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
        <span>{paramsSummary}</span>
        <span>
          {slice.clean.used}/{slice.clean.total} quotes · med spread{' '}
          {slice.clean.medianRelSpread !== null
            ? `${(slice.clean.medianRelSpread * 100).toFixed(1)}%`
            : '—'}
        </span>
      </div>
    </div>
  );
}
