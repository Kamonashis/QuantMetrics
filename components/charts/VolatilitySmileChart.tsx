'use client';

import { EmptyState } from '@/components/ui/empty-state';
import { ChartTooltip, useChartTooltip } from '@/components/ui/chart-tooltip';

import React, { useMemo } from 'react';
import type { SliceOut } from '@/components/views/volatility/types';
import { sviTotalVariance, fitLocalSmile, evalLocalSmileIv, type LocalSmileFit } from '@/lib/options';
import { AlgoSelect } from '@/components/ui/AlgoSelect';
import { useChartSize } from '@/lib/use-chart-size';

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
 * Smile slice: raw quote IVs (dots) with the chosen fit overlay.
 * Residuals are visible by construction — that honesty is the point of the chart.
 * 'svi' = server-side raw SVI fit; 'ssvi'/'spline' = refit client-side from raw quotes.
 */
export function VolatilitySmileChart({ slice, fitMethod = 'svi', onFitMethodChange }: VolatilitySmileChartProps) {
  const { ref, size } = useChartSize<HTMLDivElement>();
  const { tooltipState, showTooltip, hideTooltip } = useChartTooltip();
  // Local refit for ssvi/spline from the raw quotes in the slice
  const localFit = useMemo<LocalSmileFit | null>(() => {
    if (!slice || fitMethod === 'svi') return null;
    const pts = slice.points.map((p) => ({ k: p.k, iv: p.iv, weight: p.weight }));
    return fitLocalSmile(pts, slice.t, fitMethod);
  }, [slice, fitMethod]);

  // Effective curve writer: chosen fit → fallback to server SVI → else nothing
  const activeMethod: SmileFitMethod | 'svi-fallback' =
    fitMethod === 'svi' ? 'svi' : localFit ? fitMethod : 'svi-fallback';
  const usedViaFallback = fitMethod !== 'svi' && localFit === null;

  const fitIvAt = (k: number): number | null => {
    const T = slice!.t;
    if (activeMethod === 'spline' || activeMethod === 'ssvi') {
      return localFit ? evalLocalSmileIv(localFit, k, T) : null;
    }
    if (slice!.svi) {
      return Math.sqrt(Math.max(1e-12, sviTotalVariance(slice!.svi as { a: number; b: number; rho: number; m: number; sigma: number }, k)) / T);
    }
    return null;
  };

  const fitParamsSummary = (): string => {
    if (activeMethod === 'ssvi' && localFit?.model === 'ssvi') {
      return `ρ ${localFit.rho.toFixed(2)} · η ${localFit.eta.toFixed(2)} · γ ${localFit.gamma.toFixed(2)} · θ ${(localFit.atmTotalVariance * 100).toFixed(2)} (w₀)`;
    }
    if (activeMethod === 'spline' && localFit?.model === 'spline') {
      return `${localFit.xs.length} knots · natural cubic in w = σ²·T`;
    }
    if (slice?.svi) {
      return `a ${slice.svi.a.toFixed(4)} · b ${slice.svi.b.toFixed(3)} · ρ ${slice.svi.rho.toFixed(2)} · m ${slice.svi.m.toFixed(3)} · σ ${slice.svi.sigma.toFixed(3)}`;
    }
    return 'no fit available';
  };

  const fitRmseIv = (): string => {
    if (localFit) return `${(localFit.rmseIv * 100).toFixed(2)} vol pts`;
    if (slice?.svi) return `${(slice.svi.rmseIv * 100).toFixed(2)} vol pts`;
    return 'no fit';
  };

  if (!slice || slice.points.length < 3) {
    return (
      <EmptyState title="No usable quotes in this slice" detail="Wide or zero-bid quotes are filtered out rather than fitted" className="h-56" />
    );
  }

  const width = size.width;
  const height = size.height;
  const ready = width > 0;
  const pad = { l: 36, r: 12, t: 12, b: 26 };

  const ks = slice.points.map((p) => p.k);
  const kMin = Math.min(...ks);
  const kMax = Math.max(...ks);
  const padK = Math.max(0.02, (kMax - kMin) * 0.06);

  const ivs = slice.points.map((p) => p.iv);
  const fitIvs: number[] = [];
  for (let i = 0; i < 48; i++) {
    const k = kMin - padK + ((kMax + padK - (kMin - padK)) * i) / 47;
    const iv = fitIvAt(k);
    if (iv !== null) fitIvs.push(iv);
  }
  const allIv = [...ivs, ...fitIvs];
  const ivMin = Math.min(...allIv) * 0.97;
  const ivMax = Math.max(...allIv) * 1.03;
  const spanIv = Math.max(1e-6, ivMax - ivMin);

  const xOf = (k: number) => pad.l + ((k - (kMin - padK)) / (kMax + padK - (kMin - padK))) * (width - pad.l - pad.r);
  const yOf = (iv: number) => pad.t + (1 - (iv - ivMin) / spanIv) * (height - pad.t - pad.b);

  const fitPath = fitIvs.length === 48
    ? Array.from({ length: 48 }, (_, i) => {
        const k = kMin - padK + ((kMax + padK - (kMin - padK)) * i) / 47;
        const iv = fitIvAt(k)!;
        return `${i === 0 ? 'M' : 'L'}${xOf(k).toFixed(2)},${yOf(iv).toFixed(2)}`;
      }).join(' ')
    : '';

  const yTicks = [ivMin, ivMin + spanIv / 2, ivMax];
  const methodLabel = activeMethod === 'svi-fallback' ? 'SVI (fallback)' : FIT_LABELS[activeMethod as SmileFitMethod];

  return (
    <div>
      <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
        <AlgoSelect
          ariaLabel="Smile fitting method"
          value={fitMethod}
          onChange={(v) => onFitMethodChange?.(v as SmileFitMethod)}
          options={[
            { value: 'svi', label: 'SVI', hint: 'server fit · raw SVI' },
            { value: 'ssvi', label: 'SSVI', hint: 'surface SVI · client refit' },
            { value: 'spline', label: 'Spline', hint: 'natural cubic · client refit' },
          ]}
          className="w-56"
        />
        {usedViaFallback && (
          <span className="rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wide text-amber-500">
            {fitMethod} needs ≥5 quotes — showing SVI
          </span>
        )}
      </div>

      <div ref={ref} className="chart-frame relative">
      {ready && (
      <svg width={width} height={height} className="block h-full w-full" role="img" aria-label="Volatility smile" onMouseLeave={hideTooltip}>
        {/* gridlines */}
        {yTicks.map((iv, i) => (
          <g key={i}>
            <line x1={pad.l} x2={width - pad.r} y1={yOf(iv)} y2={yOf(iv)} stroke="rgba(128,128,138,0.18)" strokeDasharray="3 4" />
            <text x={4} y={yOf(iv) + 3} fontSize="9" fill="#8b8b93" fontFamily="var(--font-mono)">
              {(iv * 100).toFixed(1)}%
            </text>
          </g>
        ))}
        {/* zero moneyness reference */}
        <line x1={xOf(0)} x2={xOf(0)} y1={pad.t} y2={height - pad.b} stroke="rgba(222,192,132,0.35)" strokeDasharray="2 3" />

        {/* Fit line */}
        {fitPath && <path d={fitPath} fill="none" stroke="#DEC084" strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round" />}

        {/* raw quote dots */}
        {slice.points.map((p, i) => (
          <circle
            key={i}
            cx={xOf(p.k)}
            cy={yOf(p.iv)}
            r={3.2}
            fill={p.type === 'call' ? 'transparent' : 'rgba(236,236,241,0.75)'}
            stroke={p.type === 'call' ? '#DEC084' : 'rgba(236,236,241,0.75)'}
            strokeWidth="0.9"
            opacity="0.9"
            onMouseMove={(e) => showTooltip(e, [`${p.type.toUpperCase()} K=${p.strike}`, `k=${p.k.toFixed(3)} · IV=${(p.iv * 100).toFixed(2)}%`, `mid=${p.mid.toFixed(2)} · spread ${(p.relSpread * 100).toFixed(1)}%`])}
          />
        ))}

        {/* x axis labels */}
        <text x={xOf(kMin)} y={height - 8} fontSize="9" fill="#8b8b93" fontFamily="var(--font-mono)">k {kMin.toFixed(2)}</text>
        <text x={xOf(0)} y={height - 8} fontSize="9" fill="#8b8b93" fontFamily="var(--font-mono)" textAnchor="middle">0 (FWD)</text>
        <text x={xOf(kMax)} y={height - 8} fontSize="9" fill="#8b8b93" fontFamily="var(--font-mono)" textAnchor="end">k +{kMax.toFixed(2)}</text>
      </svg>
      )}
      <ChartTooltip state={tooltipState} />
      </div>

      {/* Fit diagnostics strip */}
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-[var(--hairline)] pt-2 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
        <span>{methodLabel} RMSE <strong className="text-zinc-800 dark:text-zinc-100">{fitRmseIv()}</strong></span>
        <span>{methodLabel} {fitParamsSummary()}</span>
        {slice.svi && slice.svi.arbitrageFlags.length > 0 && activeMethod === 'svi' && (
          <span className="rounded border border-rose-500/40 bg-rose-500/10 px-1.5 py-0.5 text-rose-500 dark:text-rose-400">
            arbitrage flags: {slice.svi.arbitrageFlags.join(', ')}
          </span>
        )}
        <span className="text-zinc-400">○ calls · ● puts · line = {methodLabel} fit</span>
      </div>
    </div>
  );
}
