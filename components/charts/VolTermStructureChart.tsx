'use client';

import React from 'react';
import { useChartSize } from '@/lib/use-chart-size';

export interface TermStructurePoint {
  dte: number;
  iv: number | null;
}

/**
 * IV term structure. Can plot on the vol axis (σ) or the total-variance axis
 * (w = σ²T) — the quantity that is actually additive across maturities and a
 * variance swap prices. Reference band = 1y 10/90 percentile of ^VIX (or closest
 * available index series), labelled as reference only.
 */
export function VolTermStructureChart({
  points,
  axisMode,
  band,
  iv30Marker,
}: {
  points: TermStructurePoint[];
  axisMode: 'vol' | 'totalVariance';
  band?: { low: number; high: number; label: string } | null;
  iv30Marker?: number | null;
}) {
  const { ref, size } = useChartSize<HTMLDivElement>();
  const valid = points.filter((p): p is { dte: number; iv: number } => p.iv !== null && isFinite(p.iv) && p.iv > 0);
  if (valid.length < 2) {
    return (
      <div className="flex h-44 items-center justify-center rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] text-xs text-zinc-400">
        Not enough expiries to draw the curve
      </div>
    );
  }

  const width = size.width;
  const height = size.height;
  const ready = width > 0;
  const pad = { l: 40, r: 12, t: 14, b: 26 };

  const yOf = (p: { dte: number; iv: number }) =>
    axisMode === 'vol' ? p.iv * 100 : p.iv * p.iv * (p.dte / 365) * 1e4;

  const years = valid.map((p) => p.dte);
  const xMin = Math.min(...years);
  const xMax = Math.max(...years);
  const spanX = Math.max(1, xMax - xMin);

  const ys = valid.map(yOf);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (band) {
    if (axisMode === 'vol') {
      yMin = Math.min(yMin, band.low);
      yMax = Math.max(yMax, band.high);
    }
  }
  const spanY = Math.max(1e-6, yMax - yMin) * 1.15;
  yMin -= spanY * 0.08;
  yMax += spanY * 0.08;

  const xOf = (dte: number) => pad.l + ((dte - xMin) / spanX) * (width - pad.l - pad.r);
  const yPix = (v: number) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * (height - pad.t - pad.b);

  const linePath = valid
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${xOf(p.dte).toFixed(2)},${yPix(yOf(p)).toFixed(2)}`)
    .join(' ');

  const bandPix =
    band && axisMode === 'vol'
      ? { y1: yPix(band.high), y2: yPix(band.low) }
      : null;

  return (
    <div>
      <div ref={ref} className="chart-frame">
      {ready && (
      <svg width={width} height={height} className="block h-full w-full" role="img" aria-label="IV term structure">
        {/* reference band */}
        {bandPix && (
          <rect x={pad.l} y={bandPix.y1} width={width - pad.l - pad.r} height={Math.max(1, bandPix.y2 - bandPix.y1)} fill="rgba(222,192,132,0.07)" />
        )}
        {bandPix && (
          <>
            <line x1={pad.l} x2={width - pad.r} y1={bandPix.y1} y2={bandPix.y1} stroke="rgba(222,192,132,0.35)" strokeDasharray="3 4" />
            <line x1={pad.l} x2={width - pad.r} y1={bandPix.y2} y2={bandPix.y2} stroke="rgba(222,192,132,0.35)" strokeDasharray="3 4" />
          </>
        )}

        {/* IV30 vertical marker */}
        {iv30Marker && (
          <line
            x1={xOf(30)} x2={xOf(30)} y1={pad.t} y2={height - pad.b}
            stroke="rgba(236,236,241,0.18)" strokeDasharray="2 4"
          />
        )}

        <path d={linePath} fill="none" stroke="#DEC084" strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round" />
        {valid.map((p) => (
          <circle key={p.dte} cx={xOf(p.dte)} cy={yPix(yOf(p))} r="3.4" fill="#0B0B0D" stroke="#DEC084" strokeWidth="1.05">
            <title>{`${p.dte}d: ${(p.iv * 100).toFixed(2)}% IV${axisMode === 'totalVariance' ? ` · w=${(p.iv * p.iv * (p.dte / 365) * 1e4).toFixed(1)} bp` : ''}`}</title>
          </circle>
        ))}

        {/* axes */}
        <text x={pad.l - 6} y={pad.t + 4} fontSize="8.5" fill="#8b8b93" textAnchor="end" fontFamily="var(--font-mono)">
          {axisMode === 'vol' ? (yMax).toFixed(0) + '%' : (yMax).toFixed(0) + 'bp'}
        </text>
        <text x={pad.l - 6} y={height - pad.b} fontSize="8.5" fill="#8b8b93" textAnchor="end" fontFamily="var(--font-mono)">
          {axisMode === 'vol' ? (yMin).toFixed(0) + '%' : (yMin).toFixed(0)}
        </text>
        {valid.filter((_, i) => i % 2 === 0).map((p) => (
          <text key={p.dte} x={xOf(p.dte)} y={height - 8} fontSize="8.5" fill="#8b8b93" textAnchor="middle" fontFamily="var(--font-mono)">
            {p.dte}d
          </text>
        ))}
      </svg>
      )}
      </div>
      {band && axisMode === 'vol' && (
        <p className="mt-1.5 font-mono text-[9px] text-zinc-400">
          dashed band = {band.label} · full curve labelled; markers = chain ATM IV
        </p>
      )}
    </div>
  );
}
