'use client';

import React from 'react';
import type { SliceOut } from '@/components/views/volatility/types';
import { sviTotalVariance } from '@/lib/options';

/**
 * Smile slice: raw quote IVs (dots, calls hollow / puts solid) with the SVI fit overlay.
 * Residuals are visible by construction — that honesty is the point of the chart.
 */
export function VolatilitySmileChart({ slice }: { slice: SliceOut | null }) {
  if (!slice || slice.points.length < 3) {
    return (
      <div className="flex h-56 items-center justify-center rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] text-xs text-zinc-400">
        No usable quotes in this slice
      </div>
    );
  }

  const width = 720;
  const height = 260;
  const pad = { l: 36, r: 12, t: 12, b: 26 };

  const ks = slice.points.map((p) => p.k);
  const kMin = Math.min(...ks);
  const kMax = Math.max(...ks);
  const padK = Math.max(0.02, (kMax - kMin) * 0.06);

  const ivs = slice.points.map((p) => p.iv);
  const fitIvs: number[] = slice.svi
    ? Array.from({ length: 48 }, (_, i) => {
        const k = kMin - padK + ((kMax + padK - (kMin - padK)) * i) / 47;
        return Math.sqrt(Math.max(1e-12, sviTotalVariance(slice.svi as { a: number; b: number; rho: number; m: number; sigma: number }, k)) / slice.t);
      })
    : [];
  const allIv = [...ivs, ...fitIvs];
  const ivMin = Math.min(...allIv) * 0.97;
  const ivMax = Math.max(...allIv) * 1.03;
  const spanIv = Math.max(1e-6, ivMax - ivMin);

  const xOf = (k: number) => pad.l + ((k - (kMin - padK)) / (kMax + padK - (kMin - padK))) * (width - pad.l - pad.r);
  const yOf = (iv: number) => pad.t + (1 - (iv - ivMin) / spanIv) * (height - pad.t - pad.b);

  const fitPath = slice.svi
    ? Array.from({ length: 48 }, (_, i) => {
        const k = kMin - padK + ((kMax + padK - (kMin - padK)) * i) / 47;
        const iv = Math.sqrt(Math.max(1e-12, sviTotalVariance(slice.svi as { a: number; b: number; rho: number; m: number; sigma: number }, k)) / slice.t);
        return `${i === 0 ? 'M' : 'L'}${xOf(k).toFixed(2)},${yOf(iv).toFixed(2)}`;
      }).join(' ')
    : '';

  const yTicks = [ivMin, ivMin + spanIv / 2, ivMax];

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[560px]" role="img" aria-label="Volatility smile">
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

        {/* SVI fit line */}
        {fitPath && <path d={fitPath} fill="none" stroke="#DEC084" strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round" />}

        {/* raw quote dots: calls hollow gold, puts solid zinc */}
        {slice.points.map((p, i) => (
          <circle
            key={i}
            cx={xOf(p.k)}
            cy={yOf(p.iv)}
            r={p.type === 'call' ? 3.2 : 3.2}
            fill={p.type === 'call' ? 'transparent' : 'rgba(236,236,241,0.75)'}
            stroke={p.type === 'call' ? '#DEC084' : 'rgba(236,236,241,0.75)'}
            strokeWidth="0.9"
            opacity="0.9"
          >
            <title>{`${p.type.toUpperCase()} K=${p.strike}\nk=${p.k.toFixed(3)} IV=${(p.iv * 100).toFixed(2)}%\nmid=${p.mid.toFixed(2)} · spread ${(p.relSpread * 100).toFixed(1)}%`}</title>
          </circle>
        ))}

        {/* x axis labels */}
        <text x={xOf(kMin)} y={height - 8} fontSize="9" fill="#8b8b93" fontFamily="var(--font-mono)">k {kMin.toFixed(2)}</text>
        <text x={xOf(0)} y={height - 8} fontSize="9" fill="#8b8b93" fontFamily="var(--font-mono)" textAnchor="middle">0 (FWD)</text>
        <text x={xOf(kMax)} y={height - 8} fontSize="9" fill="#8b8b93" fontFamily="var(--font-mono)" textAnchor="end">k +{kMax.toFixed(2)}</text>
      </svg>

      {/* Fit diagnostics strip */}
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-[var(--hairline)] pt-2 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
        <span>SVI RMSE <strong className="text-zinc-800 dark:text-zinc-100">{slice.svi ? `${(slice.svi.rmseIv * 100).toFixed(2)} vol pts` : 'no fit'}</strong></span>
        {slice.svi && (
          <span>fit a {slice.svi.a.toFixed(4)} · b {slice.svi.b.toFixed(3)} · ρ {slice.svi.rho.toFixed(2)} · m {slice.svi.m.toFixed(3)} · σ {slice.svi.sigma.toFixed(3)}</span>
        )}
        {slice.svi && slice.svi.arbitrageFlags.length > 0 && (
          <span className="rounded border border-rose-500/40 bg-rose-500/10 px-1.5 py-0.5 text-rose-500 dark:text-rose-400">
            arbitrage flags: {slice.svi.arbitrageFlags.join(', ')}
          </span>
        )}
        <span className="text-zinc-400">○ calls · ● puts · line = SVI fit</span>
      </div>
    </div>
  );
}
