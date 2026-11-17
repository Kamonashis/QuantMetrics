'use client';

import React, { useMemo } from 'react';
import type { MarketRnd } from '@/lib/rnd';

/**
 * Cumulative distribution of the market-price risk-neutral density —
 * the CDF step of the Breeden–Litzenberger pipeline (density → CDT by
 * cumulative trapezoid). x-axis re-expressed as log-return k = ln(K/S₀).
 */
export function RndCdfChart({ rnd }: { rnd: MarketRnd }) {
  const width = 680;
  const height = 150;
  const pad = { l: 38, r: 12, t: 10, b: 22 };

  const pts = useMemo(
    () => rnd.strikes.map((K, i) => ({ k: Math.log(K / rnd.spot), p: rnd.cdf[i] })),
    [rnd]
  );

  const kMin = pts[0]?.k ?? -0.3;
  const kMax = pts[pts.length - 1]?.k ?? 0.3;
  const spanK = Math.max(1e-6, kMax - kMin);
  const xPix = (k: number) => pad.l + ((k - kMin) / spanK) * (width - pad.l - pad.r);
  const yPix = (p: number) => pad.t + (1 - p) * (height - pad.t - pad.b);

  const path = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${xPix(p.k).toFixed(2)},${yPix(p.p).toFixed(2)}`).join(' ');

  const cdfAt = (K: number): number => {
    const k = Math.log(K / rnd.spot);
    if (k <= pts[0].k) return pts[0].p;
    if (k >= pts[pts.length - 1].k) return pts[pts.length - 1].p;
    for (let i = 1; i < pts.length; i++) {
      if (k <= pts[i].k) {
        const f = (k - pts[i - 1].k) / Math.max(1e-12, pts[i].k - pts[i - 1].k);
        return pts[i - 1].p + f * (pts[i].p - pts[i - 1].p);
      }
    }
    return 1;
  };

  const tail = {
    deep: cdfAt(rnd.spot * 0.8),
    ten: cdfAt(rnd.spot * 0.9),
    spot: cdfAt(rnd.spot),
    up10: cdfAt(rnd.spot * 1.1),
    up20: cdfAt(rnd.spot * 1.2),
  };

  const marker = (k: number, label: string) => (
    <g key={label}>
      <line x1={xPix(k)} x2={xPix(k)} y1={pad.t} y2={height - pad.b} stroke="rgba(236,236,241,0.12)" strokeDasharray="2 3" />
      <text x={xPix(k)} y={pad.t + 8} fontSize="8" fill="#8b8b93" fontFamily="var(--font-mono)" textAnchor="middle">
        {label}
      </text>
    </g>
  );

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[560px]" role="img" aria-label="Risk-neutral cumulative distribution function">
        {marker(Math.log(0.8), '−20%')}
        {marker(Math.log(0.9), '−10%')}
        {marker(0, 'spot')}
        {marker(Math.log(1.1), '+10%')}
        {marker(Math.log(1.2), '+20%')}
        {/* 50% line */}
        <line x1={pad.l} x2={width - pad.r} y1={yPix(0.5)} y2={yPix(0.5)} stroke="rgba(236,236,241,0.18)" strokeDasharray="2 3" />
        <text x={pad.l - 4} y={yPix(0.5) + 3} fontSize="8" fill="#8b8b93" fontFamily="var(--font-mono)" textAnchor="end">50%</text>
        <path
          d={`${path} L${xPix(pts[pts.length - 1].k).toFixed(2)},${yPix(0).toFixed(2)} L${xPix(pts[0].k).toFixed(2)},${yPix(0).toFixed(2)} Z`}
          fill="rgba(222,192,132,0.10)"
          stroke="none"
        />
        <path d={path} fill="none" stroke="#DEC084" strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round" />
        {/* y ticks */}
        {[0, 1].map((p) => (
          <text key={p} x={pad.l - 4} y={yPix(p) + 3} fontSize="8" fill="#8b8b93" fontFamily="var(--font-mono)" textAnchor="end">
            {p === 0 ? '0' : '1'}
          </text>
        ))}
      </svg>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[9px] text-zinc-400">
        <span>CDF of the market-price RND — trapezoid d(Σ density)</span>
        <span className="tabular-nums">P(&lt;0.8S) {(tail.deep * 100).toFixed(1)}%</span>
        <span className="tabular-nums">P(&lt;0.9S) {(tail.ten * 100).toFixed(1)}%</span>
        <span className="tabular-nums">P(&lt;S) {(tail.spot * 100).toFixed(1)}%</span>
        <span className="tabular-nums">P(&lt;1.1S) {(tail.up10 * 100).toFixed(1)}%</span>
        <span className="tabular-nums">P(&lt;1.2S) {(tail.up20 * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}
