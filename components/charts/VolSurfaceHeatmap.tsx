'use client';

import React from 'react';
import type { SliceOut } from '@/components/views/volatility/types';

/**
 * Implied-vol surface heatmap: expiries (rows) × log-moneyness k buckets (cols),
 * cell colour ∝ IV. Gaps stay empty — the surface is only as real as the quotes.
 */
export function VolSurfaceHeatmap({ slices }: { slices: SliceOut[] }) {
  // Moneyness buckets of width 0.1 from -0.45 to +0.45
  const buckets: { k0: number; label: string }[] = [];
  for (let k0 = -0.45; k0 < 0.45; k0 += 0.1) {
    buckets.push({ k0: Number(k0.toFixed(1)), label: k0.toFixed(1) });
  }

  const grid = slices.map((slice) => {
    const cells = buckets.map(({ k0 }) => {
      const inBucket = slice.points.filter((p) => p.k >= k0 && p.k < k0 + 0.1);
      if (inBucket.length === 0) return { iv: null as number | null, n: 0 };
      const avg = inBucket.reduce((s, p) => s + p.iv, 0) / inBucket.length;
      return { iv: avg, n: inBucket.length };
    });
    return { slice, cells };
  });

  // Colour scale from all observed IVs (min → max)
  const allIv = slices.flatMap((s) => s.points.map((p) => p.iv));
  const minIv = allIv.length ? Math.min(...allIv) : 0;
  const maxIv = allIv.length ? Math.max(...allIv) : 1;
  const span = Math.max(1e-6, maxIv - minIv);

  const cellStyle = (iv: number | null): React.CSSProperties => {
    if (iv === null) return {};
    const t = (iv - minIv) / span; // 0..1
    const a = 0.1 + t * 0.9;
    return { background: `rgba(222, 192, 132, ${a.toFixed(3)})` };
  };

  return (
    <div className="overflow-x-auto rounded-xl border border-white/[0.07] bg-[#0F0F12] p-3">
      <div className="min-w-[760px]">
        {/* Header row: bucket labels */}
        <div className="grid" style={{ gridTemplateColumns: `72px repeat(${buckets.length}, 1fr)` }}>
          <div className="px-2 py-1.5 font-mono text-[9px] uppercase tracking-wider text-zinc-500">
            DTE ↓ · k →
          </div>
          {buckets.map((b) => (
            <div key={b.label} className="px-1 py-1.5 text-center font-mono text-[9px] text-zinc-500">
              {b.label}
            </div>
          ))}
        </div>

        {/* Surface rows */}
        {grid.map(({ slice, cells }) => (
          <div
            key={slice.expirationDate}
            className="grid items-stretch border-t border-white/[0.06]"
            style={{ gridTemplateColumns: `72px repeat(${buckets.length}, 1fr)` }}
          >
            <div className="flex flex-col justify-center px-2 py-1">
              <span className="font-mono text-[10px] font-semibold text-zinc-300">
                {slice.dte}d
              </span>
              <span className="font-mono text-[8px] text-zinc-600">{slice.expirationDate.slice(5)}</span>
            </div>
            {cells.map((cell, i) => (
              <div
                key={i}
                title={cell.iv !== null ? `k ∈ [${buckets[i].label}, ${(buckets[i].k0 + 0.1).toFixed(1)})\nIV ${(cell.iv * 100).toFixed(1)}% · n=${cell.n}` : 'no quotes'}
                className="flex h-9 items-center justify-center font-mono text-[9px] tabular-nums text-[#141310] transition-transform duration-150 hover:scale-[1.08] hover:z-10"
                style={cellStyle(cell.iv)}
              >
                {cell.iv !== null ? (cell.iv * 100).toFixed(0) : ''}
              </div>
            ))}
          </div>
        ))}

        {/* Scale legend */}
        <div className="mt-3 flex items-center gap-2 font-mono text-[9px] text-zinc-500">
          <span>IV {(minIv * 100).toFixed(0)}%</span>
          <div className="h-2.5 flex-1 rounded-full" style={{ background: 'linear-gradient(90deg, rgba(222,192,132,0.1), rgba(222,192,132,0.55), rgba(222,192,132,1))' }} />
          <span>{(maxIv * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
}
