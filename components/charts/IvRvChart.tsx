'use client';

import React from 'react';

export interface IvRvPoint {
  date: string;
  iv: number | null; // annualized IV, decimal
  rv: number | null; // annualized RV, decimal
}

/**
 * IV vs RV, dual panel: vol lines on top, IV/RV ratio below shaded red where IV < RV.
 * Both series must be aligned by date by the caller.
 */
export function IvRvChart({ points, ivLabel }: { points: IvRvPoint[]; ivLabel: string }) {
  const data = points.filter((p) => (p.iv !== null && isFinite(p.iv)) || (p.rv !== null && isFinite(p.rv)));
  if (data.length < 30) {
    return (
      <div className="flex h-44 items-center justify-center rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] text-xs text-zinc-400">
        Not enough overlapping history for the IV/RV pair
      </div>
    );
  }

  const width = 640;
  const height = 210;
  const pad = { l: 38, r: 12, t: 10, b: 22 };
  const gap = 14;
  const topH = 140;
  const botH = height - topH - gap - pad.b;

  const ivMax = Math.max(...data.map((p) => p.iv ?? 0));
  const rvMax = Math.max(...data.map((p) => p.rv ?? 0));
  const yMax = Math.max(ivMax, rvMax) * 1.12;
  const yMin = 0;

  const xOf = (i: number) => pad.l + (i / (data.length - 1)) * (width - pad.l - pad.r);
  const yTop = (v: number) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * topH;
  const yBot = (ratio: number) => {
    const rMax = 1.8;
    const rMin = 0.6;
    const c = Math.min(rMax, Math.max(rMin, ratio));
    const top = pad.t + topH + gap;
    return top + (1 - (c - rMin) / (rMax - rMin)) * botH;
  };

  const ivPath = data.map((p, i) => (p.iv === null ? null : `${i === 0 ? 'M' : 'L'}${xOf(i).toFixed(2)},${yTop(p.iv).toFixed(2)}`)).filter(Boolean).join(' ');
  const rvPath = data.map((p, i) => (p.rv === null ? null : `${i === 0 ? 'M' : 'L'}${xOf(i).toFixed(2)},${yTop(p.rv).toFixed(2)}`)).filter(Boolean).join(' ');
  const ratioPath = data.map((p, i) => {
    if (p.iv === null || p.rv === null || p.rv === 0) return null;
    return `${i === 0 ? 'M' : 'L'}${xOf(i).toFixed(2)},${yBot(p.iv / p.rv).toFixed(2)}`;
  }).filter(Boolean).join(' ');

  // Red shading where ratio < 1 (IV < RV)
  const belowOne: string[] = [];
  let bucketStart: number | null = null;
  data.forEach((p, i) => {
    const below = p.iv !== null && p.rv !== null && p.rv > 0 && p.iv / p.rv < 1;
    if (below && bucketStart === null) bucketStart = i;
    if (!below && bucketStart !== null) {
      belowOne.push(`${xOf(bucketStart).toFixed(2)},${xOf(i - 1).toFixed(2)}`);
      bucketStart = null;
    }
  });
  if (bucketStart !== null) belowOne.push(`${xOf(bucketStart).toFixed(2)},${xOf(data.length - 1).toFixed(2)}`);

  const top = pad.t + topH + gap;

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[520px]" role="img" aria-label="Implied versus realized volatility">
        {/* top panel gridlines */}
        {[0.5, 1].map((f, i) => (
          <g key={i}>
            <line x1={pad.l} x2={width - pad.r} y1={pad.t + topH * f} y2={pad.t + topH * f} stroke="rgba(128,128,138,0.14)" strokeDasharray="3 4" />
          </g>
        ))}
        {/* bottom panel ratio = 1 line */}
        <line x1={pad.l} x2={width - pad.r} y1={yBot(1)} y2={yBot(1)} stroke="rgba(128,128,138,0.35)" strokeDasharray="4 3" />

        {/* red shading for IV < RV regimes */}
        {belowOne.map((seg, i) => {
          const [x1, x2] = seg.split(',').map(Number);
          return <rect key={i} x={x1} y={top} width={Math.max(1, x2 - x1)} height={botH} fill="rgba(224,101,95,0.12)" />;
        })}

        <path d={ivPath} fill="none" stroke="#DEC084" strokeWidth="1.8" strokeLinejoin="round" />
        <path d={rvPath} fill="none" stroke="#7dd3fc" strokeWidth="1.5" strokeLinejoin="round" strokeDasharray="4 2.5" opacity="0.9" />
        <path d={ratioPath} fill="none" stroke="rgba(236,236,241,0.6)" strokeWidth="1.2" />

        {/* labels */}
        <text x={pad.l - 6} y={pad.t + 8} fontSize="8.5" fill="#DEC084" textAnchor="end" fontFamily="var(--font-mono)">IV {(yMax * 100).toFixed(0)}%</text>
        <text x={pad.l - 6} y={top + 8} fontSize="8.5" fill="#8b8b93" textAnchor="end" fontFamily="var(--font-mono)">ratio 1.8</text>
        <text x={pad.l - 6} y={height - pad.b} fontSize="8.5" fill="#8b8b93" textAnchor="end" fontFamily="var(--font-mono)">0.6</text>
        <text x={pad.l} y={height - 4} fontSize="8.5" fill="#8b8b93" fontFamily="var(--font-mono)">{data[0].date.slice(0, 7)}</text>
        <text x={width - pad.r} y={height - 4} fontSize="8.5" fill="#8b8b93" fontFamily="var(--font-mono)" textAnchor="end">{data[data.length - 1].date.slice(0, 7)}</text>
      </svg>
      <div className="mt-1.5 flex flex-wrap items-center gap-4 font-mono text-[9px] text-zinc-400">
        <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 rounded bg-[#DEC084]" /> {ivLabel}</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-0 w-4 border-t border-dashed border-[#7dd3fc]" /> RV 20d (ticker)</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-3 bg-rose-500/30" /> IV &lt; RV (underpriced risk)</span>
      </div>
    </div>
  );
}
