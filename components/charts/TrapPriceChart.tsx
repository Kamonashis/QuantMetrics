'use client';

import React from 'react';
import { useChartSize } from '@/lib/use-chart-size';

/**
 * Daily close line with the three dealer levels etched in:
 * call wall (gold dash), put wall (rose dash), gamma flip (zinc dotted).
 * Deliberately minimal — the point is the proximity, not the price history.
 */
export function TrapPriceChart({
  closes,
  spot,
  callWall,
  putWall,
  flip,
}: {
  closes: { date: string; close: number }[];
  spot: number;
  callWall: number | null;
  putWall: number | null;
  flip: number | null;
}) {
  const { ref, size } = useChartSize<HTMLDivElement>();
  if (closes.length < 10) {
    return (
      <div className="flex h-44 items-center justify-center rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] text-xs text-zinc-400">
        Not enough price history
      </div>
    );
  }

  const data = closes.slice(-120);
  const width = size.width;
  const height = size.height;
  const ready = width > 0;
  const pad = { l: 38, r: 26, t: 12, b: 20 };

  const levels = [callWall, putWall, flip, spot].filter((v): v is number => v !== null && isFinite(v));
  const all = data.map((d) => d.close).concat(levels);
  const yMin = Math.min(...all) * 0.985;
  const yMax = Math.max(...all) * 1.015;
  const span = Math.max(1e-6, yMax - yMin);

  const xOf = (i: number) => pad.l + (i / (data.length - 1)) * (width - pad.l - pad.r);
  const yOf = (v: number) => pad.t + (1 - (v - yMin) / span) * (height - pad.t - pad.b);

  const linePath = data.map((d, i) => `${i === 0 ? 'M' : 'L'}${xOf(i).toFixed(2)},${yOf(d.close).toFixed(2)}`).join(' ');
  const areaPath = `${linePath} L${xOf(data.length - 1).toFixed(2)},${height - pad.b} L${xOf(0).toFixed(2)},${height - pad.b} Z`;

  const levelLine = (v: number | null, color: string, label: string, dash = '5 4') => {
    if (v === null || !isFinite(v)) return null;
    const y = yOf(v);
    return (
      <g key={label}>
        <line x1={pad.l} x2={width - pad.r} y1={y} y2={y} stroke={color} strokeWidth="0.85" strokeDasharray={dash} opacity="0.85" />
        <text x={width - pad.r + 2} y={y + 3} fontSize="8" fill={color} fontFamily="var(--font-mono)" textAnchor="start">
          {label}
        </text>
      </g>
    );
  };

  return (
    <div ref={ref} className="h-[170px] sm:h-[190px] 2xl:h-[220px] w-full contain-[layout_paint]">
      {ready && (
      <svg width={width} height={height} className="block h-full w-full" role="img" aria-label="Price with dealer levels">
        <path d={areaPath} fill="rgba(222,192,132,0.07)" stroke="none" />
        {levelLine(callWall, '#DEC084', 'CW')}
        {levelLine(putWall, '#e0655f', 'PW')}
        {levelLine(flip, '#a1a1ad', 'FLIP', '2 3')}
        <path d={linePath} fill="none" stroke="#ececf1" strokeWidth="1.05" strokeLinejoin="round" strokeLinecap="round" />
        <circle cx={xOf(data.length - 1)} cy={yOf(spot)} r="3" fill="#DEC084" stroke="#0B0B0D" strokeWidth="0.9" />
        {/* y bounds */}
        <text x={pad.l - 5} y={pad.t + 6} fontSize="8.5" fill="#8b8b93" textAnchor="end" fontFamily="var(--font-mono)">
          {yMax.toFixed(1)}
        </text>
        <text x={pad.l - 5} y={height - pad.b} fontSize="8.5" fill="#8b8b93" textAnchor="end" fontFamily="var(--font-mono)">
          {yMin.toFixed(1)}
        </text>
        <text x={pad.l} y={height - 4} fontSize="8.5" fill="#8b8b93" fontFamily="var(--font-mono)">{data[0].date.slice(0, 7)}</text>
        <text x={width - pad.r} y={height - 4} fontSize="8.5" fill="#8b8b93" fontFamily="var(--font-mono)" textAnchor="end">{data[data.length - 1].date.slice(0, 7)}</text>
      </svg>
      )}
    </div>
  );
}
