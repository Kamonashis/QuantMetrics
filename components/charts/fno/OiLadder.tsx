'use client';

import React from 'react';

/**
 * Dual-sided OI ladder around spot: calls left-of-center in gold, puts right-of-center
 * in rose, spot as a vertical rule. Bars scaled to the largest single-strike OI.
 */
export function OiLadder({
  rows,
  spot,
}: {
  rows: { strike: number; callOi: number; putOi: number; totalOi: number }[];
  spot: number;
}) {
  if (rows.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] text-xs text-zinc-400">
        No strikes in ±15% band
      </div>
    );
  }

  const maxOi = Math.max(1, ...rows.flatMap((r) => [r.callOi, r.putOi]));
  const width = 640;
  const rowH = 13;
  const padL = 46;
  const padR = 46;
  const height = rows.length * rowH + 30;
  const mid = padL + (width - padL - padR) / 2;
  const half = (width - padL - padR) / 2;

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[440px]" role="img" aria-label="Dual-sided OI ladder">
        <line x1={mid} x2={mid} y1={2} y2={height - 16} stroke="rgba(236,236,241,0.3)" strokeWidth="1" strokeDasharray="3 3" />
        {rows.map((r, i) => {
          const y = i * rowH + 6;
          const callW = (r.callOi / maxOi) * half;
          const putW = (r.putOi / maxOi) * half;
          const near = Math.abs(r.strike - spot) / spot < 0.025;
          return (
            <g key={r.strike}>
              <rect x={mid - callW} y={y + 1} width={callW} height={rowH - 2} rx="1.5" fill={near ? 'rgba(222,192,132,0.95)' : 'rgba(222,192,132,0.55)'}>
                <title>{`${r.strike} call OI ${r.callOi.toLocaleString()}`}</title>
              </rect>
              <rect x={mid} y={y + 1} width={putW} height={rowH - 2} rx="1.5" fill={near ? 'rgba(224,101,95,0.95)' : 'rgba(224,101,95,0.55)'}>
                <title>{`${r.strike} put OI ${r.putOi.toLocaleString()}`}</title>
              </rect>
              <text x={padL - 5} y={y + rowH / 2 + 1} textAnchor="end" fontSize="8" fill={near ? '#DEC084' : '#8b8b93'} fontFamily="var(--font-mono)">
                {r.strike}
              </text>
            </g>
          );
        })}
        <text x={width / 2} y={height - 4} fontSize="8" fill="#8b8b93" textAnchor="middle" fontFamily="var(--font-mono)">
          gold = calls · rose = puts · dashed = spot {spot.toFixed(2)}
        </text>
      </svg>
    </div>
  );
}
